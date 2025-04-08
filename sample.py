import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import time
import json
import os

# OCC imports
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax2, gp_Vec, gp_Trsf, gp_Ax1, gp_Pnt2d, gp_Circ
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace, BRepBuilderAPI_Transform, BRepBuilderAPI_MakeEdge
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Core.TopoDS import TopoDS_Wire
from OCC.Core.GCE2d import GCE2d_MakeCircle
from OCC.Core.gp import gp_Circ2d, gp_Ax2d
from OCC.Display.SimpleGui import init_display
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
import copy

# ================== CONSTANTS AND PARAMETERS ===================
# General parameters
RANDOM_SEED = 42                   # Seed for reproducibility
DEFAULT_ERROR_THRESHOLD = 0.001    # Default error threshold for optimization termination
DEFAULT_MAX_ITERATIONS = 100       # Default maximum number of iterations for optimization

# Point cloud parameters
DEFAULT_POINTS_PER_CLOUD = 1000    # Default number of points to generate for point clouds
MIN_POINTS_FOR_CORRESPONDENCE = 10 # Minimum number of points needed for correspondence

# Normals estimation parameters
NORMALS_RADIUS = 0.15              # Radius for normal estimation
NORMALS_MAX_NN = 50                # Maximum nearest neighbors for normal estimation

# ICP parameters
ICP_MAX_CORRESPONDENCE_DISTANCE = 0.2  # Default max correspondence distance for ICP
ICP_RELATIVE_FITNESS = 1e-6            # Convergence criteria - relative fitness
ICP_RELATIVE_RMSE = 1e-6               # Convergence criteria - relative RMSE
ICP_MAX_ITERATION = 50                 # Maximum iterations for ICP

# Optimization parameters
INITIAL_DIST_THRESHOLD = 0.3       # Initial distance threshold for correspondence matching
POINT_WEIGHT_MIN = 0.3             # Minimum weight for points 
POINT_WEIGHT_SCALE = 0.7           # Scale factor for point weights
SEGMENT_WEIGHT_MIN = 0.2           # Minimum weight for segments
SEGMENT_WEIGHT_SCALE = 0.8         # Scale factor for segment weights
DELTA_THETA_SCALE = 0.4            # Scale factor for joint angle updates

# First iteration specific parameters
FIRST_ITER_CLOSEST_PERCENTAGE = 0.05   # Percentage of closest points to use in first iteration
FIRST_ITER_DAMPING = 1.0               # Damping factor for first iteration
FIRST_ITER_CONSTRAINT_WEIGHT = 0.1     # Constraint weight for first iteration
FIRST_ITER_CONTINUITY_WEIGHT = 0.2     # Continuity weight for first iteration
FIRST_ITER_MAX_ANGLE_MAIN = np.pi/15   # Maximum angle (main axis) for first iteration
FIRST_ITER_MAX_ANGLE_OTHER = np.pi/20  # Maximum angle (other axes) for first iteration

# Later iterations parameters
LATER_ITER_CONSTRAINT_WEIGHT = 0.02    # Constraint weight for later iterations
LATER_ITER_CONTINUITY_WEIGHT = 0.05    # Continuity weight for later iterations
LATER_ITER_MAX_ANGLE_MAIN = np.pi/6    # Base maximum angle (main axis) for later iterations
LATER_ITER_MAX_ANGLE_OTHER = np.pi/12  # Base maximum angle (other axes) for later iterations

# Numerical stability parameters
EPSILON = 1e-6                     # Small value to prevent division by zero
MIN_DIRECTION_NORM = 1e-10         # Minimum norm for direction vectors

# H-section proportions (when using single size value)
H_HEIGHT_RATIO = 1.8               # Web height as multiple of base size
H_WEB_THICKNESS_RATIO = 0.2        # Web thickness as fraction of base size
H_FLANGE_THICKNESS_RATIO = 0.25    # Flange thickness as fraction of base size

# Visualization parameters
ORIGINAL_COLOR = [0.7, 0.7, 0.7]   # Color for original column (gray)
DEFORMED_COLOR = [0, 0, 1]         # Color for deformed column (blue)
CURRENT_COLOR = [1, 0, 0]          # Color for current optimization state (red)
FINAL_COLOR = [0, 1, 0]            # Color for final result (green)

# ================== ARTICULATED COLUMN MODEL ===================
class ArticulatedSegment:
    def __init__(self, length, size, parent=None, cross_section='circular'):
        self.length = length
        self.size = size #radius for circular or a tuple (width, height) for rectangular
        self.cross_section = cross_section
        self.parent = parent
        self.children = []
        self.local_rotation = np.zeros(3)  # x, y, z angles in radians, Local rotation relative to parent
        
        # World space properties
        self.start_pos = np.array([0., 0., 0.])
        self.end_pos = np.array([0., 0., self.length])
        self.orientation = np.eye(3)  # 3x3 rotation matrix
        
        if parent:
            parent.children.append(self)
            self.start_pos = parent.end_pos.copy()
            self.end_pos = self.start_pos + np.array([0., 0., self.length])
            self.orientation = parent.orientation.copy() #rotations are later applied

    def update_transform(self):
        """Update segment's transform based on local rotation"""
        # Create rotation matrices for each axis
        Rx = np.array([[1, 0, 0], [0, np.cos(self.local_rotation[0]), -np.sin(self.local_rotation[0])],  [0, np.sin(self.local_rotation[0]), np.cos(self.local_rotation[0])]])
        Ry = np.array([[np.cos(self.local_rotation[1]), 0, np.sin(self.local_rotation[1])], [0, 1, 0], [-np.sin(self.local_rotation[1]), 0, np.cos(self.local_rotation[1])]])
        Rz = np.array([[np.cos(self.local_rotation[2]), -np.sin(self.local_rotation[2]), 0], [np.sin(self.local_rotation[2]), np.cos(self.local_rotation[2]), 0], [0, 0, 1]])
        
        # Combine rotations and update orientation
        local_orientation = Rz @ Ry @ Rx
        self.orientation = self.parent.orientation @ local_orientation if self.parent else local_orientation
        
        # Update positions
        if self.parent:
            self.start_pos = self.parent.end_pos.copy()
        
        # Calculate new end position
        self.end_pos = self.start_pos + self.orientation @ np.array([0., 0., self.length])
        
        # Update all children
        for child in self.children:
            child.update_transform()
    
    def rotate(self, axis, angle):
        """Rotate segment around specified axis"""
        if axis in ['x', 'y', 'z']:
            axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
            self.local_rotation[axis_idx] = angle
            self.update_transform()

    def create_circular_base(self):
        """Create a circular base for the prism using direct 3D methods"""
        radius = self.size if isinstance(self.size, (int, float)) else self.size[0]
        circle = gp_Circ(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), radius)   # Create a 3D circle directly
        edge = BRepBuilderAPI_MakeEdge(circle).Edge() # Create an edge from the circle
        wire = BRepBuilderAPI_MakeWire(edge).Wire() # Create a wire and face
        face = BRepBuilderAPI_MakeFace(wire).Face()
        return face
    
    def create_rectangular_base(self):
        width, height = self.size if isinstance(self.size, tuple) else (self.size, self.size)
        wire_builder = BRepBuilderAPI_MakeWire() # Create a wire in the shape of a rectangle
        # Define the four corners of the rectangle
        points = [gp_Pnt(-width/2, -height/2, 0), gp_Pnt(width/2, -height/2, 0), gp_Pnt(width/2, height/2, 0), gp_Pnt(-width/2, height/2, 0)]
        # Create edges for the wire
        edge1 = BRepBuilderAPI_MakeEdge(points[0], points[1]).Edge()
        edge2 = BRepBuilderAPI_MakeEdge(points[1], points[2]).Edge()
        edge3 = BRepBuilderAPI_MakeEdge(points[2], points[3]).Edge()
        edge4 = BRepBuilderAPI_MakeEdge(points[3], points[0]).Edge()
        wire_builder.Add(edge1)
        wire_builder.Add(edge2)
        wire_builder.Add(edge3)
        wire_builder.Add(edge4)
        wire = wire_builder.Wire()
        face = BRepBuilderAPI_MakeFace(wire).Face()
        return face
    
    def create_h_shaped_base(self):
        # Get dimensions from the size parameter
        # If size is a tuple, use it as (flange_width, web_height, web_thickness, flange_thickness)
        # If size is a single value, use it to create a proportional H-shape
        if isinstance(self.size, tuple) and len(self.size) >= 4:
            flange_width = self.size[0]
            web_height = self.size[1]
            web_thickness = self.size[2]
            flange_thickness = self.size[3]
        else:
            # Create proportional H-shape based on a single size value
            base_size = self.size if isinstance(self.size, (int, float)) else self.size[0]
            flange_width = base_size
            web_height = base_size * 1.8  # Make height 80% larger than width
            web_thickness = base_size * 0.2  # Web thickness is 20% of size
            flange_thickness = base_size * 0.25  # Flange thickness is 25% of size
        
        # Create wire builder
        wire_builder = BRepBuilderAPI_MakeWire()
        
        # Define the 12 corners of the H-section
        # Starting at top-left corner, going clockwise
        half_height = web_height / 2
        half_width = flange_width / 2
        half_web = web_thickness / 2
        
        points = [
            # Start from bottom left outer and move clockwise
            gp_Pnt(-half_width, -half_height, 0),
            gp_Pnt(half_width, -half_height, 0),
            gp_Pnt(half_width, -half_height + flange_thickness, 0),
            gp_Pnt(half_web, -half_height + flange_thickness, 0),
            gp_Pnt(half_web, half_height - flange_thickness, 0),
            gp_Pnt(half_width, half_height - flange_thickness, 0),
            gp_Pnt(half_width, half_height, 0),
            gp_Pnt(-half_width, half_height, 0),
            gp_Pnt(-half_width, half_height - flange_thickness, 0),
            gp_Pnt(-half_web, half_height - flange_thickness, 0),
            gp_Pnt(-half_web, -half_height + flange_thickness, 0),
            gp_Pnt(-half_width, -half_height + flange_thickness, 0),
        ]

        
        # Create edges for the wire by connecting adjacent points
        for i in range(len(points) - 1):
            edge = BRepBuilderAPI_MakeEdge(points[i], points[i + 1]).Edge()
            wire_builder.Add(edge)
        
        # Connect last point to the first point to close the shape
        edge = BRepBuilderAPI_MakeEdge(points[-1], points[0]).Edge()
        wire_builder.Add(edge)
        
        # Create wire and face
        wire = wire_builder.Wire()
        face = BRepBuilderAPI_MakeFace(wire).Face()
        return face
    
    def create_geometry(self):
        """Create visualization geometry using BRepPrimAPI_MakePrism"""
        # Create the base face based on the cross-section type
        if self.cross_section.lower() == 'circular':
            base_face = self.create_circular_base()
        elif self.cross_section.lower() == 'rectangular':
            base_face = self.create_rectangular_base()
        elif self.cross_section.lower() == 'h':
            base_face = self.create_h_shaped_base()
        else:
            # Default to circular if unknown type
            base_face = self.create_circular_base()
            
        # Calculate direction vector from start to end
        direction = self.end_pos - self.start_pos
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm < MIN_DIRECTION_NORM:
            # Handle degenerate case
            dir_vec = gp_Vec(0, 0, self.length)
        else:
            direction = direction / direction_norm
            dir_vec = gp_Vec(direction[0], direction[1], direction[2]) * self.length
        
        # Create a transformation to position the base at the start point and orient it properly
        transform = gp_Trsf()
        
        # Calculate rotation from Z-axis to the segment direction
        z_axis = np.array([0, 0, 1])
        segment_axis = direction
        
        if np.allclose(z_axis, segment_axis, atol=EPSILON):
            # No rotation needed
            rotation_axis = np.array([1, 0, 0])
            angle = 0
        elif np.allclose(z_axis, -segment_axis, atol=EPSILON):
            # 180 degree rotation around X
            rotation_axis = np.array([1, 0, 0])
            angle = np.pi
        else:
            # Cross product gives rotation axis
            rotation_axis = np.cross(z_axis, segment_axis)
            rotation_axis_norm = np.linalg.norm(rotation_axis)
            
            # Check if rotation axis is valid (not zero length and no NaN values)
            if rotation_axis_norm < MIN_DIRECTION_NORM or np.isnan(rotation_axis_norm) or np.isinf(rotation_axis_norm):
                # Fall back to a default rotation axis
                rotation_axis = np.array([1, 0, 0])
                angle = 0
            else:
                rotation_axis = rotation_axis / rotation_axis_norm
                
                # Check for NaN or infinite values
                if np.any(np.isnan(rotation_axis)) or np.any(np.isinf(rotation_axis)):
                    rotation_axis = np.array([1, 0, 0])
                    angle = 0
                
                # Dot product gives cosine of angle
                cos_angle = np.dot(z_axis, segment_axis)
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        # Set rotation - explicitly use the components rather than unpacking
        # Make sure rotation_axis components are valid numbers
        try:
            rotation_ax1 = gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(float(rotation_axis[0]), float(rotation_axis[1]), float(rotation_axis[2])))
            transform.SetRotation(rotation_ax1, angle)
        except (TypeError, ValueError) as e:
            print(f"Warning: Error setting rotation: {e}")
            print(f"Rotation axis: {rotation_axis}, Angle: {angle}")
            # Use identity rotation as fallback
            transform.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0)), 0)
        
        # Set translation to the start position
        try:
            transform.SetTranslationPart(gp_Vec(*self.start_pos))
        except (TypeError, ValueError) as e:
            print(f"Warning: Error setting translation: {e}")
            print(f"Start position: {self.start_pos}")
            # Use origin as fallback
            transform.SetTranslationPart(gp_Vec(0, 0, 0))
        
        # Apply transformation to the base face
        base_face_transformed = BRepBuilderAPI_Transform(base_face, transform).Shape()
        # Create the prism by sweeping along the direction vector
        prism = BRepPrimAPI_MakePrism(base_face_transformed, dir_vec).Shape()
        return prism
    

class ArticulatedColumn:
    def __init__(self, num_segments=5, segment_length=1.0, size=0.1, cross_section='circular'):
        self.segments = []
        
        # Create base segment
        base = ArticulatedSegment(segment_length, size, cross_section=cross_section)
        self.segments.append(base)
        
        # Create chain of segments
        for i in range(1, num_segments):
            self.segments.append(ArticulatedSegment(segment_length, size, parent=self.segments[-1], cross_section=cross_section))
    
    def bend_at_segment(self, segment_idx, axis, angle):
        """Bend the column at a specific segment"""
        if 0 <= segment_idx < len(self.segments):
            # Convert string angle expressions like 'pi/24' to float if needed
            if isinstance(angle, str) and 'pi' in angle:
                try:
                    # Replace 'pi' with 'np.pi' for evaluation
                    expr = angle.replace('pi', 'np.pi')
                    angle = eval(expr)
                    print(f"Converted angle expression '{angle}' to {angle}")
                except Exception as e:
                    print(f"Error converting angle expression '{angle}': {e}")
                    # Default to a small angle if conversion fails
                    angle = np.pi/24
            
            # Rotate the specified segment
            self.segments[segment_idx].rotate(axis, angle)
            
            # Calculate reduced angles for adjacent segments for natural bending
            reduced_angle = angle * 0.5  # Adjacent segments rotate by half
            
            # Rotate adjacent segments in the same direction
            if segment_idx > 0:
                self.segments[segment_idx - 1].rotate(axis, reduced_angle)
            
            if segment_idx < len(self.segments) - 1:
                self.segments[segment_idx + 1].rotate(axis, reduced_angle)

# ================== HELPER FUNCTIONS ===================
def column_to_point_cloud(column, total_points=DEFAULT_POINTS_PER_CLOUD):
    all_points = []
    points_per_segment = total_points // len(column.segments)
    
    np.random.seed(RANDOM_SEED) # Set a seed for reproducibility
    
    for segment in column.segments:
        if segment.cross_section.lower() == 'circular':
            # Use deterministic sampling for circular cross-section
            radius = segment.size if isinstance(segment.size, (int, float)) else segment.size[0]
            theta_count = int(np.sqrt(points_per_segment))
            z_count = points_per_segment // theta_count
            # Create evenly spaced grid with small random perturbations
            theta_values = np.linspace(0, 2*np.pi, theta_count, endpoint=False)
            z_values = np.linspace(0, segment.length, z_count)
            # Create grid of points
            theta_grid, z_grid = np.meshgrid(theta_values, z_values)
            theta = theta_grid.flatten() + np.random.normal(0, 0.05, theta_grid.size)  # Small randomness
            z = z_grid.flatten() + np.random.normal(0, 0.01, z_grid.size)  # Small randomness
            # Convert to cartesian coordinates
            local_points = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), z])
        
        elif segment.cross_section.lower() == 'rectangular':
            width, height = segment.size if isinstance(segment.size, tuple) else (segment.size, segment.size)
            # Sample points on the six faces of the box
            sides_count = 6
            points_per_side = points_per_segment // sides_count
            local_points = []
            
            # Bottom face (z=0)
            for _ in range(points_per_side):
                x = np.random.uniform(-width/2, width/2)
                y = np.random.uniform(-height/2, height/2)
                local_points.append([x, y, 0])
            
            # Top face (z=length)
            for _ in range(points_per_side):
                x = np.random.uniform(-width/2, width/2)
                y = np.random.uniform(-height/2, height/2)
                local_points.append([x, y, segment.length])
            
            # Front face (y=-height/2)
            for _ in range(points_per_side):
                x = np.random.uniform(-width/2, width/2)
                z = np.random.uniform(0, segment.length)
                local_points.append([x, -height/2, z])
            
            # Back face (y=height/2)
            for _ in range(points_per_side):
                x = np.random.uniform(-width/2, width/2)
                z = np.random.uniform(0, segment.length)
                local_points.append([x, height/2, z])
            
            # Left face (x=-width/2)
            for _ in range(points_per_side):
                y = np.random.uniform(-height/2, height/2)
                z = np.random.uniform(0, segment.length)
                local_points.append([-width/2, y, z])
            
            # Right face (x=width/2)
            for _ in range(points_per_side):
                y = np.random.uniform(-height/2, height/2)
                z = np.random.uniform(0, segment.length)
                local_points.append([width/2, y, z])
                
            local_points = np.array(local_points)
        
        elif segment.cross_section.lower() == 'h':
            # For H-section, extract dimensions
            if isinstance(segment.size, tuple) and len(segment.size) >= 4:
                flange_width = segment.size[0]
                web_height = segment.size[1]
                web_thickness = segment.size[2]
                flange_thickness = segment.size[3]
            else:
                # Create proportional H-shape based on a single size value
                base_size = segment.size if isinstance(segment.size, (int, float)) else segment.size[0]
                flange_width = base_size
                web_height = base_size * H_HEIGHT_RATIO
                web_thickness = base_size * H_WEB_THICKNESS_RATIO
                flange_thickness = base_size * H_FLANGE_THICKNESS_RATIO
            
            half_height = web_height / 2
            half_width = flange_width / 2
            half_web = web_thickness / 2
            
            # Calculate points per sub-surface to distribute proportionally
            total_area = 2 * flange_width * flange_thickness + (web_height - 2 * flange_thickness) * web_thickness
            
            # Points for the top flange
            top_flange_points = []
            points_top_flange = int(points_per_segment * (flange_width * flange_thickness / total_area))
            for _ in range(points_top_flange):
                x = np.random.uniform(-half_width, half_width)
                y = np.random.uniform(half_height - flange_thickness, half_height)
                z = np.random.uniform(0, segment.length)
                top_flange_points.append([x, y, z])
            
            # Points for the bottom flange
            bottom_flange_points = []
            points_bottom_flange = int(points_per_segment * (flange_width * flange_thickness / total_area))
            for _ in range(points_bottom_flange):
                x = np.random.uniform(-half_width, half_width)
                y = np.random.uniform(-half_height, -half_height + flange_thickness)
                z = np.random.uniform(0, segment.length)
                bottom_flange_points.append([x, y, z])
            
            # Points for the web
            web_points = []
            points_web = points_per_segment - points_top_flange - points_bottom_flange
            for _ in range(points_web):
                x = np.random.uniform(-half_web, half_web)
                y = np.random.uniform(-half_height + flange_thickness, half_height - flange_thickness)
                z = np.random.uniform(0, segment.length)
                web_points.append([x, y, z])
            
            # Combine all points
            local_points = np.array(top_flange_points + bottom_flange_points + web_points)
        
        else:
            # Default to circular if unknown cross-section
            radius = 0.1  # Default radius
            theta_count = int(np.sqrt(points_per_segment))
            z_count = points_per_segment // theta_count
            theta_values = np.linspace(0, 2*np.pi, theta_count, endpoint=False)
            z_values = np.linspace(0, segment.length, z_count)
            theta_grid, z_grid = np.meshgrid(theta_values, z_values)
            theta = theta_grid.flatten() + np.random.normal(0, 0.05, theta_grid.size)
            z = z_grid.flatten() + np.random.normal(0, 0.01, z_grid.size)
            local_points = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), z])
        
        # Transform to world space
        transformed_points = local_points @ segment.orientation.T + segment.start_pos
        all_points.append(transformed_points)
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(all_points))
    
    return pcd

#calculate the RMSE

def calculate_rmse(source_points, target_points):
    differences = source_points - target_points
    squared_diffs = np.square(differences)
    mean_squared_diff = np.mean(squared_diffs)
    rmse = np.sqrt(mean_squared_diff)
    return rmse

# ================== JACOBIAN AND POINT MATCHING ===================
def compute_jacobian(column, source_points, target_points, point_weights=None, segment_weights=None):
    """Compute the Jacobian matrix with weighted influences"""
    num_segments = len(column.segments)
    num_points = len(source_points)
    num_params = num_segments * 3
    
    # Initialize Jacobian and error
    J = np.zeros((num_points * 3, num_params))
    error_vectors = target_points - source_points
    error = error_vectors.flatten()
    
    # Default weights if not provided
    if point_weights is None:
        point_weights = np.ones(num_points)
        
    if segment_weights is None:
        segment_weights = np.ones(num_segments)
    
    # Find which segment each point belongs to
    point_segments = np.zeros(num_points, dtype=int)
    for i, point in enumerate(source_points):
        point_segments[i] = find_closest_segment(column, point)
    
    # For each point, calculate Jacobian entries
    for i, point in enumerate(source_points):
        closest_segment_idx = point_segments[i]
        point_weight = point_weights[i]
        
        # Calculate entries for all segments that affect this point
        for j in range(closest_segment_idx + 1):
            segment = column.segments[j]
            segment_weight = segment_weights[j]
            combined_weight = point_weight * segment_weight
            
            # For each rotation axis
            for k in range(3):
                param_idx = j * 3 + k
                
                # Get rotation axis in world space
                if k == 0:  # x-axis
                    rotation_axis = segment.orientation @ np.array([1, 0, 0])
                elif k == 1:  # y-axis
                    rotation_axis = segment.orientation @ np.array([0, 1, 0])
                else:  # z-axis
                    rotation_axis = segment.orientation @ np.array([0, 0, 1])
                
                # ICPIK formula: J = v_j × (s_i - p_j)
                cross_product = np.cross(rotation_axis, point - segment.start_pos)
                J[i*3:i*3+3, param_idx] = cross_product * combined_weight
    
    return J, error

def find_closest_segment(column, point):
    """Find the index of the closest segment to a given point"""
    closest_idx = 0
    min_dist = float('inf')
    
    for i, segment in enumerate(column.segments):
        # Calculate distance to segment line
        start = segment.start_pos
        end = segment.end_pos
        
        segment_vec = end - start
        segment_len = np.linalg.norm(segment_vec)
        segment_dir = segment_vec / segment_len if segment_len > MIN_DIRECTION_NORM else np.array([0, 0, 1])
        
        to_point = point - start
        projection = np.dot(to_point, segment_dir)
        projection = max(0, min(segment_len, projection))
        
        closest_point = start + projection * segment_dir
        dist = np.linalg.norm(point - closest_point)
        
        if dist < min_dist:
            min_dist = dist
            closest_idx = i
    return closest_idx

def find_quality_correspondences(kinematic_pcd, deformed_pcd, distance_threshold=ICP_MAX_CORRESPONDENCE_DISTANCE):
    """Find quality correspondences with fallbacks and filtering"""
    # Ensure normals are calculated
    kinematic_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=NORMALS_RADIUS, max_nn=NORMALS_MAX_NN))
    deformed_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=NORMALS_RADIUS, max_nn=NORMALS_MAX_NN))
    
    # Use more iterations for better convergence
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=ICP_RELATIVE_FITNESS, relative_rmse=ICP_RELATIVE_RMSE, max_iteration=ICP_MAX_ITERATION)
    # Try point-to-point ICP
    try:
        result = o3d.pipelines.registration.registration_icp(kinematic_pcd, deformed_pcd, max_correspondence_distance=distance_threshold, init=np.eye(4), estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(), criteria=criteria)
        correspondences = np.asarray(result.correspondence_set)
    except Exception as e:
        print(f"Point-to-point ICP failed: {e}")
        correspondences = np.array([])  # Empty array instead of None
    
    # If too few correspondences, try with relaxed parameters
    if len(correspondences) < 100:
        print(f"  Warning: {len(correspondences)} correspondences found. Trying relaxed parameters...")
        distance_threshold *= 1.5  # Increase the threshold to capture more matches
        try:
            result = o3d.pipelines.registration.registration_icp(kinematic_pcd, deformed_pcd, max_correspondence_distance=distance_threshold, init=np.eye(4), estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(), criteria=criteria)
            correspondences = np.asarray(result.correspondence_set)
        except Exception as e:
            print(f"Relaxed ICP failed: {e}")
            correspondences = np.array([])  # Empty array instead of None
    
    print(f'Number of correspondences found: {len(correspondences)}')
    return correspondences

# ================== PROGRESS TRACKING ===================
class OptimizationTracker:
    """
    Tracks and logs progress during optimization process.
    Provides timing information, iteration metrics, and performance statistics.
    """
    def __init__(self, name="ICPIK Optimization", log_to_file=True):
        self.name = name
        self.start_time = None
        self.iteration_times = []
        self.iteration_metrics = {}
        self.log_to_file = log_to_file
        self.log_file = None
        
        # Initialize log file if needed
        if log_to_file:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = open(f"optimization_log_{timestamp}.txt", "w")
            self.log_file.write(f"=== {self.name} ===\n")
            self.log_file.write(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            self.log_file.flush()
    
    def start(self):
        """Start tracking a new optimization run"""
        self.start_time = time.time()
        self.iteration_times = []
        self.iteration_metrics = {
            'rmse': [],
            'avg_distance': [],
            'max_distance': [],
            'correspondences': [],
            'closest_points': []
        }
        self.log(f"Started {self.name}")
    
    def log_iteration(self, iteration, metrics):
        """Log metrics for current iteration"""
        current_time = time.time()
        if self.start_time is None:
            self.start_time = current_time
        
        elapsed = current_time - self.start_time
        if len(self.iteration_times) > 0:
            iter_duration = current_time - self.iteration_times[-1]
        else:
            iter_duration = elapsed
        
        self.iteration_times.append(current_time)
        
        # Store metrics
        for key, value in metrics.items():
            if key in self.iteration_metrics:
                self.iteration_metrics[key].append(value)
        
        # Format log message
        message = f"Iteration {iteration} - "
        message += f"Time: {elapsed:.2f}s (+"
        message += f"{iter_duration:.2f}s) | "
        
        # Add metrics to message
        metrics_str = " | ".join([f"{k}: {v}" for k, v in metrics.items() if k != 'delta_theta'])
        message += metrics_str
        
        self.log(message)
        
        # Return elapsed time and iteration time
        return elapsed, iter_duration
    
    def log(self, message):
        """Log a custom message"""
        print(message)
        if self.log_to_file and self.log_file:
            self.log_file.write(message + "\n")
            self.log_file.flush()
    
    def finish(self):
        """Complete tracking and report final statistics"""
        if self.start_time is None:
            self.log("Warning: finish() called before start()")
            return
        
        total_duration = time.time() - self.start_time
        iterations = len(self.iteration_times)
        
        # Calculate statistics
        avg_iter_time = total_duration / max(1, iterations)
        
        # Generate summary message
        summary = f"\n=== Optimization Summary ===\n"
        summary += f"Total duration: {total_duration:.2f} seconds\n"
        summary += f"Iterations: {iterations}\n"
        summary += f"Average iteration time: {avg_iter_time:.2f} seconds\n"
        
        if 'rmse' in self.iteration_metrics and len(self.iteration_metrics['rmse']) > 0:
            initial_rmse = self.iteration_metrics['rmse'][0]
            final_rmse = self.iteration_metrics['rmse'][-1]
            improvement = (initial_rmse - final_rmse) / initial_rmse * 100
            summary += f"Initial RMSE: {initial_rmse:.6f}\n"
            summary += f"Final RMSE: {final_rmse:.6f}\n"
            summary += f"Improvement: {improvement:.2f}%\n"
        
        self.log(summary)
        
        if self.log_to_file and self.log_file:
            self.log_file.close()
            self.log_file = None
        
        return total_duration, iterations

# ================== INCREMENTAL OPTIMIZATION IMPLEMENTATION ===================
def apply_incremental_optimization(deformed_pcd, kinematic_column, max_iterations=DEFAULT_MAX_ITERATIONS, error_threshold=DEFAULT_ERROR_THRESHOLD):
    frames = []
    all_errors = []
    all_rmse_values = []
    
    # Initialize the tracker
    tracker = OptimizationTracker("ICPIK Optimization")
    tracker.start()
    
    #visualization
    original_pcd = column_to_point_cloud(ArticulatedColumn(
        num_segments=len(kinematic_column.segments), 
        segment_length=kinematic_column.segments[0].length, 
        size=kinematic_column.segments[0].size,
        cross_section=kinematic_column.segments[0].cross_section))
    original_pcd.paint_uniform_color(ORIGINAL_COLOR)  # Gray
    
    deformed_pcd_vis = copy.deepcopy(deformed_pcd)
    deformed_pcd_vis.paint_uniform_color(DEFORMED_COLOR)    # Blue
    
    current_pcd = column_to_point_cloud(kinematic_column)
    current_pcd.paint_uniform_color(CURRENT_COLOR)         # Red
    frames.append([original_pcd, deformed_pcd_vis, current_pcd])
     
    # Main optimization loop
    num_segments = len(kinematic_column.segments)
    
    # Set an initial distance threshold that will adapt
    dist_threshold = INITIAL_DIST_THRESHOLD
    
    for iteration in range(max_iterations):
        iter_start_time = time.time()
        iteration_metrics = {}
        
        # 1. Find correspondences
        kinematic_pcd = column_to_point_cloud(kinematic_column)
        kinematic_pcd.estimate_normals()
        
        # Adaptive distance threshold - starts higher and gradually decreases
        dist_threshold = INITIAL_DIST_THRESHOLD * (1.0 - min(0.9, iteration / max_iterations))
        iteration_metrics['dist_threshold'] = f"{dist_threshold:.4f}"
        
        correspondences = find_quality_correspondences(kinematic_pcd, deformed_pcd, dist_threshold)
        iteration_metrics['correspondences'] = len(correspondences)
        
        if len(correspondences) < MIN_POINTS_FOR_CORRESPONDENCE * 2:
            tracker.log(f"  Warning: Only {len(correspondences)} correspondences found.")
            # Try with a more relaxed threshold just for this iteration
            temp_threshold = dist_threshold * 1.5
            tracker.log(f"  Trying with relaxed threshold: {temp_threshold:.4f}")
            correspondences = find_quality_correspondences(kinematic_pcd, deformed_pcd, temp_threshold)
            iteration_metrics['correspondences'] = f"{len(correspondences)} (relaxed)"
            
            if len(correspondences) < MIN_POINTS_FOR_CORRESPONDENCE:
                tracker.log("  Still insufficient correspondences. Reducing constraint weights and continuing.")
                # We'll continue with what we have, but with gentler constraints
        
        # 2. Get source and target points
        kinematic_points = np.asarray(kinematic_pcd.points)
        deformed_points = np.asarray(deformed_pcd.points)
        source_points = kinematic_points[correspondences[:, 0]]
        target_points = deformed_points[correspondences[:, 1]]

        rmse = calculate_rmse(source_points, target_points)
        iteration_metrics['rmse'] = rmse
        all_rmse_values.append(rmse)
        
        # 3. Calculate distances between corresponding points
        distances = np.linalg.norm(target_points - source_points, axis=1)
        avg_distance = np.mean(distances)
        max_distance = np.max(distances)
        
        iteration_metrics['avg_distance'] = f"{avg_distance:.6f}"
        iteration_metrics['max_distance'] = f"{max_distance:.6f}"
        
        # Early termination check
        if max_distance < error_threshold:
            tracker.log(f"  Reached error threshold ({error_threshold}). Terminating early.")
            break
        
        # 4. Sort points by distance (smallest first)
        sorted_indices = np.argsort(distances)
        
        # First iteration: focus on only the closest 5% of points
        if iteration == 0:
            closest_percentage = FIRST_ITER_CLOSEST_PERCENTAGE
        else:
            # Gradually increase percentage after first iteration
            closest_percentage = 0.1 + 0.7 * ((iteration-1) / (max_iterations-1))
            
        num_closest = max(int(len(sorted_indices) * closest_percentage), MIN_POINTS_FOR_CORRESPONDENCE)
        closest_indices = sorted_indices[:num_closest]
        
        iteration_metrics['closest_points'] = f"{num_closest} ({closest_percentage:.2%})"
        
        # 5. Filter to just the closest points
        close_source_points = source_points[closest_indices]
        close_target_points = target_points[closest_indices]
        
        # 6. Identify which segments these points belong to
        point_segments = np.zeros(len(close_source_points), dtype=int)
        for i, point in enumerate(close_source_points):
            point_segments[i] = find_closest_segment(kinematic_column, point)
            
        # Print the distribution of closest points across segments
        seg_distribution = np.bincount(point_segments, minlength=num_segments)
        iteration_metrics['segment_distribution'] = str(seg_distribution)
        
        # 7. Calculate segment participation weights
        # Segments with more close points get higher priority
        segment_counts = np.zeros(num_segments)
        for seg_idx in point_segments:
            segment_counts[seg_idx] += 1
        
        if iteration == 0:
            # Give higher weights to segments with more points and zero weight to segments with no points
            segment_weights = np.zeros(num_segments)
            for i in range(num_segments):
                if segment_counts[i] > 0:
                    segment_weights[i] = 1.0  # Uniform weight for segments with points
                    
            # Filter out segments with zero weights to prevent unnecessary movement
            active_segments = np.where(segment_weights > 0)[0]
            if len(active_segments) == 0:
                tracker.log("  Warning: No active segments found. Using default weights.")
                segment_weights = np.ones(num_segments) * SEGMENT_WEIGHT_MIN
            else:
                iteration_metrics['active_segments'] = str(active_segments)
        else:
            # Normal weight calculation for subsequent iterations
            # Normalize and add base weight to ensure all segments have some influence
            segment_weights = segment_counts / max(segment_counts.max(), 1) 
            segment_weights = SEGMENT_WEIGHT_MIN + SEGMENT_WEIGHT_SCALE * segment_weights  # Ensure minimum weight
        
        # 8. Calculate point weights - give more weight to closer points
        # Inverse distance weighting
        point_distances = distances[closest_indices]
        # Prevent division by zero by adding a small epsilon
        inv_distances = 1.0 / (point_distances + EPSILON)
        point_weights = inv_distances / inv_distances.max()  # Normalize to [0,1]
        
        # Apply non-linear scaling to emphasize very close points
        point_weights = point_weights ** 2  # Square to emphasize differences
        # Ensure minimum weight
        point_weights = POINT_WEIGHT_MIN + POINT_WEIGHT_SCALE * point_weights
        
        # 9. Compute Jacobian with our weights
        J, error_vector = compute_jacobian(kinematic_column, close_source_points, close_target_points, point_weights=point_weights, segment_weights=segment_weights)
        current_error = np.linalg.norm(error_vector)
        all_errors.append(current_error) 
        iteration_metrics['error'] = f"{current_error:.6f}"
        
        # 10. Solve using Damped Least Squares with adaptive damping
        JTJ = J.T @ J
        
        # Adaptive damping based on current iteration and error
        # Much higher damping in first iteration to prevent large jumps
        if iteration == 0:
            adaptive_damping = FIRST_ITER_DAMPING  # Very high damping for first iteration
        else:
            # Gradually reduce damping in later iterations
            progress_factor = min(1.0, (iteration-1) / (max_iterations * 0.7))
            error_factor = min(1.0, current_error / (INITIAL_DIST_THRESHOLD * 10))
            adaptive_damping = 0.1 * (1.0 - progress_factor * 0.8) * (1.0 + error_factor)
        
        iteration_metrics['damping'] = f"{adaptive_damping:.4f}"
        
        lambda_I = adaptive_damping * np.eye(JTJ.shape[0])
        JTe = J.T @ error_vector
        
        # 11. Add directional bias to favor natural bending
        # Higher constraining for first iteration
        if iteration == 0:
            constraint_weight = FIRST_ITER_CONSTRAINT_WEIGHT  # Stronger constraint for first iteration
        else:
            constraint_weight = LATER_ITER_CONSTRAINT_WEIGHT  # Lighter constraint for later iterations    
        constraint_matrix = np.eye(JTJ.shape[0]) * constraint_weight
        
        # Determine main bending direction
        error_directions = close_target_points - close_source_points
        mean_direction = np.mean(np.abs(error_directions), axis=0)
        main_axis = np.argmax(mean_direction)
        secondary_axes = [i for i in range(3) if i != main_axis]
        
        # Determine proper direction of bend
        mean_error_direction = np.mean(error_directions, axis=0)
        main_axis_direction = np.sign(mean_error_direction[main_axis])
        iteration_metrics['main_axis'] = f"{main_axis} (dir: {main_axis_direction})"
        
        # Apply axis-specific constraints
        for i in range(num_segments):
            # First iteration: Apply constraint to all segments
            if iteration == 0:
                # For first iteration, only reduce constraint on segments with points
                if segment_counts[i] > 0:
                    constraint_matrix[i*3 + main_axis, i*3 + main_axis] *= 0.5
                    
                # Keep strong constraints on other axes
                for j in secondary_axes:
                    constraint_matrix[i*3 + j, i*3 + j] *= 2.0
            else:
                # Later iterations: standard constraints
                # Less constraint on main bending axis
                constraint_matrix[i*3 + main_axis, i*3 + main_axis] *= 0.5
                
                # More constraint on other axes to discourage unnecessary rotation
                for j in secondary_axes:
                    constraint_matrix[i*3 + j, i*3 + j] *= 2.0
                
        # 12. Add continuity constraints to encourage smooth bending
        continuity_matrix = np.zeros(JTJ.shape)
        
        # First iteration: stronger continuity constraints
        if iteration == 0:
            continuity_weight = FIRST_ITER_CONTINUITY_WEIGHT  # Strong continuity constraint for first iteration
        else:
            continuity_weight = LATER_ITER_CONTINUITY_WEIGHT  # Normal continuity for later iterations
        
        for i in range(1, num_segments - 1):
            # Each middle segment should ideally have a rotation similar to average of neighbors
            for axis in range(3):
                # Penalize deviation from average of neighboring segments
                curr_idx = i * 3 + axis
                prev_idx = (i-1) * 3 + axis
                next_idx = (i+1) * 3 + axis
                
                # Add entries to penalize deviation from average of neighbors
                continuity_matrix[curr_idx, curr_idx] += 2 * continuity_weight
                continuity_matrix[curr_idx, prev_idx] -= continuity_weight
                continuity_matrix[curr_idx, next_idx] -= continuity_weight
        
        # 13. Solve the system with all constraints
        augmented_system = JTJ + lambda_I + constraint_matrix + continuity_matrix
        delta_theta = np.linalg.solve(augmented_system, JTe)
        
        # 14. Scale motion 
        delta_theta *= DELTA_THETA_SCALE

        # 15. Apply joint limits with gradually increasing freedom
        if iteration == 0:
            max_angle_main = FIRST_ITER_MAX_ANGLE_MAIN  # Very limited for first iteration
            max_angle_other = FIRST_ITER_MAX_ANGLE_OTHER
        else:
            max_angle_factor = min(1.0, (iteration-1) / (max_iterations * 0.6))
            max_angle_main = LATER_ITER_MAX_ANGLE_MAIN * (1.0 + max_angle_factor)
            max_angle_other = LATER_ITER_MAX_ANGLE_OTHER * (1.0 + max_angle_factor)
            
        limited_count = 0
        for i in range(num_segments):
            for j in range(3):
                param_idx = i * 3 + j
                # Use appropriate angle limit based on axis
                if j == main_axis:
                    max_angle = max_angle_main
                else:
                    max_angle = max_angle_other               
                current_angle = kinematic_column.segments[i].local_rotation[j]
                new_angle = current_angle + delta_theta[param_idx]
                if abs(new_angle) > max_angle:
                    # Clip to the limit
                    delta_theta[param_idx] = np.sign(new_angle) * max_angle - current_angle
                    limited_count += 1
        
        iteration_metrics['angle_limits'] = f"{limited_count} joints limited"
        
        # 16. Apply changes to the kinematic model
        for i in range(num_segments):
            segment = kinematic_column.segments[i]
            for j in range(3):
                param_idx = i * 3 + j
                segment.local_rotation[j] += delta_theta[param_idx]
            segment.update_transform()
        
        # 17. Save frame for visualization
        # Always save the first iteration frame and less frequently after that
        if iteration == 0 or iteration % 5 == 0 or iteration == max_iterations - 1:
            current_pcd = column_to_point_cloud(kinematic_column)
            current_pcd.paint_uniform_color(CURRENT_COLOR)
            frames.append([original_pcd, deformed_pcd_vis, current_pcd])
        
        # Log iteration results
        delta_theta_norm = np.linalg.norm(delta_theta)
        iteration_metrics['delta_theta_norm'] = f"{delta_theta_norm:.6f}"
        _, iter_time = tracker.log_iteration(iteration + 1, iteration_metrics)
        
        # 18. Check for early convergence (but not in first few iterations)
        if iteration > 10 and current_error < error_threshold * 2:
            # If we're getting close, do a full evaluation
            kinematic_pcd = column_to_point_cloud(kinematic_column)
            full_correspondences = find_quality_correspondences(kinematic_pcd, deformed_pcd, dist_threshold * 0.5)
            
            if len(full_correspondences) > 0:
                full_source = kinematic_points[full_correspondences[:, 0]]
                full_target = deformed_points[full_correspondences[:, 1]]
                full_distances = np.linalg.norm(full_target - full_source, axis=1)
                full_max_error = np.max(full_distances)
                
                tracker.log(f"  Full evaluation max error: {full_max_error:.6f}")
                
                if full_max_error < error_threshold:
                    tracker.log("  Early convergence achieved. Terminating.")
                    break
    
    # Final result
    final_pcd = column_to_point_cloud(kinematic_column)
    final_pcd.paint_uniform_color(FINAL_COLOR)  # Green for final result
    frames.append([original_pcd, deformed_pcd_vis, final_pcd])
    
    # Complete tracking
    tracker.finish()
    
    return kinematic_column, all_errors, frames, all_rmse_values

def calculate_result_vs_groundtruth_error(deformed_column, result_column):
    """Calculate the distance between the ICPIK result and ground truth for each node"""
    num_segments = len(deformed_column.segments)
    node_indices = list(range(num_segments))
    errors = []
    
    for i in range(num_segments):
        # Calculate the Euclidean distance between result and ground truth positions
        gt_pos = deformed_column.segments[i].end_pos
        result_pos = result_column.segments[i].end_pos
        
        # Calculate distance
        distance = np.linalg.norm(result_pos - gt_pos)
        errors.append(distance)
        
        print(f"Node {i}: Ground Truth position: {gt_pos}, Result position: {result_pos}, Error: {distance:.4f}")
    
    return node_indices, errors

def plot_result_vs_groundtruth_error(deformed_column, result_column):
    """Plot the distance between ICPIK result and ground truth for each node"""
    node_indices, errors = calculate_result_vs_groundtruth_error(deformed_column, result_column)
    
    plt.figure(figsize=(12, 6))
    plt.bar(node_indices, errors, color='skyblue', alpha=0.7)
    plt.plot(node_indices, errors, 'ro-', linewidth=2)
    plt.xlabel('Node Number', fontsize=20)
    plt.ylabel('Euclidean Distance', fontsize=20)
    plt.title('Distance Between ICPIK Result and Ground Truth for Each Node', fontsize=24, pad=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    # Set y-axis limit to 0.05
    plt.ylim(0, 0.8)
    # Set small font size for tick numbers
    plt.xticks(node_indices, fontsize=10)  # Small font for x-axis tick labels
    plt.yticks(fontsize=10)  # Small font for y-axis tick labels
    # Add error values above each bar with a moderate font size
    for i, d in enumerate(errors):
        plt.text(i, d + 0.001, f'{d:.3f}', ha='center', fontsize=10)  # Adjusted vertical offset
    plt.tight_layout()
    plt.show()
    return node_indices, errors

#node calc

def calculate_node_displacements(original_column, result_column):

    num_segments = len(original_column.segments)
    node_indices = list(range(num_segments))
    displacements = []
    
    for i in range(num_segments):
        # Calculate displacement at the end position of each segment
        original_pos = original_column.segments[i].end_pos
        result_pos = result_column.segments[i].end_pos
        
        # Calculate Euclidean distance
        distance = np.linalg.norm(result_pos - original_pos)
        displacements.append(distance)
        
        print(f"Node {i}: Original position: {original_pos}, Result position: {result_pos}, Distance: {distance:.4f}")
    
    return node_indices, displacements

def plot_node_displacements(original_column, result_column):
    """
    Plot the Euclidean distance between corresponding nodes in the original and result columns.
    
    Args:
        original_column: The initial straight column
        result_column: The final deformed column after optimization
    """
    node_indices, displacements = calculate_node_displacements(original_column, result_column)
    
    plt.figure(figsize=(12, 6))
    plt.bar(node_indices, displacements, color='skyblue', alpha=0.7)
    plt.plot(node_indices, displacements, 'ro-', linewidth=2)
    plt.xlabel('Node Number', fontsize=20)
    plt.ylabel('Euclidean Distance', fontsize=20)
    plt.title('Displacement between Original and Deformed Column Nodes', fontsize=24, pad=20)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Set small font size for tick numbers
    plt.xticks(node_indices, fontsize=10)  # Small font for x-axis tick labels
    plt.yticks(fontsize=10)  # Small font for y-axis tick labels

    # Add displacement values above each bar with a moderate font size
    for i, d in enumerate(displacements):
        plt.text(i, d + 0.01, f'{d:.3f}', ha='center', fontsize=10)  # Set font size for the data labels

    plt.tight_layout()
    plt.show()
    return node_indices, displacements

# ================== ENHANCED VISUALIZATION ===================
def visualize_optimization_results(original_column, deformed_column, result_column, frames, errors, rmse_values):
    """
    Creates a comprehensive visualization of optimization results with multiple views:
    1. 3D comparison view of original, deformed, and result columns
    2. Interactive animation of optimization process
    3. Error metrics plots
    4. Node displacement visualization
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D
    
    plt.style.use('seaborn-v0_8-whitegrid')  # Use a modern style
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('Articulated Column Optimization Results', fontsize=16, fontweight='bold')
    
    # 1. RMSE Plot (log scale)
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(rmse_values, 'b-', linewidth=2, marker='o', markersize=3)
    ax1.set_yscale('log')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('RMSE (log scale)')
    ax1.set_title('Convergence Progress')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Node displacement plot
    ax2 = fig.add_subplot(2, 3, 2)
    node_indices, displacements = calculate_node_displacements(original_column, result_column)
    ax2.bar(node_indices, displacements, color='blue', alpha=0.7)
    ax2.set_xlabel('Segment Index')
    ax2.set_ylabel('Displacement Magnitude')
    ax2.set_title('Node Displacements')
    
    # 3. Result vs Ground Truth error
    ax3 = fig.add_subplot(2, 3, 3)
    node_indices, gt_errors = calculate_result_vs_groundtruth_error(deformed_column, result_column)
    ax3.bar(node_indices, gt_errors, color='red', alpha=0.7)
    ax3.set_xlabel('Segment Index')
    ax3.set_ylabel('Error (distance)')
    ax3.set_title('Result vs Ground Truth Error')
    
    # 4. 3D visualization of columns (static comparison)
    ax4 = fig.add_subplot(2, 3, (4, 6), projection='3d')
    ax4.set_title('Column Comparison')
    
    # Plot segment positions for each column
    def plot_column(column, color, label, alpha=1.0, linewidth=2):
        # Extract segment endpoints
        points = np.array([segment.start_pos for segment in column.segments] + 
                         [column.segments[-1].end_pos])
        # Plot the column as a line
        ax4.plot(points[:, 0], points[:, 1], points[:, 2], 
                 color=color, alpha=alpha, linewidth=linewidth, label=label)
        # Plot the segment joints as points
        ax4.scatter(points[:, 0], points[:, 1], points[:, 2], 
                    color=color, alpha=alpha, s=30)
    
    # Plot all three columns
    plot_column(original_column, 'gray', 'Original', alpha=0.7)
    plot_column(deformed_column, 'blue', 'Deformed (Ground Truth)', alpha=0.8)
    plot_column(result_column, 'green', 'Optimization Result', alpha=1.0, linewidth=3)
    
    # Add coordinate system indicators
    origin = np.zeros(3)
    axis_length = max([np.max(np.abs([segment.end_pos for segment in column.segments])) 
                       for column in [original_column, deformed_column, result_column]]) * 0.3
    
    # X, Y, Z axes as arrows
    ax4.quiver(origin[0], origin[1], origin[2], axis_length, 0, 0, color='red', arrow_length_ratio=0.1)
    ax4.quiver(origin[0], origin[1], origin[2], 0, axis_length, 0, color='green', arrow_length_ratio=0.1)
    ax4.quiver(origin[0], origin[1], origin[2], 0, 0, axis_length, color='blue', arrow_length_ratio=0.1)
    
    # Add text labels for axes
    ax4.text(axis_length*1.1, 0, 0, "X", color='red', fontsize=12)
    ax4.text(0, axis_length*1.1, 0, "Y", color='green', fontsize=12)
    ax4.text(0, 0, axis_length*1.1, "Z", color='blue', fontsize=12)
    
    # Set equal aspect ratio
    ax4.set_box_aspect([1, 1, 1])
    ax4.legend(loc='upper left')
    
    # Set view angle
    ax4.view_init(elev=30, azim=45)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
    plt.savefig('optimization_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a separate animation for the optimization process
    print("Creating optimization animation...")
    fig_anim = plt.figure(figsize=(10, 8))
    ax_anim = fig_anim.add_subplot(111, projection='3d')
    
    # Setup for animation
    num_frames = len(frames)
    
    # Function to update the animation
    def update_frame(frame_idx):
        ax_anim.clear()
        frame = frames[frame_idx]
        
        # Get the point cloud data from the frame
        original_pc, deformed_pc, current_pc = frame
        
        # Convert Open3D point clouds to numpy arrays
        original_points = np.asarray(original_pc.points)
        deformed_points = np.asarray(deformed_pc.points)
        current_points = np.asarray(current_pc.points)
        
        # Get colors
        original_colors = np.asarray(original_pc.colors)
        deformed_colors = np.asarray(deformed_pc.colors)
        current_colors = np.asarray(current_pc.colors)
        
        # Only plot a subset of points for performance
        sample_size = min(500, len(original_points))
        indices = np.random.choice(len(original_points), sample_size, replace=False)
        
        # Plot the point clouds
        ax_anim.scatter(original_points[indices, 0], original_points[indices, 1], original_points[indices, 2], 
                    c=original_colors[indices], s=5, alpha=0.3, label='Original')
        ax_anim.scatter(deformed_points[indices, 0], deformed_points[indices, 1], deformed_points[indices, 2], 
                    c=deformed_colors[indices], s=5, alpha=0.3, label='Deformed (Target)')
        ax_anim.scatter(current_points[indices, 0], current_points[indices, 1], current_points[indices, 2], 
                    c=current_colors[indices], s=10, alpha=1.0, label='Current')
        
        # Add iteration number as title
        ax_anim.set_title(f'Optimization Progress - Iteration {frame_idx if frame_idx < num_frames-1 else "Final Result"}')
        
        # Set labels
        ax_anim.set_xlabel('X')
        ax_anim.set_ylabel('Y')
        ax_anim.set_zlabel('Z')
        
        # Set equal aspect ratio and fixed limits
        ax_anim.set_box_aspect([1, 1, 1])
        
        # Add legend
        ax_anim.legend()
        
        # Set view angle
        ax_anim.view_init(elev=30, azim=(45 + frame_idx * 2) % 360)  # Rotate view for better visibility
        
        return ax_anim,
    
    # Create animation
    anim = FuncAnimation(fig_anim, update_frame, frames=range(num_frames), interval=500, blit=False)
    
    # Save animation
    try:
        anim.save('optimization_animation.gif', writer='pillow', fps=2, dpi=100)
        print("Animation saved as 'optimization_animation.gif'")
    except Exception as e:
        print(f"Could not save animation: {e}")
    
    plt.show()
    
    return fig, anim

# ================== CONFIGURATION MANAGEMENT ===================
class ConfigManager:
    """
    Manages algorithm configuration parameters, allowing them to be loaded from a JSON file
    or passed directly. Provides defaults if specific parameters are not provided.
    """
    def __init__(self, config_path=None, config_dict=None):
        # Initialize with default values
        self.config = {
            # General parameters
            "random_seed": RANDOM_SEED,
            "error_threshold": DEFAULT_ERROR_THRESHOLD,
            "max_iterations": DEFAULT_MAX_ITERATIONS,
            
            # Point cloud parameters
            "points_per_cloud": DEFAULT_POINTS_PER_CLOUD,
            "min_points_for_correspondence": MIN_POINTS_FOR_CORRESPONDENCE,
            
            # Normals estimation parameters
            "normals_radius": NORMALS_RADIUS,
            "normals_max_nn": NORMALS_MAX_NN,
            
            # ICP parameters
            "icp_max_correspondence_distance": ICP_MAX_CORRESPONDENCE_DISTANCE,
            "icp_relative_fitness": ICP_RELATIVE_FITNESS,
            "icp_relative_rmse": ICP_RELATIVE_RMSE,
            "icp_max_iteration": ICP_MAX_ITERATION,
            
            # Optimization parameters
            "initial_dist_threshold": INITIAL_DIST_THRESHOLD,
            "point_weight_min": POINT_WEIGHT_MIN,
            "point_weight_scale": POINT_WEIGHT_SCALE,
            "segment_weight_min": SEGMENT_WEIGHT_MIN,
            "segment_weight_scale": SEGMENT_WEIGHT_SCALE,
            "delta_theta_scale": DELTA_THETA_SCALE,
            
            # First iteration specific parameters
            "first_iter_closest_percentage": FIRST_ITER_CLOSEST_PERCENTAGE,
            "first_iter_damping": FIRST_ITER_DAMPING,
            "first_iter_constraint_weight": FIRST_ITER_CONSTRAINT_WEIGHT,
            "first_iter_continuity_weight": FIRST_ITER_CONTINUITY_WEIGHT,
            "first_iter_max_angle_main": FIRST_ITER_MAX_ANGLE_MAIN,
            "first_iter_max_angle_other": FIRST_ITER_MAX_ANGLE_OTHER,
            
            # Later iterations parameters
            "later_iter_constraint_weight": LATER_ITER_CONSTRAINT_WEIGHT,
            "later_iter_continuity_weight": LATER_ITER_CONTINUITY_WEIGHT,
            "later_iter_max_angle_main": LATER_ITER_MAX_ANGLE_MAIN,
            "later_iter_max_angle_other": LATER_ITER_MAX_ANGLE_OTHER,
            
            # Numerical stability parameters
            "epsilon": EPSILON,
            "min_direction_norm": MIN_DIRECTION_NORM,
            
            # H-section proportions
            "h_height_ratio": H_HEIGHT_RATIO,
            "h_web_thickness_ratio": H_WEB_THICKNESS_RATIO,
            "h_flange_thickness_ratio": H_FLANGE_THICKNESS_RATIO,
            
            # Visualization parameters
            "original_color": ORIGINAL_COLOR,
            "deformed_color": DEFORMED_COLOR,
            "current_color": CURRENT_COLOR,
            "final_color": FINAL_COLOR,
            
            # Experiment configuration
            "experiment": {
                "cross_section": "circular",
                "size": 0.1,
                "num_segments": 10,
                "segment_length": 0.25,
                "deformation": [
                    {"segment": 9, "axis": "y", "angle": "pi/24"},
                    {"segment": 8, "axis": "y", "angle": "pi/20"},
                    {"segment": 7, "axis": "y", "angle": "pi/20"},
                    {"segment": 6, "axis": "y", "angle": "pi/24"}
                ]
            }
        }
        
        # Load from file if provided
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
        
        # Override with passed dictionary if provided
        if config_dict:
            self.update_config(config_dict)
    
    def load_from_file(self, config_path):
        """Load configuration from a JSON file"""
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
            self.update_config(file_config)
            print(f"Configuration loaded from {config_path}")
        except Exception as e:
            print(f"Error loading configuration from {config_path}: {e}")
    
    def update_config(self, config_dict):
        """Update configuration with values from a dictionary"""
        # Recursively update nested dictionaries
        def update_nested_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    update_nested_dict(d[k], v)
                else:
                    # Handle special case for angle expressions (e.g., "pi/24")
                    if isinstance(v, str) and 'pi' in v:
                        try:
                            # Replace 'pi' with 'np.pi' for evaluation
                            expr = v.replace('pi', 'np.pi')
                            d[k] = eval(expr)
                        except:
                            print(f"Warning: Could not evaluate angle expression '{v}' for '{k}'")
                            d[k] = v
                    else:
                        d[k] = v
        
        update_nested_dict(self.config, config_dict)
    
    def save_to_file(self, config_path):
        """Save current configuration to a JSON file"""
        # Convert numpy values and special values to serializable format
        def prepare_for_json(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            elif isinstance(value, dict):
                return {k: prepare_for_json(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [prepare_for_json(item) for item in value]
            # Add handling for float32/64 and other numpy types
            elif hasattr(value, 'item') and callable(getattr(value, 'item')):
                return value.item()
            # Format pi-based values as strings for readability
            elif isinstance(value, float):
                # Check if it's a simple fraction of pi
                for denom in [2, 3, 4, 6, 8, 12, 15, 16, 20, 24, 30, 32]:
                    if abs(value - np.pi/denom) < 1e-10:
                        return f"pi/{denom}"
                return value
            else:
                return value
        
        prepared_config = prepare_for_json(self.config)
        
        try:
            with open(config_path, 'w') as f:
                json.dump(prepared_config, f, indent=2)
            print(f"Configuration saved to {config_path}")
        except Exception as e:
            print(f"Error saving configuration to {config_path}: {e}")
    
    def get(self, key, default=None):
        """Get a configuration value, with optional default"""
        # Support nested keys with dot notation
        if '.' in key:
            parts = key.split('.')
            value = self.config
            for part in parts:
                if part in value:
                    value = value[part]
                else:
                    return default
            return value
        return self.config.get(key, default)
    
    def set(self, key, value):
        """Set a configuration value"""
        # Support nested keys with dot notation
        if '.' in key:
            parts = key.split('.')
            target = self.config
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]
            target[parts[-1]] = value
        else:
            self.config[key] = value

# Create a default config manager
config_manager = ConfigManager()

# Add function to export default configuration
def export_default_config(path="config.json"):
    """Export the default configuration to a JSON file"""
    config_manager.save_to_file(path)
    return path

# Update the run_incremental_optimization_experiment function to use the configuration
def run_incremental_optimization_experiment(config=None, config_file=None):
    """
    Run the incremental optimization experiment with the specified configuration.
    
    Args:
        config: A dictionary of configuration parameters to override the defaults
        config_file: Path to a JSON configuration file
    """
    # Create configuration with defaults
    if config_file:
        cfg = ConfigManager(config_path=config_file)
    else:
        cfg = ConfigManager(config_dict=config)
    
    # Extract experiment parameters
    cross_section = cfg.get("experiment.cross_section")
    size = cfg.get("experiment.size")
    num_segments = cfg.get("experiment.num_segments")
    segment_length = cfg.get("experiment.segment_length")
    
    # Set random seed for reproducibility
    np.random.seed(cfg.get("random_seed"))
    
    # For rectangular cross-section, use a tuple for size
    if cross_section.lower() == 'rectangular' and not isinstance(size, tuple):
        size = (size, size * 0.8)  # Default to a non-square rectangle if not specified
    
    # Create original column (straight)
    original_column = ArticulatedColumn(num_segments=num_segments, 
                                       segment_length=segment_length, 
                                       size=size, 
                                       cross_section=cross_section)
    
    # Create ground truth deformed column with a clear bend
    deformed_column = copy.deepcopy(original_column)

    # Apply deformations from configuration
    deformations = cfg.get("experiment.deformation", [])
    for deform in deformations:
        segment = deform.get("segment")
        axis = deform.get("axis")
        angle = deform.get("angle")
        
        # Convert string angle expressions if needed
        if isinstance(angle, str) and 'pi' in angle:
            try:
                # Replace 'pi' with 'np.pi' for evaluation
                expr = angle.replace('pi', 'np.pi')
                angle = eval(expr)
                print(f"Converted deformation angle expression '{angle}' to {angle}")
            except Exception as e:
                print(f"Error converting angle expression '{angle}': {e}")
                # Default to a small angle if conversion fails
                angle = np.pi/24
        
        if segment is not None and axis and angle is not None:
            deformed_column.bend_at_segment(segment, axis, angle)
    
    # Create point cloud from the deformed column
    deformed_pcd = column_to_point_cloud(deformed_column, total_points=cfg.get("points_per_cloud"))
    
    # Create a kinematic model starting from straight position
    kinematic_column = copy.deepcopy(original_column)
    
    # Run incremental optimization
    print("Running incremental optimization...")
    start_time = time.time()
    result_column, errors, frames, rmse_values = apply_incremental_optimization(
        deformed_pcd, 
        kinematic_column, 
        max_iterations=cfg.get("max_iterations"), 
        error_threshold=cfg.get("error_threshold")
    )
    elapsed_time = time.time() - start_time
    print(f"Optimization completed in {elapsed_time:.2f} seconds")
    
    # Use enhanced visualization
    visualize_optimization_results(original_column, deformed_column, result_column, frames, errors, rmse_values)
    
    # Final visualization with OpenCascade
    display, start_display, _, _ = init_display()
    
    # Original column (white)
    for segment in original_column.segments:
        shape = segment.create_geometry()
        display.DisplayShape(shape, color="WHITE", update=False)
    
    # Deformed column (blue)
    for segment in deformed_column.segments:
        shape = segment.create_geometry()
        display.DisplayShape(shape, color="BLUE", update=False)
    
    # Result column (green)
    for segment in result_column.segments:
        shape = segment.create_geometry()
        display.DisplayShape(shape, color="GREEN", update=True)
    
    display.FitAll()
    start_display() 
    
    return original_column, deformed_column, result_column, errors, rmse_values, frames, None, None

# Update main function to support command-line configuration
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Inverse Kinematics Point Cloud Optimization")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--export-config", action="store_true", help="Export default configuration to config.json")
    parser.add_argument("--cross-section", type=str, choices=["circular", "rectangular", "h"], 
                       help="Cross-section type for the column")
    parser.add_argument("--size", type=float, help="Size parameter for the column")
    args = parser.parse_args()
    
    if args.export_config:
        path = export_default_config()
        print(f"Default configuration exported to {path}")
    elif args.config:
        # Run with the specified configuration file
        run_incremental_optimization_experiment(config_file=args.config)
    else:
        # Create config from command line arguments
        config = {}
        if args.cross_section:
            config["experiment"] = {"cross_section": args.cross_section}
        if args.size:
            if "experiment" not in config:
                config["experiment"] = {}
            config["experiment"]["size"] = args.size
        
        # Run with the specified or default configuration
        run_incremental_optimization_experiment(config=config)

# ================== BENCHMARKING ===================
class PerformanceBenchmark:
    """
    Benchmarks the algorithm performance with different parameter sets.
    Compares results and generates reports.
    """
    def __init__(self):
        self.results = {}
        self.default_config = ConfigManager().config
    
    def run_benchmark_set(self, benchmark_configs, name="benchmark", save_report=True):
        """
        Run a set of benchmarks with different configurations
        
        Args:
            benchmark_configs: List of dictionaries with configuration overrides
            name: Name for this benchmark set
            save_report: Whether to save a detailed report
        """
        print(f"=== Starting benchmark set: {name} ===")
        self.results[name] = []
        
        for i, config_override in enumerate(benchmark_configs):
            config_name = config_override.get("name", f"Config {i+1}")
            print(f"\nRunning benchmark {i+1}/{len(benchmark_configs)}: {config_name}")
            
            # Create a tracker that logs to a file
            tracker = OptimizationTracker(f"Benchmark {name} - {config_name}", log_to_file=True)
            
            # Create a deep copy of the default config
            test_config = copy.deepcopy(self.default_config)
            
            # Apply the override values
            if "override" in config_override:
                # Recursively update the config
                def update_config(target, source):
                    for k, v in source.items():
                        if isinstance(v, dict) and k in target and isinstance(target[k], dict):
                            update_config(target[k], v)
                        else:
                            # Handle angle expressions
                            if isinstance(v, str) and 'pi' in v:
                                try:
                                    # Replace 'pi' with 'np.pi' for evaluation
                                    expr = v.replace('pi', 'np.pi')
                                    target[k] = eval(expr)
                                    tracker.log(f"Converted angle expression '{v}' to {target[k]}")
                                except Exception as e:
                                    tracker.log(f"Error converting angle expression '{v}': {e}")
                                    target[k] = v
                            else:
                                target[k] = v
                
                update_config(test_config, config_override["override"])
            
            # Create columns using the configuration
            tracker.start()
            
            # Extract experiment parameters
            cross_section = test_config["experiment"]["cross_section"]
            size = test_config["experiment"]["size"]
            num_segments = test_config["experiment"]["num_segments"]
            segment_length = test_config["experiment"]["segment_length"]
            
            try:
                # Set random seed for reproducibility
                np.random.seed(test_config["random_seed"])
                
                # Create columns and run test
                original_column = ArticulatedColumn(num_segments=num_segments, 
                                                  segment_length=segment_length, 
                                                  size=size, 
                                                  cross_section=cross_section)
                
                # Create ground truth deformed column
                deformed_column = copy.deepcopy(original_column)
                
                # Apply deformations from configuration
                deformations = test_config["experiment"]["deformation"]
                for deform in deformations:
                    segment = deform["segment"]
                    axis = deform["axis"]
                    angle = deform["angle"]
                    
                    # Convert angle expressions if needed
                    if isinstance(angle, str) and 'pi' in angle:
                        try:
                            expr = angle.replace('pi', 'np.pi')
                            angle = eval(expr)
                            tracker.log(f"Converted benchmark angle expression '{angle}' to {angle}")
                        except Exception as e:
                            tracker.log(f"Error converting angle expression '{angle}': {e}")
                            angle = np.pi/24
                    
                    deformed_column.bend_at_segment(segment, axis, angle)
                
                # Create point cloud from the deformed column
                deformed_pcd = column_to_point_cloud(deformed_column, total_points=test_config["points_per_cloud"])
                
                # Create a kinematic model starting from straight position
                kinematic_column = copy.deepcopy(original_column)
                
                # Run incremental optimization
                tracker.log(f"Running optimization with {config_name}")
                start_time = time.time()
                
                result_column, errors, frames, rmse_values = apply_incremental_optimization(
                    deformed_pcd, 
                    kinematic_column, 
                    max_iterations=test_config["max_iterations"], 
                    error_threshold=test_config["error_threshold"]
                )
                
                elapsed_time = time.time() - start_time
                
                # Calculate final error metrics
                final_rmse = rmse_values[-1] if rmse_values else None
                convergence_iterations = len(rmse_values)
                
                # Calculate error vs ground truth
                node_indices, gt_errors = calculate_result_vs_groundtruth_error(deformed_column, result_column)
                max_gt_error = np.max(gt_errors) if len(gt_errors) > 0 else None
                avg_gt_error = np.mean(gt_errors) if len(gt_errors) > 0 else None
                
                # Store the results
                result = {
                    "name": config_name,
                    "config": config_override,
                    "time": elapsed_time,
                    "iterations": convergence_iterations,
                    "final_rmse": final_rmse,
                    "max_gt_error": max_gt_error,
                    "avg_gt_error": avg_gt_error,
                    "error_values": errors,
                    "rmse_values": rmse_values
                }
                
                self.results[name].append(result)
                
                # Save visualization if requested
                if config_override.get("save_visualization", False):
                    # Create a simplified visualization and save it
                    fig = plt.figure(figsize=(12, 8))
                    
                    # RMSE plot
                    ax1 = fig.add_subplot(211)
                    ax1.plot(rmse_values, 'b-', linewidth=2)
                    ax1.set_yscale('log')
                    ax1.set_xlabel('Iteration')
                    ax1.set_ylabel('RMSE (log scale)')
                    ax1.set_title(f'Convergence for {config_name}')
                    ax1.grid(True)
                    
                    # GT error plot
                    ax2 = fig.add_subplot(212)
                    ax2.bar(node_indices, gt_errors, color='red', alpha=0.7)
                    ax2.set_xlabel('Segment Index')
                    ax2.set_ylabel('Error vs Ground Truth')
                    ax2.set_title('Segment Errors Compared to Ground Truth')
                    
                    plt.tight_layout()
                    plt.savefig(f"benchmark_{name}_{config_name.replace(' ', '_')}.png", dpi=300)
                    plt.close(fig)
                
                tracker.log(f"Benchmark completed: Time={elapsed_time:.2f}s, Iterations={convergence_iterations}, Final RMSE={final_rmse:.6f}, GT Error={avg_gt_error:.6f}")
                
            except Exception as e:
                tracker.log(f"Error in benchmark: {e}")
                import traceback
                tracker.log(traceback.format_exc())
                
                # Add error result
                result = {
                    "name": config_name,
                    "config": config_override,
                    "error": str(e),
                    "time": None,
                    "iterations": None,
                    "final_rmse": None,
                    "max_gt_error": None,
                    "avg_gt_error": None
                }
                
                self.results[name].append(result)
            
            finally:
                # End tracking
                tracker.finish()
        
        # Generate report
        if save_report:
            self.generate_report(name)
    
    def generate_report(self, benchmark_name):
        """Generate a detailed report for a benchmark set"""
        if benchmark_name not in self.results:
            print(f"No results found for benchmark: {benchmark_name}")
            return
        
        results = self.results[benchmark_name]
        
        # Create report file
        report_filename = f"benchmark_report_{benchmark_name}.txt"
        with open(report_filename, "w") as f:
            f.write(f"=== Benchmark Report: {benchmark_name} ===\n\n")
            
            # Summary table
            f.write("Summary:\n")
            f.write("-" * 110 + "\n")
            f.write(f"{'Config Name':<25} {'Time (s)':<12} {'Iterations':<12} {'Final RMSE':<15} {'Avg GT Error':<15} {'Max GT Error':<15}\n")
            f.write("-" * 110 + "\n")
            
            for result in results:
                name = result.get("name", "Unknown")
                time = f"{result.get('time', 'N/A'):.2f}" if result.get('time') is not None else "N/A"
                iterations = result.get("iterations", "N/A")
                final_rmse = f"{result.get('final_rmse', 'N/A'):.6f}" if result.get('final_rmse') is not None else "N/A"
                avg_gt_error = f"{result.get('avg_gt_error', 'N/A'):.6f}" if result.get('avg_gt_error') is not None else "N/A"
                max_gt_error = f"{result.get('max_gt_error', 'N/A'):.6f}" if result.get('max_gt_error') is not None else "N/A"
                
                f.write(f"{name:<25} {time:<12} {iterations:<12} {final_rmse:<15} {avg_gt_error:<15} {max_gt_error:<15}\n")
            
            f.write("-" * 110 + "\n\n")
            
            # Detailed results
            f.write("Detailed Results:\n\n")
            
            for result in results:
                f.write(f"Configuration: {result.get('name', 'Unknown')}\n")
                f.write("-" * 80 + "\n")
                
                if "error" in result:
                    f.write(f"ERROR: {result['error']}\n")
                    f.write("-" * 80 + "\n\n")
                    continue
                
                f.write(f"Execution time: {result.get('time', 'N/A'):.2f} seconds\n")
                f.write(f"Iterations: {result.get('iterations', 'N/A')}\n")
                f.write(f"Final RMSE: {result.get('final_rmse', 'N/A'):.6f}\n")
                f.write(f"Avg GT Error: {result.get('avg_gt_error', 'N/A'):.6f}\n")
                f.write(f"Max GT Error: {result.get('max_gt_error', 'N/A'):.6f}\n\n")
                
                # Parameter overrides
                f.write("Parameter Overrides:\n")
                if "config" in result and "override" in result["config"]:
                    # Format the overrides
                    def format_overrides(overrides, prefix=""):
                        lines = []
                        for k, v in overrides.items():
                            if isinstance(v, dict):
                                lines.append(f"{prefix}{k}:")
                                lines.extend(format_overrides(v, prefix + "  "))
                            else:
                                lines.append(f"{prefix}{k}: {v}")
                        return lines
                    
                    override_lines = format_overrides(result["config"]["override"])
                    for line in override_lines:
                        f.write(f"  {line}\n")
                else:
                    f.write("  No overrides specified\n")
                
                f.write("-" * 80 + "\n\n")
        
        print(f"Benchmark report saved to {report_filename}")
        
        # Create a visualization comparing RMSE convergence
        self.plot_benchmark_comparison(benchmark_name)
    
    def plot_benchmark_comparison(self, benchmark_name):
        """Create a visual comparison of RMSE convergence for different configurations"""
        if benchmark_name not in self.results:
            print(f"No results found for benchmark: {benchmark_name}")
            return
        
        results = self.results[benchmark_name]
        
        # Create figure for RMSE comparison
        plt.figure(figsize=(12, 8))
        
        for result in results:
            if "error" in result:
                continue
            
            name = result.get("name", "Unknown")
            rmse_values = result.get("rmse_values", [])
            
            if rmse_values:
                plt.plot(rmse_values, label=name, linewidth=2, marker='o', markersize=3, markevery=5)
        
        plt.yscale('log')
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('RMSE (log scale)', fontsize=14)
        plt.title(f'RMSE Convergence Comparison - {benchmark_name}', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save the figure
        plt.savefig(f"benchmark_comparison_{benchmark_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a bar chart comparing final metrics
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        names = []
        times = []
        final_rmses = []
        avg_errors = []
        
        for result in results:
            if "error" in result:
                continue
            
            names.append(result.get("name", "Unknown"))
            times.append(result.get("time", 0))
            final_rmses.append(result.get("final_rmse", 0))
            avg_errors.append(result.get("avg_gt_error", 0))
        
        # Plot execution times
        ax1.bar(names, times, color='blue', alpha=0.7)
        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Execution Time')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot final RMSE
        ax2.bar(names, final_rmses, color='green', alpha=0.7)
        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('Final RMSE')
        ax2.set_title('Final RMSE')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot GT error
        ax3.bar(names, avg_errors, color='red', alpha=0.7)
        ax3.set_xlabel('Configuration')
        ax3.set_ylabel('Avg Ground Truth Error')
        ax3.set_title('Ground Truth Error')
        ax3.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"benchmark_metrics_{benchmark_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Benchmark visualizations saved as benchmark_comparison_{benchmark_name}.png and benchmark_metrics_{benchmark_name}.png")

# Example of how to use the benchmark system
def run_example_benchmark():
    """Run an example benchmark with different parameter sets"""
    benchmark = PerformanceBenchmark()
    
    # Define benchmark configurations
    configs = [
        {
            "name": "Default",
            "override": {},  # No changes from default
            "save_visualization": True
        },
        {
            "name": "Higher Point Density",
            "override": {
                "points_per_cloud": 5000  # 5x default
            },
            "save_visualization": True
        },
        {
            "name": "Tighter Error Threshold",
            "override": {
                "error_threshold": 0.0005  # Half of default
            },
            "save_visualization": True
        },
        {
            "name": "Lower Constraint Weights",
            "override": {
                "first_iter_constraint_weight": 0.05,
                "later_iter_constraint_weight": 0.01
            },
            "save_visualization": True
        },
        {
            "name": "More Aggressive Damping",
            "override": {
                "first_iter_damping": 2.0,
                "delta_theta_scale": 0.2
            },
            "save_visualization": True
        }
    ]
    
    # Run the benchmark
    benchmark.run_benchmark_set(configs, name="parameter_tuning")
    
    return benchmark

# Add command for running benchmarks to the main function
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Inverse Kinematics Point Cloud Optimization")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--export-config", action="store_true", help="Export default configuration to config.json")
    parser.add_argument("--cross-section", type=str, choices=["circular", "rectangular", "h"], 
                       help="Cross-section type for the column")
    parser.add_argument("--size", type=float, help="Size parameter for the column")
    parser.add_argument("--benchmark", action="store_true", help="Run example benchmark")
    args = parser.parse_args()
    
    if args.benchmark:
        print("Running benchmark mode...")
        benchmark = run_example_benchmark()
    elif args.export_config:
        path = export_default_config()
        print(f"Default configuration exported to {path}")
    elif args.config:
        # Run with the specified configuration file
        run_incremental_optimization_experiment(config_file=args.config)
    else:
        # Create config from command line arguments
        config = {}
        if args.cross_section:
            config["experiment"] = {"cross_section": args.cross_section}
        if args.size:
            if "experiment" not in config:
                config["experiment"] = {}
            config["experiment"]["size"] = args.size
        
        # Run with the specified or default configuration
        run_incremental_optimization_experiment(config=config)

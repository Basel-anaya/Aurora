# Vehicle Registration Card OCR System

An advanced OCR system built on Qwen 2.5 Vision Language Model to extract structured information from vehicle registration cards.

## Features

- **High-Accuracy OCR**: Leverages Qwen 2.5 Vision model for superior text recognition performance
- **Multi-Side Processing**: Handles both front and back sides of vehicle registration cards
- **Image Enhancement**: Improves image quality before processing to increase accuracy
- **Structured Data Extraction**: Outputs standardized JSON with vehicle, owner, and registration details
- **Validation and Normalization**: Automatically validates and normalizes extracted data
- **Multiple Output Formats**: Supports JSON, YAML, and CSV output formats
- **Performance Optimization**: Includes ONNX Runtime and TensorRT integrations for faster inference

## System Architecture

```
OCR System
│
├── API Layer
│   ├── Upload Service
│   ├── Validation Service
│   └── Webhook Integration
│
├── Core OCR Engine
│   ├── Qwen Vision Service
│   ├── Image Processor
│   ├── Data Extractor
│   ├── Registry & Optimizer
│   └── Model Interface
│
├── Storage Layer
│   ├── Image Storage
│   ├── Database Integration
│   └── Audit Logging
│
├── Monitoring & Operations
│   ├── Performance Metrics
│   ├── Logging System
│   ├── Alerting Service
│   └── Dashboard
│
└── Utilities
    ├── Configuration
    ├── Helpers
    ├── Validators
    └── Image Processing
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- 16GB+ RAM (32GB+ recommended for production)

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/vehicle-registration-ocr.git
   cd vehicle-registration-ocr
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download Qwen 2.5 model (automatically done on first run, or manually):
   ```bash
   python -c "from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration; AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct-AWQ'); Qwen2_5_VLForConditionalGeneration.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct-AWQ')"
   ```

## Usage

### Command Line Interface

Process a single registration card:

```bash
python main.py --front path/to/front_image.jpg --back path/to/back_image.jpg --output-dir results
```

Options:
- `--front`: Path to front side image (required)
- `--back`: Path to back side image (optional)
- `--output-dir`: Directory to save results (default: 'output')
- `--save-enhanced`: Save enhanced images
- `--no-flash-attention`: Disable flash attention
- `--format`: Output format (json, yaml, csv) (default: json)

### Python API

```python
from core.qwen_service import QwenVisionService
from core.processor import RegistrationCardProcessor

# Initialize services
qwen_service = QwenVisionService()
processor = RegistrationCardProcessor(qwen_service=qwen_service)

# Process registration card
result = processor.process_registration_card(
    front_image_path="path/to/front_image.jpg",
    back_image_path="path/to/back_image.jpg",
    save_enhanced=True,
    output_dir="output"
)

# Access extracted data
vehicle_make = result.get('vehicle', {}).get('make')
owner_name = result.get('owner', {}).get('name')
expiry_date = result.get('registration', {}).get('expiry_date')
```

## Docker Deployment

Build and run the Docker container:

```bash
docker build -t vehicle-registration-ocr .
docker run -p 8000:8000 -v $(pwd)/images:/app/images vehicle-registration-ocr
```

## Performance Optimization

For improved performance, the system supports:

1. **Flash Attention 2**: Enables faster processing with lower memory usage
2. **ONNX Runtime**: Speeds up inference on CPUs and GPUs
3. **TensorRT**: Further optimizes performance on NVIDIA GPUs
4. **Quantization**: Reduces model size and increases inference speed

Configure these optimizations in `config.py` or via environment variables.

## Model Information

This system uses the Qwen 2.5 Vision Language Model:

- **Model**: Qwen2.5-VL-7B-Instruct-AWQ
- **Size**: 7 billion parameters (compressed with AWQ)
- **Context Window**: Supports up to 16K tokens
- **Visual Tokens**: 4-16384 tokens per image
- **Multimodal Understanding**: Processes both text and images together

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
black .
flake8 .
```

## License

[MIT License](LICENSE)

## Contributors

- Your Name - Initial work
# Aurora

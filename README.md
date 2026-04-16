# localOCR

Local OCR text extraction from images and PDFs using EasyOCR.

## Overview

This project provides a Python-based OCR (Optical Character Recognition) system that extracts text from images and PDF files locally using EasyOCR. No cloud services or API keys required - everything runs on your machine.

## Features

- **Local Processing**: Extract text entirely on your machine
- **EasyOCR Integration**: Uses the EasyOCR engine
- **GPU Support**: Automatically uses GPU if available
- **Batch Processing**: Extract text from multiple images/PDFs efficiently
- **Easy-to-Use API**: Simple Python interface for integration into other projects
- **Multi-format Support**: Works with PNG, JPG, JPEG, BMP, GIF, WebP, and PDF files
- **PDF Processing**: Extracts text from PDFs with text layers or converts scanned PDFs to images for OCR

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or your preferred package manager
- (Optional) CUDA 11.8+ for GPU acceleration

### Setup

1. Clone the repository:
```bash
cd d:\AI\localai\localOCR
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# or
source venv/bin/activate  # On Linux/macOS
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: The first installation may take some time as it downloads the model files (~2GB).

## Usage

### Web Interface

Start the web server:
```bash
python web.py
```

Then open your browser and navigate to:
```
http://localhost:5000
```

Features:
- Drag and drop image/PDF upload
- Real-time text extraction
- Copy extracted text to clipboard
- Support for multiple formats (PNG, JPG, JPEG, BMP, GIF, WebP, PDF)
- Model and device information display

### Command Line

Extract text from a single image or PDF:
```bash
python main.py path/to/image.jpg
python main.py path/to/document.pdf
```

### Python API

```python
from localocr import OCRExtractor

# Initialize the extractor
extractor = OCRExtractor()

# Extract text from a single image or PDF
text = extractor.extract_text("image.jpg")
print(text)

# Extract text from a PDF
pdf_text = extractor.extract_text("document.pdf")
print(pdf_text)

# Extract text from multiple files (images and PDFs)
texts = extractor.extract_text_batch(["image1.jpg", "document.pdf", "image2.png"])
for i, text in enumerate(texts):
    print(f"File {i+1}: {text}")
```

### Custom Model

```python
from localocr import OCRExtractor

# Use different languages with EasyOCR
extractor = OCRExtractor(languages=['en', 'ru'])  # Specify languages
text = extractor.extract_text("image.jpg")
```

### REST API

The web server provides the following endpoints:

**Health Check**
```bash
GET /api/health
```

**Model Information**
```bash
GET /api/model-info
```
Returns:
```json
{
    "model": "EasyOCR",
    "device": "cuda" or "cpu"
}
```

**Extract Text from Image**
```bash
POST /api/extract
```
Request: multipart/form-data with `file` field
Response:
```json
{
    "success": true,
    "filename": "image.jpg",
    "text": "Extracted text content..."
}
```

## Project Structure

```
localOCR/
├── src/
│   └── localocr/
│       ├── __init__.py
│       └── ocr.py          # Main OCR extractor module
├── templates/
│   └── index.html          # Web interface template
├── static/
│   ├── style.css           # Web interface styling
│   └── script.js           # Web interface JavaScript
├── main.py                 # CLI entry point
├── web.py                  # Flask web server
├── pyproject.toml          # Project metadata and dependencies
├── requirements.txt        # Pip dependencies
├── README.md              # This file
└── .github/
    └── copilot-instructions.md
```

## Model Information

- **Model Name**: EasyOCR
- **Source**: [EasyOCR GitHub](https://github.com/JaidedAI/EasyOCR)
- **Type**: OCR engine based on deep learning

## Performance Notes

- First run will download model files (~2GB)
- GPU acceleration recommended for batch processing
- Processing time depends on image size and complexity

## Troubleshooting

### Out of Memory
If you encounter CUDA out of memory errors:
1. Reduce batch size
2. Use CPU instead: `torch.device("cpu")`
3. Increase system memory

### Model Download Issues
If the model download fails:
1. Check internet connection
2. Ensure sufficient disk space (>3GB)
3. Try again - HuggingFace may have temporary issues

## Development

Install development dependencies:
```bash
pip install -e ".[dev]"
```

Run tests:
```bash
pytest
```

Format code:
```bash
black src/ main.py
isort src/ main.py
```

## License

[Specify your license here]

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## References

- [EasyOCR on GitHub](https://github.com/JaidedAI/EasyOCR)
- [Transformers Library](https://huggingface.co/docs/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/)

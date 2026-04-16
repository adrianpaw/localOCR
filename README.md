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
- **Signature Detection**: Computer vision-based signature detection using OpenCV
- **Signature Extraction**: Extract signature regions from documents
- **Web Interface**: User-friendly web UI for text extraction and signature detection

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
**Request Formats:**

1. **Multipart Form Data** (traditional file upload):
   - Content-Type: `multipart/form-data`
   - Field: `file` (required) - the image/PDF file
   - Parameter: `detect_signatures` (optional) - set to `true` to enable signature detection

2. **JSON Body with Base64** (programmatic API):
   - Content-Type: `application/json`
   - JSON structure:
   ```json
   {
     "file_base64": "base64-encoded-file-data",
     "file_name": "optional_filename.png",
     "file_extension": "optional_extension",
     "detect_signatures": true
   }
   ```
   - `file_base64`: Required base64-encoded file content (can include data URL prefix)
   - `file_name`: Optional filename for reference
   - `file_extension`: Optional file extension (default: 'png')
   - `detect_signatures`: Optional boolean (default: false)

**Response (without signature detection):**
```json
{
    "success": true,
    "filename": "image.jpg",
    "text": "Extracted text content..."
}
```
**Response (with signature detection):**
```json
{
    "success": true,
    "filename": "image.jpg",
    "text": "Extracted text content...",
    "signatures": [
        {
            "id": 0,
            "bbox": [100, 200, 150, 50],
            "area": 7500,
            "aspect_ratio": 3.0,
            "line_density": 0.18,
            "solidity": 0.72,
            "center": [175, 225]
        }
    ],
    "signature_count": 1,
    "has_signatures": true
}
```

**Detect Signatures Only**
```bash
POST /api/detect-signatures
```
**Request Formats:**

1. **Multipart Form Data**:
   - Content-Type: `multipart/form-data`
   - Field: `file` (required) - the image/PDF file

2. **JSON Body with Base64**:
   - Content-Type: `application/json`
   - JSON structure:
   ```json
   {
     "file_base64": "base64-encoded-file-data",
     "file_name": "optional_filename.png"
   }
   ```

**Response:**
```json
{
    "success": true,
    "filename": "image.jpg",
    "signatures": [...],
    "signature_count": 2,
    "has_signatures": true
}
```

### Signature Detection

The system now includes computer vision-based signature detection using OpenCV. This feature can be used standalone or integrated with OCR text extraction.

**Python API Example:**
```python
from localocr import OCRExtractor

# Initialize extractor
extractor = OCRExtractor(languages=['en', 'ru'])

# Detect signatures only
signatures = extractor.detect_signatures("document.png")
print(f"Found {len(signatures)} signatures")

# Extract text with signature detection
result = extractor.extract_with_signatures("document.png")
print(f"Text: {result['text'][:100]}...")
print(f"Signatures: {result['signature_count']}")

# Standalone signature detector
from localocr.signature_detector import SignatureDetector
detector = SignatureDetector()
signatures = detector.detect_signatures("document.png")
```

**Signature Detection Parameters:**
- `min_signature_area`: Minimum area in pixels (default: 500)
- `max_signature_area`: Maximum area in pixels (default: 50000)
- `aspect_ratio_range`: Acceptable width/height ratio (default: 0.3-3.0)
- `line_density_threshold`: Minimum edge density (default: 0.15)
- `solidity_threshold`: Minimum contour solidity (default: 0.5)

**Demo Script:**
Run the demonstration script to see signature detection in action:
```bash
python demo_signature_detection.py
```

## Project Structure

```
localOCR/
├── src/
│   └── localocr/
│       ├── __init__.py
│       ├── ocr.py                 # Main OCR extractor module
│       └── signature_detector.py  # Signature detection module
├── templates/
│   └── index.html          # Web interface template
├── static/
│   ├── style.css           # Web interface styling
│   └── script.js           # Web interface JavaScript
├── main.py                 # CLI entry point
├── web.py                  # Flask web server
├── demo_signature_detection.py  # Signature detection demo
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

## Deployment

This application can be deployed to various cloud platforms. Below are instructions for popular hosting services.

### Prerequisites

1. Ensure your code is pushed to a Git repository (GitHub, GitLab, etc.)
2. Have an account on the chosen hosting platform

### Render.com (Recommended)

1. **Create a new Web Service** on [Render](https://render.com)
2. **Connect your repository**
3. **Configure settings**:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn web:app`
   - **Environment Variables**: Add any needed variables from `.env.example`
4. **Click Deploy**

Render will automatically build and deploy your application. The first build may take longer due to downloading OCR models (~2GB).

### Railway.app

1. **Create a new project** on [Railway](https://railway.app)
2. **Connect your repository**
3. **Railway will automatically detect** the Python project and install dependencies
4. **Add environment variables** if needed
5. **Deploy**

### PythonAnywhere

1. **Create a new Web App** on [PythonAnywhere](https://pythonanywhere.com)
2. **Upload your code** via Git or manual upload
3. **Create a virtual environment** and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure WSGI file** to point to `web:app`
5. **Reload the application**

### Environment Variables

For production, set the following environment variables (optional):

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `5000` | Port to listen on |
| `HOST` | `0.0.0.0` | Host to bind to |
| `OCR_LANGUAGES` | `en,ru` | Comma-separated language codes |
| `OCR_GPU` | `true` | Whether to use GPU if available |
| `MAX_CONTENT_LENGTH_MB` | `50` | Maximum upload size in MB |
| `UPLOAD_FOLDER` | `uploads` | Temporary upload directory |
| `FLASK_DEBUG` | `false` | Enable Flask debug mode |

### Notes

- The first request will trigger model download, which may take several minutes.
- GPU acceleration may not be available on all hosting platforms.
- For large files, ensure sufficient memory and disk space.

## License

[Specify your license here]

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## References

- [EasyOCR on GitHub](https://github.com/JaidedAI/EasyOCR)
- [Transformers Library](https://huggingface.co/docs/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/)

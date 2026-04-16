"""
Flask web interface for localOCR
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add src directory to path so localocr can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from localocr import OCRExtractor

# Load environment variables (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure JSON encoder to preserve Cyrillic characters
app.json.ensure_ascii = False
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH_MB', 50)) * 1024 * 1024  # Default 50MB
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif', 'webp', 'pdf'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize OCR extractor (lazy loading)
ocr_extractor = None
model_error = None


def get_ocr_extractor():
    """Get or initialize OCR extractor"""
    global ocr_extractor, model_error
    if ocr_extractor is None and model_error is None:
        try:
            # Read languages from environment variable
            languages_str = os.getenv('OCR_LANGUAGES', 'en,ru')
            languages = [lang.strip() for lang in languages_str.split(',') if lang.strip()]
            gpu = os.getenv('OCR_GPU', 'true').lower() == 'true'
            
            logger.info(f"Initializing OCR extractor with languages: {languages}, GPU: {gpu}")
            ocr_extractor = OCRExtractor(languages=languages, gpu=gpu)
            logger.info("OCR extractor initialized successfully")
        except Exception as e:
            model_error = str(e)
            logger.error(f"Error loading OCR model: {model_error}")
    return ocr_extractor


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/api/extract', methods=['POST'])
def extract_text():
    """Extract text from uploaded image"""
    try:
        # Check if model is loaded
        extractor = get_ocr_extractor()
        if extractor is None:
            return jsonify({'error': f'Model loading failed: {model_error}'}), 500
        
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': f'File type not allowed. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"Processing file: {filename}")
        
        # Extract text
        text = extractor.extract_text(filepath)
        
        # Clean up
        os.remove(filepath)
        
        logger.info(f"Successfully extracted text from {filename}")
        
        return jsonify({
            'success': True,
            'filename': filename,
            'text': text
        }), 200
    
    except Exception as e:
        logger.error(f"Error during text extraction: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok'}), 200


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    try:
        extractor = get_ocr_extractor()
        if extractor is None:
            return jsonify({'error': f'Model not loaded: {model_error}', 'loaded': False}), 503
        
        return jsonify({
            'loaded': True,
            'languages': extractor.languages,
            'gpu': extractor.gpu,
            'model': 'EasyOCR'
        }), 200
    except Exception as e:
        logger.error(f"Error in model-info endpoint: {str(e)}")
        return jsonify({'error': str(e), 'loaded': False}), 500


if __name__ == '__main__':
    # Get host and port from environment variables (for production)
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    
    logger.info(f"Starting server on {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)

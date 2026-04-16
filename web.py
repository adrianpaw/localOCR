"""
Flask web interface for localOCR
"""

import os
import sys
import json
import logging
import base64
import tempfile
from pathlib import Path

# Add src directory to path so localocr can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import uuid
import re
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


def secure_filename_unicode(filename):
    """
    Custom secure filename that preserves Unicode characters.
    Adds UUID prefix to ensure uniqueness and avoid collisions.
    """
    if not filename or filename.strip() == '':
        return str(uuid.uuid4())[:8]
    
    # Extract extension
    if '.' in filename:
        name_part, ext = filename.rsplit('.', 1)
        ext = ext.lower()
    else:
        name_part = filename
        ext = None
    
    # Clean the name part: keep Unicode letters, digits, spaces, dots, dashes, underscores
    # Remove potentially dangerous characters
    # This regex keeps Unicode word characters, spaces, dots, dashes, underscores
    name_part = re.sub(r'[^\w\s\.\-_]', '', name_part, flags=re.UNICODE)
    # Replace multiple spaces with single underscore
    name_part = re.sub(r'\s+', '_', name_part)
    # Remove leading/trailing underscores/dots
    name_part = name_part.strip('._')
    
    # If name is empty after cleaning, use a placeholder
    if not name_part:
        name_part = 'file'
    
    # Add UUID prefix for uniqueness (first 8 chars)
    uuid_prefix = str(uuid.uuid4())[:8]
    
    # Construct final filename
    if ext:
        result = f"{uuid_prefix}_{name_part}.{ext}"
    else:
        result = f"{uuid_prefix}_{name_part}"
    
    return result


def detect_file_extension_from_bytes(file_data: bytes) -> str:
    """
    Detect file extension from binary data using magic numbers.
    
    Args:
        file_data: Binary file data
        
    Returns:
        File extension without dot (e.g., 'png', 'jpg', 'pdf')
    """
    # Check PDF
    if file_data.startswith(b'%PDF'):
        return 'pdf'
    
    # Check PNG
    if file_data.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'png'
    
    # Check JPEG
    if file_data.startswith(b'\xff\xd8\xff'):
        return 'jpg'
    
    # Check JPEG 2000
    if file_data.startswith(b'\x00\x00\x00\x0cjP  \r\n\x87\n'):
        return 'jp2'
    
    # Check BMP
    if file_data.startswith(b'BM'):
        return 'bmp'
    
    # Check GIF
    if file_data.startswith(b'GIF87a') or file_data.startswith(b'GIF89a'):
        return 'gif'
    
    # Check WebP
    if file_data.startswith(b'RIFF') and len(file_data) > 12 and file_data[8:12] == b'WEBP':
        return 'webp'
    
    # Default to png if unknown
    return 'png'


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/api/extract', methods=['POST'])
def extract_text():
    """Extract text from uploaded image with optional signature detection"""
    try:
        # Check if model is loaded
        extractor = get_ocr_extractor()
        if extractor is None:
            return jsonify({'error': f'Model loading failed: {model_error}'}), 500
        
        filepath = None
        filename = None
        detect_signatures = False
        
        # Check if request is JSON (for base64 upload)
        if request.is_json:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'Invalid JSON body'}), 400
            
            # Get detect_signatures from JSON
            detect_signatures = data.get('detect_signatures', False)
            
            # Check for base64 file data
            if 'file' not in data and 'file_base64' not in data:
                return jsonify({'error': 'No file provided in JSON body. Provide "file" (multipart) or "file_base64" (base64 string)'}), 400
            
            # Handle base64 file
            if 'file_base64' in data:
                base64_data = data['file_base64']
                # Remove data URL prefix if present
                if ',' in base64_data:
                    base64_data = base64_data.split(',', 1)[1]
                
                try:
                    file_data = base64.b64decode(base64_data)
                except Exception as e:
                    return jsonify({'error': f'Invalid base64 data: {str(e)}'}), 400
                
                # Determine file extension from binary data or use provided
                file_extension = data.get('file_extension')
                if not file_extension:
                    file_extension = detect_file_extension_from_bytes(file_data)
                
                if 'file_name' in data:
                    filename = secure_filename_unicode(data['file_name'])
                    # Ensure filename has correct extension
                    if not filename.lower().endswith(f'.{file_extension}'):
                        # Add extension if missing
                        if '.' in filename:
                            # Replace existing extension
                            name_part = filename.rsplit('.', 1)[0]
                            filename = f"{name_part}.{file_extension}"
                        else:
                            filename = f"{filename}.{file_extension}"
                else:
                    filename = f"upload_{uuid.uuid4().hex[:8]}.{file_extension}"
                
                # Save to temporary file
                temp_dir = tempfile.gettempdir()
                filepath = os.path.join(temp_dir, filename)
                with open(filepath, 'wb') as f:
                    f.write(file_data)
                
                logger.info(f"Processing base64 file: {filename}, detect_signatures: {detect_signatures}")
        
        # If not JSON or no base64, check for multipart form file
        if filepath is None:
            if 'file' not in request.files:
                return jsonify({'error': 'No file provided. Use multipart form with "file" field or JSON with "file_base64"'}), 400
            
            file = request.files['file']
            
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({'error': f'File type not allowed. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
            
            # Get detect_signatures from form if not already set from JSON
            if not request.is_json:
                detect_signatures = request.form.get('detect_signatures', 'false').lower() == 'true'
            
            # Save file
            filename = secure_filename_unicode(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            logger.info(f"Processing multipart file: {filename}, detect_signatures: {detect_signatures}")
        
        response_data = {
            'success': True,
            'filename': filename,
            'text': None,
            'signatures': None,
            'signature_count': 0,
            'has_signatures': False
        }
        
        if detect_signatures:
            # Extract text and detect signatures
            result = extractor.extract_with_signatures(filepath)
            response_data.update({
                'text': result['text'],
                'signatures': result['signatures'],
                'signature_count': result['signature_count'],
                'has_signatures': result['has_signatures']
            })
        else:
            # Extract text only
            text = extractor.extract_text(filepath)
            response_data['text'] = text
        
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)
        
        logger.info(f"Successfully processed {filename}")
        
        return jsonify(response_data), 200
    
    except Exception as e:
        logger.error(f"Error during text extraction: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect-signatures', methods=['POST'])
def detect_signatures():
    """Detect signatures in uploaded image (without text extraction)"""
    try:
        # Check if model is loaded
        extractor = get_ocr_extractor()
        if extractor is None:
            return jsonify({'error': f'Model loading failed: {model_error}'}), 500
        
        filepath = None
        filename = None
        
        # Check if request is JSON (for base64 upload)
        if request.is_json:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'Invalid JSON body'}), 400
            
            # Check for base64 file data
            if 'file' not in data and 'file_base64' not in data:
                return jsonify({'error': 'No file provided in JSON body. Provide "file" (multipart) or "file_base64" (base64 string)'}), 400
            
            # Handle base64 file
            if 'file_base64' in data:
                base64_data = data['file_base64']
                # Remove data URL prefix if present
                if ',' in base64_data:
                    base64_data = base64_data.split(',', 1)[1]
                
                try:
                    file_data = base64.b64decode(base64_data)
                except Exception as e:
                    return jsonify({'error': f'Invalid base64 data: {str(e)}'}), 400
                
                # Determine file extension from binary data or use provided
                file_extension = data.get('file_extension')
                if not file_extension:
                    file_extension = detect_file_extension_from_bytes(file_data)
                
                if 'file_name' in data:
                    filename = secure_filename_unicode(data['file_name'])
                    # Ensure filename has correct extension
                    if not filename.lower().endswith(f'.{file_extension}'):
                        # Add extension if missing
                        if '.' in filename:
                            # Replace existing extension
                            name_part = filename.rsplit('.', 1)[0]
                            filename = f"{name_part}.{file_extension}"
                        else:
                            filename = f"{filename}.{file_extension}"
                else:
                    filename = f"upload_{uuid.uuid4().hex[:8]}.{file_extension}"
                
                # Save to temporary file
                temp_dir = tempfile.gettempdir()
                filepath = os.path.join(temp_dir, filename)
                with open(filepath, 'wb') as f:
                    f.write(file_data)
                
                logger.info(f"Detecting signatures in base64 file: {filename}")
        
        # If not JSON or no base64, check for multipart form file
        if filepath is None:
            if 'file' not in request.files:
                return jsonify({'error': 'No file provided. Use multipart form with "file" field or JSON with "file_base64"'}), 400
            
            file = request.files['file']
            
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({'error': f'File type not allowed. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
            
            # Save file
            filename = secure_filename_unicode(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            logger.info(f"Detecting signatures in multipart file: {filename}")
        
        # Detect signatures
        signatures = extractor.detect_signatures(filepath)
        
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)
        
        logger.info(f"Found {len(signatures)} signatures in {filename}")
        
        return jsonify({
            'success': True,
            'filename': filename,
            'signatures': signatures,
            'signature_count': len(signatures),
            'has_signatures': len(signatures) > 0
        }), 200
    
    except Exception as e:
        logger.error(f"Error during signature detection: {str(e)}")
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

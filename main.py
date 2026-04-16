"""
Main entry point for localOCR
"""

import sys
import os
from pathlib import Path

# Add src directory to path so localocr can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from localocr import OCRExtractor


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path>")
        print("Example: python main.py image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Initialize OCR extractor
    print("Loading OCR model...")
    extractor = OCRExtractor()
    
    # Extract text
    print(f"Processing image: {image_path}")
    text = extractor.extract_text(image_path)
    
    print("\n--- Extracted Text ---")
    print(text)


if __name__ == "__main__":
    main()

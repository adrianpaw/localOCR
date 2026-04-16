"""
Basic tests for localOCR
"""

import pytest
from pathlib import Path


class TestImports:
    """Test that modules can be imported"""
    
    def test_localocr_import(self):
        """Test that localocr package can be imported"""
        import localocr
        assert hasattr(localocr, "OCRExtractor")
    
    def test_ocr_extractor_import(self):
        """Test that OCRExtractor can be imported"""
        from localocr import OCRExtractor
        assert OCRExtractor is not None

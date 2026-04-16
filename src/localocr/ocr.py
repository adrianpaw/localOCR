"""
OCR Extractor module using EasyOCR for text extraction from images and PDFs.
"""

import easyocr
import numpy as np
from PIL import Image
from typing import Union, List, Optional
from pathlib import Path
import logging
import os
import tempfile

logger = logging.getLogger(__name__)


class OCRExtractor:
    def __init__(self, languages: List[str] = None, gpu: bool = True):
        """
        Initialize the OCR Extractor with EasyOCR.

        Args:
            languages: List of language codes (e.g., ['en', 'ru']). Default is ['en', 'ru'].
            gpu: Whether to use GPU if available.
        """
        if languages is None:
            languages = ['en', 'ru']
        self.languages = languages
        self.gpu = gpu and self._check_gpu()
        
        logger.info(f"Initializing EasyOCR reader for languages: {languages}, GPU: {self.gpu}")
        try:
            self.reader = easyocr.Reader(
                lang_list=languages,
                gpu=self.gpu,
                verbose=False
            )
            logger.info("EasyOCR reader initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            raise RuntimeError(f"EasyOCR initialization failed: {e}")

    def _check_gpu(self) -> bool:
        """Check if GPU is available for EasyOCR."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _is_pdf(self, file_path: Union[str, Path]) -> bool:
        """Check if file is a PDF by extension."""
        file_path = str(file_path)
        return file_path.lower().endswith('.pdf')

    def _extract_text_from_pdf(self, pdf_path: Union[str, Path]) -> str:
        """
        Extract text from PDF file.
        
        First tries to extract text directly from PDF (if it has text layers).
        If no text is found, converts PDF pages to images and runs OCR.
        """
        pdf_path = str(pdf_path)
        
        # Try to extract text directly from PDF
        text_from_pdf = self._extract_text_from_pdf_direct(pdf_path)
        if text_from_pdf and text_from_pdf.strip():
            logger.info(f"Extracted text directly from PDF: {pdf_path}")
            return text_from_pdf
        
        # If no text found, convert to images and run OCR
        logger.info(f"No text layers found in PDF, converting to images for OCR: {pdf_path}")
        return self._extract_text_from_pdf_via_ocr(pdf_path)

    def _extract_text_from_pdf_direct(self, pdf_path: str) -> str:
        """Extract text directly from PDF using PyMuPDF (faster and more reliable than pypdf)."""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            text_parts = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text and text.strip():
                    text_parts.append(text)
            doc.close()
            return "\n".join(text_parts)
        except ImportError:
            logger.warning("PyMuPDF not installed, cannot extract text directly from PDF")
            return ""
        except Exception as e:
            logger.warning(f"Failed to extract text directly from PDF: {e}")
            return ""

    def _extract_text_from_pdf_via_ocr(self, pdf_path: str) -> str:
        """Convert PDF pages to images using PyMuPDF and run OCR."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.error("PyMuPDF not installed. Install with: pip install PyMuPDF")
            raise RuntimeError("PDF processing requires PyMuPDF. Install it first.")
        
        # Create temporary directory for images
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Open PDF with PyMuPDF
                doc = fitz.open(pdf_path)
                total_pages = len(doc)
                logger.info(f"Converting PDF with {total_pages} pages to images")
                
                # Extract text from each page
                text_parts = []
                for page_num in range(total_pages):
                    page = doc[page_num]
                    
                    # Convert page to image with good resolution for OCR
                    # Matrix 2.0 gives ~150 DPI which is good for OCR
                    mat = fitz.Matrix(2.0, 2.0)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    
                    # Save image temporarily
                    img_path = os.path.join(temp_dir, f"page_{page_num + 1}.png")
                    pix.save(img_path)
                    
                    # Extract text using existing image extraction
                    page_text = self._extract_text_from_image_file(img_path)
                    if page_text:
                        text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
                    else:
                        logger.debug(f"No text extracted from page {page_num + 1}")
                
                doc.close()
                logger.info(f"Successfully converted {total_pages} pages, extracted text from {len(text_parts)} pages")
                
                if text_parts:
                    return "\n\n".join(text_parts)
                else:
                    raise RuntimeError("No text could be extracted from any page of the PDF")
                
            except Exception as e:
                logger.error(f"Failed to convert PDF to images: {e}")
                # Close document if it's still open
                try:
                    if 'doc' in locals():
                        doc.close()
                except:
                    pass
                raise RuntimeError(f"PDF to image conversion failed: {e}")

    def _extract_text_from_image_file(self, image_path: Union[str, Path]) -> str:
        """
        Extract text from an image file (internal method).
        This is the original image extraction logic.
        """
        try:
            # Open image with PIL to handle Unicode paths
            image = Image.open(image_path).convert("RGB")
            # Convert to numpy array (RGB)
            image_np = np.array(image)
            # EasyOCR expects RGB array
            results = self.reader.readtext(image_np, detail=0, paragraph=True)
            # results is a list of strings (each paragraph)
            text = "\n".join(results)
            return text.strip()
        except Exception as e:
            raise RuntimeError(f"Error during text extraction from image: {str(e)}")

    def extract_text(self, file_path: Union[str, Path]) -> str:
        """
        Extract text from an image or PDF file.

        Args:
            file_path: Path to the image or PDF file

        Returns:
            Extracted text from the file as a single string.
        """
        file_path = str(file_path)
        
        if self._is_pdf(file_path):
            logger.info(f"Processing PDF file: {file_path}")
            return self._extract_text_from_pdf(file_path)
        else:
            logger.info(f"Processing image file: {file_path}")
            return self._extract_text_from_image_file(file_path)

    def extract_text_batch(self, image_paths: List[Union[str, Path]]) -> List[str]:
        """
        Extract text from multiple images.

        Args:
            image_paths: List of paths to image files

        Returns:
            List of extracted texts in the same order.
        """
        results = []
        for image_path in image_paths:
            text = self.extract_text(image_path)
            results.append(text)
        return results

    def extract_with_confidence(self, file_path: Union[str, Path], detail: int = 1):
        """
        Extract text with bounding boxes and confidence scores.
        
        Note: For PDF files, this method only works with detail=0 (text only)
        and will extract text without confidence scores.

        Args:
            file_path: Path to the image or PDF file
            detail: Level of detail (0 for text only, 1 for boxes and confidence)

        Returns:
            If detail=0: list of strings.
            If detail=1: list of tuples (bbox, text, confidence).
        """
        file_path = str(file_path)
        
        if self._is_pdf(file_path):
            if detail == 1:
                raise NotImplementedError(
                    "Confidence extraction with bounding boxes is not supported for PDF files. "
                    "Use detail=0 for text-only extraction."
                )
            # For PDF with detail=0, extract text as list of strings (one per page)
            text = self.extract_text(file_path)
            # Split by page markers to return list
            pages = text.split("--- Page ")
            if len(pages) > 1:
                # Remove empty first element if exists
                if not pages[0].strip():
                    pages = pages[1:]
                return [f"Page {page}" for page in pages]
            return [text] if text.strip() else []
        else:
            # Original image processing
            try:
                results = self.reader.readtext(file_path, detail=detail, paragraph=(detail == 0))
                return results
            except Exception as e:
                raise RuntimeError(f"Error during detailed text extraction: {str(e)}")

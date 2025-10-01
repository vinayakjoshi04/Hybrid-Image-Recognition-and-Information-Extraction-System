"""
OCR utilities using Tesseract (raw text only).
This module provides helper functions to:
1. Initialize Tesseract automatically (depending on OS/installation).
2. Clean noisy OCR output text.
3. Run OCR on a given image and return extracted text.
"""

import os
import re
import shutil
from pathlib import Path
from typing import Dict

from PIL import Image
import pytesseract


def _init_tesseract():
    """
    Try to set pytesseract command path automatically.
    
    - Checks environment variable `TESSERACT_CMD`
    - Checks if 'tesseract' is available in PATH
    - Falls back to common Windows installation path
    """
    env_cmd = os.environ.get("TESSERACT_CMD")
    if env_cmd and Path(env_cmd).exists():
        # If user provided path via environment variable
        pytesseract.pytesseract.tesseract_cmd = str(Path(env_cmd))
        return

    # Check if tesseract is available in system PATH
    which_cmd = shutil.which("tesseract")
    if which_cmd:
        pytesseract.pytesseract.tesseract_cmd = which_cmd
        return

    # Fallback: typical Windows installation directory
    default_win = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if Path(default_win).exists():
        pytesseract.pytesseract.tesseract_cmd = default_win


# Run Tesseract initialization when the module is imported
_init_tesseract()


def clean_ocr_text(text: str) -> str:
    """
    Basic OCR post-processing and cleaning.

    - Normalizes line breaks
    - Collapses multiple newlines into one
    - Replaces tabs/multiple spaces with a single space
    - Removes non-ASCII characters (common OCR noise)
    """
    if text is None:
        return ""
    
    # Normalize Windows/Mac line endings into Unix format
    text = re.sub(r"\r\n?", "\n", text)
    # Remove repeated blank lines
    text = re.sub(r"\n{2,}", "\n", text)
    # Replace tabs or multiple spaces with single space
    text = re.sub(r"[ \t]+", " ", text)
    # Strip out non-ASCII (OCR may introduce garbage chars)
    text = re.sub(r"[^\x00-\x7F]+", "", text)

    return text.strip()


def run_ocr(image_path: str, lang: str = "eng") -> Dict[str, str]:
    """
    Run OCR on the given image file using pytesseract.

    Args:
        image_path (str): Path to the image file
        lang (str): Language code (default: English = "eng")

    Returns:
        Dict[str, str]:
            - "extracted_text": Extracted and cleaned text
            - "error": Error message (if any)
    """
    try:
        # Open the image safely with PIL (convert to RGB for consistency)
        pil_img = Image.open(image_path).convert("RGB")
    except Exception as e:
        # Return structured error if file cannot be opened
        return {"extracted_text": "", "error": f"Cannot open image: {e}"}

    try:
        # Perform OCR with pytesseract
        raw_text = pytesseract.image_to_string(pil_img, lang=lang)
        # Clean the raw OCR output
        raw_text = clean_ocr_text(raw_text)
    except Exception as e:
        # Return structured error if OCR process fails
        return {"extracted_text": "", "error": f"Tesseract error: {e}"}

    # Return successful OCR result
    return {"extracted_text": raw_text}

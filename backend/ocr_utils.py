"""
Simple Tesseract-based OCR utilities.

Notes:
- You can set environment variable TESSERACT_CMD to point to tesseract executable (Windows).
- If Tesseract isn't found, pytesseract will raise a clear error when called.
"""

import os
import re
import shutil
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from PIL import Image
import pytesseract


def _init_tesseract():
    """
    Try to set pytesseract.pytesseract.tesseract_cmd automatically:
     - If env var TESSERACT_CMD is set and exists, use it.
     - Else try shutil.which("tesseract").
     - Else check the default Windows install path.
    If none found, do not set and let pytesseract raise a helpful error when used.
    """
    env_cmd = os.environ.get("TESSERACT_CMD")
    if env_cmd and Path(env_cmd).exists():
        pytesseract.pytesseract.tesseract_cmd = str(Path(env_cmd))
        return

    which_cmd = shutil.which("tesseract")
    if which_cmd:
        pytesseract.pytesseract.tesseract_cmd = which_cmd
        return

    default_win = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if Path(default_win).exists():
        pytesseract.pytesseract.tesseract_cmd = default_win
        return

    # Not found -- will let pytesseract throw a clear error when called.


_init_tesseract()


def clean_ocr_text(text: str) -> str:
    """Basic OCR post-processing / cleaning."""
    if text is None:
        return ""
    # normalize newlines, collapse spaces, strip non-ascii
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    return text.strip()


def run_ocr(image_path: str, lang: str = "eng") -> Dict[str, str]:
    """
    Run OCR in two passes:
      1) raw text on the original image
      2) preprocess (denoise, threshold) then OCR again for improved output

    Returns:
      {
        "raw_text": "...",
        "processed_text": "..."
      }
    """
    try:
        pil_img = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {"raw_text": "", "processed_text": "", "error": f"Cannot open image: {e}"}

    # Pass 1: raw
    try:
        raw_text = pytesseract.image_to_string(pil_img, lang=lang)
    except Exception as e:
        # Common cause: tesseract not installed or incorrect path
        return {"raw_text": "", "processed_text": "", "error": f"Tesseract error: {e}"}

    raw_text = clean_ocr_text(raw_text)

    # Pass 2: preprocessing (improve OCR quality)
    try:
        img = np.array(pil_img)[:, :, ::-1]  # RGB -> BGR for OpenCV
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Denoise
        gray = cv2.fastNlMeansDenoising(gray, None, h=10)

        # Adaptive threshold / Otsu
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Median blur
        processed = cv2.medianBlur(thresh, 3)

        # Optional morphological ops (tiny kernel to remove small noise)
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)

        processed_pil = Image.fromarray(processed)
        custom_config = r"--oem 3 --psm 6"
        improved_text = pytesseract.image_to_string(processed_pil, config=custom_config, lang=lang)
        improved_text = clean_ocr_text(improved_text)
    except Exception as e:
        # if OpenCV processing fails, fall back to raw_text only
        improved_text = ""
        # include note in returned dictionary
        return {
            "raw_text": raw_text,
            "processed_text": improved_text,
            "warning": f"Image preprocessing failed: {e}",
        }

    return {"raw_text": raw_text, "processed_text": improved_text}

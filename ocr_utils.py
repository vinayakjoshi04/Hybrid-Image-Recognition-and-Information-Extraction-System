"""
Simple Tesseract-based OCR utilities.
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


_init_tesseract()


def clean_ocr_text(text: str) -> str:
    """Basic OCR post-processing / cleaning."""
    if text is None:
        return ""
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    return text.strip()


def run_ocr(image_path: str, lang: str = "eng") -> Dict[str, str]:
    try:
        pil_img = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {"raw_text": "", "processed_text": "", "error": f"Cannot open image: {e}"}

    # Pass 1: raw OCR
    try:
        raw_text = pytesseract.image_to_string(pil_img, lang=lang)
    except Exception as e:
        return {"raw_text": "", "processed_text": "", "error": f"Tesseract error: {e}"}

    raw_text = clean_ocr_text(raw_text)

    # Pass 2: preprocessing
    try:
        img = np.array(pil_img)[:, :, ::-1]  # RGB -> BGR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.fastNlMeansDenoising(gray, None, h=10)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed = cv2.medianBlur(thresh, 3)
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)

        processed_pil = Image.fromarray(processed)
        custom_config = r"--oem 3 --psm 6"
        improved_text = pytesseract.image_to_string(processed_pil, config=custom_config, lang=lang)
        improved_text = clean_ocr_text(improved_text)
    except Exception as e:
        return {
            "raw_text": raw_text,
            "processed_text": "",
            "warning": f"Image preprocessing failed: {e}",
        }

    return {"raw_text": raw_text, "processed_text": improved_text}

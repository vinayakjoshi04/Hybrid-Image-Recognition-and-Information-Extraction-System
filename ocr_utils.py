"""
OCR utilities using Tesseract (raw text only).
"""

import os
import re
import shutil
from pathlib import Path
from typing import Dict

from PIL import Image
import pytesseract


def _init_tesseract():
    """Try to set pytesseract command path automatically."""
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


_init_tesseract()


def clean_ocr_text(text: str) -> str:
    """Basic OCR post-processing and cleaning."""
    if text is None:
        return ""
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    return text.strip()


def run_ocr(image_path: str, lang: str = "eng") -> Dict[str, str]:
    """Run OCR on the given image file (raw text only)."""
    try:
        pil_img = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {"extracted_text": "", "error": f"Cannot open image: {e}"}

    try:
        raw_text = pytesseract.image_to_string(pil_img, lang=lang)
        raw_text = clean_ocr_text(raw_text)
    except Exception as e:
        return {"extracted_text": "", "error": f"Tesseract error: {e}"}

    return {"extracted_text": raw_text}

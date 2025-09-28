"""
BLIP image captioning wrapper (Salesforce/blip-image-captioning-base).

Model is loaded lazily on first call to keep import/startup fast.
"""

from PIL import Image
import torch
from typing import Dict

# globals filled by _load_model()
_MODEL = None
_PROCESSOR = None
_DEVICE = "cpu"  # force CPU only


def _load_model():
    global _MODEL, _PROCESSOR, _DEVICE
    if _MODEL is None:
        from transformers import BlipProcessor, BlipForConditionalGeneration

        # download / cache the weights the first time
        _PROCESSOR = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _MODEL = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(_DEVICE)
    return _MODEL, _PROCESSOR, _DEVICE


def run_captioning(image_path: str, max_length: int = 60, num_beams: int = 3) -> Dict[str, str]:
    """
    Generate a single caption for the image at image_path.

    Returns: {"caption": "..."} or {"error": "..."}
    """
    try:
        model, processor, device = _load_model()
        pil_img = Image.open(image_path).convert("RGB")

        inputs = processor(images=pil_img, return_tensors="pt").to(device)
        out_ids = model.generate(**inputs, max_length=max_length, num_beams=num_beams, early_stopping=True)

        caption = processor.decode(out_ids[0], skip_special_tokens=True)
        return {"caption": caption}
    except Exception as e:
        return {"caption": "", "error": f"Captioning failed: {e}"}

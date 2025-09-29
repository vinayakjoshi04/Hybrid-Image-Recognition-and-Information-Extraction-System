"""
BLIP Image Captioning wrapper.
"""
from PIL import Image
from typing import Dict

# Global vars
_MODEL = None
_PROCESSOR = None
_DEVICE = "cpu"  # force CPU only


def _load_model():
    """Lazy load BLIP model."""
    global _MODEL, _PROCESSOR, _DEVICE
    if _MODEL is None:
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            
            _PROCESSOR = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            _MODEL = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            ).to(_DEVICE)
        except Exception as e:
            raise ImportError(f"Failed to load BLIP model. Make sure transformers is installed: {e}")
    
    return _MODEL, _PROCESSOR, _DEVICE


def run_captioning(image_path: str, max_length: int = 60, num_beams: int = 3) -> Dict[str, str]:
    """Generate a caption for an image."""
    try:
        model, processor, device = _load_model()
        pil_img = Image.open(image_path).convert("RGB")
        
        inputs = processor(images=pil_img, return_tensors="pt").to(device)
        out_ids = model.generate(**inputs, max_length=max_length, num_beams=num_beams, early_stopping=True)
        
        caption = processor.decode(out_ids[0], skip_special_tokens=True)
        return {"caption": caption}
    except Exception as e:
        return {"caption": "", "error": f"Captioning failed: {e}"}
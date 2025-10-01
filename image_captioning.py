"""
BLIP Image Captioning Wrapper.

This module provides functionality to:
1. Lazily load the BLIP model from HuggingFace only when needed.
2. Generate descriptive captions for images using a pre-trained BLIP model.

Key Components:
- `_MODEL` and `_PROCESSOR` are global variables for lazy loading the model and processor.
- `_DEVICE` defines whether to run on CPU or GPU.
- `_load_model()` ensures the model is loaded only once.
- `run_captioning()` generates captions for a given image.
"""

from PIL import Image
from typing import Dict

# ------------------------------------------------------
# Global variables for lazy loading the BLIP model
# ------------------------------------------------------
_MODEL = None         # BLIP model instance
_PROCESSOR = None     # BLIP processor instance
_DEVICE = "cpu"       # Default device; change to "cuda" if GPU is available


def _load_model():
    """
    Lazy load BLIP model and processor.

    Returns:
        tuple: (_MODEL, _PROCESSOR, _DEVICE)
    
    Notes:
    - Only loads the model if it hasn't been loaded already.
    - Uses HuggingFace transformers: BlipProcessor + BlipForConditionalGeneration.
    """
    global _MODEL, _PROCESSOR, _DEVICE

    if _MODEL is None:
        try:
            # Import BLIP processor and model dynamically
            from transformers import BlipProcessor, BlipForConditionalGeneration
            
            # Load the processor: handles image preprocessing
            _PROCESSOR = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

            # Load the BLIP model and move it to the chosen device
            _MODEL = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            ).to(_DEVICE)

        except Exception as e:
            # Provide a clear error if model loading fails
            raise ImportError(f"Failed to load BLIP model. Make sure transformers is installed: {e}")
    
    return _MODEL, _PROCESSOR, _DEVICE


def run_captioning(image_path: str, max_length: int = 60, num_beams: int = 3) -> Dict[str, str]:
    """
    Generate a caption for a given image using the BLIP model.

    Args:
        image_path (str): Path to the input image.
        max_length (int): Maximum token length of generated caption (default=60).
        num_beams (int): Number of beams for beam search (default=3).

    Returns:
        Dict[str, str]:
            - "caption": Generated caption string.
            - "error": Error message if captioning fails (optional).
    
    Notes:
    - Beam search improves caption quality by exploring multiple candidate sequences.
    - The processor handles image-to-tensor conversion.
    """
    try:
        # Load model, processor, and device (lazy loading)
        model, processor, device = _load_model()

        # Open image using PIL and convert to RGB format
        pil_img = Image.open(image_path).convert("RGB")
        
        # Preprocess image for the model: returns tensor ready for BLIP
        inputs = processor(images=pil_img, return_tensors="pt").to(device)

        # Generate caption token IDs using beam search
        out_ids = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )
        
        # Decode token IDs to readable string (skip special tokens)
        caption = processor.decode(out_ids[0], skip_special_tokens=True)
        return {"caption": caption}
    
    except Exception as e:
        # Return error in dictionary format if captioning fails
        return {"caption": "", "error": f"Captioning failed: {e}"}

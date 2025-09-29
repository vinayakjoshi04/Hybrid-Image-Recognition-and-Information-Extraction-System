"""
NLP utilities: Translation (googletrans) + Text-to-Speech (gTTS).
"""
from typing import Dict


def translate_text(text: str, target_lang: str = "en") -> Dict[str, str]:
    """Translate text to target language."""
    if not text or not text.strip():
        return {"translated_text": ""}
    
    try:
        from googletrans import Translator
        translator = Translator()
        translated = translator.translate(text, dest=target_lang)
        return {"translated_text": translated.text}
    except Exception as e:
        return {"translated_text": "", "error": f"Translation failed: {e}"}


def text_to_speech(text: str, lang: str = "en", out_path: str = "output.mp3") -> Dict[str, str]:
    """Convert text to speech and save as MP3."""
    if not text or not text.strip():
        return {"audio_path": "", "error": "No text provided for TTS"}
    
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang=lang)
        tts.save(out_path)
        return {"audio_path": out_path}
    except Exception as e:
        return {"audio_path": "", "error": f"TTS failed: {e}"}
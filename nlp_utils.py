"""
NLP utilities: Translation (googletrans) + Text-to-Speech (gTTS).

This module provides:
1. Text translation into different languages using Google Translate API (`googletrans`).
2. Text-to-Speech conversion using Google Text-to-Speech (`gTTS`).
"""

from typing import Dict


def translate_text(text: str, target_lang: str = "en") -> Dict[str, str]:
    """
    Translate input text into the specified target language.

    Args:
        text (str): Input text to be translated.
        target_lang (str): Target language code (default = "en").

    Returns:
        Dict[str, str]:
            - "translated_text": Translated text result.
            - "error": Error message if translation fails.
    """

    # Check if the text is empty or just spaces -> no translation needed
    if not text or not text.strip():
        return {"translated_text": ""}
    
    try:
        # Import translator from googletrans dynamically
        from googletrans import Translator
        translator = Translator()

        # Perform translation
        translated = translator.translate(text, dest=target_lang)

        # Return translated text in dictionary format
        return {"translated_text": translated.text}
    
    except Exception as e:
        # Handle errors (e.g., no internet, unsupported language)
        return {"translated_text": "", "error": f"Translation failed: {e}"}


def text_to_speech(text: str, lang: str = "en", out_path: str = "output.mp3") -> Dict[str, str]:
    """
    Convert text into speech and save it as an MP3 file.

    Args:
        text (str): Input text to convert to speech.
        lang (str): Language code for speech output (default = "en").
        out_path (str): Path to save the generated audio file (default = "output.mp3").

    Returns:
        Dict[str, str]:
            - "audio_path": Path to the generated MP3 file.
            - "error": Error message if TTS fails.
    """

    # Validate: input text must not be empty
    if not text or not text.strip():
        return {"audio_path": "", "error": "No text provided for TTS"}
    
    try:
        # Import Google Text-to-Speech library
        from gtts import gTTS

        # Initialize gTTS object with provided text and language
        tts = gTTS(text=text, lang=lang)

        # Save generated speech to MP3 file
        tts.save(out_path)

        # Return path of saved audio file
        return {"audio_path": out_path}
    
    except Exception as e:
        # Handle errors (e.g., invalid language, gTTS not installed, network issue)
        return {"audio_path": "", "error": f"TTS failed: {e}"}

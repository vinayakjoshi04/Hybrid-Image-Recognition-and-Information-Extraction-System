"""
Streamlit App: Hybrid OCR + Image Captioning + NLP

This app allows users to:
1. Upload an image.
2. Either:
   - Extract raw text from the image (OCR using Tesseract), OR
   - Generate a description/caption for the image (using BLIP).
3. Post-process the result by:
   - Translating the text into another language (via Google Translate).
   - Converting text into speech (via gTTS).
4. Download results as text or MP3 audio files.

Modules used:
- `ocr_utils.run_ocr`: Extracts text from an image using Tesseract OCR.
- `image_captioning.run_captioning`: Generates captions from images.
- `nlp_utils.translate_text`: Translates text into a target language.
- `nlp_utils.text_to_speech`: Converts text to MP3 speech output.
"""

import streamlit as st
import tempfile
import os

# Import custom utilities
from ocr_utils import run_ocr
from image_captioning import run_captioning
from nlp_utils import translate_text, text_to_speech

# Streamlit Page Configuration
st.set_page_config(
    page_title="Hybrid OCR & Captioning",
    page_icon="🖼️",
    layout="wide"
)

# Header and short description using HTML styling
st.markdown(
    """
    <h1 style="text-align:center; color:#4CAF50;">🖼️ Hybrid Image Recognition & NLP</h1>
    <p style="text-align:center; font-size:18px;">
        Upload an image to <b>extract text</b> 📝 or <b>generate captions</b> 📷.<br>
        Then <b>translate</b> 🌍 or <b>convert to speech</b> 🔊.
    </p>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State (keeps data persistent)

# Streamlit resets variables after each interaction, so we use session_state.
if "output_text" not in st.session_state:
    st.session_state.output_text = ""  # Holds OCR or Captioning result

if "translated_text" not in st.session_state:
    st.session_state.translated_text = ""  # Holds translated text

if "audio_path" not in st.session_state:
    st.session_state.audio_path = None  # Path to generated speech file


# Sidebar: Task Selection and Language Options

with st.sidebar:
    st.header("⚙️ Options")

    # Choose task: OCR or Image Captioning
    task_label = st.radio(
        "Choose Task",
        ("Extract Text from Image 📝", "Describe the Image 📷"),
    )
    task = "ocr" if task_label.startswith("Extract") else "caption"

    # Choose target language for translation and TTS
    target_lang = st.selectbox(
        "🌍 Select target language",
        ["en", "fr", "de", "es", "zh-cn", "ja", "ar"],
        index=0,
        key="lang_select",
    )


# File Upload Section

uploaded_file = st.file_uploader(
    "📤 Upload an image",
    type=["png", "jpg", "jpeg", "bmp", "tiff", "gif"]
)

if uploaded_file:
    # Display uploaded image preview
    st.image(uploaded_file, caption="📷 Uploaded Image", use_container_width=True)

    # Run OCR or Captioning on button click
    if st.button("🚀 Run Task", use_container_width=True):
        # Save uploaded file to a temporary location for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name  # Path to temp file

        try:
            with st.spinner("⏳ Processing..."):
                if task == "ocr":
                    # Run OCR utility
                    res = run_ocr(tmp_path)
                    st.session_state.output_text = res.get("extracted_text", "")
                else:
                    # Run Image Captioning utility
                    res = run_captioning(tmp_path)
                    st.session_state.output_text = res.get("caption", "")

                # Reset translation and audio for fresh result
                st.session_state.translated_text = ""

                # Remove old audio file if exists
                if st.session_state.audio_path and os.path.exists(st.session_state.audio_path):
                    try:
                        os.remove(st.session_state.audio_path)
                    except:
                        pass
                st.session_state.audio_path = None

            # Handle errors if any
            if res.get("error"):
                st.error(res["error"])
            else:
                # Show results for OCR or Captioning
                if task == "ocr":
                    st.subheader("🔍 Extracted Text")
                    if st.session_state.output_text:
                        st.code(st.session_state.output_text, language=None)
                        st.download_button("⬇️ Download Text", st.session_state.output_text, file_name="ocr_text.txt")
                    else:
                        st.info("No text detected.")
                else:
                    st.subheader("📝 Image Description")
                    if st.session_state.output_text:
                        st.success(st.session_state.output_text)
                        st.download_button("⬇️ Download Caption", st.session_state.output_text, file_name="image_description.txt")
                    else:
                        st.info("No description returned.")
        finally:
            # Always clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


# Post-Processing: Translation and Text-to-Speech

if st.session_state.output_text:
    st.markdown("---")
    st.subheader("🌍 Translate & 🔊 Text-to-Speech")

    # Two columns: Translation on left, TTS on right
    col1, col2 = st.columns(2)

    # Translation
    with col1:
        if st.button("🌐 Translate Text", use_container_width=True):
            with st.spinner("Translating..."):
                trans_res = translate_text(st.session_state.output_text, target_lang=target_lang)

            if trans_res.get("error"):
                st.error(trans_res["error"])
            else:
                st.session_state.translated_text = trans_res["translated_text"]
                st.success(st.session_state.translated_text)
                st.download_button("⬇️ Download Translation", st.session_state.translated_text, file_name="translated.txt")

    # Text-to-Speech
    with col2:
        if st.button("🔊 Convert to Speech", use_container_width=True):
            # Prefer translated text if available, otherwise use original
            text_for_tts = st.session_state.translated_text or st.session_state.output_text

            # Remove old audio file if exists
            if st.session_state.audio_path and os.path.exists(st.session_state.audio_path):
                try:
                    os.remove(st.session_state.audio_path)
                except:
                    pass

            # Create a temporary MP3 file for TTS output
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as audio_tmp:
                audio_path = audio_tmp.name

            # Run TTS utility
            with st.spinner("Generating speech..."):
                tts_res = text_to_speech(text_for_tts, lang=target_lang, out_path=audio_path)

            # Error handling or play/download audio
            if tts_res.get("error"):
                st.error(tts_res["error"])
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            else:
                st.session_state.audio_path = tts_res["audio_path"]
                st.audio(tts_res["audio_path"], format="audio/mp3")
                with open(tts_res["audio_path"], "rb") as f:
                    st.download_button("⬇️ Download Audio", f, file_name="output.mp3")

    # Display latest translation separately
    if st.session_state.translated_text:
        st.subheader("✅ Latest Translation")
        st.info(st.session_state.translated_text)

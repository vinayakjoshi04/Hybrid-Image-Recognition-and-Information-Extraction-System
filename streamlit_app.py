import streamlit as st
import tempfile
import os

from ocr_utils import run_ocr
from image_captioning import run_captioning
from nlp_utils import translate_text, text_to_speech

st.set_page_config(page_title="Hybrid OCR & Captioning", page_icon="🖼️", layout="wide")

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

if "output_text" not in st.session_state:
    st.session_state.output_text = ""

if "translated_text" not in st.session_state:
    st.session_state.translated_text = ""

if "audio_path" not in st.session_state:
    st.session_state.audio_path = None

with st.sidebar:
    st.header("⚙️ Options")
    task_label = st.radio(
        "Choose Task",
        ("Extract Text from Image 📝", "Describe the Image 📷"),
    )
    task = "ocr" if task_label.startswith("Extract") else "caption"

    target_lang = st.selectbox(
        "🌍 Select target language",
        ["en", "fr", "de", "es", "zh-cn", "ja", "ar"],
        index=0,
        key="lang_select",
    )

uploaded_file = st.file_uploader("📤 Upload an image", type=["png", "jpg", "jpeg", "bmp", "tiff", "gif"])

if uploaded_file:
    st.image(uploaded_file, caption="📷 Uploaded Image", use_container_width=True)

    if st.button("🚀 Run Task", use_container_width=True):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            with st.spinner("⏳ Processing..."):
                if task == "ocr":
                    res = run_ocr(tmp_path)
                    st.session_state.output_text = res.get("extracted_text", "")
                else:
                    res = run_captioning(tmp_path)
                    st.session_state.output_text = res.get("caption", "")
                st.session_state.translated_text = ""
                
                if st.session_state.audio_path and os.path.exists(st.session_state.audio_path):
                    try:
                        os.remove(st.session_state.audio_path)
                    except:
                        pass
                st.session_state.audio_path = None

            if res.get("error"):
                st.error(res["error"])
            else:
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
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

if st.session_state.output_text:
    st.markdown("---")
    st.subheader("🌍 Translate & 🔊 Text-to-Speech")

    col1, col2 = st.columns(2)

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

    with col2:
        if st.button("🔊 Convert to Speech", use_container_width=True):
            text_for_tts = st.session_state.translated_text or st.session_state.output_text

            if st.session_state.audio_path and os.path.exists(st.session_state.audio_path):
                try:
                    os.remove(st.session_state.audio_path)
                except:
                    pass
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as audio_tmp:
                audio_path = audio_tmp.name
            
            with st.spinner("Generating speech..."):
                tts_res = text_to_speech(text_for_tts, lang=target_lang, out_path=audio_path)
            
            if tts_res.get("error"):
                st.error(tts_res["error"])
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            else:
                st.session_state.audio_path = tts_res["audio_path"]
                st.audio(tts_res["audio_path"], format="audio/mp3")
                with open(tts_res["audio_path"], "rb") as f:
                    st.download_button("⬇️ Download Audio", f, file_name="output.mp3")

    if st.session_state.translated_text:
        st.subheader("✅ Latest Translation")
        st.info(st.session_state.translated_text)

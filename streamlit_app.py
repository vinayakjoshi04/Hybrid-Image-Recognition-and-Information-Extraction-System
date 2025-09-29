import streamlit as st
import tempfile
import os

from ocr_utils import run_ocr
from image_captioning import run_captioning

st.set_page_config(page_title="OCR + Captioning", page_icon="🖼️", layout="centered")
st.title("Hybrid Image Recognition and Information Extraction System")
st.write("Upload an image and choose whether you want to **extract text** or **describe the image**.")

task_label = st.radio(
    "What would you like to do?",
    ("Extract the text given in the image", "Describe the image"),
)

task = "ocr" if task_label.startswith("Extract") else "caption"

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "tiff", "gif"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded image", use_container_width=True)

    if st.button("Run"):
        # Save file temporarily for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            with st.spinner("Processing..."):
                if task == "ocr":
                    res = run_ocr(tmp_path)
                else:
                    res = run_captioning(tmp_path)

            if "error" in res and res.get("error"):
                st.error(res.get("error"))
            else:
                if task == "ocr":
                    st.subheader("🔍 Text found in the image")
                    st.code(res.get("raw_text", ""), language=None)
                    st.subheader("✨ Cleaned-up text")
                    st.code(res.get("processed_text", ""), language=None)
                    processed = res.get("processed_text", "")
                    if processed:
                        st.download_button("Download text", processed, file_name="ocr_text.txt")
                else:
                    caption = res.get("caption", "")
                    if caption:
                        st.subheader("📝 Image description")
                        st.success(caption)
                        st.download_button("Download description", caption, file_name="image_description.txt")
                    else:
                        st.info("No description returned.")
                    if res.get("error"):
                        st.warning(res["error"])

        finally:
            os.remove(tmp_path)

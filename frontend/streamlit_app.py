import streamlit as st
import requests

# Flask backend endpoint
BACKEND_URL = "http://127.0.0.1:5000/process"

st.set_page_config(page_title="OCR + Captioning", page_icon="🖼️", layout="centered")
st.title("Hybrid Image Recognition and Information Extraction System")
st.write("Upload an image and choose whether you want me to **Extract the text given in the image** or **describe the picture**.")

# More human-friendly task labels
task_label = st.radio(
    "What would you like to do?",
    ("Extract the text given in the image", "Describe the image"),
)

# Map friendly labels back to backend tasks
task = "ocr" if task_label == "Read text from the image" else "caption"

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "tiff", "gif"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded image", use_container_width=True)

    if st.button("Run"):
        file_bytes = uploaded_file.getvalue()
        files = {"file": (uploaded_file.name, file_bytes, uploaded_file.type or "application/octet-stream")}
        data = {"task": task}

        try:
            with st.spinner("Processing..."):
                resp = requests.post(BACKEND_URL, files=files, data=data, timeout=120)

            if resp.status_code != 200:
                try:
                    j = resp.json()
                    st.error(f"Backend error: {j.get('error', j)}")
                except Exception:
                    st.error(f"Backend returned status {resp.status_code}: {resp.text}")
            else:
                res = resp.json()
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
        except requests.exceptions.RequestException as e:
            st.error(f"Request to backend failed: {e}")

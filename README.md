# 🖼️ OCR + Image Captioning App

A simple web app built with **Streamlit** (frontend) and **Flask** (backend) that can:

* 🔍 **Read text from an image** (OCR)
* 📝 **Generate a description of an image** (Captioning)

The backend uses:

* **Tesseract OCR** for extracting text.
* **BLIP (Salesforce/blip-image-captioning-base)** for generating image captions.

---

## 🚀 Features

* Upload an image and choose:

  * **"Read text from the image"** → extracts text (OCR).
  * **"Describe the image"** → generates a caption (BLIP).
* Download extracted text or generated description.
* Runs fully on **CPU** (no GPU required).
* Easy-to-use, non-technical friendly interface.

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/ocr-captioning-app.git
cd ocr-captioning-app
```

### 1️⃣ Backend (Flask API)

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the backend:

```bash
python backend/app.py
```

This will start a Flask server at `http://127.0.0.1:5000`.

### 2️⃣ Frontend (Streamlit App)

In another terminal, run:

```bash
streamlit run frontend/streamlit_app.py
```

This will launch the web app at `http://localhost:8501`.

---

## 🖥️ Usage

1. Open the Streamlit app in your browser.
2. Upload an image (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.gif`).
3. Choose one of the options:

   * **Read text from the image** → Extracts text using OCR.
   * **Describe the image** → Generates a caption.
4. View the results and optionally download them as `.txt`.

---

## 📁 Project Structure

```
.
├── backend/
│   ├── app.py              # Flask API
│   ├── ocr.py              # OCR helper
│   ├── captioning.py       # BLIP captioning helper
│   └── requirements.txt
├── frontend/
│   └── streamlit_app.py    # Streamlit frontend
└── README.md
```

---

## ⚙️ Requirements

* Python 3.8+
* Flask
* Streamlit
* Transformers
* Torch
* Pillow
* Tesseract OCR installed locally

On Ubuntu/Debian:

```bash
sudo apt-get install tesseract-ocr
```

On Windows, download [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki).

---

## 📸 Demo

* **OCR Example**
  Input: ![example-image](docs/sample_ocr.png)
  Output: `"Hello World"`

* **Captioning Example**
  Input: ![example-image](docs/sample_caption.png)
  Output: `"A cat sitting on a wooden chair."`

---

## 🤝 Contributing

Pull requests are welcome!
For major changes, please open an issue first to discuss what you’d like to change.

---

## 📜 License

This project is licensed under the MIT License.

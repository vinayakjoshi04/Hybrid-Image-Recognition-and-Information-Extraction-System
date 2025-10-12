# 🖼️ Hybrid Image Recognition & NLP Application

A powerful Streamlit-based web application that combines OCR (Optical Character Recognition), AI-powered image captioning, translation, and text-to-speech capabilities into one seamless interface.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

https://hybrid-image-recognition-and-information-extraction-system-6xd.streamlit.app/

## ✨ Features

### 📝 OCR (Optical Character Recognition)
- Extract text from images using Tesseract OCR
- Advanced image preprocessing for improved accuracy
- Support for multiple languages
- Raw and cleaned text output
- Downloadable text files

### 📷 AI Image Captioning
- Generate natural language descriptions of images
- Powered by BLIP (Bootstrapping Language-Image Pre-training)
- High-quality, context-aware captions
- Downloadable caption files

### 🌍 Multi-Language Translation
- Translate extracted text or captions to 8+ languages
- Support for: English, Hindi, French, German, Spanish, Chinese, Japanese, Arabic
- Powered by Google Translate API
- Downloadable translations

### 🔊 Text-to-Speech
- Convert text to natural-sounding speech
- Multi-language audio generation
- MP3 format output
- In-browser audio playback
- Downloadable audio files

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Tesseract OCR (system installation required)
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/hybrid-image-nlp.git
   cd hybrid-image-nlp
   ```

2. **Install Tesseract OCR**

   **Windows:**
   - Download installer from [GitHub Tesseract Releases](https://github.com/UB-Mannheim/tesseract/wiki)
   - Install to default location: `C:\Program Files\Tesseract-OCR`
   - Or set custom path via environment variable `TESSERACT_CMD`

   **macOS:**
   ```bash
   brew install tesseract
   ```

   **Linux (Ubuntu/Debian):**
   ```bash
   sudo apt-get update
   sudo apt-get install tesseract-ocr
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   - The app will automatically open at `http://localhost:8501`

## 📁 Project Structure

```
hybrid-image-nlp/
│
├── app.py                    # Main Streamlit application
├── ocr_utils.py             # OCR functionality with preprocessing
├── image_captioning.py      # BLIP image captioning module
├── nlp_utils.py             # Translation and TTS utilities
├── requirements.txt         # Python dependencies
├── README.md               # Project documentation
└── .gitignore              # Git ignore file
```

## 🎯 Usage

### 1. Extract Text from Images (OCR)

1. Select "Extract Text from Image 📝" from the sidebar
2. Upload an image (PNG, JPG, JPEG, BMP, TIFF, GIF)
3. Click "🚀 Run Task"
4. View raw and cleaned OCR output
5. Download the extracted text

### 2. Generate Image Captions

1. Select "Describe the Image 📷" from the sidebar
2. Upload an image
3. Click "🚀 Run Task"
4. View the AI-generated caption
5. Download the caption

### 3. Translate Text

1. After extracting text or generating a caption
2. Select target language from the sidebar
3. Click "🌐 Translate Text"
4. View and download the translation

### 4. Convert to Speech

1. After extracting text or translating
2. Select target language for audio
3. Click "🔊 Convert to Speech"
4. Listen to the audio in-browser
5. Download the MP3 file

## 🛠️ Technical Details

### Core Technologies

- **Streamlit**: Web application framework
- **Tesseract OCR**: Text extraction engine
- **OpenCV**: Image preprocessing
- **BLIP**: Image captioning model (Salesforce)
- **Transformers**: Hugging Face library for AI models
- **Google Translate API**: Translation service
- **gTTS**: Google Text-to-Speech

### Image Preprocessing Pipeline

1. Grayscale conversion
2. Noise reduction (Non-local Means Denoising)
3. Adaptive thresholding (Otsu's method)
4. Median blur filtering
5. Morphological operations

### Supported Languages

| Language | Code | Translation | TTS |
|----------|------|-------------|-----|
| English | en | ✅ | ✅ |
| Hindi | hi | ✅ | ✅ |
| French | fr | ✅ | ✅ |
| German | de | ✅ | ✅ |
| Spanish | es | ✅ | ✅ |
| Chinese | zh-cn | ✅ | ✅ |
| Japanese | ja | ✅ | ✅ |
| Arabic | ar | ✅ | ✅ |

## 📸 Supported Image Formats

- PNG (.png)
- JPEG (.jpg, .jpeg)
- BMP (.bmp)
- TIFF (.tiff)
- GIF (.gif)

## ⚙️ Configuration

### Environment Variables

Set `TESSERACT_CMD` to specify custom Tesseract installation path:

**Windows:**
```bash
set TESSERACT_CMD=C:\Custom\Path\tesseract.exe
```

**Linux/macOS:**
```bash
export TESSERACT_CMD=/usr/local/bin/tesseract
```

### Model Configuration

The BLIP model is automatically downloaded on first run. Default settings:
- Model: `Salesforce/blip-image-captioning-base`
- Device: CPU (can be modified for GPU support)
- Max Caption Length: 60 tokens
- Beam Search: 3 beams

## 🐛 Troubleshooting

### Common Issues

**1. Tesseract not found error**
- Ensure Tesseract is installed and in system PATH
- Set `TESSERACT_CMD` environment variable
- Verify installation: `tesseract --version`

**2. Model download fails**
- Check internet connection
- Ensure sufficient disk space (~1GB for models)
- Try manually downloading models from Hugging Face

**3. Translation errors**
- `googletrans` may need specific version: `pip install googletrans==4.0.0rc1`
- Check internet connectivity
- Try alternative translation services if persistent

**4. Memory issues**
- BLIP model requires ~2GB RAM
- Close other applications
- Consider using GPU for better performance

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) - Google's OCR engine
- [Salesforce BLIP](https://github.com/salesforce/BLIP) - Image captioning model
- [Streamlit](https://streamlit.io/) - Web app framework
- [Hugging Face](https://huggingface.co/) - Transformer models
- [gTTS](https://github.com/pndurette/gTTS) - Text-to-Speech library

## 📧 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter) - your.email@example.com

Project Link: [https://github.com/yourusername/hybrid-image-nlp](https://github.com/yourusername/hybrid-image-nlp)

---

⭐ If you find this project useful, please consider giving it a star on GitHub!
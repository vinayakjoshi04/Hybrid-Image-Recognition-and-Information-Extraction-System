from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import uuid
import traceback

from ocr_utils import run_ocr
from image_captioning import run_captioning

# Config
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff", "gif"}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)  # allow Streamlit (or other frontends) to call this API


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/process", methods=["POST"])
def process():
    """
    POST /process
      - multipart/form-data:
        - 'file': the image file
        - 'task': 'ocr' or 'caption'
    returns JSON:
      - if task == 'ocr': {"raw_text": "...", "processed_text": "..."}
      - if task == 'caption': {"caption": "..."}
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded (missing 'file' part)"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return (
                jsonify(
                    {
                        "error": "Unsupported file extension. Allowed: "
                        + ", ".join(sorted(ALLOWED_EXTENSIONS))
                    }
                ),
                400,
            )

        task = (request.form.get("task") or request.args.get("task") or "").strip().lower()
        if task not in {"ocr", "caption"}:
            return jsonify({"error": "Missing or invalid 'task'. Use 'ocr' or 'caption'."}), 400

        # Save file with a unique name
        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_name)
        file.save(file_path)

        # Run selected task
        if task == "ocr":
            result = run_ocr(file_path)
        else:
            result = run_captioning(file_path)

        # Remove uploaded file (optional)
        try:
            os.remove(file_path)
        except Exception:
            pass

        return jsonify(result)
    except Exception as exc:
        trace = traceback.format_exc()
        # For development it's handy to return the trace. Remove 'trace' in production.
        return jsonify({"error": "Internal server error", "details": str(exc), "trace": trace}), 500


if __name__ == "__main__":
    # debug=True for local dev; set to False in production
    app.run(host="0.0.0.0", port=5000, debug=True)

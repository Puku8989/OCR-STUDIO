"""
Web interface for the Offline OCR Engine.

Run:
    python app.py

Then open http://localhost:5000 in your browser.
"""

import os
import tempfile
from pathlib import Path

from flask import Flask, render_template, request, jsonify

from ocr_engine import extract_text, extract_text_detailed

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff", "tif", "webp", "gif"}


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/ocr", methods=["POST"])
def ocr_endpoint():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    if file.filename == "" or not _allowed(file.filename):
        return jsonify({"error": f"Invalid file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"}), 400

    lang = request.form.get("lang", "en")
    backend = request.form.get("backend") or None

    # Save to a temp file, run OCR, clean up
    suffix = Path(file.filename).suffix
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp)
            tmp_path = tmp.name

        result = extract_text_detailed(tmp_path, lang=lang, backend=backend)

        lines_data = [
            {
                "text": ln.text,
                "confidence": round(ln.confidence, 4),
                "bbox": ln.bbox,
            }
            for ln in result.lines
        ]

        return jsonify({
            "text": result.raw_text,
            "backend": result.backend,
            "language": result.language,
            "line_count": len(result.lines),
            "lines": lines_data,
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    finally:
        if "tmp_path" in locals():
            os.unlink(tmp_path)


if __name__ == "__main__":
    print("\n  🔍  OCR Web UI → http://localhost:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)

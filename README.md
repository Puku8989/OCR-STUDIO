# Offline OCR Engine

A fully offline OCR solution for extracting text from images, built with **PaddleOCR** (primary) and **Tesseract** (fallback). After the initial model download, no internet connection is required.

---

## Features

- **Dual-backend architecture** — PaddleOCR for best accuracy; automatic Tesseract fallback
- **100 % offline** after first run (models are cached locally)
- **Layout preservation** — paragraph breaks are reconstructed from spatial data
- **Modular language support** — English out of the box; 12 languages pre-mapped, easily extensible
- **Detailed output** — per-line bounding boxes, confidence scores, and structured results
- **Simple API** — one function call: `extract_text(image_path)`

---

## Installation

### 1. Create a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **GPU acceleration (optional):**  
> Replace `paddlepaddle` with `paddlepaddle-gpu` and pass `use_gpu=True` to the API.

### 3. First-run model download

The first time you call `extract_text()`, PaddleOCR will automatically download its detection, recognition, and angle-classification models (~100 MB total). After that, everything runs offline.

### 4. Tesseract setup (fallback — optional)

Only needed if PaddleOCR cannot be installed on your platform.

| Platform | Install command |
|----------|----------------|
| **Windows** | Download the installer from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) and add it to `PATH` |
| **macOS** | `brew install tesseract` |
| **Ubuntu / Debian** | `sudo apt install tesseract-ocr` |
| **Fedora** | `sudo dnf install tesseract` |

---

## Quick Start

### Python API

```python
from ocr_engine import extract_text

# Basic extraction
text = extract_text("document.png")
print(text)

# With options
text = extract_text(
    "document.png",
    lang="en",          # language tag
    use_gpu=False,      # set True for GPU acceleration
    backend=None,       # "paddleocr", "tesseract", or None (auto)
)
```

### Detailed output (bounding boxes + confidence)

```python
from ocr_engine import extract_text_detailed

result = extract_text_detailed("document.png")

print(f"Backend: {result.backend}")
for line in result.lines:
    print(f"  [{line.confidence:.0%}] {line.text}")

# Full reconstructed text
print(result.raw_text)
```

### Command line

```bash
# Simple
python ocr_engine.py photo.jpg

# Detailed with Tesseract
python ocr_engine.py photo.jpg --backend tesseract --detailed

# Different language
python ocr_engine.py photo.jpg --lang fr
```

### Demo script

```bash
python demo.py path/to/image.png
python demo.py path/to/image.png --detailed
```

---

## Adding a New Language

Edit the `LANGUAGE_MAP` dictionary in `ocr_engine.py`:

```python
LANGUAGE_MAP = {
    # ... existing entries ...
    "vi": {"paddle": "vi", "tesseract": "vie"},   # Vietnamese
}
```

Then call with `lang="vi"`. PaddleOCR will download the appropriate model on first use.

For Tesseract, install the corresponding language pack:
```bash
# Ubuntu example
sudo apt install tesseract-ocr-vie
```

---

## Supported Languages (pre-mapped)

| Tag | Language | PaddleOCR | Tesseract |
|-----|----------|-----------|-----------|
| `en` | English | ✅ | ✅ |
| `ch` | Chinese (Simplified) | ✅ | ✅ |
| `fr` | French | ✅ | ✅ |
| `de` | German | ✅ | ✅ |
| `es` | Spanish | ✅ | ✅ |
| `pt` | Portuguese | ✅ | ✅ |
| `it` | Italian | ✅ | ✅ |
| `ja` | Japanese | ✅ | ✅ |
| `ko` | Korean | ✅ | ✅ |
| `ar` | Arabic | ✅ | ✅ |
| `hi` | Hindi | ✅ | ✅ |
| `ru` | Russian | ✅ | ✅ |

---

## Project Structure

```
ocr image/
├── ocr_engine.py      # Core module — import this
├── demo.py            # Quick demo script
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: paddlepaddle` | Run `pip install paddlepaddle paddleocr` |
| `TesseractNotFoundError` | Install Tesseract and add to PATH (see table above) |
| Poor accuracy on small text | Pre-scale the image to ≥300 DPI before passing to `extract_text()` |
| `CUDA out of memory` | Use `use_gpu=False` or reduce image size |
| First run is slow | Models are downloading; subsequent runs are instant |

---

## License

MIT

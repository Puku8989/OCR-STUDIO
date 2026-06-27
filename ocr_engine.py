"""
Offline OCR Engine
==================
Extracts text from images using PaddleOCR (primary) or Tesseract (fallback).
Works completely offline after the initial model download.

Usage:
    from ocr_engine import extract_text
    text = extract_text("path/to/image.png")
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("ocr_engine")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("[%(levelname)s] %(name)s — %(message)s")
    )
    logger.addHandler(_handler)

# ---------------------------------------------------------------------------
# Tesseract path configuration (Windows)
# ---------------------------------------------------------------------------
# Auto-detect Tesseract install location on Windows so pytesseract works
# even when Tesseract is not added to the system PATH.
if sys.platform == "win32":
    _tess_candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Tesseract-OCR\tesseract.exe"),
        os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    for _tess_path in _tess_candidates:
        if os.path.isfile(_tess_path):
            try:
                import pytesseract  # type: ignore
                pytesseract.pytesseract.tesseract_cmd = _tess_path
                logger.info("Tesseract found at: %s", _tess_path)
            except ImportError:
                pass
            break

# ---------------------------------------------------------------------------
# Supported languages mapping
# ---------------------------------------------------------------------------
# Maps a common language tag to the identifiers used by each backend.
# Add new entries here to support additional languages.
LANGUAGE_MAP: dict[str, dict[str, str]] = {
    "en": {"paddle": "en", "tesseract": "eng"},
    "ch": {"paddle": "ch", "tesseract": "chi_sim"},
    "fr": {"paddle": "french", "tesseract": "fra"},
    "de": {"paddle": "german", "tesseract": "deu"},
    "es": {"paddle": "es", "tesseract": "spa"},
    "pt": {"paddle": "pt", "tesseract": "por"},
    "it": {"paddle": "it", "tesseract": "ita"},
    "ja": {"paddle": "japan", "tesseract": "jpn"},
    "ko": {"paddle": "korean", "tesseract": "kor"},
    "ar": {"paddle": "ar", "tesseract": "ara"},
    "hi": {"paddle": "hi", "tesseract": "hin"},
    "ru": {"paddle": "ru", "tesseract": "rus"},
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class OCRLine:
    """A single detected text line with its bounding box and confidence."""
    text: str
    confidence: float
    bbox: list[list[int]]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]


@dataclass
class OCRResult:
    """Complete OCR result preserving spatial layout."""
    lines: list[OCRLine] = field(default_factory=list)
    raw_text: str = ""
    backend: str = ""
    language: str = "en"

    def __str__(self) -> str:
        return self.raw_text


# ---------------------------------------------------------------------------
# Backend: PaddleOCR
# ---------------------------------------------------------------------------

class PaddleBackend:
    """Primary OCR backend using PaddleOCR."""

    def __init__(self, lang: str = "en", use_gpu: bool = False) -> None:
        self.lang = lang
        self.use_gpu = use_gpu
        self._engine = None

    def _get_paddle_lang(self, lang: str) -> str:
        entry = LANGUAGE_MAP.get(lang)
        if entry and "paddle" in entry:
            return entry["paddle"]
        return lang

    def initialize(self) -> None:
        """Load PaddleOCR models. Downloads on first run, then works offline."""
        from paddleocr import PaddleOCR  # type: ignore

        paddle_lang = self._get_paddle_lang(self.lang)
        logger.info("Initializing PaddleOCR (lang=%s, gpu=%s)…", paddle_lang, self.use_gpu)
        self._engine = PaddleOCR(
            use_angle_cls=True,
            lang=paddle_lang,
            use_gpu=self.use_gpu,
            show_log=False,
        )
        logger.info("PaddleOCR ready.")

    def extract(self, image: np.ndarray) -> OCRResult:
        """Run OCR on a numpy image array."""
        if self._engine is None:
            self.initialize()

        result = self._engine.ocr(image, cls=True)
        lines: list[OCRLine] = []

        if result and result[0]:
            for entry in result[0]:
                bbox, (text, conf) = entry
                # Convert bbox coords to ints
                bbox_int = [[int(x), int(y)] for x, y in bbox]
                lines.append(OCRLine(text=text, confidence=conf, bbox=bbox_int))

        # Sort lines top-to-bottom, then left-to-right for formatting
        lines.sort(key=lambda ln: (ln.bbox[0][1], ln.bbox[0][0]))

        raw_text = _reconstruct_text(lines)
        return OCRResult(
            lines=lines,
            raw_text=raw_text,
            backend="paddleocr",
            language=self.lang,
        )


# ---------------------------------------------------------------------------
# Backend: Tesseract
# ---------------------------------------------------------------------------

class TesseractBackend:
    """Fallback OCR backend using Tesseract via pytesseract."""

    def __init__(self, lang: str = "en") -> None:
        self.lang = lang
        self._available: Optional[bool] = None

    def _get_tess_lang(self, lang: str) -> str:
        entry = LANGUAGE_MAP.get(lang)
        if entry and "tesseract" in entry:
            return entry["tesseract"]
        return lang

    def initialize(self) -> None:
        """Verify that Tesseract is installed and accessible."""
        import pytesseract  # type: ignore

        try:
            pytesseract.get_tesseract_version()
            self._available = True
            logger.info("Tesseract is available.")
        except Exception as exc:
            self._available = False
            raise RuntimeError(
                "Tesseract is not installed or not on PATH. "
                "Install it from https://github.com/tesseract-ocr/tesseract"
            ) from exc

    def extract(self, image: np.ndarray) -> OCRResult:
        """Run OCR on a numpy image array."""
        import pytesseract  # type: ignore

        if self._available is None:
            self.initialize()

        tess_lang = self._get_tess_lang(self.lang)
        pil_image = Image.fromarray(image)

        # Get detailed data for structured output
        data = pytesseract.image_to_data(
            pil_image, lang=tess_lang, output_type=pytesseract.Output.DICT
        )

        lines: list[OCRLine] = []
        current_line_num = -1
        current_text_parts: list[str] = []
        current_confs: list[float] = []
        current_bbox: Optional[list[list[int]]] = None

        for i in range(len(data["text"])):
            word = data["text"][i].strip()
            conf = float(data["conf"][i])
            line_num = data["line_num"][i]

            if conf < 0 or not word:
                continue

            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            word_bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

            if line_num != current_line_num:
                # Flush previous line
                if current_text_parts and current_bbox is not None:
                    avg_conf = sum(current_confs) / len(current_confs) if current_confs else 0.0
                    lines.append(OCRLine(
                        text=" ".join(current_text_parts),
                        confidence=avg_conf / 100.0,
                        bbox=current_bbox,
                    ))
                current_line_num = line_num
                current_text_parts = [word]
                current_confs = [conf]
                current_bbox = word_bbox
            else:
                current_text_parts.append(word)
                current_confs.append(conf)
                # Expand bbox to encompass new word
                if current_bbox is not None:
                    current_bbox = [
                        [min(current_bbox[0][0], word_bbox[0][0]), min(current_bbox[0][1], word_bbox[0][1])],
                        [max(current_bbox[1][0], word_bbox[1][0]), min(current_bbox[1][1], word_bbox[1][1])],
                        [max(current_bbox[2][0], word_bbox[2][0]), max(current_bbox[2][1], word_bbox[2][1])],
                        [min(current_bbox[3][0], word_bbox[3][0]), max(current_bbox[3][1], word_bbox[3][1])],
                    ]

        # Flush last line
        if current_text_parts and current_bbox is not None:
            avg_conf = sum(current_confs) / len(current_confs) if current_confs else 0.0
            lines.append(OCRLine(
                text=" ".join(current_text_parts),
                confidence=avg_conf / 100.0,
                bbox=current_bbox,
            ))

        raw_text = _reconstruct_text(lines)
        return OCRResult(
            lines=lines,
            raw_text=raw_text,
            backend="tesseract",
            language=self.lang,
        )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _reconstruct_text(lines: list[OCRLine]) -> str:
    """
    Reconstruct multi-line text preserving vertical spacing.

    Inserts a blank line when the vertical gap between consecutive lines
    exceeds 1.5× the average line height — approximating paragraph breaks.
    """
    if not lines:
        return ""

    # Compute average line height
    heights = [abs(ln.bbox[2][1] - ln.bbox[0][1]) for ln in lines]
    avg_height = sum(heights) / len(heights) if heights else 20

    parts: list[str] = [lines[0].text]
    for prev, curr in zip(lines, lines[1:]):
        gap = curr.bbox[0][1] - prev.bbox[2][1]
        if gap > avg_height * 1.5:
            parts.append("")  # blank line for paragraph break
        parts.append(curr.text)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _load_image(image_path: str | Path) -> np.ndarray:
    """Load an image from disk and return an RGB numpy array."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    img = Image.open(path)

    # Convert to RGB (handles RGBA, grayscale, palette, etc.)
    if img.mode != "RGB":
        img = img.convert("RGB")

    return np.array(img)


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

# Module-level backend cache (lazy-initialized)
_backends: dict[str, PaddleBackend | TesseractBackend] = {}


def _get_backend(
    lang: str = "en",
    use_gpu: bool = False,
    backend: Optional[str] = None,
) -> PaddleBackend | TesseractBackend:
    """
    Return an initialized backend, creating it on first use.

    Priority: PaddleOCR → Tesseract (unless *backend* is explicitly set).
    """
    cache_key = f"{backend or 'auto'}:{lang}:{use_gpu}"

    if cache_key in _backends:
        return _backends[cache_key]

    if backend == "tesseract":
        be = TesseractBackend(lang=lang)
        be.initialize()
        _backends[cache_key] = be
        return be

    if backend == "paddleocr" or backend is None:
        try:
            be = PaddleBackend(lang=lang, use_gpu=use_gpu)
            be.initialize()
            _backends[cache_key] = be
            return be
        except Exception as exc:
            if backend == "paddleocr":
                raise RuntimeError(
                    "PaddleOCR could not be initialized. "
                    "Make sure paddlepaddle and paddleocr are installed."
                ) from exc
            logger.warning("PaddleOCR unavailable (%s), falling back to Tesseract.", exc)

    # Fallback to Tesseract
    be = TesseractBackend(lang=lang)
    be.initialize()
    _backends[cache_key] = be
    return be


def extract_text(
    image_path: str | Path,
    *,
    lang: str = "en",
    use_gpu: bool = False,
    backend: Optional[str] = None,
    preserve_layout: bool = True,
) -> str:
    """
    Extract text from an image file.

    Parameters
    ----------
    image_path : str | Path
        Path to the image file (PNG, JPG, BMP, TIFF, etc.).
    lang : str
        Language tag (default ``"en"``).  See ``LANGUAGE_MAP`` for all
        supported tags.  Additional languages can be added there.
    use_gpu : bool
        Enable GPU acceleration for PaddleOCR (requires paddlepaddle-gpu).
    backend : str | None
        Force a specific backend: ``"paddleocr"`` or ``"tesseract"``.
        If ``None``, PaddleOCR is tried first with automatic Tesseract
        fallback.
    preserve_layout : bool
        If ``True`` (default), attempt to preserve paragraph breaks.

    Returns
    -------
    str
        The extracted text.

    Raises
    ------
    FileNotFoundError
        If the image file does not exist.
    RuntimeError
        If no OCR backend could be initialized.
    """
    image = _load_image(image_path)
    be = _get_backend(lang=lang, use_gpu=use_gpu, backend=backend)
    result = be.extract(image)
    return result.raw_text


def extract_text_detailed(
    image_path: str | Path,
    *,
    lang: str = "en",
    use_gpu: bool = False,
    backend: Optional[str] = None,
) -> OCRResult:
    """
    Extract text with full metadata (bounding boxes, confidence scores).

    Returns an :class:`OCRResult` containing individual :class:`OCRLine`
    objects for each detected line.
    """
    image = _load_image(image_path)
    be = _get_backend(lang=lang, use_gpu=use_gpu, backend=backend)
    return be.extract(image)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Simple CLI: ``python ocr_engine.py <image_path> [--lang en] [--backend paddleocr|tesseract]``"""
    import argparse

    parser = argparse.ArgumentParser(description="Offline OCR — extract text from images")
    parser.add_argument("image", help="Path to the image file")
    parser.add_argument("--lang", default="en", help="Language tag (default: en)")
    parser.add_argument("--backend", choices=["paddleocr", "tesseract"], default=None)
    parser.add_argument("--gpu", action="store_true", help="Use GPU (PaddleOCR only)")
    parser.add_argument("--detailed", action="store_true", help="Print per-line details")
    args = parser.parse_args()

    if args.detailed:
        result = extract_text_detailed(
            args.image, lang=args.lang, use_gpu=args.gpu, backend=args.backend,
        )
        print(f"Backend : {result.backend}")
        print(f"Language: {result.language}")
        print(f"Lines   : {len(result.lines)}")
        print("-" * 60)
        for i, line in enumerate(result.lines, 1):
            print(f"  [{i}] (conf={line.confidence:.2f}) {line.text}")
        print("-" * 60)
        print(result.raw_text)
    else:
        text = extract_text(
            args.image, lang=args.lang, use_gpu=args.gpu, backend=args.backend,
        )
        print(text)


if __name__ == "__main__":
    main()

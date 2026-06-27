"""
Quick demo — run OCR on a sample image.

Usage:
    python demo.py <image_path>
    python demo.py <image_path> --detailed
"""

import sys
from pathlib import Path

from ocr_engine import extract_text, extract_text_detailed


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python demo.py <image_path> [--detailed]")
        sys.exit(1)

    image_path = sys.argv[1]
    detailed = "--detailed" in sys.argv

    if not Path(image_path).exists():
        print(f"Error: file not found — {image_path}")
        sys.exit(1)

    if detailed:
        result = extract_text_detailed(image_path)
        print(f"Backend : {result.backend}")
        print(f"Language: {result.language}")
        print(f"Lines   : {len(result.lines)}\n")
        for i, line in enumerate(result.lines, 1):
            print(f"  [{i:>3}] conf={line.confidence:.2%}  │ {line.text}")
        print("\n" + "═" * 60)
        print(result.raw_text)
    else:
        text = extract_text(image_path)
        print(text)


if __name__ == "__main__":
    main()

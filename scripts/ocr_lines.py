#!/usr/bin/env python3
"""Simple OCR over data/lines/ that preserves original case.

Creates `data/lines_transcriptions.txt` with lines like:
<relative-path>\t<transcription>

Requirements:
- Python packages: `pytesseract`, `opencv-python`, `pillow`
- System: Tesseract OCR installed and on PATH (or set `--tesseract-cmd`).
"""
import os
import argparse
from pathlib import Path

import cv2
import pytesseract


def transcribe_image(path, psm=7, oem=3):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return ""
    # basic preprocessing: adaptive threshold often helps; keep as optional
    try:
        _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        pil_mode = th
    except Exception:
        pil_mode = img

    config = f"--oem {oem} --psm {psm}"
    # do NOT change case — return exactly as tesseract outputs
    text = pytesseract.image_to_string(pil_mode, config=config)
    # strip only leading/trailing whitespace/newlines, keep internal casing
    return text.strip()


def find_images(folder):
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    for p in sorted(Path(folder).glob("**/*")):
        if p.suffix.lower() in exts and p.is_file():
            yield p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lines-dir", default="data/lines", help="Folder with line images")
    parser.add_argument("--out", default="data/lines_transcriptions.txt", help="Output file")
    parser.add_argument("--tesseract-cmd", default=None, help="Full path to tesseract executable (optional)")
    parser.add_argument("--psm", type=int, default=7, help="Tesseract PSM (page segmentation mode)")
    args = parser.parse_args()

    # Load default tesseract cmd from configs/config.yaml if present.
    cfg_path = Path("configs/config.yaml")
    default_cmd = None
    if cfg_path.exists():
        try:
            import yaml

            cfg = yaml.safe_load(cfg_path.read_text())
            default_cmd = cfg.get("ocr", {}).get("tesseract_cmd") if cfg else None
        except Exception:
            try:
                for line in cfg_path.read_text().splitlines():
                    if "tesseract_cmd" in line:
                        default_cmd = line.split(":", 1)[1].strip().strip('"').strip("'")
                        break
            except Exception:
                default_cmd = None

    final_cmd = args.tesseract_cmd or default_cmd
    if final_cmd:
        pytesseract.pytesseract.tesseract_cmd = final_cmd
        try:
            tessdata = Path(final_cmd).parent / "tessdata"
            if tessdata.exists():
                os.environ["TESSDATA_PREFIX"] = str(tessdata)
            else:
                parent = Path(final_cmd).parent
                if (parent / "tessdata").exists():
                    os.environ["TESSDATA_PREFIX"] = str(parent / "tessdata")
                else:
                    os.environ["TESSDATA_PREFIX"] = str(parent)
        except Exception:
            pass

    lines_dir = Path(args.lines_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fh:
        for img_path in find_images(lines_dir):
            rel = os.path.relpath(img_path, start=os.getcwd())
            text = transcribe_image(img_path, psm=args.psm)
            fh.write(f"{rel}\t{text}\n")
            print(f"{rel}: {text}")


if __name__ == "__main__":
    main()

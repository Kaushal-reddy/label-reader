"""Stage 2: baseline. Barcode decode + OCR on the full, uncropped photos.

Run with:  python baseline.py   (from the repo folder, with the venv active)

No cropping, no preprocessing beyond grayscale, no tuning. The point is to
measure how far the libraries get on their own, so later stages have a number
to beat.

Outputs (all in output/baseline/):
  results.csv          one row per photo: what was decoded, whether it matched, timings
  ocr_text/<photo>.txt the raw text Tesseract returned, so you can see what it "saw"
  boxes/<photo>.jpg    the photo with a green box around every barcode ZBar found
"""

import os
import re
import time

import native_libs  # noqa: I001  (must come before pyzbar: lets it find ZBar on macOS)
import cv2
import numpy as np
import pandas as pd
import pytesseract
from pyzbar import pyzbar

PHOTOS_DIR = "photos"
OUT_DIR = os.path.join("output", "baseline")

# One barcode ZBar found: (barcode type, decoded value, (x, y, width, height)).
Barcode = tuple[str, str, tuple[int, int, int, int]]

# pytesseract runs whatever tesseract_cmd names. Point it at the real install
# (the Windows installer doesn't add Tesseract to PATH). If none is found, keep
# the default so pytesseract raises its own clear "tesseract is not installed".
pytesseract.pytesseract.tesseract_cmd = native_libs.find_tesseract() or "tesseract"


def normalize(text: str) -> str:
    """Uppercase and keep only letters and digits.

    OCR output is full of spaces, line breaks and stray punctuation, and labels
    write the serial as "SN: ...", "S/N:..." etc. Stripping all of that lets us
    compare characters only. Example: "S/N: scbm nv00" -> "SNSCBMNV00".
    """
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def found(true_serial: str, text: str) -> bool:
    """True if the serial appears anywhere in the text (after normalizing both).

    This is deliberately lenient: at this stage we only ask "is the serial
    somewhere in what the tool returned?", not "did we pick the right line".
    Picking the right line is Stage 6's job (validation).
    """
    return normalize(true_serial) in normalize(text)


def read_barcodes(gray: np.ndarray) -> list[Barcode]:
    """Run ZBar on the whole image. Returns a list of (type, value, rect)."""
    results = []
    for b in pyzbar.decode(gray):
        # b.data is bytes; most labels use plain ASCII, "replace" avoids a crash if not.
        value = b.data.decode("utf-8", errors="replace")
        results.append((b.type, value, b.rect))
    return results


def save_box_image(img: np.ndarray, barcodes: list[Barcode], path: str) -> None:
    """Draw a green rectangle around each barcode ZBar found, for checking by eye."""
    vis = img.copy()
    for btype, value, (x, y, w, h) in barcodes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 4)
        cv2.putText(vis, btype, (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2)
    cv2.imwrite(path, vis)


def main() -> None:
    os.makedirs(os.path.join(OUT_DIR, "ocr_text"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "boxes"), exist_ok=True)

    truth = pd.read_csv("truth.csv")
    rows = []

    for filename, true_serial in zip(truth.filename, truth.true_serial):
        stem = os.path.splitext(filename)[0]
        img = cv2.imread(os.path.join(PHOTOS_DIR, filename))
        if img is None:
            # cv2.imread returns None instead of raising when the file is missing/unreadable.
            print(f"Could not read {filename}, skipping")
            continue
        # Both ZBar and Tesseract work on intensity, so colour adds nothing here.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- Barcode ---
        t0 = time.perf_counter()
        barcodes = read_barcodes(gray)
        barcode_s = time.perf_counter() - t0

        barcode_values = [value for _, value, _ in barcodes]
        barcode_ok = any(found(true_serial, v) for v in barcode_values)
        save_box_image(img, barcodes, os.path.join(OUT_DIR, "boxes", f"{stem}.jpg"))

        # --- OCR ---
        # Default settings: English, page segmentation mode 3 (fully automatic
        # page layout). Not tuned on purpose; this is the baseline.
        t0 = time.perf_counter()
        ocr_text = pytesseract.image_to_string(gray)
        ocr_s = time.perf_counter() - t0

        ocr_ok = found(true_serial, ocr_text)
        with open(os.path.join(OUT_DIR, "ocr_text", f"{stem}.txt"), "w", encoding="utf-8") as f:
            f.write(ocr_text)

        rows.append({
            "filename": filename,
            "true_serial": true_serial,
            "barcodes_found": len(barcodes),
            "barcode_values": " | ".join(f"{t}:{v}" for t, v, _ in barcodes),
            "barcode_ok": barcode_ok,
            "ocr_ok": ocr_ok,
            "either_ok": barcode_ok or ocr_ok,
            "barcode_s": round(barcode_s, 3),
            "ocr_s": round(ocr_s, 3),
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "results.csv"), index=False)

    # --- Summary ---
    n = len(df)
    print(df[["filename", "barcodes_found", "barcode_ok", "ocr_ok", "barcode_s", "ocr_s"]]
          .to_string(index=False))
    print()
    print(f"Photos:            {n}")
    print(f"Barcode correct:   {df.barcode_ok.sum()}/{n}")
    print(f"OCR found serial:  {df.ocr_ok.sum()}/{n}")
    print(f"Either correct:    {df.either_ok.sum()}/{n}")
    print(f"Mean time/photo:   barcode {df.barcode_s.mean():.2f}s, OCR {df.ocr_s.mean():.2f}s")


if __name__ == "__main__":
    main()

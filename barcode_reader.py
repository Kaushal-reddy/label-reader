"""Stage 3: barcode reading, with retries on modified images when the plain decode fails.

Run with:  python barcode_reader.py   (from the repo folder, with the venv active)

Idea: ZBar either decodes a barcode or returns nothing (it almost never returns a
wrong value, because barcodes carry a checksum). So if the plain photo fails we
can safely try other versions of it - bigger, sharper, rotated - and stop at the
first one that decodes. Extra attempts cost time but can't add wrong answers.

This script runs EVERY variant on EVERY photo (not just until the first success),
so you can see exactly which change rescued which photo:
  output/barcode/variant_matrix.csv   photo x variant table: serial found? (True/False)
  output/barcode/results.csv          what the real reader (stop at first success) returned
  output/barcode/variants/<photo>/    the image each variant fed to ZBar
"""

import os
import time

import native_libs  # noqa: F401, I001  (must come before pyzbar: lets it find ZBar on macOS)
import cv2
import numpy as np
import pandas as pd
from pyzbar import pyzbar
from pyzbar.pyzbar import ZBarSymbol

from baseline import found  # same lenient "is the serial in there?" check as Stage 2

PHOTOS_DIR = "photos"
OUT_DIR = os.path.join("output", "barcode")


# ---------- image helpers ----------

def upscale(gray: np.ndarray, factor: float) -> np.ndarray:
    # INTER_CUBIC interpolates smoothly between pixels, so thin bars become
    # several pixels wide instead of blocky 1-pixel steps.
    return cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


def sharpen(gray: np.ndarray) -> np.ndarray:
    # Unsharp mask: subtract a blurred copy to boost edges.
    # result = 1.5*image - 0.5*blurred  -> bar edges get crisper.
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
    return cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)


def rotate(gray: np.ndarray, angle: float) -> np.ndarray:
    """Rotate around the centre, growing the canvas so the corners aren't cut off."""
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    # Work out the size of the rotated image's bounding box...
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
    # ...and shift so the image lands in the middle of the new canvas.
    m[0, 2] += new_w / 2 - w / 2
    m[1, 2] += new_h / 2 - h / 2
    return cv2.warpAffine(gray, m, (new_w, new_h), borderValue=255)


def clahe(gray: np.ndarray) -> np.ndarray:
    # CLAHE = local contrast boost. Helps when part of the image is in shadow,
    # because it stretches contrast tile-by-tile instead of over the whole image.
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


# Each variant is (name, function that turns the grayscale photo into the image to try).
# Order matters for the real reader: cheapest / most likely first.
VARIANTS = [
    ("original",       lambda g: g),
    ("upscale2x",      lambda g: upscale(g, 2)),
    ("upscale3x",      lambda g: upscale(g, 3)),
    ("upscale2x_sharp", lambda g: sharpen(upscale(g, 2))),
    ("clahe_2x",       lambda g: upscale(clahe(g), 2)),
    # Rotations: ZBar scans rows and columns, so a barcode tilted ~20-40 degrees
    # can fall between the two. Rotating brings it back to roughly straight.
    *[(f"rot{a:+d}_2x", (lambda a: lambda g: rotate(upscale(g, 2), a))(a))
      for a in (-30, -15, 15, 30)],
]

# The variants the real reader actually uses. Chosen from the variant matrix on
# our 11 photos: only upscale3x rescued a photo the original missed
# (Laptop_2_Glare). The others rescued nothing, so they'd only cost time.
# With a bigger photo set this list should be re-checked.
READER_VARIANTS = ["original", "upscale3x"]

# Only look for barcode types that IT equipment labels actually use. Without this,
# ZBar also tries types like I25 (Interleaved 2 of 5), which has no mandatory
# checksum: on Mouse_2_Sideangle it "decoded" the mousepad texture as I25 '132342'.
# Code128 and QR both carry checksums, so a decode from them is trustworthy.
SYMBOLS = [ZBarSymbol.CODE128, ZBarSymbol.CODE39, ZBarSymbol.QRCODE, ZBarSymbol.EAN13]


# ---------- decoding ----------

def decode(img: np.ndarray) -> list[str]:
    """Return the list of decoded values (strings) ZBar finds in img."""
    return [b.data.decode("utf-8", errors="replace")
            for b in pyzbar.decode(img, symbols=SYMBOLS)]


def read_barcode(gray: np.ndarray) -> tuple[list[str], str | None]:
    """The actual reader used by later stages.

    Tries each reader variant in order and returns (values, variant_name) for the
    first one where ZBar decodes anything. Returns ([], None) if every variant fails.
    """
    makers = dict(VARIANTS)
    for name in READER_VARIANTS:
        values = decode(makers[name](gray))
        if values:
            return values, name
    return [], None


# ---------- experiment ----------

def main() -> None:
    truth = pd.read_csv("truth.csv")
    matrix_rows, result_rows = [], []

    for filename, true_serial in zip(truth.filename, truth.true_serial):
        stem = os.path.splitext(filename)[0]
        gray = cv2.cvtColor(cv2.imread(os.path.join(PHOTOS_DIR, filename)), cv2.COLOR_BGR2GRAY)
        var_dir = os.path.join(OUT_DIR, "variants", stem)
        os.makedirs(var_dir, exist_ok=True)

        # 1) Diagnostic: every variant, so we learn which ones matter.
        row = {"filename": filename}
        for name, make in VARIANTS:
            img = make(gray)
            cv2.imwrite(os.path.join(var_dir, f"{name}.jpg"), img)
            row[name] = any(found(true_serial, v) for v in decode(img))
        matrix_rows.append(row)

        # 2) The real reader: stop at the first variant that decodes.
        t0 = time.perf_counter()
        values, variant = read_barcode(gray)
        elapsed = time.perf_counter() - t0
        result_rows.append({
            "filename": filename,
            "true_serial": true_serial,
            "values": " | ".join(values),
            "variant_used": variant,
            "barcode_ok": any(found(true_serial, v) for v in values),
            "seconds": round(elapsed, 3),
        })

    matrix = pd.DataFrame(matrix_rows)
    results = pd.DataFrame(result_rows)
    matrix.to_csv(os.path.join(OUT_DIR, "variant_matrix.csv"), index=False)
    results.to_csv(os.path.join(OUT_DIR, "results.csv"), index=False)

    # Print the matrix compactly: '#' = serial found, '.' = not found.
    names = [n for n, _ in VARIANTS]
    print("Which variant found the serial ('#' = yes):\n")
    print(f"{'':24}" + " ".join(f"{i:>2}" for i in range(len(names))))
    for r in matrix_rows:
        print(f"{r['filename']:24}" + " ".join(f"{'#' if r[n] else '.':>2}" for n in names))
    print("\nColumns: " + ", ".join(f"{i}={n}" for i, n in enumerate(names)))

    n = len(results)
    print("\nReal reader (stop at first success):")
    print(results[["filename", "variant_used", "barcode_ok", "seconds"]].to_string(index=False))
    # Compare against whatever the last baseline.py run produced (if it exists).
    baseline_path = os.path.join("output", "baseline", "results.csv")
    if os.path.exists(baseline_path):
        base = pd.read_csv(baseline_path)
        print(f"\nBarcode correct: {results.barcode_ok.sum()}/{n}   "
              f"(Stage 2 baseline: {base.barcode_ok.sum()}/{len(base)})")
    else:
        print(f"\nBarcode correct: {results.barcode_ok.sum()}/{n}   (run baseline.py to compare)")
    print(f"Mean time/photo: {results.seconds.mean():.2f}s   max: {results.seconds.max():.2f}s")


if __name__ == "__main__":
    main()

"""Stage 3: barcode reading, with retries on modified images when the plain decode fails.

Run with:  python barcode_reader.py   (from the repo folder, with the venv active)

Idea: if the plain photo fails, try other versions of it - bigger, sharper,
rotated - and stop at the first one that decodes. Barcodes carry a checksum, so
we first assumed extra attempts cost time but couldn't add wrong answers. The
403-image EAN set showed otherwise: on blurry, small barcodes ZBar sometimes
reads a wrong EAN-13 that still passes the check digit (1 and 7, or 2 and 8,
differ by one bar width). So a product barcode is also read with a few more
variants, and validation only trusts it when two variants agree (CONFIRM_VARIANTS).

This script runs EVERY variant on EVERY photo (not just until the first success),
so you can see exactly which change rescued which photo:
  output/barcode/variant_matrix.csv   photo x variant table: serial found? (True/False)
  output/barcode/results.csv          what the real reader (stop at first success) returned
  output/barcode/variants/<photo>/    a preview of the image each variant fed to ZBar
                                      (shrunk to PREVIEW_MAX_SIDE; ZBar got full size)

Another dataset folder (its own photos + truth.csv) can be given instead:
  python barcode_reader.py photos/ean_dataset
Results then go to output/<folder name>/barcode/ (without the variant images,
which would be thousands of files), and truth.csv rows with a `checked`
column that isn't "yes" are skipped, so unverified values never count.
"""

import os
import sys
import time
from typing import NamedTuple

import native_libs  # noqa: F401, I001  (must come before pyzbar: lets it find ZBar on macOS)
import cv2
import numpy as np
import pandas as pd
from pyzbar import pyzbar
from pyzbar.pyzbar import ZBarSymbol

from baseline import found, load_truth, out_dir_for  # found: same lenient check as Stage 2

PHOTOS_DIR = "photos"
OUT_DIR = os.path.join("output", "barcode")

# The saved variant images are only for looking at, so they're shrunk to at most
# this many pixels on the long side. Saving them full size was the slow part of
# this script: a 3x variant of a 36-megapixel photo is a 329-megapixel image,
# and JPEG-compressing it took longer than decoding it (~4-5 min per photo).
PREVIEW_MAX_SIDE = 1600


def preview(img: np.ndarray) -> np.ndarray:
    """A shrunk copy for saving to disk (never bigger than the original)."""
    scale = PREVIEW_MAX_SIDE / max(img.shape[:2])
    if scale >= 1:
        return img
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


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

# Variants that re-read a product barcode (EAN-13) once READER_VARIANTS found one.
# A product barcode is accepted without a second source (OCR can't read the small
# digits), so one read isn't enough: on the 403-image EAN set the reader above
# accepted 216 barcodes and 8 were wrong, each read by a single variant with a
# valid check digit. Counting how many variants read the same value, and treating
# any other EAN-13 they read as a conflict, makes validate.py accept 178 with
# 0 wrong (44 more go to review). These two were the best of the 9 for that;
# rotating found wrong values as often as it confirmed right ones.
CONFIRM_VARIANTS = ["original", "upscale2x", "clahe_2x"]

# Only look for barcode types that IT equipment labels and retail products actually
# use. Without this, ZBar also tries types like I25 (Interleaved 2 of 5), which has
# no mandatory checksum: on Mouse_2_Sideangle it "decoded" the mousepad texture as
# I25 '132342'. Code128, QR and EAN-13 all carry checksums - but see resolve():
# a checksum alone did not stop one wrong EAN-13 read.
SYMBOLS = [ZBarSymbol.CODE128, ZBarSymbol.CODE39, ZBarSymbol.QRCODE, ZBarSymbol.EAN13]

# QR codes are 2-D and ZBar always gives them quality 1, so votes only mean
# something for the 1-D (bar) types.
LINEAR_TYPES = {"CODE128", "CODE39", "EAN13"}
# A value must have at least this many times the votes of any other value read
# from the same barcode to be kept (see resolve()). On the Monster can the true
# value had 3.6-168x the phantom's votes; 3 is just below the smallest margin
# seen, so this is tuned on very little data.
MIN_VOTE_RATIO = 3


# ---------- decoding ----------

class BarcodeRead(NamedTuple):
    values: list[str]          # what the barcode(s) say, after resolve()
    variant: str | None        # which image variant they were read from
    conflicts: list[str]       # other values ZBar read from the same barcode(s), rejected
    agreeing: dict[str, int]   # value -> how many image variants read it (see CONFIRM_VARIANTS)


def overlaps(a, b) -> bool:
    """Do two pyzbar rects (left, top, width, height) overlap?"""
    return (a.left < b.left + b.width and b.left < a.left + a.width
            and a.top < b.top + b.height and b.top < a.top + a.height)


def resolve(decoded: list) -> tuple[list[str], list[str]]:
    """Turn ZBar's raw results into (values to trust, conflicting values rejected).

    ZBar can read ONE barcode as two different values. On the curved Monster can
    it read the true EAN-13 9342866001670 and also 9348266007670 - from the same
    spot, and the wrong one passes the EAN-13 check digit too. What separates
    them is ZBar's `quality`: for bar codes it's how many of its scan lines
    agreed on the value, i.e. votes. The true value got 122 votes, the wrong
    one 14.

    So results of the same 1-D type at overlapping positions are treated as
    one barcode. If they disagree, the most-voted value is kept only if it has
    at least MIN_VOTE_RATIO times the votes of every other value; otherwise
    none is trusted. Either way the losers are returned as conflicts, so
    validation can refuse to auto-accept a barcode that was read two ways.
    """
    groups: list[list] = []
    for d in decoded:
        if d.type in LINEAR_TYPES:
            group = next((g for g in groups if g[0].type == d.type
                          and any(overlaps(d.rect, o.rect) for o in g)), None)
            if group is not None:
                group.append(d)
                continue
        groups.append([d])

    values, conflicts = [], []
    for g in groups:
        votes: dict[str, int] = {}
        for d in g:
            v = d.data.decode("utf-8", errors="replace")
            votes[v] = votes.get(v, 0) + d.quality
        ranked = sorted(votes, key=votes.get, reverse=True)
        if len(ranked) == 1 or votes[ranked[0]] >= MIN_VOTE_RATIO * votes[ranked[1]]:
            values.append(ranked[0])
            conflicts += ranked[1:]
        else:
            conflicts += ranked          # too close to call: trust none of them
    return values, conflicts


def decode(img: np.ndarray) -> list[str]:
    """Return the decoded values (strings) ZBar finds in img, after resolve()."""
    return resolve(pyzbar.decode(img, symbols=SYMBOLS))[0]


def read_barcode(gray: np.ndarray) -> BarcodeRead:
    """The actual reader used by later stages.

    Tries each reader variant in order and stops at the first one where a value
    survives resolve(). If that found a product barcode, confirm() re-reads it.
    Conflicts are collected from every variant tried.
    Returns BarcodeRead([], None, conflicts, {}) if every variant fails.
    """
    makers = dict(VARIANTS)
    conflicts = []
    for i, name in enumerate(READER_VARIANTS):
        values, rejected = resolve(pyzbar.decode(makers[name](gray), symbols=SYMBOLS))
        conflicts += [c for c in rejected if c not in conflicts]
        if values:
            agreeing = confirm(gray, values, conflicts, tried=READER_VARIANTS[:i + 1])
            return BarcodeRead(values, name, conflicts, agreeing)
    return BarcodeRead([], None, conflicts, {})


def is_product_code(v: str) -> bool:
    """13 digits: an EAN-13 product number (validate.py also checks the check digit)."""
    return len(v) == 13 and v.isdigit()


def confirm(gray: np.ndarray, values: list[str], conflicts: list[str],
            tried: list[str]) -> dict[str, int]:
    """Count how many image variants read each value.

    Only product barcodes are re-read: they're the ones accepted on the barcode
    alone. Serial barcodes are checked against OCR instead, so re-reading them
    would only cost time. Any other product number a variant reads is added to
    conflicts (in place), because it means the barcode was read two ways.
    """
    agreeing = {v: 1 for v in values}
    if not any(is_product_code(v) for v in values):
        return agreeing
    makers = dict(VARIANTS)
    for name in CONFIRM_VARIANTS:
        if name in tried:
            continue
        read, rejected = resolve(pyzbar.decode(makers[name](gray), symbols=SYMBOLS))
        for v in read:
            if v in agreeing:
                agreeing[v] += 1               # this variant read the same value
            elif is_product_code(v):
                rejected.append(v)             # it read a different product number
        conflicts += [c for c in rejected if c not in agreeing and c not in conflicts]
    return agreeing


# ---------- experiment ----------

def main(folder: str | None = None) -> None:
    photos_dir, out_dir, truth = folder or PHOTOS_DIR, out_dir_for(folder, "barcode"), load_truth(folder)
    save_variants = not folder
    os.makedirs(out_dir, exist_ok=True)
    matrix_rows, result_rows, reads = [], [], []

    for filename, true_serial in zip(truth.filename, truth.true_serial):
        stem = os.path.splitext(filename)[0]
        gray = cv2.cvtColor(cv2.imread(os.path.join(photos_dir, filename)), cv2.COLOR_BGR2GRAY)
        var_dir = os.path.join(out_dir, "variants", stem)
        if save_variants:
            os.makedirs(var_dir, exist_ok=True)

        # 1) Diagnostic: every variant, so we learn which ones matter.
        row = {"filename": filename}
        for name, make in VARIANTS:
            img = make(gray)
            if save_variants:
                cv2.imwrite(os.path.join(var_dir, f"{name}.jpg"), preview(img))
            row[name] = any(found(true_serial, v) for v in decode(img))
        matrix_rows.append(row)

        # 2) The real reader: stop at the first variant that decodes.
        t0 = time.perf_counter()
        read = read_barcode(gray)
        elapsed = time.perf_counter() - t0
        reads.append(read)
        values, variant, conflicts, agreeing = read
        result_rows.append({
            "filename": filename,
            "true_serial": true_serial,
            "values": " | ".join(values),
            "variant_used": variant,
            "conflicts": " | ".join(conflicts),
            "agreeing": " | ".join(f"{v}:{n}" for v, n in agreeing.items()),
            "barcode_ok": any(found(true_serial, v) for v in values),
            "seconds": round(elapsed, 3),
        })

    matrix = pd.DataFrame(matrix_rows)
    results = pd.DataFrame(result_rows)
    matrix.to_csv(os.path.join(out_dir, "variant_matrix.csv"), index=False)
    results.to_csv(os.path.join(out_dir, "results.csv"), index=False)

    # Print the matrix compactly: '#' = serial found, '.' = not found.
    names = [n for n, _ in VARIANTS]
    print("Which variant found the serial ('#' = yes):\n")
    print(f"{'':24}" + " ".join(f"{i:>2}" for i in range(len(names))))
    for r in matrix_rows:
        print(f"{r['filename']:24}" + " ".join(f"{'#' if r[n] else '.':>2}" for n in names))
    print("\nColumns: " + ", ".join(f"{i}={n}" for i, n in enumerate(names)))

    n = len(results)
    print("\nReal reader (stop at first success):")
    print(results[["filename", "variant_used", "barcode_ok", "conflicts", "seconds"]].to_string(index=False))
    # Compare against whatever the last baseline.py run produced (if it exists).
    baseline_path = os.path.join("output", "baseline", "results.csv")
    if folder:
        print(f"\nBarcode correct: {results.barcode_ok.sum()}/{n}")
        # What Stage 6 would do with each read on the barcode alone (these sets are
        # product barcodes, and the product rule doesn't use OCR). Imported here
        # because validate.py imports this file.
        from validate import barcode_candidates, decide
        decisions = [decide(barcode_candidates(r.values), [], [], r.conflicts, r.agreeing)
                     for r in reads]
        results["decision"] = [d[0] for d in decisions]
        results["serial"] = [d[1] for d in decisions]
        results["reason"] = [d[3] for d in decisions]
        accepted = results[results.decision == "accepted"]
        wrong = accepted[accepted.serial != accepted.true_serial]
        print(f"Accepted on the barcode alone: {len(accepted)}/{n}   wrong: {len(wrong)}   "
              f"needs review: {(results.decision != 'accepted').sum()} "
              f"(nothing read: {(results['values'] == '').sum()})")
        if len(wrong):
            print(wrong[["filename", "true_serial", "values", "agreeing"]].to_string(index=False))
        results.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    elif os.path.exists(baseline_path):
        base = pd.read_csv(baseline_path)
        print(f"\nBarcode correct: {results.barcode_ok.sum()}/{n}   "
              f"(Stage 2 baseline: {base.barcode_ok.sum()}/{len(base)})")
    else:
        print(f"\nBarcode correct: {results.barcode_ok.sum()}/{n}   (run baseline.py to compare)")
    print(f"Mean time/photo: {results.seconds.mean():.2f}s   max: {results.seconds.max():.2f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

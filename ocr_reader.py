"""Stage 5: OCR on the label crops from Stage 4, with preprocessing.

Run with:  python ocr_reader.py   (from the repo folder, with the venv active)

The key finding of this stage: OCR'ing a whole label crop fails when the crop
contains a barcode or QR code. On Mouse_2 the full crop gave 'aa', but the
S/N line cut out on its own gave the serial exactly right.
Tesseract's page-layout step mistakes the bars for huge characters and loses
the text next to them. So the main method here splits each crop into single
text lines (classical OpenCV again), skips anything too tall to be a line of
text (barcodes, QR codes, logos), and OCRs each line on its own.

Per line we also:
  - fix polarity: a crop can contain BOTH dark-on-light and light-on-dark
    text (Razer: silver strip + dark body), so polarity is decided per line
  - resize so the text is ~30 px tall, the size Tesseract works best at
  - try it upside down too, and keep whichever Tesseract is more confident
    about (Stage 4 can't tell up from down). Uses no knowledge of the truth.

Like Stage 3, this compares methods on every photo:
  output/ocr/variant_matrix.csv       photo x method: serial found? + chars wrong
  output/ocr/text/<photo>__<method>.txt   what Tesseract read
  output/ocr/lines/<photo>_<crop>_<line>.png   each text line as fed to Tesseract
"""

import atexit
import glob
import os
import sys
import re
import time

import cv2
import numpy as np
import pandas as pd
import pytesseract
from pytesseract import Output

import baseline  # noqa: F401  (sets pytesseract's path to the tesseract program)
from baseline import found, load_truth, normalize, out_dir_for, valid_ean13
from label_finder import crop_rotated, find_label_boxes
from pyzbar import pyzbar
from pyzbar.pyzbar import ZBarSymbol
from tess_direct import load_engine

PHOTOS_DIR = "photos"
OUT_DIR = os.path.join("output", "ocr")

TARGET_TEXT_HEIGHT = 30   # px; Tesseract is tuned for text roughly this tall
# A serial has >= 10 characters and a character is ~0.6x as wide as it is tall,
# so a line narrower than ~4 text-heights can't hold one. Skipping those lines
# drops short label text ('12M') AND the short fragments that textures produce:
# a mousepad crop on Mouse_1 had 34 "lines" and cost ~10 s of OCR.
MIN_LINE_WIDTH = 4        # in text heights
ORIENTATION_SAMPLE = 3    # decide upside-down from this many of the widest lines
# Pieces of text on the same row closer than this (in character heights) are one
# line. Sized for the gaps between the digit groups under a product barcode.
ROW_MERGE_GAP = 1.5

# An axis-aligned box inside a crop: (x, y, width, height), as cv2.boundingRect gives.
Box = tuple[int, int, int, int]
LINE_PAD = 0.35           # margin added around each line, as a fraction of text height


# ---------- OCR helpers ----------

# Tesseract's engine loaded once into this process (see tess_direct.py): ~0.006 s
# per small image instead of ~0.15 s for pytesseract, which starts a tesseract process
# every call. Checked to give identical text on all 228 images this pipeline
# OCRs. None if the library isn't found -> fall back to pytesseract.
ENGINE = load_engine()
if ENGINE is not None:
    atexit.register(ENGINE.close)   # free Tesseract's memory when Python exits


def ocr_with_confidence(img: np.ndarray, config: str, whitelist: str = "") -> tuple[str, float]:
    """Run Tesseract and return (text, mean word confidence 0-100).

    config is pytesseract-style, e.g. "--psm 7". whitelist, if given, limits the
    characters Tesseract may output (e.g. "0123456789").
    """
    if ENGINE is not None:
        psm = int(re.search(r"--psm (\d+)", config).group(1))
        pairs = ENGINE.words(img, psm, whitelist)
    else:
        if whitelist:
            config += f" -c tessedit_char_whitelist={whitelist}"
        d = pytesseract.image_to_data(img, config=config, output_type=Output.DICT)
        pairs = [(word, float(conf)) for word, conf in zip(d["text"], d["conf"])]
    words, confs = [], []
    for word, conf in pairs:
        # conf is -1 for layout entries (blocks/lines) that aren't words.
        if word.strip() and conf >= 0:
            words.append(word)
            confs.append(conf)
    return " ".join(words), (float(np.mean(confs)) if confs else 0.0)


def ocr_best_orientation(img: np.ndarray, config: str) -> tuple[str, float]:
    """OCR img as-is and upside down; return (text, conf) of the more confident one."""
    up = ocr_with_confidence(img, config)
    down = ocr_with_confidence(cv2.rotate(img, cv2.ROTATE_180), config)
    return up if up[1] >= down[1] else down


def invert_if_dark(gray: np.ndarray) -> np.ndarray:
    """Make the background light. The background is most of the image, so if
    the image is mostly dark, the text must be the light part -> invert."""
    return cv2.bitwise_not(gray) if np.median(gray) < 127 else gray


# ---------- method 1: whole crop (the reference) ----------

def read_whole_crop(crop: np.ndarray) -> str:
    """Stage-2 style: hand Tesseract the whole crop as one block of text."""
    return ocr_best_orientation(invert_if_dark(crop), "--psm 6")[0]


# ---------- method 2: line by line ----------

def find_text_lines(gray: np.ndarray) -> list[Box]:
    """Return (x, y, w, h) boxes of the text lines in a crop, top to bottom.

    Same idea as Stage 4 but at the scale of characters instead of labels:
      edges -> estimate character height -> smear edges sideways so the letters
      of one line join up (but lines above/below don't) -> one box per line.
    """
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    # Inside a label crop Otsu works: there really are two groups (flat vs edge).
    _, edges = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Character height = the typical height of small connected edge blobs.
    n, _, stats, _ = cv2.connectedComponentsWithStats(edges)
    heights = stats[1:, cv2.CC_STAT_HEIGHT]                    # [0] is the background
    heights = heights[(heights >= 8) & (heights < gray.shape[0] * 0.4)]
    if len(heights) == 0:
        return []
    char_h = int(np.median(heights))

    # Remove long straight lines (label outlines, panel dividers). Without this,
    # every text line touches the label's outline and they all join into one
    # giant blob (that's what happened on Mouse_1). An "opening" with a long thin
    # kernel keeps ONLY straight runs at least that long - letters are too short
    # to survive it - so subtracting the result removes just the long lines.
    long_h = cv2.morphologyEx(edges, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (char_h * 6, 1)))
    long_v = cv2.morphologyEx(edges, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (1, char_h * 3)))
    edges = cv2.subtract(edges, cv2.bitwise_or(long_h, long_v))

    # A kernel 1 px tall and ~1 character wide joins letters and words on the same
    # line but can't bridge the gap to the next line.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, char_h), 1))
    smeared = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    # RETR_CCOMP, not RETR_EXTERNAL: if the label's outline survives as a closed
    # ring, EXTERNAL returns only that ring and hides every text line inside it
    # (this happened on the ASUS mouse labels). CCOMP returns outer boundaries at
    # every depth; hierarchy[...][3] == -1 marks an outer boundary (not a hole).
    contours, hierarchy = cv2.findContours(smeared, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []

    pieces = []
    for c, (_, _, _, parent) in zip(contours, hierarchy[0]):
        if parent != -1:
            continue    # this contour is the inside edge of a hole, not a shape
        x, y, w, h = cv2.boundingRect(c)
        if h < 0.5 * char_h or h > 2.5 * char_h:
            continue    # too small = noise; too tall = barcode, QR code, logo
        pieces.append((x, y, w, h))

    # Merge pieces of the same line BEFORE the width filter. The digits under a
    # product barcode are printed in groups ("4 006381 333931") with the
    # barcode's long guard bars between them, so each group came out as its own
    # "line" - and the lone leading '4' was then thrown away as too short.
    lines = merge_same_row(pieces, max_gap=ROW_MERGE_GAP * char_h)
    lines = [b for b in lines if b[2] >= MIN_LINE_WIDTH * b[3]]   # too short for a serial
    return sorted(lines, key=lambda b: (b[1], b[0]))


def merge_same_row(boxes: list[Box], max_gap: float) -> list[Box]:
    """Join boxes that sit on the same row with at most max_gap pixels between them.

    "Same row" = they overlap vertically by at least half the smaller box's height.
    Boxes are visited left to right, so each one only needs checking against
    the right-hand end of the rows built so far.
    """
    rows: list[Box] = []
    for x, y, w, h in sorted(boxes):                       # sorted by x
        for i, (rx, ry, rw, rh) in enumerate(rows):
            overlap = min(y + h, ry + rh) - max(y, ry)
            gap = x - (rx + rw)
            if overlap >= 0.5 * min(h, rh) and gap <= max_gap:
                x0, y0 = min(x, rx), min(y, ry)
                rows[i] = (x0, y0, max(x + w, rx + rw) - x0, max(y + h, ry + rh) - y0)
                break
        else:
            rows.append((x, y, w, h))
    return rows


def line_image(gray: np.ndarray, box: Box) -> np.ndarray:
    """Cut out one line, light background, text ~TARGET_TEXT_HEIGHT tall, with a margin.

    The margin is ADDED as plain background colour rather than cut from the
    crop. Cutting it from the crop grabbed slivers of the lines above and below
    (labels have tight line spacing), and with --psm 7 ("exactly one line")
    Tesseract then returned nothing at all for Laptop_2's SN line.
    """
    x, y, w, h = box
    line = invert_if_dark(gray[y:y + h, x:x + w])
    scale = TARGET_TEXT_HEIGHT / h
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    line = cv2.resize(line, None, fx=scale, fy=scale, interpolation=interp)
    pad = int(TARGET_TEXT_HEIGHT * LINE_PAD) + 1
    background = int(np.median(line))
    return cv2.copyMakeBorder(line, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=background)


def read_lines(crop: np.ndarray, save_prefix: str | None = None) -> str:
    """OCR every text line separately (--psm 7 = 'this image is one line of text').

    Orientation is decided ONCE per crop, not per line: a crop is either upside
    down or it isn't, and a single short line (e.g. '12M') is too little evidence -
    on Laptop_2 Tesseract was more confident about some lines read upside down.
    To save time it's decided from a sample: the ORIENTATION_SAMPLE widest lines
    are read both ways, the confidences summed, and the remaining lines are then
    read in the winning orientation only. (Every pytesseract call starts a new
    tesseract process, ~0.15 s each, so the number of calls is what costs time.)
    """
    imgs = [line_image(crop, box) for box in find_text_lines(crop)]
    if save_prefix:
        for i, img in enumerate(imgs):
            cv2.imwrite(f"{save_prefix}_{i}.png", img)
    if not imgs:
        return ""

    sample = sorted(range(len(imgs)), key=lambda i: -imgs[i].shape[1])[:ORIENTATION_SAMPLE]
    up = {i: ocr_with_confidence(imgs[i], "--psm 7") for i in sample}
    down = {i: ocr_with_confidence(cv2.rotate(imgs[i], cv2.ROTATE_180), "--psm 7") for i in sample}
    flipped = sum(c for _, c in down.values()) > sum(c for _, c in up.values())

    texts = []
    for i, img in enumerate(imgs):
        done = down if flipped else up
        if i in done:
            texts.append(done[i][0])      # already read during the sample
        else:
            texts.append(ocr_with_confidence(
                cv2.rotate(img, cv2.ROTATE_180) if flipped else img, "--psm 7")[0])
    if flipped:
        texts.reverse()                   # upside down: last line is really the first
    return "\n".join(t for t in texts if t)


# ---------- measuring ----------

def get_crops(img: np.ndarray) -> list[np.ndarray]:
    """Stage 4: grayscale label crops, at full resolution."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return [cv2.cvtColor(crop_rotated(img, r), cv2.COLOR_BGR2GRAY)
            for r in find_label_boxes(gray)]


def levenshtein(a: str, b: str) -> int:
    """Edit distance: fewest single-character inserts/deletes/swaps turning a into b."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def closest_match(true_serial: str, text: str) -> tuple[int, str]:
    """The serial-length stretch of the OCR text closest to the true serial:
    (characters wrong, that stretch). "" if the text has nothing at all.
    Checked line by line, so a serial is never stitched together across lines.
    """
    t = normalize(true_serial)
    best, best_s = len(t), ""
    for line in text.splitlines():
        s = normalize(line)
        for L in (len(t) - 1, len(t), len(t) + 1):      # allow one missing/extra char
            for i in range(0, max(1, len(s) - L + 1)):
                d = levenshtein(t, s[i:i + L])
                if d < best:
                    best, best_s = d, s[i:i + L]
    return best, best_s


def chars_wrong(true_serial: str, text: str) -> int:
    """How close did OCR get? The fewest character edits needed to turn some
    serial-length stretch of the OCR text into the true serial.
    0 = exact, 1 = one character off (e.g. O read instead of 0), etc.
    """
    return closest_match(true_serial, text)[0]


METHODS = [
    ("whole_crop", lambda crop, prefix: read_whole_crop(crop)),
    ("lines",      lambda crop, prefix: read_lines(crop, prefix)),
    ("barcode_digits", lambda crop, prefix: "\n".join(read_barcode_digits(crop))),
]


# ---------- method 3: the digits printed next to a product barcode ----------
#
# On product barcodes (EAN-13) the digits are printed touching the bars, with the
# long "guard bars" running down between the digit groups. Methods 1 and 2 both
# fail there: the line finder merges digits + bars into one tall blob and skips
# it, and whole-crop OCR is confused by the bars. So this method uses the barcode
# itself as a map: ZBar says where the bars are, the digits are right next to them.

DIGIT_BAND = 0.16      # height of the strip searched for digits, as a fraction of the bar width
BAR_ROW_MATCH = 0.85   # a row still belongs to the bars if >= 85% of it matches a nearby bar row


def bar_rows(ink: np.ndarray, y_mid: int) -> tuple[int, int]:
    """First and last row of the bars, starting from a row known to cross them.

    ink is True where the bar columns are dark. ZBar's box doesn't cover the
    bars' full height, so walk outwards one row at a time while each row still
    has the same black/white pattern as the row a few pixels further in. The
    first row that doesn't is where the digits (or blank label) begin. Comparing
    with a NEARBY row, not one fixed row, tolerates bars that lean slightly
    (curved cans, photos taken at an angle).
    """
    lag = max(2, ink.shape[1] // 100)

    def same(row: int, ref: int) -> bool:
        return (ink[row] == ink[ref]).mean() >= BAR_ROW_MATCH

    top = bottom = y_mid
    while top > 0 and same(top - 1, min(top - 1 + lag, y_mid)):
        top -= 1
    while bottom < ink.shape[0] - 1 and same(bottom + 1, max(bottom + 1 - lag, y_mid)):
        bottom += 1
    return top, bottom


def erase_guard_bars(band: np.ndarray, bar_side_row: int, max_width: int) -> np.ndarray:
    """Remove thin dark shapes that touch the bars' side of the strip.

    Those are the guard bars continuing into the digit row - the thing that glued
    digits and bars together. Digits don't reach that edge, so they survive.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(band.astype(np.uint8))
    for i in range(1, n):                     # 0 is the background
        x, y, w, h, _ = stats[i]
        if y <= bar_side_row <= y + h - 1 and w <= max_width:
            band[labels == i] = False
    return band


def read_digit_band(band: np.ndarray) -> list[str]:
    """OCR one strip (True = ink) as digits only; return any valid EAN-13 inside it."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(band.astype(np.uint8))
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    heights = heights[heights >= 0.3 * band.shape[0]]   # digit-sized shapes
    if len(heights) < 6:
        return []                                       # no row of digits in this strip
    img = np.where(band, 0, 255).astype(np.uint8)       # black digits on white
    scale = TARGET_TEXT_HEIGHT / float(np.median(heights))
    img = cv2.resize(img, None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    pad = TARGET_TEXT_HEIGHT // 2
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    out = []
    # Both ways up: Stage 4 can't tell whether the crop is upside down.
    for im in (img, cv2.rotate(img, cv2.ROTATE_180)):
        digits = re.sub(r"\D", "", ocr_with_confidence(im, "--psm 7", "0123456789")[0])
        for i in range(len(digits) - 12):
            if valid_ean13(digits[i:i + 13]):
                out.append(digits[i:i + 13])
    return out


def read_barcode_digits(crop: np.ndarray) -> list[str]:
    """Valid EAN-13 numbers read from the digits printed beside each EAN barcode in a crop.

    This is OCR of the PRINTED digits; the barcode only says where to look. The
    printed digits and the bars are separate print, so this is still an
    independent reading for Stage 6 to compare with the decoded barcode.
    """
    out = []
    for b in pyzbar.decode(crop, symbols=[ZBarSymbol.EAN13]):
        x, y, w, h = b.rect
        cols = crop[:, x:x + w]
        thresh, _ = cv2.threshold(cols, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        top, bottom = bar_rows(cols < thresh, min(y + h // 2, crop.shape[0] - 1))

        band_h = int(DIGIT_BAND * w)
        # Wider than the bars: the first digit sits outside them - on the left if
        # the crop is upright, on the RIGHT if it's upside down. So widen both sides.
        margin = int(0.13 * w)
        x0, x1 = max(0, x - margin), min(crop.shape[1], x + w + margin)
        ink = crop[:, x0:x1] < thresh
        # Strip above the bars (bars at its bottom edge) and below (bars at its top).
        for lo, hi, bar_side in ((max(0, top - band_h), top, "bottom"),
                                 (bottom + 1, min(crop.shape[0], bottom + 1 + band_h), "top")):
            if hi - lo < 5:
                continue
            band = ink[lo:hi].copy()
            band = erase_guard_bars(band, band.shape[0] - 1 if bar_side == "bottom" else 0,
                                    max_width=max(3, w // 50))
            out += read_digit_band(band)
    return list(dict.fromkeys(out))   # unique, in the order found


def read_ocr(img: np.ndarray) -> str:
    """The reader later stages use: Stage 4 crops, then BOTH methods on every crop.

    The two methods fail on different photos (whole-crop reads the ASUS labels,
    line-by-line reads the Razer labels and Laptop_2), so we keep all the text
    and let Stage 6 (validation) decide which candidate is the serial.
    Returns one string, one line of OCR output per line.
    """
    texts = []
    for crop in get_crops(img):
        texts.append(read_whole_crop(crop))
        texts.append(read_lines(crop))
        texts.extend(read_barcode_digits(crop))   # product barcodes: the printed digits
    return "\n".join(texts)


def main(folder: str | None = None) -> None:
    photos_dir, out_dir, truth = folder or PHOTOS_DIR, out_dir_for(folder, "ocr"), load_truth(folder)
    for sub in ("text", "lines"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    for old in glob.glob(os.path.join(out_dir, "lines", "*.png")):
        os.remove(old)

    rows = []
    for filename, true_serial in zip(truth.filename, truth.true_serial):
        stem = os.path.splitext(filename)[0]
        crops = get_crops(cv2.imread(os.path.join(photos_dir, filename)))
        row = {"filename": filename, "n_crops": len(crops)}
        all_text = []
        for name, method in METHODS:
            t0 = time.perf_counter()
            texts = []
            for i, crop in enumerate(crops):
                prefix = os.path.join(out_dir, "lines", f"{stem}_{i}")
                texts.append(f"[crop {i}]\n{method(crop, prefix)}")
            joined = "\n".join(texts)
            all_text.append(joined)
            row[f"{name}_ok"] = found(true_serial, joined)
            row[f"{name}_wrong"] = chars_wrong(true_serial, joined)
            row[f"{name}_s"] = round(time.perf_counter() - t0, 2)
            with open(os.path.join(out_dir, "text", f"{stem}__{name}.txt"), "w", encoding="utf-8") as f:
                f.write(joined)
        # What read_ocr() returns: both methods' text together.
        combined = "\n".join(all_text)
        row["combined_ok"] = found(true_serial, combined)
        row["combined_wrong"] = chars_wrong(true_serial, combined)
        row["combined_s"] = round(sum(row[f"{n}_s"] for n, _ in METHODS), 2)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "variant_matrix.csv"), index=False)
    print(df.to_string(index=False))
    n = len(df)
    baseline_csv = os.path.join(out_dir_for(folder, "baseline"), "results.csv")
    if os.path.exists(baseline_csv):
        base = pd.read_csv(baseline_csv)
        print(f"\nSerial found exactly (stage 2, full photo): {base.ocr_ok.sum()}/{len(base)}")
    for name in [m for m, _ in METHODS] + ["combined"]:
        print(f"Serial found exactly ({name:10}):           {df[f'{name}_ok'].sum()}/{n}"
              f"   within 1 char: {(df[f'{name}_wrong'] <= 1).sum()}/{n}"
              f"   mean {df[f'{name}_s'].mean():.1f}s/photo")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

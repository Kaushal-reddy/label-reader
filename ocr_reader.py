"""Stage 5: OCR on the label crops from Stage 4, with preprocessing.

Run with:  .venv\\Scripts\\python.exe ocr_reader.py

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

import glob
import os
import re
import time

import cv2
import numpy as np
import pandas as pd
import pytesseract
from pytesseract import Output

import baseline  # noqa: F401  (sets pytesseract's path to tesseract.exe)
from baseline import found, normalize
from label_finder import crop_rotated, find_label_boxes

PHOTOS_DIR = "photos"
OUT_DIR = os.path.join("output", "ocr")

TARGET_TEXT_HEIGHT = 30   # px; Tesseract is tuned for text roughly this tall
# A serial has >= 10 characters and a character is ~0.6x as wide as it is tall,
# so a line narrower than ~4 text-heights can't hold one. Skipping those lines
# drops short label text ('12M') AND the short fragments that textures produce:
# a mousepad crop on Mouse_1 had 34 "lines" and cost ~10 s of OCR.
MIN_LINE_WIDTH = 4        # in text heights
ORIENTATION_SAMPLE = 3    # decide upside-down from this many of the widest lines

# An axis-aligned box inside a crop: (x, y, width, height), as cv2.boundingRect gives.
Box = tuple[int, int, int, int]
LINE_PAD = 0.35           # margin added around each line, as a fraction of text height


# ---------- OCR helpers ----------

def ocr_with_confidence(img: np.ndarray, config: str) -> tuple[str, float]:
    """Run Tesseract and return (text, mean word confidence 0-100)."""
    d = pytesseract.image_to_data(img, config=config, output_type=Output.DICT)
    words, confs = [], []
    for word, conf in zip(d["text"], d["conf"]):
        # conf is -1 for layout entries (blocks/lines) that aren't words.
        if word.strip() and float(conf) >= 0:
            words.append(word)
            confs.append(float(conf))
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

    lines = []
    for c, (_, _, _, parent) in zip(contours, hierarchy[0]):
        if parent != -1:
            continue    # this contour is the inside edge of a hole, not a shape
        x, y, w, h = cv2.boundingRect(c)
        if h < 0.5 * char_h or h > 2.5 * char_h:
            continue    # too small = noise; too tall = barcode, QR code, logo
        if w < MIN_LINE_WIDTH * h:
            continue    # too short to hold a serial (see MIN_LINE_WIDTH)
        lines.append((x, y, w, h))
    return sorted(lines, key=lambda b: (b[1], b[0]))


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
    tesseract.exe, ~0.15 s each, so the number of calls is what costs time.)
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


def chars_wrong(true_serial: str, text: str) -> int:
    """How close did OCR get? The fewest character edits needed to turn some
    serial-length stretch of the OCR text into the true serial.
    0 = exact, 1 = one character off (e.g. O read instead of 0), etc.
    Checked line by line, so a serial is never stitched together across lines.
    """
    t = normalize(true_serial)
    best = len(t)
    for line in text.splitlines():
        s = normalize(line)
        for L in (len(t) - 1, len(t), len(t) + 1):      # allow one missing/extra char
            for i in range(0, max(1, len(s) - L + 1)):
                best = min(best, levenshtein(t, s[i:i + L]))
    return best


METHODS = [
    ("whole_crop", lambda crop, prefix: read_whole_crop(crop)),
    ("lines",      lambda crop, prefix: read_lines(crop, prefix)),
]


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
    return "\n".join(texts)


def main() -> None:
    for sub in ("text", "lines"):
        os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)
    for old in glob.glob(os.path.join(OUT_DIR, "lines", "*.png")):
        os.remove(old)

    truth = pd.read_csv("truth.csv")
    rows = []
    for filename, true_serial in zip(truth.filename, truth.true_serial):
        stem = os.path.splitext(filename)[0]
        crops = get_crops(cv2.imread(os.path.join(PHOTOS_DIR, filename)))
        row = {"filename": filename, "n_crops": len(crops)}
        all_text = []
        for name, method in METHODS:
            t0 = time.perf_counter()
            texts = []
            for i, crop in enumerate(crops):
                prefix = os.path.join(OUT_DIR, "lines", f"{stem}_{i}")
                texts.append(f"[crop {i}]\n{method(crop, prefix)}")
            joined = "\n".join(texts)
            all_text.append(joined)
            row[f"{name}_ok"] = found(true_serial, joined)
            row[f"{name}_wrong"] = chars_wrong(true_serial, joined)
            row[f"{name}_s"] = round(time.perf_counter() - t0, 2)
            with open(os.path.join(OUT_DIR, "text", f"{stem}__{name}.txt"), "w", encoding="utf-8") as f:
                f.write(joined)
        # What read_ocr() returns: both methods' text together.
        combined = "\n".join(all_text)
        row["combined_ok"] = found(true_serial, combined)
        row["combined_wrong"] = chars_wrong(true_serial, combined)
        row["combined_s"] = round(sum(row[f"{n}_s"] for n, _ in METHODS), 2)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "variant_matrix.csv"), index=False)
    print(df.to_string(index=False))
    n = len(df)
    base = pd.read_csv(os.path.join("output", "baseline", "results.csv"))
    print(f"\nSerial found exactly (stage 2, full photo): {base.ocr_ok.sum()}/{n}")
    for name in [m for m, _ in METHODS] + ["combined"]:
        print(f"Serial found exactly ({name:10}):           {df[f'{name}_ok'].sum()}/{n}"
              f"   within 1 char: {(df[f'{name}_wrong'] <= 1).sum()}/{n}"
              f"   mean {df[f'{name}_s'].mean():.1f}s/photo")


if __name__ == "__main__":
    main()

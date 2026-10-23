"""Turn photos/Barcode_Dataset (a downloaded object-detection set) into a test
set for this project, and run the Stage 3 variant matrix on it.

Run with:  python -m scripts.prepare_ean_dataset   (from the repo folder, with the venv active)

The downloaded set only has BOXES (Pascal VOC .xml: "there is a barcode here"),
not the numbers the barcodes hold, so it can't be scored like truth.csv yet.
This script:

  1. picks the images with exactly one annotated barcode and nothing else
     (truth.csv holds one value per photo) and COPIES them, with their .xml,
     to photos/ean_dataset/. Barcode_Dataset itself is not changed.
  2. decodes every image with every Stage 3 variant (barcode_reader.VARIANTS)
     and saves the matrix to output/ean_dataset/variant_matrix.csv.
  3. writes photos/ean_dataset/truth.csv, PRE-FILLED with what ZBar read.
     These are suggestions, not truth: ZBar reads are what we want to test,
     so every value must be checked against the digits printed under the
     bars. Column `checked` starts as "no"; set it to "yes" once a row is
     checked (or corrected). Rows ZBar couldn't read are blank: type those.
  4. draws review sheets, photos/ean_dataset/review/page_NNN.jpg: 12 zoomed
     barcode crops per page, each with its truth.csv row number and the
     suggested value, so checking is mostly comparing two numbers by eye.
     Green = one clean read, orange = ZBar read conflicting numbers (check
     carefully), red = no read (type the digits).

Without checked truth the matrix can only say "decoded a valid EAN-13", not
"decoded the RIGHT one". Re-run barcode_reader.py on the folder once checked:
  python -m label_reader.barcode_reader photos/ean_dataset
"""

import os
import re
import shutil
import time
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from label_reader import native_libs  # noqa: F401, I001  (must come before pyzbar: lets it find ZBar on macOS)
import cv2
import numpy as np
import pandas as pd
from pyzbar import pyzbar

from label_reader.barcode_reader import READER_VARIANTS, SYMBOLS, VARIANTS, resolve

SOURCE_DIR = os.path.join("photos", "Barcode_Dataset")
DEST_DIR = os.path.join("photos", "ean_dataset")
OUT_DIR = os.path.join("output", "ean_dataset")
TILES_PER_PAGE, TILE_COLS = 12, 3
TILE_W, TILE_H, HEADER_H = 600, 340, 46
WORKERS = max(1, (os.cpu_count() or 2) - 1)


def check_digit_ok(code: str) -> bool:
    """EAN-13 check digit (same rule as validate.check_digit_ok; copied so this
    script doesn't load the OCR engine in every worker process)."""
    digits = [int(c) for c in code]
    total = sum(d * (3 if i % 2 == 0 else 1) for i, d in enumerate(reversed(digits[:-1])))
    return (10 - total % 10) % 10 == digits[-1]


def is_ean13(v: str) -> bool:
    return bool(re.fullmatch(r"\d{13}", v)) and check_digit_ok(v)


def source_group(stem: str) -> str | None:
    """Which original collection an image came from (by its file name)."""
    if stem.startswith("Foto"):
        return "Foto"
    if stem.startswith(("0510", "0610")):
        return "0510/0610"
    if re.match(r"P\w*I\w*C\w*T", stem):      # PICT0021, PICfggfhT0018, ...
        return "PICT"
    return None                              # the QR sets: not product barcodes


def read_box(xml_path: str) -> tuple[list[str], tuple[int, int, int, int] | None]:
    """(class names of all objects, box of the first one) from a VOC .xml."""
    objs = ET.parse(xml_path).getroot().findall("object")
    names = [o.findtext("name") for o in objs]
    if not objs:
        return names, None
    b = objs[0].find("bndbox")
    return names, tuple(int(float(b.findtext(k))) for k in ("xmin", "ymin", "xmax", "ymax"))


def decode_variants(path: str) -> dict:
    """Run every Stage 3 variant on one image (runs in a worker process)."""
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    out = {}
    for name, make in VARIANTS:
        t0 = time.perf_counter()
        kept, rejected = resolve(pyzbar.decode(make(gray), symbols=SYMBOLS))
        out[name] = {"kept": kept, "rejected": rejected, "s": time.perf_counter() - t0}
    return out


def suggest(reads: dict) -> tuple[str, str, str]:
    """(suggested value, note, colour key) from every variant's reads."""
    votes = Counter(v for r in reads.values() for v in r["kept"] if is_ean13(v))
    rejected = sorted({v for r in reads.values() for v in r["rejected"]})
    if not votes:
        return "", "no read: type the digits", "none"
    best, n = votes.most_common(1)[0]
    others = sorted((set(votes) | set(rejected)) - {best})
    if others:
        return best, f"CHECK: ZBar also read {' '.join(others)}", "conflict"
    return best, f"read by {n} of {len(reads)} variants", "clean"


def tile(img: np.ndarray, box, row: int, filename: str, value: str, kind: str) -> np.ndarray:
    """Zoomed crop of the barcode with its row number and suggested value."""
    x0, y0, x1, y1 = box
    # The box usually covers the bars; the digits are printed just below.
    pw, ph = int(0.10 * (x1 - x0)), int(0.30 * (y1 - y0))
    crop = img[max(0, y0 - ph // 3):min(img.shape[0], y1 + ph), max(0, x0 - pw):min(img.shape[1], x1 + pw)]
    s = min((TILE_W - 10) / crop.shape[1], (TILE_H - HEADER_H - 10) / crop.shape[0])
    crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    t = np.full((TILE_H, TILE_W, 3), 255, np.uint8)
    y, x = HEADER_H + (TILE_H - HEADER_H - crop.shape[0]) // 2, (TILE_W - crop.shape[1]) // 2
    t[y:y + crop.shape[0], x:x + crop.shape[1]] = crop
    colour = {"clean": (0, 140, 0), "conflict": (0, 120, 230), "none": (0, 0, 210)}[kind]
    cv2.rectangle(t, (0, 0), (TILE_W - 1, HEADER_H), colour, -1)
    label = value if value else "NO READ - type it"
    cv2.putText(t, f"{row}", (8, 33), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(t, label, (90, 33), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (255, 255, 255), 2)
    cv2.putText(t, filename[:40], (8, TILE_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)
    cv2.rectangle(t, (0, 0), (TILE_W - 1, TILE_H - 1), (200, 200, 200), 1)
    return t


def main() -> None:
    # 1) pick and copy
    picked, skipped = [], Counter()
    for f in sorted(os.listdir(SOURCE_DIR)):
        stem, ext = os.path.splitext(f)
        if ext == ".xml" or source_group(stem) is None:
            continue
        names, box = read_box(os.path.join(SOURCE_DIR, stem + ".xml"))
        if names != ["barcode"]:
            skipped[f"{source_group(stem)}: {'+'.join(sorted(names))}"] += 1
            continue
        picked.append({"filename": f, "group": source_group(stem), "box": box})
    os.makedirs(os.path.join(DEST_DIR, "review"), exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    for p in picked:
        stem = os.path.splitext(p["filename"])[0]
        for name in (p["filename"], stem + ".xml"):
            if not os.path.exists(os.path.join(DEST_DIR, name)):
                shutil.copy2(os.path.join(SOURCE_DIR, name), os.path.join(DEST_DIR, name))
    print(f"Copied {len(picked)} images to {DEST_DIR}/  (skipped: {dict(skipped)})")

    # 2) variant matrix, in parallel (every variant of every image)
    print(f"Decoding {len(picked)} images x {len(VARIANTS)} variants with {WORKERS} processes...")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(WORKERS) as pool:
        results = list(pool.map(decode_variants, [os.path.join(DEST_DIR, p["filename"]) for p in picked],
                                chunksize=4))
    print(f"  done in {time.perf_counter() - t0:.0f} s")

    names = [n for n, _ in VARIANTS]
    matrix, truth = [], []
    for p, reads in zip(picked, results):
        value, note, kind = suggest(reads)
        p.update(value=value, note=note, kind=kind)
        row = {"filename": p["filename"], "group": p["group"], "suggested": value, "status": kind}
        for n in names:
            row[n] = " ".join(v for v in reads[n]["kept"] if is_ean13(v))
            row[f"{n}_conflict"] = " ".join(reads[n]["rejected"])
            row[f"{n}_s"] = round(reads[n]["s"], 3)
        matrix.append(row)
        truth.append({"filename": p["filename"], "true_serial": value, "checked": "no", "note": note})
    m = pd.DataFrame(matrix)
    m.to_csv(os.path.join(OUT_DIR, "variant_matrix.csv"), index=False)

    # 3) pre-filled truth.csv - never overwrite one that's being checked
    truth_path = os.path.join(DEST_DIR, "truth.csv")
    if os.path.exists(truth_path):
        truth_path = os.path.join(DEST_DIR, "truth_suggested.csv")
        print(f"  {DEST_DIR}/truth.csv already exists; suggestions written to {truth_path} instead")
    pd.DataFrame(truth).to_csv(truth_path, index=False)

    # 4) review sheets, in truth.csv order (row number = line in the CSV minus 1)
    for page in range(0, len(picked), TILES_PER_PAGE):
        tiles = []
        for i, p in enumerate(picked[page:page + TILES_PER_PAGE], start=page + 1):
            img = cv2.imread(os.path.join(DEST_DIR, p["filename"]))
            tiles.append(tile(img, p["box"], i, p["filename"], p["value"], p["kind"]))
        while len(tiles) % TILE_COLS:
            tiles.append(np.full((TILE_H, TILE_W, 3), 255, np.uint8))
        rows = [np.hstack(tiles[r:r + TILE_COLS]) for r in range(0, len(tiles), TILE_COLS)]
        cv2.imwrite(os.path.join(DEST_DIR, "review", f"page_{page // TILES_PER_PAGE + 1:03d}.jpg"),
                    np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])

    # ---------- summary ----------
    n = len(m)
    ok = {v: m[v] != "" for v in names}
    print(f"\nDecoded a valid EAN-13 (correctness unknown until truth.csv is checked), of {n}:")
    any_ok = np.logical_or.reduce([ok[v] for v in names])
    t = pd.DataFrame({
        "decoded": [int(ok[v].sum()) for v in names],
        "only_this_variant": [int((ok[v] & (sum(ok[w] for w in names) == 1)).sum()) for v in names],
        "variants_disagree": [int(((m[v] != "") & (m[v] != m.suggested)).sum()) for v in names],
        "conflicts_rejected": [int((m[f"{v}_conflict"] != "").sum()) for v in names],
        "mean_s": [round(m[f"{v}_s"].mean(), 2) for v in names],
    }, index=names)
    print(t.to_string())
    print(f"\nAny variant: {int(any_ok.sum())}/{n}")
    reader = np.logical_or.reduce([ok[v] for v in READER_VARIANTS])
    print(f"Current reader ({' + '.join(READER_VARIANTS)}): {int(reader.sum())}/{n}")
    # Greedy: which variants, added one at a time, decode the most images?
    chosen, covered = [], np.zeros(n, bool)
    while True:
        gain = {v: int((ok[v] & ~covered).sum()) for v in names if v not in chosen}
        best = max(gain, key=gain.get) if gain else None
        if not best or gain[best] == 0:
            break
        chosen.append(best)
        covered |= ok[best]
        print(f"  + {best:16} -> {int(covered.sum())}/{n}")
    print("\nPer source:")
    g = m.assign(any=any_ok, reader=reader, original=ok["original"]).groupby("group")
    print(pd.DataFrame({"images": g.size(), "original": g.original.sum(), "current_reader": g.reader.sum(),
                        "any_variant": g["any"].sum()}).to_string())
    print(f"\nTo check: {(m.status == 'clean').sum()} clean reads (green), "
          f"{(m.status == 'conflict').sum()} conflicts (orange), {(m.status == 'none').sum()} no read (red)")
    print(f"Review sheets: {DEST_DIR}/review/   truth to check: {truth_path}")


if __name__ == "__main__":
    main()

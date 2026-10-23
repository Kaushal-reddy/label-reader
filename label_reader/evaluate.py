"""Stage 7: measure accuracy per stage against truth.csv, and time per item.

Run with:  python -m label_reader.evaluate   (from the repo folder, with the venv active)

One pass over all photos that runs every stage and measures it the same way,
so the numbers can be compared side by side. Each stage's own script printed
its own numbers while we built it; this is the single, consistent report.

Measures (every one compared with truth.csv):
  Stage 2  barcode / OCR on the raw photo, no tuning
  Stage 3  barcode with retries + barcode-type filter
  Stage 4  label crops: did any crop come out? does the barcode still read on the crops?
  Stage 5  OCR on the crops: exact, and within 1 character (near misses like O vs 0)
  Stage 6  final decision: accepted & correct, accepted & WRONG, needs review

Timing: the full pipeline (what validate.py runs per photo) is timed RUNS times
and the median is reported, because single timings on one machine varied by ~20%
between runs.

Outputs:
  output/evaluation/per_photo.csv   every measurement for every photo
  output/evaluation/summary.md      the summary tables (the source of BENCHMARKS.md)

analyze.py turns per_photo.csv into figures and confusion matrices.
"""

import os
import sys
import statistics
import time
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pytesseract

from label_reader import baseline
from label_reader.baseline import found, load_truth, out_dir_for
from label_reader.barcode_reader import read_barcode
from label_reader.label_finder import crop_rotated, find_label_boxes
from label_reader.ocr_reader import closest_match, get_crops, read_barcode_digits, read_lines, read_whole_crop
from label_reader.validate import barcode_candidates, decide, ocr_candidates

PHOTOS_DIR = "photos"
OUT_DIR = os.path.join("output", "evaluation")
RUNS = 3   # timing repeats per photo


def timed(fn: Callable[..., Any], *args: Any) -> tuple[Any, float]:
    """Run fn(*args) and return (result, seconds)."""
    t0 = time.perf_counter()
    result = fn(*args)
    return result, time.perf_counter() - t0


def evaluate_photo(img: np.ndarray, true_serial: str) -> dict[str, Any]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    r = {}

    # Stage 2: raw photo, untuned (same calls as baseline.py)
    raw_barcodes = baseline.read_barcodes(gray)
    r["s2_barcode_ok"] = any(found(true_serial, v) for _, v, _ in raw_barcodes)
    raw_text = pytesseract.image_to_string(gray)
    r["s2_ocr_ok"] = found(true_serial, raw_text)

    # Stage 3: barcode with retries, full photo
    values, _, conflicts, agreeing = read_barcode(gray)
    r["s3_conflicts"] = " ".join(conflicts)
    r["s3_agreeing"] = " ".join(f"{v}:{n}" for v, n in agreeing.items())
    r["s3_barcode_ok"] = any(found(true_serial, v) for v in values)

    # Stage 4: label crops
    crops_bgr = [crop_rotated(img, rect) for rect in find_label_boxes(gray)]
    r["s4_n_crops"] = len(crops_bgr)
    crop_values = []
    for c in crops_bgr:
        crop_values += read_barcode(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY))[0]
    r["s4_barcode_on_crops_ok"] = any(found(true_serial, v) for v in crop_values)

    # Stage 5: OCR on the crops (each method, and all together = read_ocr)
    crops = get_crops(img)
    whole = "\n".join(read_whole_crop(c) for c in crops)
    lines = "\n".join(read_lines(c) for c in crops)
    digits = "\n".join(d for c in crops for d in read_barcode_digits(c))
    both = whole + "\n" + lines + "\n" + digits
    for name, text in (("whole", whole), ("lines", lines), ("digits", digits), ("combined", both)):
        r[f"s5_{name}_ok"] = found(true_serial, text)
        r[f"s5_{name}_wrong"], closest = closest_match(true_serial, text)
    r["s5_combined_closest"] = closest   # what OCR read where the serial should be (analyze.py)

    # Stage 6: the decision, from exactly what validate.py would see
    bc = barcode_candidates(values)
    ocr_lab, ocr_unlab = ocr_candidates(both)
    status, serial, source, reason = decide(bc, ocr_lab, ocr_unlab, conflicts, agreeing)
    r.update({"s6_status": status, "s6_serial": serial, "s6_source": source, "s6_reason": reason,
              "s6_correct": serial == true_serial,
              "s6_barcode_candidates": " ".join(bc),
              "s6_ocr_labelled": " ".join(ocr_lab), "s6_ocr_unlabelled": " ".join(ocr_unlab)})
    return r


def time_pipeline(img: np.ndarray) -> dict[str, float]:
    """Time the real per-photo pipeline (validate.py's steps), split by step."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, t_labels = timed(find_label_boxes, gray)
    values_t = timed(read_barcode, gray)
    crops_t = timed(get_crops, img)
    crops = crops_t[0]
    _, t_ocr = timed(lambda: "\n".join(read_whole_crop(c) + "\n" + read_lines(c) + "\n"
                                       + "\n".join(read_barcode_digits(c)) for c in crops))
    _, t_decide = timed(lambda: decide(barcode_candidates(values_t[0][0]), [], []))
    # get_crops repeats the label finding, so OCR time = crop cutting + reading.
    return {"t_label_find": t_labels, "t_barcode": values_t[1],
            "t_ocr": crops_t[1] - t_labels + t_ocr, "t_decide": t_decide}


def pct(k: int, n: int) -> str:
    return f"{k}/{n} ({100 * k / n:.0f}%)"


def main(folder: str | None = None) -> None:
    photos_dir, out_dir, truth = folder or PHOTOS_DIR, out_dir_for(folder, "evaluation"), load_truth(folder)
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for filename, true_serial in zip(truth.filename, truth.true_serial):
        img = cv2.imread(os.path.join(photos_dir, filename))
        row = {"filename": filename, "true_serial": true_serial}
        row.update(evaluate_photo(img, true_serial))
        runs = [time_pipeline(img) for _ in range(RUNS)]
        for key in runs[0]:
            row[key] = round(statistics.median(run[key] for run in runs), 2)
        row["t_total"] = round(sum(row[k] for k in ("t_label_find", "t_barcode", "t_ocr", "t_decide")), 2)
        rows.append(row)
        print(f"{filename:24} {row['s6_status']:13} {row['t_total']:.1f}s")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "per_photo.csv"), index=False)
    n = len(df)
    acc = df[df.s6_status == "accepted"]
    rev = df[df.s6_status != "accepted"]

    stage_table = [
        ("2", "Barcode, raw photo", pct(df.s2_barcode_ok.sum(), n)),
        ("2", "OCR, raw photo (serial anywhere in text)", pct(df.s2_ocr_ok.sum(), n)),
        ("3", "Barcode with retries + type filter", pct(df.s3_barcode_ok.sum(), n)),
        ("4", "At least one label crop found", pct((df.s4_n_crops > 0).sum(), n)),
        ("4", "Barcode on the crops", pct(df.s4_barcode_on_crops_ok.sum(), n)),
        ("5", "OCR on crops, whole crop: exact", pct(df.s5_whole_ok.sum(), n)),
        ("5", "OCR on crops, line by line: exact", pct(df.s5_lines_ok.sum(), n)),
        ("5", "OCR on crops, barcode digits: exact", pct(df.s5_digits_ok.sum(), n)),
        ("5", "OCR on crops, combined: exact", pct(df.s5_combined_ok.sum(), n)),
        ("5", "OCR on crops, combined: within 1 character", pct((df.s5_combined_wrong <= 1).sum(), n)),
    ]
    final_table = [
        ("Accepted (no human needed)", pct(len(acc), n)),
        ("Accepted and correct", pct(acc.s6_correct.sum(), n)),
        ("Accepted and WRONG", pct((~acc.s6_correct).sum(), n)),
        ("Precision (correct / accepted)",
         f"{acc.s6_correct.sum()}/{len(acc)} ({100 * acc.s6_correct.mean():.0f}%)" if len(acc) else "n/a"),
        ("Needs review", pct(len(rev), n)),
        ("Needs review, best guess already correct", pct(rev.s6_correct.sum(), n)),
    ]
    time_table = [(k.replace("t_", ""), f"{df[k].mean():.2f}", f"{df[k].max():.2f}")
                  for k in ("t_label_find", "t_barcode", "t_ocr", "t_decide", "t_total")]

    truth_path = os.path.join(folder, 'truth.csv') if folder else 'truth.csv'
    md = [f"Measured on {n} photos ({truth_path}). Times: median of {RUNS} runs per photo.\n",
          "### Accuracy per stage\n", "| Stage | Measure | Result |", "|---|---|---|"]
    md += [f"| {s} | {m} | {r} |" for s, m, r in stage_table]
    md += ["\n### Final decisions (Stage 6: accept only if barcode and OCR agree, "
           "or a product barcode is read one way only, by 2+ image variants)\n",
           "| Measure | Result |", "|---|---|"]
    md += [f"| {m} | {r} |" for m, r in final_table]
    md += ["\n### Time per photo (seconds)\n", "| Step | Mean | Max |", "|---|---|---|"]
    md += [f"| {s} | {a} | {b} |" for s, a, b in time_table]
    md += ["\n### Per photo\n", "| Photo | Decision | Serial | Correct | Reason |", "|---|---|---|---|---|"]
    md += [f"| {r.filename} | {r.s6_status} | {r.s6_serial} | {'yes' if r.s6_correct else 'no'} | {r.s6_reason} |"
           for r in df.itertuples()]
    summary = "\n".join(md)
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print("\n" + summary)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

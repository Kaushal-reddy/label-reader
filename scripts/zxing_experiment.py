"""Experiment: does a second barcode decoder (zxing-cpp) help ZBar?

Run with:  python -m scripts.zxing_experiment                    (our 26 photos)
      or:  python -m scripts.zxing_experiment photos/ean_dataset (the 403-photo EAN set)
Needs validate.py to have been run on the same folder first: its OCR candidates
(validation.csv) are reused, so only the barcode step is re-run.

The learned detector (branch barcode-detection) showed ZBar's failures on the EAN
set are decoding failures: the barcode is found in all 403 photos, but ZBar reads
nothing in 178. zxing-cpp (Apache-2.0) is an independent decoder with its own
binarizer and scanning, so this measures whether it reads bars ZBar can't, and
which way of combining the two is safest. The result (README, Stage 3): zxing_first
accepted the most photos with none wrong, so barcode_reader.read_barcode now
works that way. This script keeps the comparison reproducible.

zxing-cpp reads the same image variants as ZBar's confirmation step
(barcode_reader.CONFIRM_VARIANTS: original, upscale2x, clahe_2x) through
barcode_reader.zxing_decode, which pads UPC-A to the 13 digits ZBar returns.

Every policy goes through the SAME Stage 6 decision (validate.decide), with the
OCR candidates validate.py found:
  zbar      ZBar alone, the reader before zxing-cpp (baseline)
  zxing     zxing-cpp alone
  fallback  ZBar; zxing-cpp only when ZBar reads nothing
  zxing_first  zxing-cpp; ZBar only when zxing-cpp reads nothing (what
            barcode_reader.read_barcode does now)
  either    both decoders' values pooled; a product barcode needs 2 agreeing
            variants from ONE decoder (as ZBar alone does); any disagreement is a conflict
  both      both decoders' values pooled; a product barcode needs BOTH decoders
            to read it (decoder agreement replaces variant agreement)

Outputs: output/zxing/ (or output/<dataset>/zxing/): per_photo.csv, summary.md
"""

import os
import sys
import time

import cv2
import pandas as pd

from label_reader.baseline import load_truth, out_dir_for
from label_reader.barcode_reader import CONFIRM_VARIANTS, VARIANTS, is_product_code, zbar_read, zxing_decode
from label_reader.validate import barcode_candidates, decide

PHOTOS_DIR = "photos"
POLICIES = ["zbar", "zxing", "fallback", "zxing_first", "either", "both"]


def same_id(a: str, b: str) -> bool:
    """Equal, counting a 12-digit UPC-A and its zero-padded EAN-13 as the same number."""
    pad = lambda s: s.zfill(13) if s.isdigit() and len(s) == 12 else s
    return pad(a) == pad(b)


def zxing_read(gray) -> tuple[list[str], list[str], dict[str, int], set[str]]:
    """Like barcode_reader.zxing_read(), plus every value any variant read (for the raw-read counts)."""
    makers = dict(VARIANTS)
    per_variant = [zxing_decode(makers[name](gray)) for name in CONFIRM_VARIANTS]
    values = next((r for r in per_variant if r), [])          # first variant that read anything
    agreeing = {v: sum(v in r for r in per_variant) for v in values}
    everything = {v for r in per_variant for v in r}
    conflicts = sorted(v for v in everything if is_product_code(v) and v not in values)
    return values, conflicts, agreeing, everything


def pooled(a: list[str], b: list[str]) -> list[str]:
    return a + [v for v in b if v not in a]


def policies(zb, zx) -> dict[str, tuple[list[str], list[str], dict[str, int]]]:
    """(values, conflicts, agreeing) to give validate.decide, per policy."""
    zx_values, zx_conflicts, zx_agreeing, zx_all = zx
    values = pooled(zb.values, zx_values)
    conflicts = pooled(zb.conflicts, zx_conflicts)
    zb_all = set(zb.values) | set(zb.agreeing)
    return {
        "zbar": (zb.values, zb.conflicts, zb.agreeing),
        "zxing": (zx_values, zx_conflicts, zx_agreeing),
        "fallback": (zb.values, zb.conflicts, zb.agreeing) if zb.values
                    else (zx_values, pooled(zb.conflicts, zx_conflicts), zx_agreeing),
        "zxing_first": (zx_values, zx_conflicts, zx_agreeing) if zx_values
                       else (zb.values, zb.conflicts, zb.agreeing),
        "either": (values, conflicts,
                   {v: max(zb.agreeing.get(v, 0), zx_agreeing.get(v, 0)) for v in values}),
        "both": (values, conflicts, {v: (v in zb_all) + (v in zx_all) for v in values}),
    }


def md_table(t: pd.DataFrame) -> str:
    """A markdown table (same layout as analyze.py, without needing tabulate)."""
    lines = ["| " + " | ".join(map(str, t.columns)) + " |", "|" + "---|" * len(t.columns)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in t.itertuples(index=False)]
    return "\n".join(lines)


def split(cell: str) -> list[str]:
    return cell.split() if isinstance(cell, str) else []


def main(folder: str | None = None) -> None:
    photos_dir = folder or PHOTOS_DIR
    out_dir = out_dir_for(folder, "zxing")
    os.makedirs(out_dir, exist_ok=True)
    truth = dict(zip(*[load_truth(folder)[c] for c in ("filename", "true_serial")]))
    ocr = pd.read_csv(os.path.join(out_dir_for(folder, ""), "validation.csv"), dtype=str).set_index("filename")

    rows = []
    for i, filename in enumerate(truth, 1):
        true = truth[filename]
        gray = cv2.cvtColor(cv2.imread(os.path.join(photos_dir, filename)), cv2.COLOR_BGR2GRAY)
        t0 = time.perf_counter()
        zb = zbar_read(gray)               # ZBar on its own, as before zxing-cpp
        t_zbar = time.perf_counter() - t0
        t0 = time.perf_counter()
        zx = zxing_read(gray)
        t_zxing = time.perf_counter() - t0

        ocr_lab, ocr_unlab = split(ocr.at[filename, "ocr_labelled"]), split(ocr.at[filename, "ocr_unlabelled"])
        row = {"filename": filename, "true_serial": true, "t_zbar": round(t_zbar, 3), "t_zxing": round(t_zxing, 3),
               "zbar_values": " ".join(zb.values), "zxing_values": " ".join(zx[0]),
               "zbar_read_ok": any(same_id(v, true) for v in zb.values),
               "zxing_read_ok": any(same_id(v, true) for v in zx[3]),
               "zbar_wrong_product": any(is_product_code(v) and not same_id(v, true) for v in zb.values),
               "zxing_wrong_product": any(is_product_code(v) and not same_id(v, true) for v in zx[3])}
        for name, (values, conflicts, agreeing) in policies(zb, zx).items():
            status, serial, _, reason = decide(barcode_candidates(values), ocr_lab, ocr_unlab, conflicts, agreeing)
            row |= {f"{name}_status": status, f"{name}_serial": serial, f"{name}_reason": reason,
                    f"{name}_correct": same_id(serial, true)}
        # Sanity check: validate.py (zxing-cpp first since this experiment) must
        # make the same decision as the zxing_first policy here.
        row["zxing_first_matches_validate"] = row["zxing_first_status"] == ocr.at[filename, "status"]
        rows.append(row)
        if i % 50 == 0:
            print(f"  {i}/{len(truth)} photos")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "per_photo.csv"), index=False)

    summary = []
    for name in POLICIES:
        acc = df[df[f"{name}_status"] == "accepted"]
        rev = df[df[f"{name}_status"] != "accepted"]
        summary.append({"policy": name, "accepted": len(acc), "accepted_wrong": int((~acc[f"{name}_correct"]).sum()),
                        "review_guess_correct": int(rev[f"{name}_correct"].sum()),
                        "nothing_read": int((df[f"{name}_serial"].fillna("") == "").sum())})
    summary = pd.DataFrame(summary)
    reads = pd.DataFrame([{"decoder": d, "read_correct": int(df[f"{d}_read_ok"].sum()),
                           "read_a_wrong_product_number": int(df[f"{d}_wrong_product"].sum()),
                           "seconds_mean": round(df[f"t_{d}"].mean(), 3), "seconds_max": round(df[f"t_{d}"].max(), 3)}
                          for d in ("zbar", "zxing")])
    only = {"zxing reads, ZBar doesn't": int((df.zxing_read_ok & ~df.zbar_read_ok).sum()),
            "ZBar reads, zxing doesn't": int((df.zbar_read_ok & ~df.zxing_read_ok).sum())}
    wrong = df[[any(df.at[i, f"{p}_status"] == "accepted" and not df.at[i, f"{p}_correct"] for p in POLICIES)
                for i in df.index]]

    text = (f"# zxing-cpp vs ZBar: {len(df)} photos ({photos_dir})\n\n"
            f"zxing_first reproduces validate.py on {int(df.zxing_first_matches_validate.sum())}/{len(df)} photos.\n\n"
            f"## Raw reads (any image variant)\n\n{md_table(reads)}\n\n"
            + "".join(f"- {k}: {v}\n" for k, v in only.items())
            + f"\n## Decisions (all through validate.decide, with validate.py's OCR)\n\n{md_table(summary)}\n\n"
            f"## Accepted and wrong under any policy: {len(wrong)}\n\n"
            + (md_table(wrong[["filename"] + [f"{p}_status" for p in POLICIES]]) if len(wrong) else "none")
            + "\n")
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

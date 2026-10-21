"""Stage 3b, measured: does the learned detector (barcode_detector.py) help
the barcode reader on the EAN set, without letting wrong numbers through?

Two steps, which can run on different machines:

  python evaluate_detector.py train      # GPU machine: trains one detector per fold
  python evaluate_detector.py evaluate   # this machine: detect -> crop -> ZBar -> Stage 6

`train` only needs torch, torchvision, OpenCV, numpy and pandas (no ZBar or
Tesseract), plus photos/Barcode_Dataset and photos/ean_dataset/truth.csv.
Copy models/folds/ back here before running `evaluate`.

Honest scoring: grouped cross-validation. The 403 checked EAN images are split
into FOLDS groups, keeping every photo of one product (same true_serial) in the
same fold, because they are near-identical frames. Each fold's detector is
trained on everything EXCEPT that fold and scored only on it, so every EAN image
is scored by a model that never saw it or its near-copies. Also left out of
training: the other images from the same three photo collections (Foto, PICT,
0510/0610). They show products we have no value for, so they could be the test
fold's products. The QR images (a separate collection) are always used.

`evaluate` reads each test photo four ways, and every one goes through the same
Stage 6 decision (validate.decide: checksum, vote margin, 2 agreeing variants):
  full       the current reader on the whole photo (the baseline to beat)
  crops      the reader on the detector's crops only
  fallback   the whole photo first; the crops only if it read nothing
  merged     both: values and conflicts pooled, agreeing variants added up
             (a read of the crop counts as extra evidence for the same value).
             The riskiest: a crop is the same pixels as the photo, so one crop
             read plus one full-photo variant already meets the 2-variant bar.
             In the smoke test it accepted a photo the full read sent to review.

Outputs in output/detector/:
  per_photo.csv   every measurement for every photo
  summary.md      the tables printed at the end
"""

import argparse
import os
import sys
import time

import cv2
import pandas as pd
import torch
from torchvision.ops import box_iou

import barcode_detector as bd

FOLDS = 5
FOLD_DIR = os.path.join("models", "folds")
FOLDS_CSV = os.path.join(FOLD_DIR, "folds.csv")
OUT_DIR = os.path.join("output", "detector")
POLICIES = ["full", "crops", "fallback", "merged"]


# ---------- folds ----------

def assign_folds(truth: pd.DataFrame, folds: int = FOLDS) -> pd.DataFrame:
    """filename, true_serial, fold: each product's photos all in one fold.

    Biggest products first, each into the fold with the fewest photos so far,
    so the folds come out about the same size. Deterministic: the GPU machine
    and this one get the same split from the same truth.csv.
    """
    sizes = truth.groupby("true_serial").size()
    order = sorted(sizes.index, key=lambda serial: (-sizes[serial], serial))
    count, fold_of = [0] * folds, {}
    for serial in order:
        k = count.index(min(count))
        fold_of[serial] = k
        count[k] += sizes[serial]
    out = truth[["filename", "true_serial"]].copy()
    out["fold"] = out.true_serial.map(fold_of)
    return out.sort_values("filename").reset_index(drop=True)


def fold_items(fold: int, folds: pd.DataFrame) -> tuple[list, list]:
    """(training items, test items) for one fold, as (image, xml) paths."""
    test_names = set(folds.filename[folds.fold == fold])
    train_names = set(folds.filename[folds.fold != fold])
    train, test = [], []
    for item in bd.dataset_items():
        name = bd.item_name(item)
        if name in test_names:
            # Scored on the photos/ean_dataset copy (same file, checked truth beside it).
            stem = os.path.splitext(name)[0]
            test.append((os.path.join(bd.EAN_DIR, name), os.path.join(bd.EAN_DIR, stem + ".xml")))
        elif name in train_names or bd.source_group(os.path.splitext(name)[0]) is None:
            train.append(item)
        # else: same collections as the test products, value unknown -> left out
    return train, test


def load_folds() -> pd.DataFrame:
    """The split the fold models were trained on, checked against today's truth.csv."""
    if not os.path.exists(FOLDS_CSV):
        sys.exit(f"{FOLDS_CSV} not found: run `python evaluate_detector.py train` first "
                 f"(or copy models/folds/ from the training machine).")
    saved = pd.read_csv(FOLDS_CSV, dtype={"true_serial": str})
    now = assign_folds(bd.checked_truth(), int(saved.fold.max()) + 1)
    if not saved.reset_index(drop=True).equals(now[saved.columns].reset_index(drop=True)):
        sys.exit(f"{FOLDS_CSV} doesn't match photos/ean_dataset/truth.csv any more (rows or "
                 f"values changed since training). Re-train the folds so the test photos "
                 f"stay unseen.")
    return saved


def train_folds(args) -> None:
    folds = assign_folds(bd.checked_truth(), args.folds)
    os.makedirs(FOLD_DIR, exist_ok=True)
    folds.to_csv(FOLDS_CSV, index=False)
    print(f"{len(folds)} checked EAN photos, {folds.true_serial.nunique()} products, "
          f"{args.folds} folds: {folds.fold.value_counts().sort_index().tolist()} photos each")
    for k in (args.only if args.only is not None else range(args.folds)):
        train_items, test_items = fold_items(k, folds)
        groups = bd.product_groups(train_items, bd.checked_truth())
        tr, val = bd.split_val(train_items, bd.VAL_FRACTION, seed=k,
                               group_of=lambda item: groups[bd.item_name(item)])
        out = os.path.join(FOLD_DIR, f"fold{k}.pt")
        print(f"\nFold {k}: train {len(tr)}, validation {len(val)}, test {len(test_items)} (never seen)")
        bd.train(tr, out, val, epochs=args.epochs, batch_size=args.batch_size, arch=args.arch,
                 workers=args.workers, seed=k, resume=args.resume)


# ---------- evaluation ----------

def merge_reads(reads: list) -> tuple[list[str], list[str], dict[str, int]]:
    """Pool several BarcodeReads of crops of ONE photo.

    Overlapping detections are crops of the same barcode, so a value's
    agreeing count is its best single crop's, not the sum: two near-identical
    crops reading the same wrong value are not two independent confirmations.
    """
    values, conflicts, agreeing = [], [], {}
    for r in reads:
        values += [v for v in r.values if v not in values]
        conflicts += [c for c in r.conflicts if c not in conflicts]
        for v, n in r.agreeing.items():
            agreeing[v] = max(agreeing.get(v, 0), n)
    return values, [c for c in conflicts if c not in values], agreeing


def decide_read(values, conflicts, agreeing, true_serial, decide, barcode_candidates) -> dict:
    status, serial, _, reason = decide(barcode_candidates(values), [], [], conflicts, agreeing)
    return {"status": status, "serial": serial, "reason": reason,
            "read_ok": true_serial in values, "accepted_wrong": status == "accepted" and serial != true_serial}


def evaluate() -> None:
    # Imported here so `train` doesn't need ZBar / Tesseract installed.
    from barcode_reader import read_barcode
    from validate import barcode_candidates, decide

    folds = load_folds()
    os.makedirs(OUT_DIR, exist_ok=True)
    rows, box_rows = [], []
    for k in sorted(folds.fold.unique()):
        path = os.path.join(FOLD_DIR, f"fold{k}.pt")
        if not os.path.exists(path):
            sys.exit(f"{path} not found: train it, or copy it from the training machine.")
        det = bd.BarcodeDetector(path)
        _, test_items = fold_items(k, folds)
        box_rows.append({"fold": k, "photos": len(test_items),
                         **bd.evaluate_boxes(det.model, test_items, det.dev)})
        truth = dict(zip(folds.filename, folds.true_serial))
        print(f"Fold {k}: {len(test_items)} photos", flush=True)
        for img_path, xml_path in test_items:
            name = os.path.basename(img_path)
            true_serial = truth[name]
            img = bd.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            t0 = time.perf_counter()
            full = read_barcode(gray)
            t_full = time.perf_counter() - t0
            t0 = time.perf_counter()
            dets = det.detect(img)
            t_detect = time.perf_counter() - t0
            t0 = time.perf_counter()
            crop_reads = [read_barcode(c) for c in det.crops(img, dets)]
            t_crops = time.perf_counter() - t0

            # Did a box land on the true barcode? (IoU >= 0.5 with an annotated one)
            true_boxes, true_labels = bd.read_voc(xml_path)
            true_boxes = true_boxes[true_labels == bd.CLASSES.index("barcode")]
            ious = [float(box_iou(torch.tensor(b[None]), torch.from_numpy(true_boxes)).max())
                    for b, _, _ in dets] if len(true_boxes) else []

            c_values, c_conflicts, c_agree = merge_reads(crop_reads)
            m_values = full.values + [v for v in c_values if v not in full.values]
            m_conflicts = [c for c in dict.fromkeys(full.conflicts + c_conflicts) if c not in m_values]
            m_agree = {v: full.agreeing.get(v, 0) + c_agree.get(v, 0) for v in m_values}
            reads = {
                "full": (full.values, full.conflicts, full.agreeing),
                "crops": (c_values, c_conflicts, c_agree),
                "fallback": ((full.values, full.conflicts, full.agreeing) if full.values
                             else (c_values, c_conflicts, c_agree)),
                "merged": (m_values, m_conflicts, m_agree),
            }
            row = {"fold": k, "filename": name, "group": bd.source_group(os.path.splitext(name)[0]),
                   "true_serial": true_serial, "n_boxes": len(dets),
                   "top_score": round(dets[0][2], 3) if dets else 0.0,
                   "best_iou": round(max(ious), 3) if ious else 0.0,
                   "box_found": bool(ious) and max(ious) >= bd.IOU_MATCH,
                   "full_values": " ".join(full.values), "crop_values": " ".join(c_values),
                   "t_full": round(t_full, 3), "t_detect": round(t_detect, 3), "t_crops": round(t_crops, 3)}
            for p, (v, c, a) in reads.items():
                for key, val in decide_read(v, c, a, true_serial, decide, barcode_candidates).items():
                    row[f"{p}_{key}"] = val
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "per_photo.csv"), index=False)
    summarize(df, pd.DataFrame(box_rows))


def md_table(df: pd.DataFrame, index: bool = False) -> str:
    """A DataFrame as a Markdown table (like analyze.py's, no extra package needed)."""
    t = df.reset_index() if index else df
    lines = ["| " + " | ".join(map(str, t.columns)) + " |", "|" + "---|" * len(t.columns)]
    return "\n".join(lines + ["| " + " | ".join(map(str, r)) + " |" for r in t.itertuples(index=False)])


def summarize(df: pd.DataFrame, boxes: pd.DataFrame) -> None:
    n = len(df)
    policy_table = pd.DataFrame([{
        "policy": p,
        "read correctly": int(df[f"{p}_read_ok"].sum()),
        "accepted": int((df[f"{p}_status"] == "accepted").sum()),
        "accepted WRONG": int(df[f"{p}_accepted_wrong"].sum()),
        "review, guess correct": int(((df[f"{p}_status"] != "accepted")
                                      & (df[f"{p}_serial"] == df.true_serial)).sum()),
        "nothing read": int((df[f"{p}_serial"] == "").sum()),
    } for p in POLICIES])
    by_group = df.groupby("group").agg(
        photos=("filename", "size"), box_found=("box_found", "sum"),
        **{f"{p} accepted": (f"{p}_status", lambda s: int((s == "accepted").sum())) for p in POLICIES})
    timing = pd.DataFrame({"mean": df[["t_full", "t_detect", "t_crops"]].mean(),
                           "max": df[["t_full", "t_detect", "t_crops"]].max()}).round(2)
    # Photos the detector rescued (crops accepted, full photo didn't), and the
    # reverse, so a change in totals can be traced to actual photos.
    gained = df[(df.fallback_status == "accepted") & (df.full_status != "accepted")]
    wrong = df[df[[f"{p}_accepted_wrong" for p in POLICIES]].any(axis=1)]

    md = [f"Grouped {df.fold.nunique()}-fold cross-validation on {n} checked EAN photos: "
          "every photo scored by a detector that never saw it or its product.\n",
          "### Decisions (all through Stage 6)\n", md_table(policy_table),
          "\n### Detection on the test folds (AP@0.5; recall at score >= "
          f"{bd.MIN_SCORE})\n", md_table(boxes.round(3)),
          "\n### By photo collection\n", md_table(by_group, index=True),
          "\n### Time per photo (seconds)\n", md_table(timing, index=True),
          f"\n### Accepted thanks to the detector (fallback, not full): {len(gained)}\n",
          md_table(gained[["filename", "group", "true_serial", "fallback_reason"]])
          if len(gained) else "none",
          f"\n### Accepted and WRONG under any policy: {len(wrong)}\n",
          md_table(wrong[["filename", "true_serial"] + [f"{p}_serial" for p in POLICIES]
                         + [f"{p}_status" for p in POLICIES]]) if len(wrong) else "none"]
    summary = "\n".join(md)
    with open(os.path.join(OUT_DIR, "summary.md"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print("\n" + summary)


def main() -> None:
    p = argparse.ArgumentParser(description="Grouped cross-validation of the barcode detector on the EAN set.")
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train", help="train one detector per fold (GPU machine)")
    t.add_argument("--folds", type=int, default=FOLDS)
    t.add_argument("--only", type=int, nargs="+", help="train only these folds, e.g. --only 0 1")
    t.add_argument("--epochs", type=int, default=bd.EPOCHS)
    t.add_argument("--batch-size", type=int, default=bd.BATCH_SIZE)
    t.add_argument("--arch", default=bd.ARCH, choices=list(bd.ARCHITECTURES))
    t.add_argument("--workers", type=int, default=8, help="data-loading processes")
    t.add_argument("--resume", action="store_true", help="continue each fold from its _last.pt")
    sub.add_parser("evaluate", help="score the fold models through the real reader (needs ZBar)")
    args = p.parse_args()
    train_folds(args) if args.command == "train" else evaluate()


if __name__ == "__main__":
    main()

"""Stage 3b: a learned barcode / QR detector, to find codes the plain reader misses.

Train with:  python barcode_detector.py --out models/barcode_detector.pt
(from the repo folder; needs the packages in requirements-detector.txt and is
meant for a CUDA GPU, e.g. a DGX Spark. It runs on a Mac too, just slowly.)

Why: on the 403-image EAN set, ZBar read nothing at all on 178 images, and the
hand-tuned label finder (Stage 4) found a crop in only 59 of them: its edge
thresholds were tuned on serial labels in 4032 px photos. Finding "where is
the barcode?" is what object detectors are good at, and the downloaded dataset
(photos/Barcode_Dataset) already has a box drawn around every code.

The detector only proposes WHERE to look. The crops go through the same
Stage 3 reader and Stage 6 rules as the full photo (checksum, vote margin,
2 agreeing variants), so it can add reads or send more photos to review, but
it can't make the pipeline trust a number it wouldn't otherwise trust.
evaluate_detector.py measures whether that holds.

Model: Faster R-CNN with a ResNet-50 FPN v2 backbone from torchvision,
pretrained on COCO (everyday objects) and fine-tuned on two classes, barcode
and qr. torchvision is BSD-licensed, so it can be used in a company's internal
tools without conditions (Ultralytics YOLO is AGPL-3.0, which would not allow
that without a paid licence).

Training (train()):
  - augmentation that moves the boxes with the image: 90-degree rotations
    (vertical barcodes are common), flips, zoom-in crops, brightness/contrast/
    gamma/colour changes, Gaussian and motion blur, JPEG artefacts
  - multi-scale: each batch is resized to a random shorter side of 640-1024 px
  - SGD with a 1-epoch warmup, then a cosine learning-rate decay
  - bf16 mixed precision on CUDA (about 2x faster, same accuracy)
  - after every epoch: AP@0.5 on a held-out validation split; the best epoch's
    weights are kept, and a resumable checkpoint is written (--resume)

This script's own main() trains ONE detector on every image in the dataset.
That model has seen the EAN test images, so it must NOT be used to measure
accuracy on them; evaluate_detector.py trains separate models per fold that
never see the images they are scored on.
"""

import argparse
import csv
import math
import os
import random
import re
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.models.detection import (fasterrcnn_mobilenet_v3_large_fpn,
                                          fasterrcnn_resnet50_fpn_v2)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou

DATASET_DIR = os.path.join("photos", "Barcode_Dataset")
EAN_DIR = os.path.join("photos", "ean_dataset")   # the checked EAN test set (a subset of DATASET_DIR)
MODEL_PATH = os.path.join("models", "barcode_detector.pt")

# Index 0 is reserved for "background" by torchvision's detectors.
CLASSES = ["background", "barcode", "qr"]
# One box in Foto(737) is labelled "d"; it's a barcode (checked by eye).
CLASS_FIXES = {"d": "barcode"}

# resnet50_fpn_v2 is the most accurate Faster R-CNN in torchvision (46.7 COCO
# box AP vs 32.8 for mobilenet_v3_large_fpn). The MobileNet one is ~5x faster
# on a CPU, if speed on the station's PC ever matters more.
ARCHITECTURES = {"resnet50_fpn_v2": fasterrcnn_resnet50_fpn_v2,
                 "mobilenet_v3_large_fpn": fasterrcnn_mobilenet_v3_large_fpn}
ARCH = "resnet50_fpn_v2"
# Shorter image side during training: one of these at random per batch, so the
# model sees codes at many sizes. Inference uses the last (largest) one, so a
# small label in a 4032x2268 photo is still ~1/2 of its pixels after resizing.
TRAIN_SIZES = tuple(range(640, 1025, 64))
MAX_SIZE = 1824               # longest side cap (1024 x 16/9, a phone photo's shape)
# All 5 ResNet stages are fine-tuned: barcodes look nothing like COCO's
# objects, so even the early "edge and texture" layers can usefully adapt.
TRAINABLE_LAYERS = 5

EPOCHS = 40
BATCH_SIZE = 8
BASE_LR = 0.02                # for a batch of 16 (torchvision's recipe); scaled linearly
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 1
VAL_FRACTION = 0.1            # of the training images, held out to pick the best epoch
IOU_MATCH = 0.5               # a detection counts as finding a code at IoU >= 0.5

# Boxes below this confidence are dropped at inference. Low on purpose: a false
# box only costs one extra ZBar call on a crop, a missed barcode costs a read.
MIN_SCORE = 0.3
MAX_CROPS = 3
# Detected boxes are grown by this fraction on every side before cropping:
# ZBar needs a quiet zone of white around the bars, and boxes are never exact.
CROP_MARGIN = 0.15


def device() -> torch.device:
    """NVIDIA GPU (CUDA) or Apple GPU (MPS) if there is one, else the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------- data ----------

def read_voc(xml_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Boxes (N x 4: x1, y1, x2, y2) and class indices (N) from a Pascal VOC .xml."""
    boxes, labels = [], []
    for obj in ET.parse(xml_path).getroot().findall("object"):
        name = obj.findtext("name", "")
        name = CLASS_FIXES.get(name, name)
        b = obj.find("bndbox")
        if b is None:
            continue
        x1, y1, x2, y2 = (float(b.findtext(k, "0")) for k in ("xmin", "ymin", "xmax", "ymax"))
        if name in CLASSES and x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])
            labels.append(CLASSES.index(name))
    return np.array(boxes, dtype=np.float32).reshape(-1, 4), np.array(labels, dtype=np.int64)


def dataset_items(folder: str = DATASET_DIR) -> list[tuple[str, str]]:
    """(image path, xml path) for every image in the dataset that has an .xml."""
    items = []
    for f in sorted(os.listdir(folder)):
        stem, ext = os.path.splitext(f)
        xml = os.path.join(folder, stem + ".xml")
        if ext.lower() in (".jpg", ".jpeg", ".png") and os.path.exists(xml):
            items.append((os.path.join(folder, f), xml))
    return items


def checked_truth(folder: str = EAN_DIR) -> pd.DataFrame:
    """Rows of <folder>/truth.csv marked checked=yes, padding stripped.

    Same rules as baseline.load_truth(), repeated here because baseline.py
    loads ZBar and Tesseract, which the training machine doesn't need.
    """
    truth = pd.read_csv(os.path.join(folder, "truth.csv"), dtype=str, keep_default_na=False)
    truth.columns = truth.columns.str.strip()
    truth = truth.apply(lambda col: col.str.strip())
    if "checked" in truth.columns:
        truth = truth[(truth.checked == "yes") & (truth.true_serial != "")]
    return truth


def source_group(stem: str) -> str | None:
    """Which original collection a dataset image came from (by its file name).
    Same rule as prepare_ean_dataset.source_group()."""
    if stem.startswith("Foto"):
        return "Foto"
    if stem.startswith(("0510", "0610")):
        return "0510/0610"
    if re.match(r"P\w*I\w*C\w*T", stem):      # PICT0021, PICfggfhT0018, ...
        return "PICT"
    return None                              # the QR sets: not product barcodes


def item_name(item: tuple[str, str]) -> str:
    return os.path.basename(item[0])


def product_groups(items: list[tuple[str, str]], truth: pd.DataFrame) -> dict[str, str]:
    """item file name -> group, for keeping near-identical photos together.

    EAN images are grouped by product (their checked true_serial). Other images
    from the same three photo collections show products we have no value for,
    so they get one shared group per collection. The QR images are each their own.
    """
    product = dict(zip(truth.filename, truth.true_serial))
    out = {}
    for item in items:
        name = item_name(item)
        src = source_group(os.path.splitext(name)[0])
        out[name] = (f"product {product[name]}" if name in product
                     else f"collection {src}" if src else f"image {name}")
    return out


def imread(path: str) -> np.ndarray:
    """cv2.imread, but a missing or unreadable file is an error instead of None."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"can't read {path}")
    return img


def to_tensor(bgr: np.ndarray) -> torch.Tensor:
    """OpenCV's BGR uint8 image -> torchvision's RGB float tensor (3 x H x W, 0..1)."""
    return torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255


# ---------- augmentation (every geometric step moves the boxes too) ----------

def rotate90(img: np.ndarray, boxes: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Rotate by k x 90 degrees counter-clockwise. Exact for boxes: they stay boxes."""
    for _ in range(k % 4):
        w = img.shape[1]
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # A point (x, y) moves to (y, w - x).
        boxes = np.stack([boxes[:, 1], w - boxes[:, 2], boxes[:, 3], w - boxes[:, 0]], axis=1)
    return img, boxes


def flip(img: np.ndarray, boxes: np.ndarray, horizontal: bool) -> tuple[np.ndarray, np.ndarray]:
    boxes = boxes.copy()
    if horizontal:
        w = img.shape[1]
        boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        return img[:, ::-1].copy(), boxes
    h = img.shape[0]
    boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
    return img[::-1].copy(), boxes


def zoom_crop(img: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cut out a random window that still contains every box whole (a "zoom in")."""
    h, w = img.shape[:2]
    if len(boxes):
        ux1, uy1 = boxes[:, :2].min(axis=0)
        ux2, uy2 = boxes[:, 2:].max(axis=0)
    else:
        ux1, uy1, ux2, uy2 = w / 2, h / 2, w / 2, h / 2
    x1, y1 = random.uniform(0, ux1), random.uniform(0, uy1)
    x2, y2 = random.uniform(ux2, w), random.uniform(uy2, h)
    x1, y1, x2, y2 = int(x1), int(y1), int(math.ceil(x2)), int(math.ceil(y2))
    if x2 - x1 < 32 or y2 - y1 < 32:
        return img, boxes
    return img[y1:y2, x1:x2].copy(), boxes - np.array([x1, y1, x1, y1], dtype=np.float32)


def photometric(img: np.ndarray) -> np.ndarray:
    """Lighting, colour, focus and compression changes; boxes are unaffected."""
    alpha, beta = random.uniform(0.6, 1.4), random.uniform(-40, 40)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)            # contrast, brightness
    if random.random() < 0.3:                                          # gamma: shadows / glare
        g = random.uniform(0.6, 1.6)
        img = cv2.LUT(img, (255 * (np.arange(256) / 255) ** g).astype(np.uint8))
    if random.random() < 0.3:                                          # colour saturation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.5, 1.5), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if random.random() < 0.25:                                         # out of focus
        img = cv2.GaussianBlur(img, (0, 0), random.uniform(0.5, 2.5))
    elif random.random() < 0.15:                                       # hand shake
        n = random.choice([5, 7, 9, 11])
        kernel = np.zeros((n, n), np.float32)
        kernel[n // 2] = 1 / n
        rot = cv2.getRotationMatrix2D((n / 2 - 0.5, n / 2 - 0.5), random.uniform(0, 180), 1)
        img = cv2.filter2D(img, -1, cv2.warpAffine(kernel, rot, (n, n)))
    if random.random() < 0.3:                                          # JPEG artefacts
        q = random.randint(30, 90)
        decoded = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
        if decoded is not None:
            img = decoded
    return img


def augment(img: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if random.random() < 0.5:
        img, boxes = rotate90(img, boxes, random.randint(1, 3))
    if random.random() < 0.5:
        img, boxes = flip(img, boxes, horizontal=True)
    if random.random() < 0.2:
        img, boxes = flip(img, boxes, horizontal=False)
    if random.random() < 0.3:
        img, boxes = zoom_crop(img, boxes)
    return photometric(img), boxes


class VocDataset(torch.utils.data.Dataset):
    """Images + boxes. With train=True each image is randomly augmented."""

    def __init__(self, items: list[tuple[str, str]], train: bool):
        self.items, self.train = items, train

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        img_path, xml_path = self.items[i]
        img = imread(img_path)
        boxes, labels = read_voc(xml_path)
        if self.train:
            img, boxes = augment(img, boxes)
        target = {"boxes": torch.from_numpy(np.ascontiguousarray(boxes, dtype=np.float32)).reshape(-1, 4),
                  "labels": torch.from_numpy(labels)}
        return to_tensor(img), target


def collate(batch):
    """Detectors take a list of images (different sizes), not one stacked tensor."""
    return tuple(zip(*batch))


# ---------- model ----------

def build_model(arch: str = ARCH, pretrained: bool = True) -> torch.nn.Module:
    """Faster R-CNN with its last layer swapped for our classes.

    pretrained=False builds the empty network, for loading our own saved weights
    (nothing is downloaded).
    """
    model = ARCHITECTURES[arch](
        weights="DEFAULT" if pretrained else None, weights_backbone=None,
        trainable_backbone_layers=TRAINABLE_LAYERS if pretrained else None,
        min_size=TRAIN_SIZES, max_size=MAX_SIZE)
    old = model.roi_heads.box_predictor
    assert isinstance(old, FastRCNNPredictor)
    model.roi_heads.box_predictor = FastRCNNPredictor(old.cls_score.in_features, len(CLASSES))
    return model


def save_weights(model: torch.nn.Module, path: str, arch: str, info: dict) -> None:
    """Weights plus what's needed to rebuild the same network (BarcodeDetector)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"arch": arch, "classes": CLASSES, "train_sizes": TRAIN_SIZES, "max_size": MAX_SIZE,
                "state_dict": model.state_dict(), **info}, path)


# ---------- measuring ----------

def average_precision(detections: list[tuple[int, float, bool]], n_true: int) -> float:
    """Area under the precision-recall curve (all-point, as in Pascal VOC 2010+).

    detections: (image index, score, matched a true box?) for one class, all images.
    """
    if n_true == 0:
        return float("nan")
    hits = np.array([m for _, _, m in sorted(detections, key=lambda d: -d[1])], dtype=float)
    if not len(hits):
        return 0.0
    tp = np.cumsum(hits)
    recall = tp / n_true
    precision = tp / np.arange(1, len(hits) + 1)
    # Make precision never increase as recall grows, then sum the area in steps.
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    recall = np.concatenate([[0.0], recall])
    return float(np.sum((recall[1:] - recall[:-1]) * precision))


def match(pred_boxes: torch.Tensor, true_boxes: torch.Tensor) -> list[bool]:
    """For each prediction (most confident first): does it claim a true box?
    Each true box can be claimed once, as in VOC/COCO scoring."""
    if len(true_boxes) == 0 or len(pred_boxes) == 0:
        return [False] * len(pred_boxes)
    iou = box_iou(pred_boxes, true_boxes)
    taken, out = set(), []
    for row in iou:
        j = int(row.argmax())
        ok = row[j] >= IOU_MATCH and j not in taken
        if ok:
            taken.add(j)
        out.append(bool(ok))
    return out


@torch.no_grad()
def evaluate_boxes(model: torch.nn.Module, items: list[tuple[str, str]], dev: torch.device) -> dict:
    """AP@0.5 per class, and recall at MIN_SCORE, on items (no augmentation)."""
    model.eval()
    dets = {c: [] for c in range(1, len(CLASSES))}
    n_true = {c: 0 for c in dets}
    found = {c: 0 for c in dets}
    for i, (img_path, xml_path) in enumerate(items):
        true_boxes, true_labels = read_voc(xml_path)
        out = model([to_tensor(imread(img_path)).to(dev)])[0]
        for c in dets:
            t = torch.from_numpy(true_boxes[true_labels == c])
            keep = out["labels"].cpu() == c
            boxes, scores = out["boxes"].cpu()[keep], out["scores"].cpu()[keep]
            matched = match(boxes, t)
            dets[c] += [(i, float(s), m) for s, m in zip(scores, matched)]
            n_true[c] += len(t)
            found[c] += sum(m for s, m in zip(scores, matched) if s >= MIN_SCORE)
    result = {}
    for c, name in enumerate(CLASSES[1:], start=1):
        result[f"{name}_ap50"] = average_precision(dets[c], n_true[c])
        result[f"{name}_recall"] = found[c] / n_true[c] if n_true[c] else float("nan")
        result[f"{name}_n"] = n_true[c]
    return result


# ---------- training ----------

def split_val(items: list[tuple[str, str]], fraction: float, seed: int,
              group_of=lambda item: item[0]) -> tuple[list, list]:
    """(train, validation) split, keeping each group whole on one side.

    group_of(item) names an item's group; by default every image is its own.
    evaluate_detector.py groups the EAN images by product, because the photos
    of one product are near-identical frames: one in training and one in
    validation would make validation look better than it is.
    """
    groups: dict[str, list] = {}
    for item in items:
        groups.setdefault(group_of(item), []).append(item)
    names = sorted(groups)
    random.Random(seed).shuffle(names)
    train_items, val_items = [], []
    for name in names:
        (val_items if len(val_items) < fraction * len(items) else train_items).extend(groups[name])
    return train_items, val_items


def selection_score(row: dict, epoch: int) -> float:
    """Which epoch's weights to keep: the best barcode AP on validation (the class
    that matters here); if validation had no barcodes, the mean AP; with no
    validation at all, simply the latest epoch."""
    aps = [row[k] for k in row if k.endswith("_ap50") and not math.isnan(row[k])]
    if "barcode_ap50" in row and not math.isnan(row["barcode_ap50"]):
        return row["barcode_ap50"]
    return sum(aps) / len(aps) if aps else float(epoch)


def train(items: list[tuple[str, str]], out_path: str, val_items: list[tuple[str, str]] | None = None,
          epochs: int = EPOCHS, batch_size: int = BATCH_SIZE, arch: str = ARCH, workers: int = 8,
          seed: int = 0, resume: bool = False) -> None:
    """Fine-tune the detector on items, keep the epoch with the best barcode AP
    on val_items, and save it to out_path. With no val_items the last epoch is kept.

    Also writes <out_path minus .pt>_last.pt (resumable) and _log.csv.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dev = device()
    stem = os.path.splitext(out_path)[0]
    last_path, log_path = stem + "_last.pt", stem + "_log.csv"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    model = build_model(arch).to(dev)
    loader = torch.utils.data.DataLoader(
        VocDataset(items, train=True), batch_size=batch_size, shuffle=True, collate_fn=collate,
        num_workers=workers, pin_memory=dev.type == "cuda", persistent_workers=workers > 0,
        drop_last=len(items) > batch_size)
    params = [p for p in model.parameters() if p.requires_grad]
    lr = BASE_LR * batch_size / 16
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=WEIGHT_DECAY)
    steps, warmup = epochs * len(loader), WARMUP_EPOCHS * len(loader)
    # Linear warmup from 1% of lr (a pretrained model knocked by big first steps
    # forgets what it knew), then a cosine fall to 0 by the last step.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (
        0.01 + 0.99 * s / warmup if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup)))))
    # bf16 autocast: matrix maths in 16-bit on the GPU, ~2x faster. No loss
    # scaling needed (bf16 keeps float32's range); torchvision's box ops cast
    # back to float32 where they need it.
    use_amp = dev.type == "cuda"

    start, best = 0, -1.0
    if not resume and os.path.exists(log_path):
        os.remove(log_path)                        # a fresh run starts a fresh log
    if resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start, best = ck["epoch"], ck["best"]
        print(f"  resuming after epoch {start} (best barcode AP so far {best:.3f})")

    print(f"  {len(items)} training images, {len(val_items or [])} validation, "
          f"{arch}, batch {batch_size}, lr {lr:.4f}, device {dev}{', bf16' if use_amp else ''}")
    for epoch in range(start, epochs):
        model.train()
        t0, total = time.perf_counter(), 0.0
        for images, targets in loader:
            images = [im.to(dev, non_blocking=True) for im in images]
            targets = [{k: v.to(dev, non_blocking=True) for k, v in t.items()} for t in targets]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                losses = model(images, targets)              # train mode: returns the losses
                loss = torch.stack(list(losses.values())).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item()
        row = {"epoch": epoch + 1, "loss": round(total / len(loader), 4),
               "seconds": round(time.perf_counter() - t0, 1)}
        if val_items:
            row |= {k: round(v, 4) if isinstance(v, float) else v
                    for k, v in evaluate_boxes(model, val_items, dev).items()}
        score = selection_score(row, epoch)
        if score > best:
            best = score
            save_weights(model, out_path, arch, {"epoch": epoch + 1, "val": row})
        row["best"] = round(best, 4)
        print("  " + "  ".join(f"{k} {v}" for k, v in row.items()), flush=True)
        with open(log_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if f.tell() == 0:
                w.writeheader()
            w.writerow(row)
        torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch + 1, "best": best}, last_path)


# ---------- using a trained model ----------

class BarcodeDetector:
    """Loads a trained model once; detect() and crops() then run on any photo."""

    def __init__(self, path: str = MODEL_PATH):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck["classes"] != CLASSES:
            raise ValueError(f"{path} was trained on classes {ck['classes']}, expected {CLASSES}")
        self.dev = device()
        self.model = build_model(ck["arch"], pretrained=False)
        self.model.load_state_dict(ck["state_dict"])
        self.model.to(self.dev).eval()

    @torch.no_grad()
    def detect(self, bgr: np.ndarray, min_score: float = MIN_SCORE) -> list[tuple[np.ndarray, str, float]]:
        """[(box x1,y1,x2,y2, class name, score), ...], most confident first."""
        out = self.model([to_tensor(bgr).to(self.dev)])[0]
        return [(b.cpu().numpy(), CLASSES[int(lab)], float(s))
                for b, lab, s in zip(out["boxes"], out["labels"], out["scores"]) if s >= min_score]

    def crops(self, bgr: np.ndarray, detections=None, max_crops: int = MAX_CROPS) -> list[np.ndarray]:
        """Grayscale crops around the most confident detections, with a margin."""
        h, w = bgr.shape[:2]
        out = []
        for (x1, y1, x2, y2), _, _ in (self.detect(bgr) if detections is None else detections)[:max_crops]:
            mx, my = CROP_MARGIN * (x2 - x1), CROP_MARGIN * (y2 - y1)
            x1, y1 = int(max(0, x1 - mx)), int(max(0, y1 - my))
            x2, y2 = int(min(w, x2 + mx)), int(min(h, y2 + my))
            if x2 - x1 >= 8 and y2 - y1 >= 8:
                out.append(cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY))
        return out


def main() -> None:
    p = argparse.ArgumentParser(description="Train one barcode/QR detector on the whole dataset.")
    p.add_argument("--out", default=MODEL_PATH)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--arch", default=ARCH, choices=list(ARCHITECTURES))
    p.add_argument("--workers", type=int, default=8, help="data-loading processes")
    p.add_argument("--val-fraction", type=float, default=VAL_FRACTION,
                   help="held out to pick the best epoch; 0 = train on everything, keep the last epoch")
    p.add_argument("--resume", action="store_true", help="continue from <out>_last.pt")
    args = p.parse_args()

    items = dataset_items()
    # Group near-identical photos so validation isn't flattered (needs the
    # checked EAN truth.csv; without it every image is its own group).
    truth_path = os.path.join(EAN_DIR, "truth.csv")
    groups = product_groups(items, checked_truth() if os.path.exists(truth_path)
                            else pd.DataFrame(columns=["filename", "true_serial"]))
    train_items, val_items = split_val(items, args.val_fraction, seed=0,
                                       group_of=lambda item: groups[item_name(item)])
    print(f"Training on {DATASET_DIR}: {len(items)} images")
    train(train_items, args.out, val_items, epochs=args.epochs, batch_size=args.batch_size,
          arch=args.arch, workers=args.workers, resume=args.resume)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()

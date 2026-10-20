"""Stage 4: find and crop the label(s) in each photo with classical OpenCV.

Run with:  python label_finder.py   (from the repo folder, with the venv active)

Why not just threshold for a bright rectangle? Because our labels come in both
polarities: white label on a black mouse (ASUS), and dark label with white text
on a dark chassis (laptops, Razer). What every label DOES have is lots of
sharp, high-contrast detail packed close together - text and barcodes. So:

  1. shrink      work on a 1000 px tall copy: fast, and every size below means
                 the same thing whatever resolution the phone used
  2. gradient    morphological gradient = dilate - erode. Bright where pixel
                 values change sharply (character edges), dark on flat areas.
                 Works for dark-on-light AND light-on-dark text.
  3. edges       keep only STRONG edges (gradient > EDGE_THRESHOLD). Printed
                 text is near-black next to near-white, so its edges are
                 stronger than most texture.
  4. density     for every pixel, what fraction of its 21x21 neighbourhood is
                 edge? Text is DENSE edges; texture like the mousepad weave
                 leaves scattered specks that are bright but sparse.
                 Keep pixels where density > DENSITY_THRESHOLD.
  5. close       "closing" (dilate then erode) joins neighbouring text lines
                 into one solid blob per label
  6. contours    outline each blob and reject the ones that aren't label-like
                 (too small, touching 2+ frame edges, too few edges inside).
                 Keep the biggest MAX_LABELS of what's left.
  7. crop        cut each kept box out of the FULL-resolution photo, rotated
                 upright, so later stages get every pixel the camera captured

All thresholds were picked by looking at measurements on our 12 photos (see the
comments on each). That's the same set we measure accuracy on, so the numbers
are optimistic; more photos would be needed for an honest held-out test.

Outputs in output/labels/:
  debug/<photo>_<step>.jpg   the image after each step, to see where it goes wrong
  boxes/<photo>.jpg          the photo with the kept boxes drawn and numbered
  crops/<photo>_<n>.png      the crops themselves (PNG = no extra JPEG blur)
  results.csv                per photo: number of crops, barcode on crops, timings
"""

import glob
import os
import time

import cv2
import numpy as np
import pandas as pd

from barcode_reader import read_barcode
from baseline import found

PHOTOS_DIR = "photos"
OUT_DIR = os.path.join("output", "labels")

WORK_HEIGHT = 1000
# Step 3: why a fixed number and not Otsu's automatic threshold? On our photos
# Otsu chose 32-66, because most of each photo is texture, so it separated
# "texture vs flat" instead of "text vs everything". Text edges were mostly > 90.
EDGE_THRESHOLD = 90
# Step 4: measured inside real labels the edge fraction was ~0.44-0.64; on
# mousepad, vents and chevrons ~0.3. 0.40 sits between. At 0.25 the mousepad
# joined up with the mouse outline and swallowed Mouse_1's label into one blob.
DENSITY_WINDOW = 21
DENSITY_THRESHOLD = 0.40
CLOSE_KERNEL = 9
MIN_AREA_FRAC = 0.003     # ignore blobs smaller than 0.3% of the image
MAX_BORDERS = 1           # a blob touching 2+ frame edges is background (vents,
                          # mousepad, table edge), not a label sitting in the scene
MAX_LABELS = 3            # keep at most this many, biggest first

# OpenCV's rotated rectangle, as returned by cv2.minAreaRect:
# ((centre x, centre y), (width, height), angle in degrees)
RotatedRect = tuple[tuple[float, float], tuple[float, float], float]


def find_label_boxes(gray: np.ndarray, debug_prefix: str | None = None) -> list[RotatedRect]:
    """Return rotated rectangles ((cx, cy), (w, h), angle) in FULL-res coordinates."""
    scale = WORK_HEIGHT / gray.shape[0]
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    H, W = small.shape

    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    edges = (grad > EDGE_THRESHOLD).astype(np.float32)          # 1.0 = strong edge
    # blur() averages each 21x21 window, so on a 0/1 image it gives "fraction of edges".
    density = cv2.blur(edges, (DENSITY_WINDOW, DENSITY_WINDOW))
    dense = (density > DENSITY_THRESHOLD).astype(np.uint8) * 255
    blobs = cv2.morphologyEx(dense, cv2.MORPH_CLOSE, np.ones((CLOSE_KERNEL, CLOSE_KERNEL), np.uint8))

    if debug_prefix:
        for name, img in [("1_small", small), ("2_gradient", grad),
                          ("3_edges", (edges * 255).astype(np.uint8)),
                          ("4_density", (density * 255).astype(np.uint8)),
                          ("5_blobs", blobs)]:
            cv2.imwrite(f"{debug_prefix}_{name}.jpg", img)

    contours, _ = cv2.findContours(blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < MIN_AREA_FRAC * H * W:
            continue
        borders = (x <= 2) + (y <= 2) + (x + w >= W - 2) + (y + h >= H - 2)
        if borders > MAX_BORDERS:
            continue
        # Edge fraction inside the whole blob (closing may have padded it with flat area).
        inside = np.zeros_like(blobs)
        cv2.drawContours(inside, [c], -1, 255, thickness=-1)
        if edges[inside > 0].mean() < DENSITY_THRESHOLD:
            continue
        kept.append((w * h, cv2.minAreaRect(c)))   # smallest rotated box around the blob

    kept.sort(key=lambda k: k[0], reverse=True)
    return [((cx / scale, cy / scale), (w / scale, h / scale), a)
            for _, ((cx, cy), (w, h), a) in kept[:MAX_LABELS]]


def crop_rotated(img: np.ndarray, rect: RotatedRect, pad: float = 0.08) -> np.ndarray:
    """Cut a rotated rectangle out of img and return it upright (long side horizontal).

    pad grows the box by 8% each way so characters on the edge aren't clipped
    (the density blur shrinks blobs slightly at their edges).
    """
    (cx, cy), (w, h), angle = rect
    w, h = w * (1 + 2 * pad), h * (1 + 2 * pad)
    # Rotate the whole image around the box centre so the box becomes axis-aligned...
    m = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)
    # ...then it's a plain slice. getRectSubPix copes with boxes poking past the edge.
    crop = cv2.getRectSubPix(rotated, (int(w), int(h)), (cx, cy))
    # minAreaRect's angle convention can leave a wide label standing on its end;
    # turn it so the long side is horizontal. Text may still be upside down -
    # Stage 5 will deal with that.
    if crop.shape[0] > crop.shape[1]:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    return crop


def main() -> None:
    for sub in ("debug", "boxes", "crops"):
        os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)
    for old in glob.glob(os.path.join(OUT_DIR, "crops", "*.png")):
        os.remove(old)   # a photo may produce fewer crops than last run; don't leave stale ones

    truth = pd.read_csv("truth.csv")
    rows = []
    for filename, true_serial in zip(truth.filename, truth.true_serial):
        stem = os.path.splitext(filename)[0]
        img = cv2.imread(os.path.join(PHOTOS_DIR, filename))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        t0 = time.perf_counter()
        boxes = find_label_boxes(gray, debug_prefix=os.path.join(OUT_DIR, "debug", stem))
        crops = [crop_rotated(img, r) for r in boxes]
        find_s = time.perf_counter() - t0

        vis = img.copy()
        for i, (rect, crop) in enumerate(zip(boxes, crops)):
            pts = cv2.boxPoints(rect).astype(np.int32)     # 4 corners of the rotated box
            cv2.polylines(vis, [pts], True, (0, 255, 0), 12)
            cv2.putText(vis, str(i), tuple(pts[1]), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 10)
            cv2.imwrite(os.path.join(OUT_DIR, "crops", f"{stem}_{i}.png"), crop)
        cv2.imwrite(os.path.join(OUT_DIR, "boxes", f"{stem}.jpg"), vis)

        # Side check: does the Stage 3 barcode reader still work on the crops,
        # and is it faster than on the whole photo?
        t0 = time.perf_counter()
        values = []
        for crop in crops:
            v, _ = read_barcode(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
            values += v
        barcode_s = time.perf_counter() - t0

        rows.append({
            "filename": filename,
            "n_crops": len(crops),
            "barcode_on_crops_ok": any(found(true_serial, v) for v in values),
            "find_s": round(find_s, 3),
            "barcode_on_crops_s": round(barcode_s, 3),
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "results.csv"), index=False)
    print(df.to_string(index=False))
    n = len(df)
    print(f"\nPhotos with >=1 crop:        {(df.n_crops > 0).sum()}/{n}")
    print(f"Barcode correct on crops:    {df.barcode_on_crops_ok.sum()}/{n}")
    stage3 = os.path.join("output", "barcode", "results.csv")
    if os.path.exists(stage3):
        s3 = pd.read_csv(stage3)
        print(f"Barcode correct, full photo: {s3.barcode_ok.sum()}/{len(s3)}  (Stage 3)")
        print(f"Barcode time: crops {df.barcode_on_crops_s.mean():.2f}s vs "
              f"full photo {s3.seconds.mean():.2f}s per photo")
    print(f"Label finding time:          {df.find_s.mean():.2f}s per photo")


if __name__ == "__main__":
    main()

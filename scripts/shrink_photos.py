"""Shrink oversized photos in photos/ so every stage runs faster.

Run with:  python -m scripts.shrink_photos   (from the repo folder, with the venv active)

Every step (ZBar, OpenCV, Tesseract) works on decoded pixels, so speed depends
on the number of pixels, not the file size. Measured once on our 26 photos with
the full pipeline (validate.py's steps), resizing in memory:

  long side   accepted  wrong  mean / max seconds per photo
  original       17       0      1.97 / 9.81      (product photos are 8064x4536)
  4032 px        18       0      1.13 / 3.32      <- MAX_SIDE
  3024 px        18       0      0.90 / 2.14      but loses Mouse_2
  2016 px        15       0      0.54 / 1.47

4032 px is what the phone saves at its standard setting, and what the serial-
label photos already are, so those are left alone. Going smaller started to
lose photos. JPEG quality 75 lost Mouse_1, so photos are saved at 95: the speed
comes from the pixel count, and quality costs only disk space.

The original of every photo that gets shrunk is MOVED to photos_original/
first (photos aren't in git, so they couldn't be recovered otherwise). Running
it again does nothing: shrunk photos are already small enough.
"""

import os
import shutil

import cv2

PHOTOS_DIR = "photos"
BACKUP_DIR = "photos_original"
MAX_SIDE = 4032
JPEG_QUALITY = 95
EXTENSIONS = (".jpg", ".jpeg", ".png")


def main() -> None:
    os.makedirs(BACKUP_DIR, exist_ok=True)
    shrunk = 0
    for name in sorted(os.listdir(PHOTOS_DIR)):
        path = os.path.join(PHOTOS_DIR, name)
        if not name.lower().endswith(EXTENSIONS) or not os.path.isfile(path):
            continue
        # imread applies the EXIF rotation, and imwrite saves no EXIF, so the
        # new file looks exactly as the pipeline saw the old one. (This also
        # drops the photo's metadata, e.g. GPS location.)
        img = cv2.imread(path)
        if img is None:
            print(f"Could not read {name}, skipping")
            continue
        h, w = img.shape[:2]
        if max(h, w) <= MAX_SIDE:
            continue
        backup = os.path.join(BACKUP_DIR, name)
        if os.path.exists(backup):
            print(f"{name}: {backup} already exists, not overwriting it; skipping")
            continue
        scale = MAX_SIDE / max(h, w)
        # INTER_AREA averages the pixels each new pixel covers: the standard
        # choice for shrinking, it doesn't alias fine bars into false patterns.
        small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        before = os.path.getsize(path)
        shutil.move(path, backup)
        params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY] if name.lower().endswith((".jpg", ".jpeg")) else []
        if not cv2.imwrite(path, small, params):
            shutil.move(backup, path)   # put the original back rather than lose the photo
            raise OSError(f"could not write {path}")
        shrunk += 1
        print(f"{name}: {w}x{h} -> {small.shape[1]}x{small.shape[0]}, "
              f"{before / 1e6:.1f} MB -> {os.path.getsize(path) / 1e6:.1f} MB")
    print(f"\nShrunk {shrunk} photo(s); originals are in {BACKUP_DIR}/")


if __name__ == "__main__":
    main()

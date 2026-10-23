"""Make a small dataset of SYNTHETIC label photos, so the whole pipeline can run
where the real photos can't go (CI, a fresh clone, a Docker image).

Run with:  python -m scripts.make_synthetic_dataset [folder]   (default photos/synthetic)
Then:      python -m label_reader.validate photos/synthetic   (or any other stage)

The real photos and truth.csv hold the serial numbers of our own devices, so they
are not in the repo. This draws look-alikes instead, with FAKE values only:
  - serial labels: a white sticker on a dark device, a model line, a Code128
    barcode of the serial and "S/N: <serial>" printed under it
  - product shots: an EAN-13 with the digits under it on a coloured box; the
    numbers start with 2 (in-store use, never a real product)
  - mismatches: a serial label whose printed S/N is a different serial from
    its barcode. The pipeline must never accept these (barcode and OCR disagree)
Each photo is rotated a little, blurred and given sensor noise, and everything
comes from a fixed random seed, so the same files come out every run.

It writes <folder>/*.png and <folder>/truth.csv with a `checked` column (all
"yes"), the format every stage accepts as a dataset folder, and a `kind` column
(serial / product / mismatch) for tests/test_pipeline_e2e.py.
"""

import os
import random
import sys

import cv2
import numpy as np
import pandas as pd
import zxingcpp
from PIL import Image, ImageDraw, ImageFont

DEFAULT_DIR = os.path.join("photos", "synthetic")
SEED = 1
SERIAL_LABELS = 6
PRODUCTS = 6
MISMATCHES = 2
# No O/0, I/1 etc. look-alikes in the serials: the barcode decides the value,
# and an OCR mix-up there is a different test (validate.CONFUSABLE).
SERIAL_CHARS = "ACEFHJKMNPRTUVWXY345789"
# Drawn at about the size a label has in a real 4032 x 2268 photo: text ~80 px tall.
S = 2


def ean13(first12: str) -> str:
    total = sum(int(d) * (3 if i % 2 else 1) for i, d in enumerate(first12))
    return first12 + str((10 - total % 10) % 10)


def font(size: int) -> ImageFont.FreeTypeFont:
    # Pillow's built-in scalable font: the same on every OS, nothing to install.
    return ImageFont.load_default(size=size)


def barcode_image(text: str, fmt, scale: int) -> Image.Image:
    img = np.asarray(zxingcpp.create_barcode(text, fmt).to_image(scale=scale, add_quiet_zones=True))
    return Image.fromarray(img).convert("L")


def fake_serial(rng: random.Random) -> str:
    return "FK" + "".join(rng.choice(SERIAL_CHARS) for _ in range(10))


def serial_label(rng: random.Random, serial: str, printed: str | None = None) -> Image.Image:
    """Packed with small print like a real label: Stage 4 finds labels as areas
    dense with edges, so a sparse sticker would be found line by line instead."""
    bars = barcode_image(serial, zxingcpp.BarcodeFormat.Code128, scale=3 * S)
    bars = bars.resize((bars.width, 90 * S))
    w, h = max(bars.width + 40 * S, 560 * S), 300 * S
    label = Image.new("L", (w, h), 255)
    draw = ImageDraw.Draw(label)
    small = font(22 * S)
    draw.text((20 * S, 12 * S), f"Model: FAKE-{rng.randint(1000, 9999)}   P/N: 90FK-{rng.randint(10000, 99999)}",
              fill=0, font=small)
    draw.text((20 * S, 40 * S), "Rating: 19V 3.42A   Input: 100-240V 50/60Hz", fill=0, font=small)
    label.paste(bars, ((w - bars.width) // 2, 70 * S))
    draw.text((20 * S, 164 * S), f"S/N: {printed or serial}", fill=0, font=font(36 * S))
    draw.text((20 * S, 210 * S), "For pipeline testing only, not a real device", fill=0, font=small)
    draw.text((20 * S, 238 * S), "FCC ID: FAKE0000   Class B   Made in Testland", fill=0, font=small)
    return label


def product_box(number: str) -> Image.Image:
    bars = barcode_image(number, zxingcpp.BarcodeFormat.EAN13, scale=4 * S)
    w, h = bars.width + 80 * S, bars.height + 140 * S
    box = Image.new("L", (w, h), 255)
    box.paste(bars, (40 * S, 40 * S))
    ImageDraw.Draw(box).text((40 * S, bars.height + 60 * S), " ".join([number[0], number[1:7], number[7:]]),
                             fill=0, font=font(44 * S))
    return box


def photo(rng: random.Random, sticker: Image.Image, background: int) -> np.ndarray:
    """The sticker on a plain surface, rotated a little, blurred, with noise."""
    canvas = Image.new("L", (3200, 2400), background)
    x = rng.randint(200, canvas.width - sticker.width - 200)
    y = rng.randint(200, canvas.height - sticker.height - 200)
    canvas.paste(sticker, (x, y))
    img = np.asarray(canvas)
    m = cv2.getRotationMatrix2D((x + sticker.width / 2, y + sticker.height / 2), rng.uniform(-6, 6), 1.0)
    img = cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderValue=background)
    img = cv2.GaussianBlur(img, (5, 5), rng.uniform(0.5, 1.2))
    noise = np.random.default_rng(rng.randint(0, 2**31)).normal(0, 4, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def make(folder: str = DEFAULT_DIR) -> pd.DataFrame:
    rng = random.Random(SEED)
    os.makedirs(folder, exist_ok=True)
    rows = []
    for i in range(1, SERIAL_LABELS + 1):
        serial = fake_serial(rng)
        name = f"Serial_{i}.png"
        cv2.imwrite(os.path.join(folder, name), photo(rng, serial_label(rng, serial), background=rng.randint(30, 70)))
        rows.append({"filename": name, "true_serial": serial, "checked": "yes", "kind": "serial"})
    for i in range(1, PRODUCTS + 1):
        number = ean13("2" + "".join(rng.choice("0123456789") for _ in range(11)))
        name = f"Product_{i}.png"
        cv2.imwrite(os.path.join(folder, name), photo(rng, product_box(number), background=rng.randint(90, 160)))
        rows.append({"filename": name, "true_serial": number, "checked": "yes", "kind": "product"})
    for i in range(1, MISMATCHES + 1):
        serial, printed = fake_serial(rng), fake_serial(rng)
        name = f"Mismatch_{i}.png"
        cv2.imwrite(os.path.join(folder, name),
                    photo(rng, serial_label(rng, serial, printed), background=rng.randint(30, 70)))
        rows.append({"filename": name, "true_serial": serial, "checked": "yes", "kind": "mismatch"})
    truth = pd.DataFrame(rows)
    truth.to_csv(os.path.join(folder, "truth.csv"), index=False)
    return truth


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    truth = make(out)
    print(f"Wrote {len(truth)} synthetic photos and truth.csv to {out}/")

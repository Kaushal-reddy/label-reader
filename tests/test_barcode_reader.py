"""Stage 3 reader on synthetic barcodes: zxing-cpp first, ZBar when zxing-cpp reads nothing.

The barcodes are drawn by zxing-cpp itself, with FAKE numbers (prefix 2 = in-store
use, never a real product), so these tests need no photos.
"""

import cv2
import numpy as np
import zxingcpp

from label_reader import barcode_reader
from label_reader.barcode_reader import read_barcode


def check_digit(digits: str) -> str:
    """EAN-13 / UPC-A check digit: weights 3,1,3,1,... from the right."""
    total = sum(int(d) * (3 if i % 2 == 0 else 1) for i, d in enumerate(reversed(digits)))
    return str((10 - total % 10) % 10)


FAKE_EAN = "200000000001" + check_digit("200000000001")
FAKE_UPCA = "20000000002" + check_digit("20000000002")      # 12 digits
FAKE_SERIAL = "FAKESN0001AB"


def drawn(text: str, fmt) -> np.ndarray:
    """A grayscale photo-like image: the barcode on a white page with a margin."""
    img = np.asarray(zxingcpp.create_barcode(text, fmt).to_image(scale=4))
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.copyMakeBorder(img, 200, 200, 200, 200, cv2.BORDER_CONSTANT, value=255)


def test_blank_image_reads_nothing():
    blank = np.full((400, 600), 255, dtype=np.uint8)
    assert read_barcode(blank) == ([], None, [], {})


def test_product_barcode_is_read_by_zxing_and_confirmed_by_every_variant():
    r = read_barcode(drawn(FAKE_EAN, zxingcpp.BarcodeFormat.EAN13))
    assert r.values == [FAKE_EAN]
    assert r.variant == "zxing:original"
    assert r.agreeing == {FAKE_EAN: len(barcode_reader.CONFIRM_VARIANTS)}
    assert r.conflicts == []


def test_upca_comes_back_as_13_digits_like_zbar():
    r = read_barcode(drawn(FAKE_UPCA, zxingcpp.BarcodeFormat.UPCA))
    assert r.values == ["0" + FAKE_UPCA]


def test_serial_barcode():
    r = read_barcode(drawn(FAKE_SERIAL, zxingcpp.BarcodeFormat.Code128))
    assert r.values == [FAKE_SERIAL]


def test_zbar_reads_when_zxing_reads_nothing(monkeypatch):
    monkeypatch.setattr(barcode_reader, "zxing_decode", lambda img: [])
    r = read_barcode(drawn(FAKE_SERIAL, zxingcpp.BarcodeFormat.Code128))
    assert r.values == [FAKE_SERIAL]
    assert r.variant == "original"          # ZBar's variant names, as before


def test_other_product_number_from_another_variant_is_a_conflict(monkeypatch):
    other = "200000000003" + check_digit("200000000003")
    reads = {0: [FAKE_EAN], 1: [FAKE_EAN, other], 2: [FAKE_EAN]}
    calls = iter(range(3))
    monkeypatch.setattr(barcode_reader, "zxing_decode", lambda img: reads[next(calls)])
    r = read_barcode(np.full((10, 10), 255, dtype=np.uint8))
    assert r.values == [FAKE_EAN]
    assert r.conflicts == [other]
    assert r.agreeing == {FAKE_EAN: 3}

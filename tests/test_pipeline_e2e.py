"""The whole pipeline (Stages 3-6) on synthetic label photos with FAKE serials.

scripts/make_synthetic_dataset.py draws the photos, so this runs anywhere the
native dependencies (Tesseract, ZBar) are installed, including CI, without the
private photos. It checks the properties that matter more than any one rate:
nothing is ever accepted wrongly, and a label whose printed serial disagrees
with its barcode always goes to review.
"""

import pandas as pd
import pytest

from label_reader import validate
from label_reader.native_libs import find_tesseract
from scripts.make_synthetic_dataset import make

pytestmark = pytest.mark.skipif(find_tesseract() is None, reason="Tesseract is not installed")


@pytest.fixture(scope="module")
def results(tmp_path_factory) -> pd.DataFrame:
    """validate.py's per-photo decisions joined with the truth (run once for all tests)."""
    root = tmp_path_factory.mktemp("e2e")
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(root)                  # validate writes to output/<folder>/ under the cwd
        truth = make("synthetic")
        validate.main("synthetic")
        decided = pd.read_csv(root / "output" / "synthetic" / "validation.csv", dtype=str, keep_default_na=False)
    return decided.merge(truth, on="filename")


def test_every_photo_gets_a_decision(results):
    assert len(results) == 14
    assert set(results.status) <= {"accepted", "needs_review"}


def test_nothing_is_accepted_wrongly(results):
    accepted = results[results.status == "accepted"]
    wrong = accepted[accepted.serial != accepted.true_serial]
    assert wrong.empty, wrong[["filename", "serial", "true_serial"]].to_string()


def test_label_whose_printed_serial_disagrees_is_never_accepted(results):
    mismatches = results[results.kind == "mismatch"]
    assert len(mismatches) == 2
    assert (mismatches.status == "needs_review").all()


def test_every_product_barcode_is_accepted_on_its_own(results):
    products = results[results.kind == "product"]
    assert (products.status == "accepted").all(), products[["filename", "reason"]].to_string()
    assert (products.source == "barcode").all()


def test_most_serial_labels_are_confirmed_by_ocr(results):
    # Tesseract versions differ between OSes, so allow one or two misses.
    serials = results[results.kind == "serial"]
    assert (serials.status == "accepted").sum() >= 4, serials[["filename", "status", "reason"]].to_string()


def test_the_barcode_is_the_best_guess_even_when_sent_to_review(results):
    assert (results.serial == results.true_serial).all()

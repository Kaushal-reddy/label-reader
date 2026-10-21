"""Stage 6: validation. Turn barcode values + OCR text into one decision per photo.

Run with:  python validate.py   (from the repo folder, with the venv active)

For each photo:
  1. barcode   Stage 3 reader -> pull serial-shaped values out of what it decoded
               (e.g. the ASUS QR codes hold a URL ending in the serial)
  2. OCR       Stage 5 reader -> pull serial candidates out of the text,
               preferring ones written right after "SN" / "S/N"
  3. pattern   a serial must look like one: 10-20 letters/digits, with at least
               one letter and at least two digits
  4. decide    ACCEPT when barcode and OCR independently agree, allowing for
               characters OCR is known to confuse (O/0, A/4, ...). Then output
               the BARCODE's characters: barcodes carry a checksum, OCR doesn't.
               Or, for a retail product barcode (EAN-13), ACCEPT the barcode
               alone if its check digit is valid, ZBar read it only one way,
               and at least two image variants read the same number.
               Anything else -> NEEDS REVIEW, with the reason and best guess.

truth.csv is NOT used to make any decision. It's only read at the very end to
print a quick check of how the decisions turned out (Stage 7 does this properly).

Outputs:
  output/results.csv       accepted photos: filename, serial, source
  output/needs_review.csv  the rest: filename, best guess, reason
  output/validation.csv    every photo with all the details (used by Stage 7)
"""

import os
import sys
import re
import time
from collections import Counter

import cv2
import pandas as pd

from baseline import load_truth, normalize, out_dir_for
from barcode_reader import read_barcode
from ocr_reader import read_ocr

PHOTOS_DIR = "photos"

# A serial: 10-20 characters, letters and digits only, at least one letter and
# at least two digits. Deliberately generic - it is NOT built from our truth
# serials, so it would also accept other manufacturers' formats. It rules out
# words ("CHINA"), short codes ("12M"), dates and pure numbers.
SERIAL_RE = re.compile(r"^(?=.*[A-Z])(?=(?:.*\d){2})[A-Z0-9]{10,20}$")

# A retail product barcode number: EAN-13. (UPC-A, the 12-digit US version,
# arrives from ZBar as EAN-13 with a leading 0.)
PRODUCT_RE = re.compile(r"^\d{13}$")

# Text written right after an "SN" / "S/N" / "S/N:" key on the label.
# [^A-Z0-9]* allows the ':' and spaces in between.
LABELLED_RE = re.compile(r"S\s*/?\s*N[^A-Z0-9]*([A-Z0-9]{10,20})")

# Characters OCR often mixes up in label fonts. Each group maps to one symbol,
# so "TCNOCV" and "TCN0CV" become identical when compared. Seen on our photos:
# O/0 (Laptop_3, Mouse_1 photos) and A/4 (Laptop_2_Glare).
CONFUSABLE = str.maketrans({"O": "0", "Q": "0", "D": "0",
                            "I": "1", "L": "1",
                            "Z": "2", "A": "4", "S": "5", "G": "6", "B": "8"})

# A product barcode accepted on its own must have been read the same by at least
# this many image variants (barcode_reader.CONFIRM_VARIANTS). One read isn't
# enough: on the 403-image EAN set, 8 single reads had a valid check digit but
# were the wrong number.
MIN_AGREEING_VARIANTS = 2


def looks_like_serial(s: str) -> bool:
    return bool(SERIAL_RE.match(s))


def check_digit_ok(code: str) -> bool:
    """True if the last digit of an EAN-13 / UPC number is the right check digit.

    Rule: from the right, skipping the check digit itself, weight the digits
    3, 1, 3, 1, ...; the check digit tops the sum up to a multiple of 10.
    ZBar already checks this for EAN-13 barcodes. It's checked again here so
    the rule doesn't depend on which barcode type delivered the number (a QR
    code holding 13 digits has no such check).
    """
    digits = [int(c) for c in code]
    total = sum(d * (3 if i % 2 == 0 else 1) for i, d in enumerate(reversed(digits[:-1])))
    return (10 - total % 10) % 10 == digits[-1]


def looks_like_product_code(s: str) -> bool:
    return bool(PRODUCT_RE.match(s)) and check_digit_ok(s)


def same_modulo_confusions(a: str, b: str) -> bool:
    """True if a and b are the same string once confusable characters are merged."""
    return a.translate(CONFUSABLE) == b.translate(CONFUSABLE)


def barcode_candidates(values: list[str]) -> list[str]:
    """Serial-shaped strings from decoded barcode values.

    A value can be the serial itself (Code128 on the mice), a product number
    (EAN-13 on retail packaging), or a URL whose last part is the serial (ASUS
    QR: https://qs.asus.com/<serial>). So we split on anything that isn't a
    letter/digit and test each piece.
    """
    out = []
    for v in values:
        if looks_like_product_code(v.strip()):
            out.append(v.strip())      # the whole value is a product number
            continue
        for piece in re.split(r"[^A-Za-z0-9]+", v.upper()):
            if looks_like_serial(piece):
                out.append(piece)
    return out


def ocr_candidates(text: str) -> tuple[list[str], list[str]]:
    """Serial candidates from OCR text: (labelled, unlabelled), most-read first.

    Labelled = written right after an SN key, which is much stronger evidence.
    Counting matters because read_ocr() returns the text of two OCR methods:
    a value both read the same way gets two votes.
    """
    labelled, unlabelled = Counter(), Counter()
    for line in text.upper().splitlines():
        for m in LABELLED_RE.finditer(line):
            if looks_like_serial(m.group(1)):
                labelled[m.group(1)] += 1
        for token in re.findall(r"[A-Z0-9]+", line):
            if looks_like_serial(token):
                unlabelled[token] += 1
    return [s for s, _ in labelled.most_common()], [s for s, _ in unlabelled.most_common()]


def decide(bc: list[str], ocr_lab: list[str], ocr_unlab: list[str],
           conflicts: list[str] = (), agreeing: dict[str, int] | None = None
           ) -> tuple[str, str, str, str]:
    """Return (status, serial, source, reason). Uses only what was read from the photo.

    conflicts = other values ZBar read from the same barcode(s) and rejected
    (barcode_reader.resolve); any conflict blocks accepting a barcode on its own.
    agreeing = value -> how many image variants read it (barcode_reader.confirm);
    a product barcode needs MIN_AGREEING_VARIANTS to be accepted on its own.
    """
    agreeing = agreeing or {}
    # 1) Agreement: any barcode candidate matches any OCR candidate.
    for b in bc:
        for o in ocr_lab + ocr_unlab:
            if b == o:
                return "accepted", b, "barcode+ocr", "barcode and OCR agree exactly"
            if same_modulo_confusions(b, o):
                return ("accepted", b, "barcode+ocr",
                        f"barcode and OCR agree except OCR look-alikes (OCR read {o})")
    # 2) Product barcode. The only text that could confirm it is the same digits
    #    printed small under the bars, and OCR can't read those (0 of 14 product
    #    photos exact). So the barcode stands alone, which is only safe when the
    #    check digit is valid, ZBar read it one way only (on a curved can it once
    #    read a second, wrong number that also had a valid check digit), AND more
    #    than one image variant read it (single wrong reads pass the other two checks).
    products = list(dict.fromkeys(b for b in bc if looks_like_product_code(b)))
    if len(products) > 1:
        return ("needs_review", products[0], "barcode",
                f"several product barcodes read: {', '.join(products)}")
    if products and conflicts:
        return ("needs_review", products[0], "barcode",
                f"product barcode {products[0]}, but ZBar also read {', '.join(conflicts)} from it")
    if products and agreeing.get(products[0], 1) < MIN_AGREEING_VARIANTS:
        return ("needs_review", products[0], "barcode",
                f"product barcode {products[0]} read by only {agreeing.get(products[0], 1)} "
                "image variant, so it can't be confirmed")
    if products:
        return ("accepted", products[0], "barcode",
                f"product barcode (EAN-13): check digit valid, read one way only, "
                f"by {agreeing[products[0]]} image variants")
    # 3) Only one source, or the two disagree -> a person should check.
    if bc and (ocr_lab or ocr_unlab):
        return ("needs_review", bc[0], "barcode",
                f"barcode {bc[0]} and OCR {(ocr_lab + ocr_unlab)[0]} disagree")
    if bc:
        return "needs_review", bc[0], "barcode", "barcode only, no OCR serial to confirm it"
    if ocr_lab:
        return "needs_review", ocr_lab[0], "ocr", "OCR only (after an SN key), no barcode to confirm it"
    if ocr_unlab:
        return "needs_review", ocr_unlab[0], "ocr", "OCR only, serial-shaped text without an SN key"
    return "needs_review", "", "", "nothing serial-shaped found"


def main(folder: str | None = None) -> None:
    photos_dir, out_dir, truth = folder or PHOTOS_DIR, out_dir_for(folder, ""), load_truth(folder)
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for filename in truth.filename:
        img = cv2.imread(os.path.join(photos_dir, filename))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        t0 = time.perf_counter()
        values, _, conflicts, agreeing = read_barcode(gray)
        t_barcode = time.perf_counter() - t0
        t0 = time.perf_counter()
        text = read_ocr(img)
        t_ocr = time.perf_counter() - t0

        bc = barcode_candidates(values)
        ocr_lab, ocr_unlab = ocr_candidates(text)
        status, serial, source, reason = decide(bc, ocr_lab, ocr_unlab, conflicts, agreeing)
        rows.append({
            "filename": filename, "status": status, "serial": serial,
            "source": source, "reason": reason,
            "barcode_candidates": " ".join(bc),
            "barcode_conflicts": " ".join(conflicts),
            "ocr_labelled": " ".join(ocr_lab),
            "ocr_unlabelled": " ".join(ocr_unlab),
            "barcode_s": round(t_barcode, 2), "ocr_s": round(t_ocr, 2),
            "total_s": round(t_barcode + t_ocr, 2),
        })
        print(f"{filename:24} {status:13} {serial:17} {reason}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "validation.csv"), index=False)
    acc = df[df.status == "accepted"]
    acc[["filename", "serial", "source"]].to_csv(os.path.join(out_dir, "results.csv"), index=False)
    df[df.status != "accepted"][["filename", "serial", "source", "reason"]] \
        .rename(columns={"serial": "best_guess"}) \
        .to_csv(os.path.join(out_dir, "needs_review.csv"), index=False)

    # Quick check against truth - AFTER all decisions are made.
    t = dict(zip(truth.filename, truth.true_serial))
    df["correct"] = [s == t[f] for f, s in zip(df.filename, df.serial)]
    rev = df[df.status != "accepted"]
    n = len(df)
    print(f"\nAccepted:      {len(acc)}/{n}  (correct: {df[df.status == 'accepted'].correct.sum()}, "
          f"wrong: {(~df[df.status == 'accepted'].correct).sum()})")
    print(f"Needs review:  {len(rev)}/{n}  (best guess already correct: {rev.correct.sum()})")
    print(f"Time per photo: {df.total_s.mean():.1f}s mean, {df.total_s.max():.1f}s max")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

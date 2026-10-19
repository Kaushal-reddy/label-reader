"""Stage 0: confirm that OpenCV, pyzbar (ZBar), pytesseract (Tesseract) and pandas work.

Run with:  .venv\\Scripts\\python.exe check_setup.py

Each check runs on its own, so one failure doesn't hide the others.
The images used here are made in memory just to prove the libraries run.
They are NOT test data and are never used for accuracy numbers.
"""

import shutil

# Where the UB-Mannheim Windows installer puts tesseract.exe. It does not add
# itself to PATH, so we fall back to this location if PATH has nothing.
TESSERACT_FALLBACK = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def check_opencv() -> str:
    import cv2
    import numpy as np

    img = np.zeros((50, 100, 3), dtype=np.uint8)   # black 100x50 colour image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # any real OpenCV call
    return f"version {cv2.__version__}, made a {gray.shape} grayscale image"


def check_pyzbar() -> str:
    # Importing pyzbar.pyzbar is the step that loads libzbar-64.dll. If the DLL
    # (or the Visual C++ runtime it needs) is missing, the error happens here.
    from pyzbar import pyzbar
    import numpy as np

    blank = np.full((100, 100), 255, dtype=np.uint8)  # plain white image
    results = pyzbar.decode(blank)
    # A blank image has no barcode, so the correct answer is an empty list.
    # Getting [] back (instead of a crash) shows the decoder really ran.
    assert results == [], f"expected no barcodes, got {results}"
    return "ZBar library loaded, decoded a blank image -> []"


def check_tesseract() -> str:
    import cv2
    import numpy as np
    import pytesseract

    # pytesseract is only a wrapper: it saves the image to a temp file, runs
    # tesseract.exe on it, and reads the text back. So it must find the .exe.
    exe = shutil.which("tesseract") or TESSERACT_FALLBACK
    pytesseract.pytesseract.tesseract_cmd = exe
    version = pytesseract.get_tesseract_version()

    # Draw clean black text on white, then ask Tesseract to read it back.
    img = np.full((80, 400), 255, dtype=np.uint8)
    cv2.putText(img, "ABC123", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 4)
    # --psm 7 tells Tesseract "the image is a single line of text".
    text = pytesseract.image_to_string(img, config="--psm 7").strip()
    assert text == "ABC123", f"expected 'ABC123', OCR returned {text!r}"
    return f"Tesseract {version} at {exe}, read back {text!r}"


def check_pandas() -> str:
    import os

    import pandas as pd

    # truth.csv isn't in the public repo (real serials), so a fresh clone only
    # has the example file. Either one proves pandas can read the format.
    path = "truth.csv" if os.path.exists("truth.csv") else "truth.example.csv"
    df = pd.read_csv(path)
    expected = ["filename", "true_serial"]
    assert list(df.columns) == expected, f"columns are {list(df.columns)}"
    return f"version {pd.__version__}, {path} has columns {expected}, {len(df)} rows"


if __name__ == "__main__":
    checks = [("OpenCV", check_opencv), ("pyzbar", check_pyzbar),
              ("pytesseract", check_tesseract), ("pandas", check_pandas)]
    failed = 0
    for name, fn in checks:
        try:
            print(f"[OK]   {name}: {fn()}")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")

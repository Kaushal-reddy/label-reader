"""Stage 0: confirm that OpenCV, pyzbar (ZBar), pytesseract (Tesseract), pandas
and matplotlib work.

Run with:  python check_setup.py   (from the repo folder, with the venv active)

Each check runs on its own, so one failure doesn't hide the others.
The images used here are made in memory just to prove the libraries run.
They are NOT test data and are never used for accuracy numbers.
"""

import native_libs  # before pyzbar: lets it find ZBar on macOS


def check_opencv() -> str:
    import cv2
    import numpy as np

    img = np.zeros((50, 100, 3), dtype=np.uint8)   # black 100x50 colour image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # any real OpenCV call
    return f"version {cv2.__version__}, made a {gray.shape} grayscale image"


def check_pyzbar() -> str:
    # Importing pyzbar.pyzbar is the step that loads the ZBar library
    # (libzbar-64.dll on Windows, libzbar.dylib / libzbar.so elsewhere). If it's
    # missing ("Unable to find zbar shared library"), the error happens here.
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
    # the tesseract program on it, and reads the text back. So it must find it.
    exe = native_libs.find_tesseract()
    assert exe, "tesseract not found on PATH or in the usual install folders (see README Setup)"
    pytesseract.pytesseract.tesseract_cmd = exe
    version = pytesseract.get_tesseract_version()

    # Draw clean black text on white, then ask Tesseract to read it back.
    img = np.full((80, 400), 255, dtype=np.uint8)
    cv2.putText(img, "ABC123", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 4)
    # --psm 7 tells Tesseract "the image is a single line of text".
    text = pytesseract.image_to_string(img, config="--psm 7").strip()
    assert text == "ABC123", f"expected 'ABC123', OCR returned {text!r}"
    return f"Tesseract {version} at {exe}, read back {text!r}"


def check_tess_direct() -> str:
    # The fast path ocr_reader.py uses: Tesseract's library loaded in-process.
    # If this fails the pipeline still works, through pytesseract, just slower.
    import cv2
    import numpy as np

    from tess_direct import find_library, load_engine

    engine = load_engine()
    assert engine is not None, "library not loaded; OCR will fall back to pytesseract (slower)"
    img = np.full((80, 400), 255, dtype=np.uint8)
    cv2.putText(img, "ABC123", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 4)
    text = " ".join(word for word, _ in engine.words(img, psm=7))
    engine.close()
    assert text == "ABC123", f"expected 'ABC123', OCR returned {text!r}"
    return f"{find_library()[0]} loaded, read back {text!r}"


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


def check_matplotlib() -> str:
    # analyze.py saves its figures as files, so use the non-interactive
    # backend: it needs no window system (works over SSH, in CI, etc.).
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    fig.canvas.draw()
    plt.close(fig)
    return f"version {matplotlib.__version__}, drew a figure"


if __name__ == "__main__":
    checks = [("OpenCV", check_opencv), ("pyzbar", check_pyzbar),
              ("pytesseract", check_tesseract), ("tess_direct", check_tess_direct),
              ("pandas", check_pandas), ("matplotlib", check_matplotlib)]
    failed = 0
    for name, fn in checks:
        try:
            print(f"[OK]   {name}: {fn()}")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")

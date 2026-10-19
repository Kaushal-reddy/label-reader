"""Call Tesseract's engine directly from Python, without starting tesseract.exe.

Why: every pytesseract call saves the image to a temp file, starts a new
tesseract.exe, loads the ~15 MB English model, reads, writes the result to
another temp file and exits. On a small text-line image that overhead was
most of the ~0.15 s per call, and a photo needs dozens of calls.

Here we load Tesseract's library (libtesseract-5.dll, which the Windows
installer already puts next to tesseract.exe) into this Python process ONCE,
keep the model in memory, and pass images straight from memory.

How: Python's built-in ctypes module can call functions inside a compiled
library (DLL), as long as we tell it each function's argument and return
types. Tesseract publishes a plain C interface for exactly this ("capi.h");
the functions used below all come from it.
"""

import ctypes
import os

import numpy as np

TESSERACT_DIR = r"C:\Program Files\Tesseract-OCR"
DLL_NAME = "libtesseract-5.dll"

# Tesseract's "page iterator level" numbers (enum TessPageIteratorLevel in capi.h).
RIL_WORD = 3


class TessEngine:
    """One loaded Tesseract engine. Create it once and reuse it for every image."""

    def __init__(self, tesseract_dir: str = TESSERACT_DIR, language: str = "eng") -> None:
        # The DLL depends on other DLLs in the same folder (Leptonica etc.).
        # Since Python 3.8, Windows doesn't search PATH for those, so we point
        # it at the folder explicitly.
        os.add_dll_directory(tesseract_dir)
        lib = ctypes.CDLL(os.path.join(tesseract_dir, DLL_NAME))
        self._declare_types(lib)
        self.lib = lib

        self.api = lib.TessBaseAPICreate()
        tessdata = os.path.join(tesseract_dir, "tessdata").encode()
        # Init3 loads the language model. Returns 0 on success.
        if lib.TessBaseAPIInit3(self.api, tessdata, language.encode()) != 0:
            raise RuntimeError(f"Tesseract could not load '{language}' from {tessdata!r}")

    @staticmethod
    def _declare_types(lib: ctypes.CDLL) -> None:
        """Tell ctypes each C function's argument and return types.

        Without this ctypes assumes every value is a 32-bit int. On 64-bit
        Windows pointers are 64 bits, so an undeclared pointer return would be
        cut in half and the program would crash.
        """
        p = ctypes.c_void_p   # "some pointer" - we never look inside these objects
        sig = {
            "TessBaseAPICreate":            ([], p),
            "TessBaseAPIInit3":             ([p, ctypes.c_char_p, ctypes.c_char_p], ctypes.c_int),
            "TessBaseAPISetPageSegMode":    ([p, ctypes.c_int], None),
            "TessBaseAPISetImage":          ([p, p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int], None),
            "TessBaseAPIRecognize":         ([p, p], ctypes.c_int),
            "TessBaseAPIGetIterator":       ([p], p),
            "TessResultIteratorGetUTF8Text": ([p, ctypes.c_int], p),
            "TessResultIteratorConfidence": ([p, ctypes.c_int], ctypes.c_float),
            "TessResultIteratorNext":       ([p, ctypes.c_int], ctypes.c_int),
            "TessResultIteratorDelete":     ([p], None),
            "TessDeleteText":               ([p], None),
            "TessBaseAPIClear":             ([p], None),
            "TessBaseAPIEnd":               ([p], None),
            "TessBaseAPIDelete":            ([p], None),
        }
        for name, (args, result) in sig.items():
            fn = getattr(lib, name)
            fn.argtypes = args
            fn.restype = result

    def words(self, gray: np.ndarray, psm: int) -> list[tuple[str, float]]:
        """OCR a grayscale image; return [(word, confidence 0-100), ...] in reading order.

        psm = page segmentation mode, the same number as pytesseract's '--psm N'
        (6 = one block of text, 7 = one line of text).
        """
        lib, api = self.lib, self.api
        img = np.ascontiguousarray(gray, dtype=np.uint8)   # one byte per pixel, rows back to back
        h, w = img.shape
        lib.TessBaseAPISetPageSegMode(api, psm)
        # (pointer to the pixels, width, height, bytes per pixel, bytes per row)
        lib.TessBaseAPISetImage(api, img.ctypes.data, w, h, 1, w)
        lib.TessBaseAPIRecognize(api, None)

        out = []
        it = lib.TessBaseAPIGetIterator(api)
        if it:   # NULL pointer = nothing recognised at all
            while True:
                text_ptr = lib.TessResultIteratorGetUTF8Text(it, RIL_WORD)
                if text_ptr:
                    word = ctypes.string_at(text_ptr).decode("utf-8", errors="replace")
                    lib.TessDeleteText(text_ptr)   # Tesseract allocated it, so Tesseract frees it
                    out.append((word, float(lib.TessResultIteratorConfidence(it, RIL_WORD))))
                if not lib.TessResultIteratorNext(it, RIL_WORD):
                    break
            lib.TessResultIteratorDelete(it)
        lib.TessBaseAPIClear(api)   # drop this image's results before the next one
        return out

    def close(self) -> None:
        """Free the engine (Tesseract's memory isn't managed by Python)."""
        if self.api:
            self.lib.TessBaseAPIEnd(self.api)
            self.lib.TessBaseAPIDelete(self.api)
            self.api = None


def load_engine() -> TessEngine | None:
    """Return a TessEngine, or None if the library isn't where we expect
    (e.g. on Mac/Linux, or a different install folder). Callers then fall back
    to pytesseract, which is slower but works anywhere tesseract is installed."""
    try:
        return TessEngine()
    except (OSError, AttributeError, RuntimeError):
        return None

"""Find the two native (non-Python) dependencies, Tesseract and ZBar, on
Windows, macOS and Linux.

Import this BEFORE pyzbar: pyzbar looks for the ZBar library the moment it is
imported, and on Apple Silicon Macs it can miss Homebrew's copy (see below).

Where each OS puts things:
  Windows  Tesseract: the UB-Mannheim installer uses C:\\Program Files\\Tesseract-OCR
           and does NOT add itself to PATH. ZBar: the DLL ships inside pyzbar.
  macOS    `brew install tesseract zbar` -> /opt/homebrew (Apple Silicon) or
           /usr/local (Intel). Apps started outside a terminal (an IDE, Finder)
           may not have Homebrew on PATH, so those folders are checked directly.
  Linux    `apt install tesseract-ocr libzbar0` -> /usr/bin, /usr/lib/...
"""

import os
import shutil
import sys

WINDOWS_TESSERACT_DIR = r"C:\Program Files\Tesseract-OCR"
HOMEBREW_PREFIXES = ["/opt/homebrew", "/usr/local"]

# Checked in order when `tesseract` isn't on PATH.
TESSERACT_FALLBACKS = [os.path.join(WINDOWS_TESSERACT_DIR, "tesseract.exe")] + [
    os.path.join(prefix, "bin", "tesseract") for prefix in HOMEBREW_PREFIXES + ["/usr"]
]


def find_tesseract() -> str | None:
    """Full path of the tesseract program, or None if it isn't installed."""
    on_path = shutil.which("tesseract")
    if on_path:
        return on_path
    for path in TESSERACT_FALLBACKS:
        if os.path.isfile(path):
            return path
    return None


def _let_macos_find_homebrew_libs() -> None:
    """Add Homebrew's lib folder to the places ctypes searches on macOS.

    pyzbar (and tess_direct) locate libraries with ctypes.util.find_library,
    which on macOS only searches DYLD_FALLBACK_LIBRARY_PATH, or ~/lib,
    /usr/local/lib and /usr/lib when that's unset. That covers Intel Homebrew
    but not Apple Silicon's /opt/homebrew/lib. (Homebrew's own Python is
    patched to look there; python.org and uv Pythons are not.) find_library
    reads the variable at call time, so setting it here is enough.
    """
    if sys.platform != "darwin":
        return
    current = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH")
    # Keep macOS's defaults when the variable was unset, since setting it replaces them.
    paths = current.split(":") if current else [os.path.expanduser("~/lib"), "/usr/local/lib", "/usr/lib"]
    for prefix in HOMEBREW_PREFIXES:
        lib = os.path.join(prefix, "lib")
        if os.path.isdir(lib) and lib not in paths:
            paths.append(lib)
    os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(paths)


_let_macos_find_homebrew_libs()

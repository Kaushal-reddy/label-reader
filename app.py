"""A small asset register ("mini-ERP") on top of the label reader.

Run with:  streamlit run app.py   (from the repo folder, with the venv active)
It opens in the browser at http://localhost:8501.

Streamlit re-runs this whole file from top to bottom every time someone clicks
or uploads something, and draws whatever the code says to draw. So there are no
callbacks: the page is just "read the inputs, show the outputs", top to bottom.

Screens (picked in the sidebar):
  Scan   upload a photo -> barcode + OCR + validation (Stages 3-6) -> save it

Storage: one SQLite database file, data/assets.db (git-ignored: it holds real
serials). Scanned photos are copied to data/scans/ so a reviewer can look at
them later.
"""

import os
import sqlite3
import time
from datetime import datetime

import cv2
import numpy as np
import streamlit as st

from baseline import valid_ean13
from barcode_reader import read_barcode
from ocr_reader import read_ocr
from validate import barcode_candidates, decide, ocr_candidates

DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "assets.db")
SCANS_DIR = os.path.join(DATA_DIR, "scans")


# ---------- database ----------

def db() -> sqlite3.Connection:
    """Open the database, creating the folder and table the first time.

    One row per scanned photo. status starts as what the pipeline decided
    ("accepted" or "needs_review"); the review screen will later set it to
    "confirmed" once a person has checked it.
    """
    os.makedirs(SCANS_DIR, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row          # rows can be read by column name
    con.execute("""
        CREATE TABLE IF NOT EXISTS items (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            scanned_at  TEXT NOT NULL,
            photo_path  TEXT NOT NULL,
            serial      TEXT NOT NULL,     -- the pipeline's value (or best guess)
            item_type   TEXT NOT NULL,     -- "device" (serial label) or "product" (EAN-13)
            status      TEXT NOT NULL,     -- accepted / needs_review / confirmed
            source      TEXT,              -- barcode+ocr, barcode, ocr
            reason      TEXT,              -- why the pipeline decided what it did
            assigned_to TEXT DEFAULT '',   -- e.g. the employee who has the laptop
            notes       TEXT DEFAULT ''
        )""")
    return con


def save_item(photo_bytes: bytes, original_name: str, result: dict) -> int:
    """Store the photo in data/scans/ and add a row for it. Returns the new row id."""
    os.makedirs(SCANS_DIR, exist_ok=True)   # the very first save comes before any db() call
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    photo_path = os.path.join(SCANS_DIR, f"{stamp}_{os.path.basename(original_name)}")
    with open(photo_path, "wb") as f:
        f.write(photo_bytes)
    with db() as con:   # "with" commits the change at the end (or rolls back on an error)
        cur = con.execute(
            "INSERT INTO items (scanned_at, photo_path, serial, item_type, status, source, reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",   # ? placeholders: sqlite3 fills them in safely
            (datetime.now().isoformat(timespec="seconds"), photo_path, result["serial"],
             "product" if valid_ean13(result["serial"]) else "device",
             result["status"], result["source"], result["reason"]))
        return cur.lastrowid


def already_saved(serial: str) -> list[sqlite3.Row]:
    """Earlier rows with the same serial: a second scan of the same item."""
    with db() as con:
        return con.execute("SELECT id, scanned_at, status FROM items WHERE serial = ?",
                           (serial,)).fetchall()


# ---------- the label reader ----------

def scan_photo(img: np.ndarray) -> dict:
    """Run the same steps as validate.py on one photo, and time them."""
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
    return {"status": status, "serial": serial, "source": source, "reason": reason,
            "barcode_values": values, "conflicts": conflicts,
            "ocr_labelled": ocr_lab, "ocr_unlabelled": ocr_unlab, "ocr_text": text,
            "t_barcode": t_barcode, "t_ocr": t_ocr}


# ---------- screens ----------

def scan_screen() -> None:
    st.header("Scan a label")
    upload = st.file_uploader("Photo of a serial label or product barcode",
                              type=["jpg", "jpeg", "png"])
    if upload is None:
        st.info("Upload a photo to read it.")
        return

    photo_bytes = upload.getvalue()
    # Streamlit gives us the file's bytes; OpenCV decodes them straight from memory.
    img = cv2.imdecode(np.frombuffer(photo_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        st.error("That file couldn't be read as an image.")
        return

    left, right = st.columns([1, 1])
    left.image(img, channels="BGR", use_container_width=True)

    # Streamlit re-runs the file on every click (e.g. the Save button below), and
    # reading takes seconds. st.session_state survives re-runs, so the result is
    # kept per uploaded file and the photo is only read once.
    key = f"result_{upload.file_id}"
    if key not in st.session_state:
        with st.spinner("Reading barcode and text..."):
            st.session_state[key] = scan_photo(img)
    r = st.session_state[key]

    with right:
        if r["status"] == "accepted":
            st.success(f"Accepted: **{r['serial']}**")
        else:
            st.warning(f"Needs review. Best guess: **{r['serial'] or 'nothing read'}**")
        st.write(r["reason"])
        st.caption(f"Barcode {r['t_barcode']:.1f} s, OCR {r['t_ocr']:.1f} s")

        earlier = already_saved(r["serial"]) if r["serial"] else []
        if earlier:
            st.info(f"Already in the register {len(earlier)} time(s), first on "
                    f"{earlier[0]['scanned_at']} (status: {earlier[0]['status']}).")

        saved_key = f"saved_{upload.file_id}"
        if saved_key in st.session_state:
            st.success(f"Saved as item #{st.session_state[saved_key]}.")
        elif st.button("Save to register", type="primary", disabled=not r["serial"]):
            st.session_state[saved_key] = save_item(photo_bytes, upload.name, r)
            st.rerun()   # redraw now, so the "Saved" message replaces the button

    with st.expander("What the reader saw"):
        st.write("**Barcode values:**", r["barcode_values"] or "none")
        if r["conflicts"]:
            st.write("**Rejected barcode reads:**", r["conflicts"])
        st.write("**OCR candidates after an SN key:**", r["ocr_labelled"] or "none")
        st.write("**Other OCR candidates:**", r["ocr_unlabelled"] or "none")
        st.text_area("Raw OCR text", r["ocr_text"], height=200)


# ---------- page ----------

st.set_page_config(page_title="Label Reader – Asset Register", layout="wide")
st.sidebar.title("Asset Register")
page = st.sidebar.radio("Screen", ["Scan"])
if page == "Scan":
    scan_screen()

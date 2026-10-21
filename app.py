"""A small asset register ("mini-ERP") on top of the label reader.

Run with:  streamlit run app.py   (from the repo folder, with the venv active)
It opens in the browser at http://localhost:8501.

Streamlit re-runs this whole file from top to bottom every time someone clicks
or uploads something, and draws whatever the code says to draw. So there are no
callbacks: the page is just "read the inputs, show the outputs", top to bottom.

Screens (picked in the sidebar):
  Scan     upload a photo -> barcode + OCR + validation (Stages 3-6) -> save it
  Review   items the reader wasn't sure about: a person corrects / confirms them
  Assets   the register: search, filter, assign items to people, export to CSV

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
import pandas as pd
import streamlit as st

from baseline import valid_ean13
from barcode_reader import read_barcode
from ocr_reader import read_ocr
from validate import barcode_candidates, decide, looks_like_serial, ocr_candidates

# ASSET_DATA_DIR lets a test run use a throwaway folder instead of the real register.
DATA_DIR = os.environ.get("ASSET_DATA_DIR", "data")
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


def review_items() -> list[sqlite3.Row]:
    """Everything still waiting for a person, oldest first."""
    with db() as con:
        return con.execute("SELECT * FROM items WHERE status = 'needs_review' ORDER BY id").fetchall()


def confirm_item(item_id: int, serial: str, notes: str) -> None:
    """A person checked the photo: store their value and mark the row confirmed.

    item_type is worked out again, because the person may have changed the value.
    """
    with db() as con:
        con.execute("UPDATE items SET serial = ?, item_type = ?, status = 'confirmed', notes = ? "
                    "WHERE id = ?",
                    (serial, "product" if valid_ean13(serial) else "device", notes, item_id))


def delete_item(item_id: int) -> None:
    """Remove a bad scan (wrong photo, unreadable): its row and its copied photo."""
    with db() as con:
        row = con.execute("SELECT photo_path FROM items WHERE id = ?", (item_id,)).fetchone()
        con.execute("DELETE FROM items WHERE id = ?", (item_id,))
    if row and os.path.exists(row["photo_path"]):
        os.remove(row["photo_path"])


ASSET_COLUMNS = ["id", "serial", "item_type", "status", "assigned_to", "notes", "scanned_at", "source"]


def all_items() -> pd.DataFrame:
    """The whole register as a table, newest first."""
    with db() as con:
        df = pd.read_sql_query(f"SELECT {', '.join(ASSET_COLUMNS)} FROM items ORDER BY id DESC", con)
    # Empty SQL values (NULL) arrive as None; make them "" so search and the
    # "what changed?" comparison treat them like any other text.
    return df.fillna({"assigned_to": "", "notes": "", "source": ""})


def update_details(changes: list[tuple[str, str, int]]) -> None:
    """Save edited (assigned_to, notes, id) rows, all in one transaction."""
    with db() as con:
        con.executemany("UPDATE items SET assigned_to = ?, notes = ? WHERE id = ?", changes)


def format_problem(value: str) -> str | None:
    """Why a typed value doesn't look like an ID, or None if it does.

    Uses the same rules as the pipeline (validate.looks_like_serial,
    baseline.valid_ean13), so a person can't confirm something the reader
    itself would never accept as an ID. It's a warning, not a block: a real
    serial from another manufacturer might not fit the generic pattern.
    """
    if not value:
        return "The value is empty."
    if value.isdigit():
        if len(value) == 13 and not valid_ean13(value):
            return "13 digits, but the EAN-13 check digit is wrong - probably a typo."
        if len(value) != 13:
            return f"{len(value)} digits: a product barcode number has 13."
        return None
    if not looks_like_serial(value):
        return "Doesn't look like a serial (10-20 letters/digits, at least 1 letter and 2 digits)."
    return None


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
    left.image(img, channels="BGR", width="stretch")

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


def review_screen() -> None:
    st.header("Review queue")
    items = review_items()
    if not items:
        st.success("Nothing to review.")
        return
    st.write(f"{len(items)} item(s) the reader wasn't sure about. Check each photo, "
             "correct the value if needed, then confirm.")

    for item in items:
        # border=True draws a box around each item so they're easy to tell apart.
        with st.container(border=True):
            left, right = st.columns([1, 1])
            if os.path.exists(item["photo_path"]):
                left.image(item["photo_path"], width="stretch")
            else:
                left.warning(f"Photo missing: {item['photo_path']}")

            with right:
                st.subheader(f"Item #{item['id']}")
                st.caption(f"Scanned {item['scanned_at']}")
                st.write(f"**Why it needs review:** {item['reason']}")

                # A form groups the inputs: typing doesn't re-run the page,
                # only pressing a button does. The key keeps each item's form separate.
                with st.form(key=f"review_{item['id']}"):
                    # Uppercased and stripped: labels print serials in capitals, and
                    # stray spaces from copy-paste would make the same serial look different.
                    value = st.text_input("Serial / product number", value=item["serial"]).strip().upper()
                    notes = st.text_input("Notes (optional)", value=item["notes"] or "")
                    problem = format_problem(value)
                    if problem:
                        st.warning(problem)
                    confirm_col, delete_col = st.columns(2)
                    confirmed = confirm_col.form_submit_button("Confirm", type="primary",
                                                               disabled=not value)
                    deleted = delete_col.form_submit_button("Delete scan")

                if confirmed:
                    confirm_item(item["id"], value, notes)
                    st.toast(f"Item #{item['id']} confirmed as {value}")
                    st.rerun()   # redraw: the confirmed item leaves the queue
                if deleted:
                    # Deleting can't be undone, so ask once more.
                    st.session_state[f"really_delete_{item['id']}"] = True
                if st.session_state.get(f"really_delete_{item['id']}"):
                    st.error("Delete this scan and its photo? This can't be undone.")
                    yes, no = st.columns(2)
                    if yes.button("Yes, delete", key=f"yes_{item['id']}"):
                        delete_item(item["id"])
                        del st.session_state[f"really_delete_{item['id']}"]
                        st.rerun()
                    if no.button("Cancel", key=f"no_{item['id']}"):
                        del st.session_state[f"really_delete_{item['id']}"]
                        st.rerun()


def assets_screen() -> None:
    st.header("Assets")
    df = all_items()
    if df.empty:
        st.info("The register is empty. Scan a label to add the first item.")
        return

    # Headline numbers for the whole register (before any filter).
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Items", len(df))
    c2.metric("Devices", int((df.item_type == "device").sum()))
    c3.metric("Products", int((df.item_type == "product").sum()))
    c4.metric("Waiting for review", int((df.status == "needs_review").sum()))

    f1, f2, f3 = st.columns([2, 1, 1])
    query = f1.text_input("Search serial, person or notes").strip().lower()
    statuses = f2.multiselect("Status", ["accepted", "confirmed", "needs_review"],
                              default=["accepted", "confirmed"])
    types = f3.multiselect("Type", ["device", "product"], default=["device", "product"])

    shown = df[df.status.isin(statuses) & df.item_type.isin(types)]
    if query:
        # regex=False: treat what was typed as plain text, so e.g. "." or "(" can't break the search.
        hit = (shown.serial.str.lower().str.contains(query, regex=False)
               | shown.assigned_to.str.lower().str.contains(query, regex=False)
               | shown.notes.str.lower().str.contains(query, regex=False))
        shown = shown[hit]
    st.caption(f"Showing {len(shown)} of {len(df)} items. "
               "Edit 'Assigned to' and 'Notes' in the table, then press Save changes.")

    # data_editor is an editable table. Only assigned_to and notes can be edited:
    # the serial and status come from the reader or the review screen, so they
    # stay as they are here.
    edited = st.data_editor(
        shown, hide_index=True, width="stretch",
        disabled=[c for c in ASSET_COLUMNS if c not in ("assigned_to", "notes")],
        column_config={
            "id": st.column_config.NumberColumn("#", width="small"),
            "serial": "Serial / product number",
            "item_type": "Type",
            "status": "Status",
            "assigned_to": st.column_config.TextColumn("Assigned to"),
            "notes": st.column_config.TextColumn("Notes"),
            "scanned_at": "Scanned",
            "source": "Read from",
        },
        # A new key when the filters change, so the table starts fresh for the new rows.
        key=f"assets_{query}_{statuses}_{types}")

    # Compare the edited table with what was shown: only changed rows are saved.
    before = shown.set_index("id")[["assigned_to", "notes"]]
    after = edited.set_index("id")[["assigned_to", "notes"]].fillna("")
    changed = after[(after != before).any(axis=1)]

    save_col, export_col = st.columns(2)
    if save_col.button(f"Save changes ({len(changed)})", type="primary", disabled=changed.empty):
        update_details([(r.assigned_to.strip(), r.notes.strip(), int(i)) for i, r in changed.iterrows()])
        st.toast(f"Saved {len(changed)} item(s)")
        st.rerun()
    # Exports what's on screen (after filters), e.g. "all laptops assigned to Sam".
    export_col.download_button("Export shown rows (CSV)", shown.to_csv(index=False),
                               file_name="assets.csv", mime="text/csv")


# ---------- page ----------

st.set_page_config(page_title="Label Reader – Asset Register", layout="wide")
st.sidebar.title("Asset Register")
n_review = len(review_items())
# The options stay fixed and only their displayed text shows the count. If the
# options themselves changed ("Review (3)" -> "Review (2)"), Streamlit would see
# a new widget and jump back to the first screen after every confirm.
page = st.sidebar.radio("Screen", ["Scan", "Review", "Assets"], key="page",
                        format_func=lambda p: f"Review ({n_review})" if p == "Review" else p)
if page == "Scan":
    scan_screen()
elif page == "Review":
    review_screen()
else:
    assets_screen()

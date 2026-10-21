"""The asset register's storage and ERP link, used by app.py (the screens).

No Streamlit in here, so every function can be called and tested on its own.

Storage: one SQLite file, data/assets.db, table `items` (one row per saved
scan); photos are copied to data/scans/. data/ is git-ignored: real serials.

ERP: the mock ERP from mock_erp.py, reached through Pratham's ErpClient
(erp_client.py) and his Stage 9 helpers (post_to_erp.py). An item is posted to
the ERP as a "receipt" against one line of a job's manifest (the list of items
the ERP expects). Posts go through an outbox (data/erp_outbox.sqlite): saved as
"pending" first, then sent, so a post is never lost when the ERP is down and
never received twice when it's retried.
"""

import os
import sqlite3
from datetime import datetime

import httpx
import pandas as pd

from baseline import valid_ean13
from erp_client import ErpClient, ErpRejected, ErpUnavailable
from post_to_erp import Match, Outbox, job_key_for, match_row, send_pending

# ASSET_DATA_DIR lets a test run use a throwaway folder instead of the real register.
DATA_DIR = os.environ.get("ASSET_DATA_DIR", "data")
DB_PATH = os.path.join(DATA_DIR, "assets.db")
SCANS_DIR = os.path.join(DATA_DIR, "scans")
OUTBOX_PATH = os.path.join(DATA_DIR, "erp_outbox.sqlite")

# items.erp_status values. "" = never sent to the ERP.
ERP_SENT = "sent"            # received on the ERP
ERP_PENDING = "pending"      # in the outbox, the ERP couldn't be reached yet
ERP_OFFLINE = "offline"      # couldn't even be matched: the ERP was down when saving
# Anything else is a reason a person has to look at it: Stage 9's categories
# (near_match, not_on_manifest, already_received, several_lines, not_confident)
# or "rejected" (the ERP said no).
ERP_LABELS = {
    "": "Not sent", ERP_SENT: "Received", ERP_PENDING: "Waiting to send", ERP_OFFLINE: "ERP was offline",
    "near_match": "Near match", "not_on_manifest": "Not on manifest",
    "already_received": "Already received", "several_lines": "Several lines match",
    "not_confident": "Not confirmed yet", "rejected": "Rejected by ERP",
}


# ---------- database ----------

def db() -> sqlite3.Connection:
    """Open the database, creating the folder and table the first time.

    One row per scanned photo. status starts as what the pipeline decided
    ("accepted" or "needs_review"); the review screen sets it to "confirmed"
    once a person has checked it.
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
    # The ERP columns came later. A register made before them gets them added
    # here, so an existing data/assets.db keeps working (a small "migration").
    have = {row["name"] for row in con.execute("PRAGMA table_info(items)")}
    for column in ("erp_job", "erp_status", "erp_detail"):
        if column not in have:
            con.execute(f"ALTER TABLE items ADD COLUMN {column} TEXT DEFAULT ''")
    return con


def item_type_of(serial: str) -> str:
    return "product" if valid_ean13(serial) else "device"


def save_item(photo_bytes: bytes, original_name: str, result: dict) -> int:
    """Store the photo in data/scans/ and add a row for it. Returns the new row id."""
    os.makedirs(SCANS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    photo_path = os.path.join(SCANS_DIR, f"{stamp}_{os.path.basename(original_name)}")
    with open(photo_path, "wb") as f:
        f.write(photo_bytes)
    with db() as con:   # "with" commits the change at the end (or rolls back on an error)
        cur = con.execute(
            "INSERT INTO items (scanned_at, photo_path, serial, item_type, status, source, reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",   # ? placeholders: sqlite3 fills them in safely
            (datetime.now().isoformat(timespec="seconds"), photo_path, result["serial"],
             item_type_of(result["serial"]), result["status"], result["source"], result["reason"]))
        return cur.lastrowid


def get_item(item_id: int) -> sqlite3.Row | None:
    with db() as con:
        return con.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()


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
                    "WHERE id = ?", (serial, item_type_of(serial), notes, item_id))


def correct_serial(item_id: int, serial: str) -> None:
    """Change the value of an already-confirmed item (e.g. to a near-match line's identifier)."""
    with db() as con:
        con.execute("UPDATE items SET serial = ?, item_type = ? WHERE id = ?",
                    (serial, item_type_of(serial), item_id))


def delete_item(item_id: int) -> None:
    """Remove a bad scan (wrong photo, unreadable): its row and its copied photo."""
    with db() as con:
        row = con.execute("SELECT photo_path FROM items WHERE id = ?", (item_id,)).fetchone()
        con.execute("DELETE FROM items WHERE id = ?", (item_id,))
    if row and os.path.exists(row["photo_path"]):
        os.remove(row["photo_path"])


ASSET_COLUMNS = ["id", "serial", "item_type", "status", "erp_status", "assigned_to", "notes",
                 "scanned_at", "source"]


def all_items() -> pd.DataFrame:
    """The whole register as a table, newest first."""
    with db() as con:
        df = pd.read_sql_query(f"SELECT {', '.join(ASSET_COLUMNS)} FROM items ORDER BY id DESC", con)
    # Empty SQL values (NULL) arrive as None; make them "" so search and the
    # "what changed?" comparison treat them like any other text.
    return df.fillna({"assigned_to": "", "notes": "", "source": "", "erp_status": ""})


def update_details(changes: list[tuple[str, str, int]]) -> None:
    """Save edited (assigned_to, notes, id) rows, all in one transaction."""
    with db() as con:
        con.executemany("UPDATE items SET assigned_to = ?, notes = ? WHERE id = ?", changes)


# ---------- ERP ----------

_client: ErpClient | None = None


def erp() -> ErpClient:
    """One ERP client for the whole app, so its login token is reused.

    Module-level, because Streamlit re-runs app.py on every click but keeps
    imported modules (like this one) loaded. Short timeout and 2 attempts: the
    person is waiting at the screen, and the outbox catches whatever fails.
    """
    global _client
    if _client is None:
        _client = ErpClient.from_env(timeout=5, max_attempts=2)
    return _client


def erp_online() -> bool:
    """Is the ERP answering? /health needs no login and is never made to fail on purpose."""
    try:
        return erp().http.get(erp().base_url + "/health", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


def set_erp(item_id: int, job_id: str, status: str, detail: str) -> None:
    with db() as con:
        con.execute("UPDATE items SET erp_job = ?, erp_status = ?, erp_detail = ? WHERE id = ?",
                    (job_id, status, detail, item_id))


def match_preview(serial: str, job_id: str) -> Match:
    """What Stage 9 would do with this value, without posting anything.

    Treated as accepted, so the answer is about the manifest only (exact / near /
    not on it / already received), not about how sure the reader was.
    """
    row = {"filename": "", "status": "accepted", "serial": serial, "reason": ""}
    return match_row(row, erp().get_lines(job_id), {})


def _outbox_result(post: dict) -> tuple[str, str]:
    if post["state"] == "sent":
        return ERP_SENT, f"received on the ERP: line {post['line_id']}, receipt {post['receipt_id']}"
    if post["state"] == "pending":
        return ERP_PENDING, f"waiting to send: {post['last_error'] or 'not sent yet'}"
    return "rejected", post["last_error"]


def post_item(item_id: int, job_id: str) -> tuple[str, str, list[dict]]:
    """Send one saved item to the ERP job. Returns (erp_status, detail, candidate lines)
    and stores the first two on the item.

    Only accepted or confirmed items are sent. The capture id is the item's photo
    path, which is unique per scan, so posting the same item again finds its
    outbox entry instead of posting twice.
    """
    item = get_item(item_id)
    client = erp()
    candidates: list[dict] = []
    try:
        job_key = job_key_for(client, job_id)
        outbox = Outbox(OUTBOX_PATH)     # new per call: an sqlite3 connection belongs to one thread
        post = outbox.find(job_key, item["photo_path"])
        if post is None:
            # Stage 9 only posts "accepted" rows; a person's confirmation counts as accepted.
            row = {"filename": item["photo_path"], "serial": item["serial"], "reason": item["reason"] or "",
                   "status": "accepted" if item["status"] in ("accepted", "confirmed") else item["status"]}
            planned: dict[int, list[str]] = {}   # posts still waiting count against a line's quantity
            for p in outbox.pending(job_key):
                planned.setdefault(p["line_id"], []).append(p["capture_id"])
            m = match_row(row, client.get_lines(job_id), planned)
            if m.action != "post" or m.line is None:
                set_erp(item_id, job_id, m.category, m.reason)
                return m.category, m.reason, m.candidates
            outbox.add(job_key, job_id, row["filename"], m.line["line_id"], m.line["identifier"])
            send_pending(client, outbox, job_key)
            post = outbox.find(job_key, item["photo_path"])
        status, detail = _outbox_result(post)
    except ErpUnavailable as e:
        status, detail = ERP_OFFLINE, f"ERP unreachable: {e}"
    except ErpRejected as e:
        status, detail = "rejected", str(e)
    set_erp(item_id, job_id, status, detail)
    return status, detail, candidates


def retry_pending(job_id: str) -> str | None:
    """Send everything waiting in the outbox for this job, and update the items.

    Returns None if the ERP answered every post, or the error that stopped it.
    """
    client = erp()
    job_key = job_key_for(client, job_id)
    outbox = Outbox(OUTBOX_PATH)
    error = send_pending(client, outbox, job_key)
    with db() as con:
        waiting = con.execute("SELECT id, photo_path FROM items WHERE erp_job = ? AND erp_status = ?",
                              (job_id, ERP_PENDING)).fetchall()
    for item in waiting:
        post = outbox.find(job_key, item["photo_path"])
        if post:
            set_erp(item["id"], job_id, *_outbox_result(post))
    return error


def unsent_items(job_id: str) -> list[sqlite3.Row]:
    """Accepted / confirmed items not yet received on this job: never sent, sent to
    another job, or saved while the ERP was offline."""
    with db() as con:
        return con.execute(
            "SELECT * FROM items WHERE status IN ('accepted', 'confirmed') "
            "AND (erp_job IS NULL OR erp_job != ? OR erp_status IN ('', ?)) ORDER BY id",
            (job_id, ERP_OFFLINE)).fetchall()


def attention_items(job_id: str) -> list[sqlite3.Row]:
    """Items the ERP step couldn't handle on its own: a person has to decide."""
    with db() as con:
        return con.execute(
            "SELECT * FROM items WHERE erp_job = ? AND erp_status NOT IN ('', ?, ?, ?) ORDER BY id",
            (job_id, ERP_SENT, ERP_PENDING, ERP_OFFLINE)).fetchall()


def pending_count(job_id: str) -> int:
    with db() as con:
        return con.execute("SELECT COUNT(*) FROM items WHERE erp_job = ? AND erp_status = ?",
                           (job_id, ERP_PENDING)).fetchone()[0]

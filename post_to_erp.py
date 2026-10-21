"""Stage 9: post validated results to the ERP, and queue everything else for review.

Run with:  python post_to_erp.py [--job JOB-0001]          (after python validate.py)
      or:  python post_to_erp.py photos/ean_dataset --job JOB-0002

The mock ERP must be running (python mock_erp.py serve). Connection settings
come from ERP_URL, ERP_CLIENT_ID and ERP_CLIENT_SECRET (see erp_client.py).

For every photo in validation.csv, match its serial against the job's manifest
lines (the assets the ERP expects), comparing uppercase letters and digits only.
A photo is POSTED automatically only when all of these hold:
  - the pipeline accepted it (barcode and OCR agreed, or a confirmed product barcode)
  - exactly one manifest line has exactly that identifier
  - that line still has quantity left (counting photos posted earlier in this run)
Everything else goes to the REVIEW QUEUE with a reason, and is never posted:
  not_confident     the pipeline wasn't sure (with a hint if its best guess matches a line)
  near_match        differs from a line only by OCR look-alikes (O/0, ...) or by one character
  not_on_manifest   no line has anything like it
  several_lines     more than one line has this identifier
  already_received  the line is full: e.g. a 2nd photo of the same item
  erp_rejected      the ERP refused the post (e.g. someone received it by hand meanwhile)

Posts go through an OUTBOX (a small SQLite file) so nothing is lost or doubled:
  1. each post is saved as "pending" with an idempotency key, BEFORE sending
  2. then sent; it ends "sent", or "rejected" (-> review), or stays "pending"
     with the last error if the ERP couldn't be reached
  3. a re-run sends the pending posts again with the same keys; the ERP
     returns the saved receipt for any it already had, so nothing is received twice

Outputs (in output/erp/, or output/<dataset>/erp/):
  posted.csv         photos received on the ERP: filename, identifier, line, receipt
  review_queue.csv   photos for a person to check: filename, best guess, reason, candidate lines
  outbox.sqlite      the outbox (keep it between runs)
"""

import argparse
import hashlib
import os
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import NamedTuple

import pandas as pd

from erp_client import ErpClient, ErpError, ErpRejected, ErpUnavailable

# Same as validate.CONFUSABLE: characters OCR mixes up in label fonts. Copied
# because importing validate.py starts the OCR engine (~0.7 s).
CONFUSABLE = str.maketrans({"O": "0", "Q": "0", "D": "0",
                            "I": "1", "L": "1",
                            "Z": "2", "A": "4", "S": "5", "G": "6", "B": "8"})


def normalize(text: str) -> str:
    """Uppercase, letters and digits only (same as baseline.normalize)."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------- matching

class Match(NamedTuple):
    action: str              # "post" | "review"
    category: str            # "exact" for posts, otherwise one of the review reasons above
    reason: str              # explanation for a person
    line: dict | None        # the line to post to (action "post" only)
    candidates: list[dict]   # lines the photo might belong to


def one_edit_apart(a: str, b: str) -> bool:
    """True if one character changed, added or removed turns a into b."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    short, long = sorted((a, b), key=len)
    return any(long[:i] + long[i + 1:] == short for i in range(len(long)))


def lookalike(a: str, b: str) -> bool:
    return a != b and a.translate(CONFUSABLE) == b.translate(CONFUSABLE)


def describe(lines: list[dict]) -> str:
    return ", ".join(f"line {line['line_id']} ({line['description']})" for line in lines)


def match_row(row: dict, lines: list[dict], planned: dict[int, list[str]]) -> Match:
    """Decide what to do with one validation.csv row.

    `planned` maps line_id -> photos already headed for that line in this run,
    so the 2nd photo of the same item counts as already received.
    """
    serial = normalize(row["serial"] or "")
    exact = [line for line in lines if serial and line["identifier"] == serial]
    looks = [line for line in lines if serial and lookalike(serial, line["identifier"])]
    one_off = [line for line in lines if serial and one_edit_apart(serial, line["identifier"])
               and line not in looks]

    if row["status"] != "accepted":
        reason = f"pipeline not confident: {row['reason']}"
        if exact:
            reason += f"; best guess matches {describe(exact)}"
        elif looks or one_off:
            reason += f"; best guess is close to {describe(looks + one_off)}"
        return Match("review", "not_confident", reason, None, exact or looks + one_off)

    if len(exact) > 1:
        return Match("review", "several_lines", f"several manifest lines match: {describe(exact)}",
                     None, exact)
    if exact:
        line = exact[0]
        earlier = planned.get(line["line_id"], [])
        if line["received"] + len(earlier) >= line["quantity"]:
            if earlier:
                reason = f"already received: {describe([line])} was matched to {earlier[-1]} earlier in this run"
            else:
                reason = (f"already received: {describe([line])} has {line['received']} of "
                          f"{line['quantity']} on the ERP")
            return Match("review", "already_received", reason, None, [line])
        return Match("post", "exact", "exact match", line, [line])
    if looks:
        return Match("review", "near_match", f"differs from {describe(looks)} only by OCR look-alike "
                     "characters (O/0, I/1, ...); not posted automatically", None, looks)
    if one_off:
        return Match("review", "near_match", f"one character different from {describe(one_off)}; "
                     "not posted automatically", None, one_off)
    return Match("review", "not_on_manifest", "not on the job's manifest", None, [])


# ---------------------------------------------------------------- outbox

class Outbox:
    """Posts waiting to be sent, and what happened to them. One row per photo per job."""

    def __init__(self, path: str):
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS outbox (
                idempotency_key TEXT PRIMARY KEY,
                job_key     TEXT NOT NULL,     -- job id + when the ERP created it (see job_key_for)
                job_id      TEXT NOT NULL,
                capture_id  TEXT NOT NULL,     -- the photo's file name
                line_id     INTEGER NOT NULL,
                identifier  TEXT NOT NULL,
                state       TEXT NOT NULL,     -- pending | sent | rejected
                attempts    INTEGER NOT NULL DEFAULT 0,
                last_error  TEXT NOT NULL DEFAULT '',
                receipt_id  INTEGER,
                updated_at  TEXT NOT NULL,
                UNIQUE (job_key, capture_id)
            )""")
        self.conn.commit()

    def find(self, job_key: str, capture_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM outbox WHERE job_key = ? AND capture_id = ?",
                                (job_key, capture_id)).fetchone()
        return dict(row) if row else None

    def pending(self, job_key: str) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM outbox WHERE job_key = ? AND state = 'pending' "
                                 "ORDER BY rowid", (job_key,)).fetchall()
        return [dict(r) for r in rows]

    def add(self, job_key: str, job_id: str, capture_id: str, line_id: int, identifier: str) -> str:
        """Save a post as pending. The idempotency key is built from what the
        post IS, so the same photo -> same line always gets the same key."""
        key = hashlib.sha256(f"{job_key}|{capture_id}|{line_id}|{identifier}".encode()).hexdigest()[:32]
        with self.conn:
            self.conn.execute("INSERT INTO outbox (idempotency_key, job_key, job_id, capture_id, line_id, "
                              "identifier, state, updated_at) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)",
                              (key, job_key, job_id, capture_id, line_id, identifier, now_text()))
        return key

    def mark(self, key: str, state: str, error: str = "", receipt_id: int | None = None) -> None:
        with self.conn:
            self.conn.execute("UPDATE outbox SET state = ?, last_error = ?, receipt_id = ?, "
                              "attempts = attempts + 1, updated_at = ? WHERE idempotency_key = ?",
                              (state, error, receipt_id, now_text(), key))


def send_pending(client: ErpClient, outbox: Outbox, job_key: str) -> str | None:
    """Send every pending post. Returns None if all were answered, or the
    error that stopped it (the ERP is unreachable: the rest stay pending)."""
    for post in outbox.pending(job_key):
        try:
            receipt = client.post_receipt(post["job_id"], post["line_id"], post["identifier"],
                                          post["capture_id"], post["idempotency_key"])
        except ErpRejected as e:
            outbox.mark(post["idempotency_key"], "rejected", error=str(e))
        except ErpUnavailable as e:
            outbox.mark(post["idempotency_key"], "pending", error=str(e))
            return str(e)
        else:
            outbox.mark(post["idempotency_key"], "sent", receipt_id=receipt["receipt_id"])
    return None


# ---------------------------------------------------------------- one run

def job_key_for(client: ErpClient, job_id: str) -> str:
    """The job id plus when the ERP created the job.

    Re-seeding the mock ERP re-creates the job, so the outbox then starts
    afresh for it instead of thinking every photo was already posted.
    """
    job = next((j for j in client.list_jobs() if j["job_id"] == job_id), None)
    if job is None:
        raise ErpRejected(404, f"job {job_id} not found on the ERP")
    return f"{job_id}@{job['created_at']}"


class Plan(NamedTuple):
    job_key: str
    lines: list[dict]
    review: dict[str, dict]     # filename -> review queue row
    new_keys: set[str]          # outbox posts added by this plan


def review_row(row: dict, m: Match) -> dict:
    return {"filename": row["filename"], "pipeline_status": row["status"], "best_guess": row["serial"],
            "category": m.category, "reason": m.reason, "candidate_lines": describe(m.candidates)}


def plan_posts(rows: list[dict], job_id: str, client: ErpClient, outbox: Outbox) -> Plan:
    """Match every photo not yet in the outbox, and save its post as pending.

    Pending posts from an earlier run are sent first, so the ERP's received
    counts are as up to date as possible before matching.
    """
    job_key = job_key_for(client, job_id)
    send_pending(client, outbox, job_key)
    lines = client.get_lines(job_id)

    planned: dict[int, list[str]] = {}
    for post in outbox.pending(job_key):          # still unsent: count them as on their way
        planned.setdefault(post["line_id"], []).append(post["capture_id"])

    review, new_keys = {}, set()
    for row in rows:
        if outbox.find(job_key, row["filename"]):
            continue                              # decided in an earlier run
        m = match_row(row, lines, planned)
        if m.action == "post" and m.line is not None:
            new_keys.add(outbox.add(job_key, job_id, row["filename"], m.line["line_id"], m.line["identifier"]))
            planned.setdefault(m.line["line_id"], []).append(row["filename"])
        else:
            review[row["filename"]] = review_row(row, m)
    return Plan(job_key, lines, review, new_keys)


@dataclass
class Report:
    posted: list[dict] = field(default_factory=list)
    review: list[dict] = field(default_factory=list)
    pending: list[dict] = field(default_factory=list)
    error: str | None = None        # why pending posts couldn't be sent


def run(rows: list[dict], job_id: str, client: ErpClient, outbox: Outbox) -> Report:
    plan = plan_posts(rows, job_id, client, outbox)
    report = Report(error=send_pending(client, outbox, plan.job_key))
    descriptions = {line["line_id"]: line["description"] for line in plan.lines}

    for row in rows:
        post = outbox.find(plan.job_key, row["filename"])
        if post is None:
            report.review.append(plan.review[row["filename"]])
        elif post["state"] == "sent":
            report.posted.append({
                "filename": row["filename"], "identifier": post["identifier"], "line_id": post["line_id"],
                "description": descriptions.get(post["line_id"], ""), "receipt_id": post["receipt_id"],
                "note": "posted now" if post["idempotency_key"] in plan.new_keys else "posted in an earlier run"})
        elif post["state"] == "rejected":
            line = {"line_id": post["line_id"], "description": descriptions.get(post["line_id"], "")}
            report.review.append({
                "filename": row["filename"], "pipeline_status": row["status"], "best_guess": row["serial"],
                "category": "erp_rejected", "reason": f"the ERP rejected the post: {post['last_error']}",
                "candidate_lines": describe([line])})
        else:
            report.pending.append({"filename": row["filename"], "line_id": post["line_id"],
                                   "last_error": post["last_error"]})
    return report


# ---------------------------------------------------------------- command line

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stage 9: post validated results to the (mock) ERP.")
    parser.add_argument("folder", nargs="?", default=None,
                        help="dataset folder given to validate.py (default: photos/)")
    parser.add_argument("--job", default="JOB-0001", help="ERP job to receive into")
    args = parser.parse_args(argv)

    out_dir = os.path.join("output", os.path.basename(os.path.normpath(args.folder))) if args.folder else "output"
    erp_dir = os.path.join(out_dir, "erp")
    validation = os.path.join(out_dir, "validation.csv")
    if not os.path.exists(validation):
        sys.exit(f"{validation} not found: run validate.py first")
    # dtype=str keeps leading zeros; keep_default_na=False keeps "" instead of NaN.
    rows = pd.read_csv(validation, dtype=str, keep_default_na=False).to_dict("records")

    try:
        with ErpClient.from_env() as client:
            report = run(rows, args.job, client, Outbox(os.path.join(erp_dir, "outbox.sqlite")))
            lines = client.get_lines(args.job)
    except ErpUnavailable as e:
        sys.exit(f"ERP unavailable: {e}\nNothing was lost: posts already in the outbox stay pending. "
                 "Start the ERP and run this again.")
    except ErpError as e:
        sys.exit(f"ERP refused: {e}")

    posted = pd.DataFrame(report.posted, columns=["filename", "identifier", "line_id", "description",
                                                  "receipt_id", "note"])
    review = pd.DataFrame(report.review, columns=["filename", "pipeline_status", "best_guess", "category",
                                                  "reason", "candidate_lines"])
    posted.to_csv(os.path.join(erp_dir, "posted.csv"), index=False)
    review.to_csv(os.path.join(erp_dir, "review_queue.csv"), index=False)

    print(f"Job {args.job}: {len(rows)} photos")
    print(f"  posted now:             {(posted.note == 'posted now').sum()}")
    print(f"  posted in earlier runs: {(posted.note != 'posted now').sum()}")
    print(f"  sent to review:         {len(review)}  ({os.path.join(erp_dir, 'review_queue.csv')})")
    for category, n in review.category.value_counts().items():
        print(f"      {category:18} {n}")
    print(f"  still pending:          {len(report.pending)}")
    if report.error:
        print(f"\nCould not send every post: {report.error}\nRun this again once the ERP is reachable.")
    expected = sum(line["quantity"] for line in lines)
    print(f"\nManifest: {sum(line['received'] for line in lines)} of {expected} expected assets received")


if __name__ == "__main__":
    main(sys.argv[1:])

"""Mock ERP: a small, generic asset-recording server to post pipeline results to.

This is a STAND-IN written for this project. It is not RazorERP, does not copy
its API, and knows nothing about how any real ERP works inside. It only has the
parts an integration has to cope with:
  - a job (one collection / pickup) with a manifest of expected assets ("lines")
  - authorised access: client id + secret -> a bearer token that expires
  - receiving an asset against a line, at most `quantity` times
  - idempotency keys, so a retried POST never receives an asset twice
  - failure injection (flags below): random 503s, lost responses, slow
    responses, rate limiting, short-lived tokens - so the client's error
    recovery can be tested, not just assumed

Run with (from the repo folder, venv active):
  python -m label_reader.erp.mock_erp seed --job JOB-0001 --from truth.csv [--leave-out Mouse_2]
  python -m label_reader.erp.mock_erp serve [--host 0.0.0.0] [--port 8000] [--fail-rate 0.3] ...
  python -m label_reader.erp.mock_erp openapi          (writes docs/erp_openapi.json)

While it runs, the API docs are at http://localhost:8000/docs.

Credentials come from the environment variables ERP_CLIENT_ID and
ERP_CLIENT_SECRET. Without them, development defaults are used (with a warning).

The database (output/erp/erp.sqlite) holds real serials when seeded from
truth.csv, so it lives in the git-ignored output/ folder.
"""

import argparse
import hashlib
import json
import math
import os
import random
import re
import secrets
import sqlite3
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

DB_PATH = os.path.join("output", "erp", "erp.sqlite")
OPENAPI_PATH = os.path.join("docs", "erp_openapi.json")

# Used when ERP_CLIENT_ID / ERP_CLIENT_SECRET aren't set. Fine on your own
# laptop; `serve` prints a warning so they're never used by accident elsewhere.
DEV_CLIENT_ID = "label-reader"
DEV_CLIENT_SECRET = "dev-secret-change-me"

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id      TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS lines (
    line_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id          TEXT NOT NULL REFERENCES jobs(job_id),
    identifier      TEXT NOT NULL,     -- normalised: uppercase letters and digits only
    identifier_type TEXT NOT NULL,     -- "serial" | "ean"
    description     TEXT NOT NULL,
    quantity        INTEGER NOT NULL,
    received        INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS receipts (
    receipt_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id          TEXT NOT NULL,
    line_id         INTEGER NOT NULL REFERENCES lines(line_id),
    identifier      TEXT NOT NULL,
    capture_id      TEXT NOT NULL,     -- which photo it came from
    client_id       TEXT NOT NULL,
    received_at     TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    request_hash    TEXT NOT NULL      -- to spot the same key reused for a different request
);
"""


@dataclass
class Settings:
    db_path: str = DB_PATH
    client_id: str = DEV_CLIENT_ID
    client_secret: str = DEV_CLIENT_SECRET
    token_ttl: float = 3600       # seconds a token stays valid
    fail_rate: float = 0.0        # chance an /api/ request fails with 503 before doing anything
    lost_rate: float = 0.0        # chance a new receipt is saved but the reply is a 500
    slow_rate: float = 0.0        # chance an /api/ request waits slow_seconds first
    slow_seconds: float = 10.0
    rate_limit: int = 0           # max /api/ requests per client per second (0 = no limit)
    seed: int | None = None       # seed for the failure dice, so tests are repeatable


def normalize(text: str) -> str:
    """Uppercase, letters and digits only (same as baseline.normalize; copied so
    the server doesn't import OpenCV and ZBar)."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def valid_ean13(s: str) -> bool:
    """13 digits with a correct check digit (same as baseline.valid_ean13)."""
    if not re.fullmatch(r"\d{13}", s):
        return False
    total = sum(int(d) * (3 if i % 2 else 1) for i, d in enumerate(s[:12]))
    return (10 - total % 10) % 10 == int(s[12])


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def connect(db_path: str) -> sqlite3.Connection:
    if os.path.dirname(db_path):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    # One connection shared by uvicorn's worker threads, so every use is
    # wrapped in a lock (see create_app).
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


# ---------------------------------------------------------------- seeding

def seed_job(db_path: str, job_id: str, lines: list[dict], name: str = "") -> None:
    """Create (or reset) a job with these expected assets.

    Each line is a dict with identifier, description and quantity. Seeding an
    existing job deletes its old lines and receipts first, so a demo can start over.
    """
    conn = connect(db_path)
    with conn:
        old_lines = "SELECT line_id FROM lines WHERE job_id = ?"
        conn.execute(f"DELETE FROM receipts WHERE line_id IN ({old_lines})", (job_id,))
        conn.execute("DELETE FROM lines WHERE job_id = ?", (job_id,))
        conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
        conn.execute("INSERT INTO jobs (job_id, name, created_at) VALUES (?, ?, ?)",
                     (job_id, name or job_id, now_text()))
        for line in lines:
            identifier = normalize(line["identifier"])
            kind = "ean" if valid_ean13(identifier) else "serial"
            conn.execute("INSERT INTO lines (job_id, identifier, identifier_type, description, quantity) "
                         "VALUES (?, ?, ?, ?, ?)",
                         (job_id, identifier, kind, line["description"], int(line["quantity"])))
    conn.close()


def lines_from_truth(truth_path: str, leave_out: list[str] | None = None) -> list[dict]:
    """One manifest line per distinct true value in truth.csv.

    Several photos show the same item (Laptop_2.jpeg, Laptop_2_Glare.jpeg, ...),
    so they become ONE line with quantity 1, described by the common start of
    their file names ("Laptop_2"). `leave_out` drops items by that description,
    so a demo can show an item that isn't on the manifest.
    """
    import pandas as pd   # only needed here; keeps `serve` start-up light

    leave_out = leave_out or []
    truth = pd.read_csv(truth_path, dtype=str)
    lines = []
    for value, group in truth.groupby("true_serial", sort=False):
        stems = [os.path.splitext(f)[0] for f in group.filename]
        description = os.path.commonprefix(stems).rstrip("_- ") or stems[0]
        lines.append({"identifier": value, "description": description, "quantity": 1})

    unknown = set(leave_out) - {line["description"] for line in lines}
    if unknown:
        raise SystemExit(f"--leave-out: no item called {', '.join(sorted(unknown))}; "
                         f"items are: {', '.join(line['description'] for line in lines)}")
    return [line for line in lines if line["description"] not in leave_out]


# ---------------------------------------------------------------- the app

class TokenRequest(BaseModel):
    client_id: str
    client_secret: str
    grant_type: str = "client_credentials"


class ReceiptRequest(BaseModel):
    job_id: str
    line_id: int
    identifier: str
    capture_id: str


def create_app(settings: Settings) -> FastAPI:
    app = FastAPI(title="Mock ERP (generic stand-in)", version="1.0",
                  description="A generic mock asset-recording API for testing the label-reader "
                              "integration. Not any vendor's real API.")
    conn = connect(settings.db_path)
    db_lock = threading.Lock()
    dice = random.Random(settings.seed)
    dice_lock = threading.Lock()
    tokens: dict[str, tuple[str, float]] = {}      # token -> (client id, expiry time)
    recent: dict[str, deque] = {}                  # client id -> times of its last requests

    def roll(chance: float) -> bool:
        with dice_lock:
            return chance > 0 and dice.random() < chance

    def query(sql: str, params: tuple = ()) -> list[dict]:
        with db_lock:
            return [dict(row) for row in conn.execute(sql, params).fetchall()]

    def api_client(authorization: str | None = Header(default=None)) -> str:
        """Runs before every /api/ endpoint: auth, rate limit, injected failures.

        Returns the client id the token belongs to.
        """
        token = (authorization or "").removeprefix("Bearer ").strip()
        client_id, expires = tokens.get(token, ("", 0.0))
        if not client_id or time.monotonic() >= expires:
            raise HTTPException(401, "missing, unknown or expired token",
                                headers={"WWW-Authenticate": "Bearer"})

        if settings.rate_limit:
            # Sliding window: the times of this client's requests in the last second.
            now = time.monotonic()
            times = recent.setdefault(client_id, deque())
            while times and times[0] <= now - 1:
                times.popleft()
            if len(times) >= settings.rate_limit:
                wait = max(1, math.ceil(times[0] + 1 - now))
                raise HTTPException(429, "rate limit exceeded", headers={"Retry-After": str(wait)})
            times.append(now)

        if roll(settings.fail_rate):
            raise HTTPException(503, "service unavailable (injected failure)")
        if roll(settings.slow_rate):
            time.sleep(settings.slow_seconds)
        return client_id

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/oauth/token")
    def get_token(body: TokenRequest) -> dict:
        """Client-credentials grant: id + secret -> bearer token."""
        ok = (body.grant_type == "client_credentials"
              and secrets.compare_digest(body.client_id, settings.client_id)
              and secrets.compare_digest(body.client_secret, settings.client_secret))
        if not ok:
            raise HTTPException(401, "invalid client credentials")
        token = secrets.token_urlsafe(32)
        tokens[token] = (body.client_id, time.monotonic() + settings.token_ttl)
        return {"access_token": token, "token_type": "Bearer", "expires_in": settings.token_ttl}

    @app.get("/api/v1/jobs")
    def list_jobs(_client: str = Depends(api_client)) -> list[dict]:
        return query("SELECT j.job_id, j.name, j.created_at, COUNT(l.line_id) AS lines, "
                     "COALESCE(SUM(l.quantity), 0) AS expected, COALESCE(SUM(l.received), 0) AS received "
                     "FROM jobs j LEFT JOIN lines l ON l.job_id = j.job_id GROUP BY j.job_id ORDER BY j.job_id")

    def require_job(job_id: str) -> None:
        if not query("SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)):
            raise HTTPException(404, f"job {job_id} not found")

    @app.get("/api/v1/jobs/{job_id}/lines")
    def list_lines(job_id: str, _client: str = Depends(api_client)) -> list[dict]:
        require_job(job_id)
        return query("SELECT line_id, identifier, identifier_type, description, quantity, received "
                     "FROM lines WHERE job_id = ? ORDER BY line_id", (job_id,))

    @app.get("/api/v1/jobs/{job_id}/receipts")
    def list_receipts(job_id: str, _client: str = Depends(api_client)) -> list[dict]:
        require_job(job_id)
        return query("SELECT receipt_id, job_id, line_id, identifier, capture_id, received_at "
                     "FROM receipts WHERE job_id = ? ORDER BY receipt_id", (job_id,))

    @app.post("/api/v1/receipts", status_code=201)
    def create_receipt(body: ReceiptRequest, client_id: str = Depends(api_client),
                       idempotency_key: str | None = Header(default=None)) -> JSONResponse:
        """Receive one asset against a manifest line.

        201 = received now. 200 = this Idempotency-Key was already used for the
        same request: nothing new is saved and the original receipt is returned.
        """
        if not idempotency_key:
            raise HTTPException(400, "Idempotency-Key header is required")
        request_hash = hashlib.sha256(json.dumps(body.model_dump(), sort_keys=True).encode()).hexdigest()
        columns = "receipt_id, job_id, line_id, identifier, capture_id, received_at"

        # The whole check-then-insert is under the lock, so two copies of the
        # same request arriving at once (a retry racing a slow original) can't
        # both be saved.
        with db_lock:
            old = conn.execute(f"SELECT {columns}, request_hash FROM receipts WHERE idempotency_key = ?",
                               (idempotency_key,)).fetchone()
            if old:
                if old["request_hash"] != request_hash:
                    raise HTTPException(422, "Idempotency-Key was already used for a different request")
                replay = dict(old)
                del replay["request_hash"]
                return JSONResponse(replay, status_code=200)

            line = conn.execute("SELECT * FROM lines WHERE line_id = ? AND job_id = ?",
                                (body.line_id, body.job_id)).fetchone()
            if line is None:
                raise HTTPException(404, f"line {body.line_id} not found in job {body.job_id}")
            if normalize(body.identifier) != line["identifier"]:
                raise HTTPException(422, f"identifier does not match line {body.line_id}")
            if line["received"] >= line["quantity"]:
                raise HTTPException(409, f"line {body.line_id} is already fully received "
                                         f"({line['received']} of {line['quantity']})")
            with conn:
                cur = conn.execute(
                    "INSERT INTO receipts (job_id, line_id, identifier, capture_id, client_id, "
                    "received_at, idempotency_key, request_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (body.job_id, body.line_id, line["identifier"], body.capture_id, client_id,
                     now_text(), idempotency_key, request_hash))
                conn.execute("UPDATE lines SET received = received + 1 WHERE line_id = ?", (body.line_id,))
            saved = dict(conn.execute(f"SELECT {columns} FROM receipts WHERE receipt_id = ?",
                                      (cur.lastrowid,)).fetchone())

        # "Lost response": the receipt IS saved, but the client sees an error.
        # Only an idempotency key on the retry stops a double receipt.
        if roll(settings.lost_rate):
            raise HTTPException(500, "internal error (injected: receipt saved, response lost)")
        return JSONResponse(saved, status_code=201)

    return app


# ---------------------------------------------------------------- command line

def credentials_from_env() -> tuple[str, str]:
    client_id = os.environ.get("ERP_CLIENT_ID", "")
    client_secret = os.environ.get("ERP_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        print("WARNING: ERP_CLIENT_ID / ERP_CLIENT_SECRET not set, using development credentials.")
        return DEV_CLIENT_ID, DEV_CLIENT_SECRET
    return client_id, client_secret


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Mock ERP (a generic stand-in, not a real vendor API).")
    parser.add_argument("--db", default=DB_PATH, help=f"database file (default {DB_PATH})")
    sub = parser.add_subparsers(dest="command", required=True)

    seed = sub.add_parser("seed", help="create or reset a job from truth.csv")
    seed.add_argument("--job", default="JOB-0001")
    seed.add_argument("--from", dest="truth", default="truth.csv", help="truth.csv to build the manifest from")
    seed.add_argument("--leave-out", action="append", default=[], metavar="ITEM",
                      help="leave this item (e.g. Mouse_2) off the manifest; can be repeated")

    serve = sub.add_parser("serve", help="run the server")
    serve.add_argument("--host", default="127.0.0.1",
                       help="127.0.0.1 = this computer only; 0.0.0.0 = reachable from your network")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--token-ttl", type=float, default=3600, help="seconds a token stays valid")
    serve.add_argument("--fail-rate", type=float, default=0.0, help="chance of a 503 per /api/ request")
    serve.add_argument("--lost-rate", type=float, default=0.0,
                       help="chance a new receipt is saved but answered with a 500")
    serve.add_argument("--slow-rate", type=float, default=0.0, help="chance of a slow /api/ response")
    serve.add_argument("--slow-seconds", type=float, default=10.0)
    serve.add_argument("--rate-limit", type=int, default=0, help="max /api/ requests per client per second")
    serve.add_argument("--seed", type=int, default=None, help="seed for the failure dice")

    sub.add_parser("openapi", help=f"write the API schema to {OPENAPI_PATH}")
    args = parser.parse_args(argv)

    if args.command == "seed":
        lines = lines_from_truth(args.truth, args.leave_out)
        seed_job(args.db, args.job, lines)
        print(f"Seeded {args.job} with {len(lines)} lines in {args.db}:")
        for line in lines:
            print(f"  {line['description']}")
        if args.leave_out:
            print(f"Left off the manifest: {', '.join(args.leave_out)}")

    elif args.command == "serve":
        import uvicorn
        client_id, client_secret = credentials_from_env()
        settings = Settings(db_path=args.db, client_id=client_id, client_secret=client_secret,
                            token_ttl=args.token_ttl, fail_rate=args.fail_rate, lost_rate=args.lost_rate,
                            slow_rate=args.slow_rate, slow_seconds=args.slow_seconds,
                            rate_limit=args.rate_limit, seed=args.seed)
        if args.host != "127.0.0.1":
            print(f"NOTE: listening on {args.host}, so other computers on your network can reach it. "
                  "The database holds real serials; only do this on a network you trust.")
        uvicorn.run(create_app(settings), host=args.host, port=args.port)

    elif args.command == "openapi":
        schema = create_app(Settings(db_path=":memory:")).openapi()
        os.makedirs(os.path.dirname(OPENAPI_PATH), exist_ok=True)
        with open(OPENAPI_PATH, "w") as f:
            json.dump(schema, f, indent=2)
            f.write("\n")
        print(f"Wrote {OPENAPI_PATH}")


if __name__ == "__main__":
    main(sys.argv[1:])

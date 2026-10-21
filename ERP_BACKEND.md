# ERP backend: handoff for the frontend session

What the backend session built on branch **`mock-erp`** (from `main` at `bfca2f8`),
and how the Streamlit app (`app.py`, branch `erp-ui`) can use it.
The full user-facing description is in `README.md`, section "ERP integration (mock)".

## 1. What changed in the plan

The original plan assumed the frontend would call the backend over HTTP: review-queue
endpoints, photo serving and CORS. `FRONTEND_API.md` shows that `app.py` calls the
pipeline **in-process** and has its own SQLite database (`data/assets.db`). So:

- **Dropped:** review-queue HTTP endpoints, photo serving and CORS. The mock ERP has none of them.
- **Integration point for `app.py`:** import `erp_client` and the Stage 9 helpers in
  `post_to_erp.py` directly (recipe in section 5). No HTTP contract to agree on.
- The mock ERP still speaks HTTP, because that is what a real ERP would do. The
  client wraps it.

## 2. Files added

| File | What it is |
|---|---|
| `mock_erp.py` | Mock ERP server (FastAPI + SQLite). A **generic stand-in**, not any vendor's API. Commands: `seed`, `serve`, `openapi` |
| `erp_client.py` | `ErpClient`: tokens, retries with backoff + jitter, `Retry-After`, idempotent receipts |
| `post_to_erp.py` | Stage 9: match `validation.csv` rows to the manifest, outbox, post or queue for review |
| `tests/` | 41 pytest tests (`conftest.py` runs the real server in a thread). **Fake serials only** |
| `pytest.ini` | `pythonpath = .`, `testpaths = tests` |
| `requirements-erp.txt` | `fastapi==0.156.8 uvicorn==0.54.0 httpx==0.28.1 pytest==9.1.1` |
| `docs/erp_openapi.json` | The mock ERP's API schema (`python mock_erp.py openapi`) |
| `README.md` | New usage rows, an "ERP integration (mock)" section, and project-layout entries |
| `.gitignore` | Added a pytest cache entry |

None of the pipeline files (`validate.py`, `barcode_reader.py`, `ocr_reader.py`, ...) were changed.

Written and tested on Python 3.14. The code uses nothing newer than 3.10 syntax, so it
should run on the frontend's 3.12, but that hasn't been tested.

## 3. Running it

```bash
pip install -r requirements.txt -r requirements-erp.txt

python mock_erp.py seed --job JOB-0001 --from truth.csv [--leave-out Mouse_2]
python mock_erp.py serve [--host 0.0.0.0] [--port 8000]      # API docs: http://localhost:8000/docs
python post_to_erp.py --job JOB-0001                          # after validate.py, in a 2nd terminal
python -m pytest tests
```

- **Seeding** creates one manifest line per distinct value in `truth.csv`, with
  quantity 1 and described by the common start of the file names (`Laptop_2`,
  `Monster`, ...). Seeding a job again resets it, deleting its lines and receipts.
- **Credentials** come from the environment variables `ERP_URL`, `ERP_CLIENT_ID`
  and `ERP_CLIENT_SECRET`. Without them, both sides fall back to the development
  defaults (`label-reader` / `dev-secret-change-me`, URL `http://127.0.0.1:8000`)
  and print a warning.
- **Failure flags** for `serve`: `--fail-rate` (random 503), `--lost-rate` (receipt
  saved, but the reply is a 500), `--slow-rate` / `--slow-seconds`, `--rate-limit N`
  (requests per client per second, 429 with `Retry-After`), `--token-ttl`, `--seed`.
- **Data on disk:** the ERP database is `output/erp/erp.sqlite` and the outbox is
  `output/erp/outbox.sqlite`. Both are git-ignored and hold real serials once seeded
  from `truth.csv`.

## 4. The mock ERP's HTTP API

Errors are FastAPI's `{"detail": "..."}`. Everything under `/api/` needs
`Authorization: Bearer <token>`.

| Method + path | Body | Answers |
|---|---|---|
| `GET /health` | – | `{"status": "ok"}`, no auth, never failure-injected |
| `POST /oauth/token` | JSON `{"client_id", "client_secret", "grant_type": "client_credentials"}` | `{"access_token", "token_type": "Bearer", "expires_in"}`; 401 for bad credentials |
| `GET /api/v1/jobs` | – | `[{job_id, name, created_at, lines, expected, received}]` |
| `GET /api/v1/jobs/{job_id}/lines` | – | `[{line_id, identifier, identifier_type: "serial"\|"ean", description, quantity, received}]`; 404 for an unknown job |
| `GET /api/v1/jobs/{job_id}/receipts` | – | `[{receipt_id, job_id, line_id, identifier, capture_id, received_at}]` |
| `POST /api/v1/receipts` | JSON `{job_id, line_id, identifier, capture_id}` + header `Idempotency-Key` | see below |

`POST /api/v1/receipts` answers:

| Status | When |
|---|---|
| 201 | Received now |
| 200 | Same key and same body as before: the saved receipt again, nothing new saved |
| 400 | No `Idempotency-Key` header |
| 404 | No such line in that job |
| 409 | The line is already fully received |
| 422 | The identifier doesn't match the line, or the key was reused with a different body |

Identifiers are compared after normalising: uppercase, letters and digits only.

## 5. Using it from `app.py` (suggested; not built)

`ErpClient` and `post_to_erp` don't import `validate.py`, `cv2` or Tesseract, so
importing them is cheap. A Review screen that sets `status = "confirmed"` could post
like this. The function below, minus the Streamlit caching, was run once against the
mock ERP with fake serials:
- the first post came back `sent`;
- calling it again returned the same `sent` entry, with no new receipt;
- a second photo of the item gave `already_received`;
- a look-alike serial gave `near_match`.

```python
import streamlit as st
from erp_client import ErpClient, ErpError, ErpUnavailable
from post_to_erp import Outbox, job_key_for, match_row, send_pending

@st.cache_resource          # one client (and one cached token) per Streamlit process
def erp() -> ErpClient:
    return ErpClient.from_env(timeout=5, max_attempts=3)   # keep UI waits short; the outbox catches the rest

def post_item(item: dict, job_id: str = "JOB-0001"):
    client = erp()
    outbox = Outbox("data/erp_outbox.sqlite")   # new per call: sqlite3 connections are per thread
    job_key = job_key_for(client, job_id)
    if (done := outbox.find(job_key, item["photo_path"])):
        return done                             # already posted / pending / rejected
    row = {"filename": item["photo_path"],      # photo_path is unique per scan -> capture id
           "status": "accepted" if item["status"] in ("accepted", "confirmed") else item["status"],
           "serial": item["serial"], "reason": item["reason"]}
    planned = {}
    for p in outbox.pending(job_key):           # posts not yet confirmed by the ERP
        planned.setdefault(p["line_id"], []).append(p["capture_id"])
    m = match_row(row, client.get_lines(job_id), planned)
    if m.action != "post" or m.line is None:
        return m                                # show m.category, m.reason, m.candidates
    outbox.add(job_key, job_id, row["filename"], m.line["line_id"], m.line["identifier"])
    error = send_pending(client, outbox, job_key)   # None = sent or rejected; str = ERP unreachable, still pending
    return outbox.find(job_key, row["filename"]), error
```

Points that matter for the UI:

- **`match_row` only posts rows whose status is `"accepted"`.** A human-confirmed item
  must be mapped to `"accepted"` by the caller, as above. A serial the operator
  corrected goes in `serial`.
- **Near matches** (OCR look-alikes like O/0, or one character different) are never
  posted automatically. `m.candidates` lists the lines the operator could pick; after
  they confirm, post again with the corrected serial.
- **A `"pending"` outbox entry is not an error.** The ERP couldn't be reached. Calling
  `send_pending(client, outbox, job_key)` later (or running `post_to_erp.py`) resends
  it with the same idempotency key, so the ERP never receives it twice.
- **Blocking:** with the defaults (`timeout=10`, `max_attempts=5`, backoff 0.5 s
  doubling up to 8 s), one call can block for close to a minute while the ERP is down.
  Use smaller values in the UI, as in the example.
- **Exceptions:** `ErpRejected` (a 4xx: `.status`, `.detail`) and `ErpUnavailable`
  (gave up after retrying) are both subclasses of `ErpError`. `job_key_for` raises
  `ErpRejected(404)` for an unknown job.
- **Outbox file:** the CLI keeps its outbox in `output/erp/`. The app can use its own
  file under `data/`, but the two won't know about each other's posts. The ERP's
  idempotency still prevents double receipts only for the same key, i.e. the same
  photo name, line and job.

## 6. Stage 9 rules (`post_to_erp.py`)

A photo is posted automatically only if all three hold:
- the pipeline **accepted** it;
- **exactly one** line has **exactly** that identifier;
- that line still has quantity left, counting photos already headed for it in this run.

Anything else goes to `review_queue.csv` with one of these categories:

| category | meaning |
|---|---|
| `not_confident` | the pipeline said needs_review; the reason includes a hint when the best guess matches or is close to a line |
| `near_match` | differs from a line only by OCR look-alikes, or by one character |
| `not_on_manifest` | nothing like it on the manifest |
| `several_lines` | more than one line has this identifier |
| `already_received` | the line is full, e.g. a second photo of the same item |
| `erp_rejected` | the ERP refused the post, e.g. someone received it by hand meanwhile |

CSV columns:
- **`review_queue.csv`:** `filename, pipeline_status, best_guess, category, reason, candidate_lines`
- **`posted.csv`:** `filename, identifier, line_id, description, receipt_id, note` (`posted now` / `posted in an earlier run`)

**Outbox:** each post is saved as `pending` with its idempotency key
(sha256 of job key, photo, line and identifier) before it is sent. Afterwards it is
`sent`, `rejected`, or still `pending` with `last_error`. The job key is the job id
plus the ERP's `created_at`, so re-seeding a job starts its outbox afresh.

## 7. Results so far

- `python -m pytest tests`: **41 passed** (about 12 s). It passed in three runs in a
  row with no warnings, and Pyright reports 0 errors on the new files.
- **Demo** on the 26 photos, with Mouse_2 left off the manifest and
  `--fail-rate 0.3 --lost-rate 0.3 --rate-limit 5 --seed 7`:
  - **Outcome:** 10 posted and 16 sent to review (8 not confident, 6 already received,
    2 not on the manifest); 10 of 11 lines received.
  - **Server responses:** 15 × 503, 2 lost-response 500s and 4 × 429.
  - **Database:** exactly one receipt per posted photo.
  - **Re-run:** a second run posted nothing new.
  - This is one run with one seed.

## 8. Open questions

- **Duplicate photos in the review queue:** 6 of the 16 review items are second
  photos of items already received. They follow the handoff ("already received" goes
  to review), but could go in a separate list so operators only see real exceptions.
  The user hasn't decided yet.
- **App integration:** section 5 is a suggestion. Nothing in `app.py` calls the ERP yet.

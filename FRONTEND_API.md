# Frontend API

What the Streamlit asset register is built against. It has two files:

| File | Role |
|---|---|
| `app.py` | The screens (Scan, Review, Assets, ERP). The only file that imports Streamlit |
| `register.py` | Storage (SQLite) and the link to the ERP. No Streamlit, so it is unit-tested (`tests/test_register.py`) |

The app talks to three things:

1. **The pipeline**, as in-process Python calls (section 1). There is no HTTP between the UI and the pipeline.
2. **Its own SQLite register**, `data/assets.db` (section 3).
3. **The mock ERP** (`mock_erp.py`, Pratham's), over HTTP. All of that goes through his `erp_client.ErpClient` and the Stage 9 helpers in `post_to_erp.py`, called from `register.py` (section 4). The app makes no raw HTTP calls of its own, except `GET /health`.

Branch `erp-ui` (it contains `product-barcodes` and `mock-erp`). Python 3.12, Streamlit 1.66.0.

---

## 1. Pipeline functions called

The calls happen in `app.scan_photo()`, in this order. It's the same sequence as
`validate.main()`, one photo at a time.

### 1.1 `barcode_reader.read_barcode`

```python
read_barcode(gray: np.ndarray) -> BarcodeRead
```

| | |
|---|---|
| **Input** | `gray`: 2-D `uint8` grayscale image, full resolution (`cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`) |
| **Returns** | `BarcodeRead`, a `NamedTuple`, unpacked as `values, _, conflicts, agreeing` |
| **Side effects** | None (writes no files) |
| **Speed** | ~0.5–1.5 s per device photo; ~14 s per 8064×4536 product photo (product barcodes are decoded by 3 image variants) |

```python
class BarcodeRead(NamedTuple):
    values: list[str]          # decoded values that survived the vote; [] if none
    variant: str | None        # "original" or "upscale3x"; None if nothing read  (unused by app)
    conflicts: list[str]       # other values read from the same barcode and rejected
    agreeing: dict[str, int]   # value -> number of image variants that read it (1-3)
```

- Barcode types read: `CODE128`, `CODE39`, `QRCODE`, `EAN13`.
- `agreeing` counts beyond 1 only for 13-digit values (product barcodes).

### 1.2 `ocr_reader.read_ocr`

```python
read_ocr(img: np.ndarray) -> str
```

| | |
|---|---|
| **Input** | `img`: 3-channel `uint8` **BGR** image (OpenCV order), full resolution |
| **Returns** | All OCR text, `\n`-separated: whole-crop text, line-by-line text, and barcode-digit reads, for up to 3 label crops |
| **Side effects** | None (writes no files) |
| **Speed** | ~0.5–2 s per photo |

### 1.3 `validate.barcode_candidates`

```python
barcode_candidates(values: list[str]) -> list[str]
```

- **Input:** `BarcodeRead.values`.
- **Returns:** the ID-shaped strings from those values, in order:
  - a value that is a valid EAN-13 is returned whole;
  - otherwise the value is uppercased and split on non-alphanumerics, and each piece that is a serial (`^(?=.*[A-Z])(?=(?:.*\d){2})[A-Z0-9]{10,20}$`) or a valid EAN-13 is kept.

### 1.4 `validate.ocr_candidates`

```python
ocr_candidates(text: str) -> tuple[list[str], list[str]]
```

- **Input:** the `read_ocr()` text.
- **Returns:** `(labelled, unlabelled)`, each unique and most-read first.
  - `labelled`: serials found right after an `SN` / `S/N` key.
  - `unlabelled`: any serial-shaped word, plus any valid EAN-13 found after joining digit groups separated by spaces or `-`.

### 1.5 `validate.decide`

```python
decide(bc: list[str], ocr_lab: list[str], ocr_unlab: list[str],
       conflicts: list[str] = (), agreeing: dict[str, int] | None = None
       ) -> tuple[str, str, str, str]      # (status, serial, source, reason)
```

| Field | Type | Values |
|---|---|---|
| `status` | `str` | `"accepted"` \| `"needs_review"` |
| `serial` | `str` | the value, or best guess when `needs_review`; `""` when nothing was read |
| `source` | `str` | `"barcode+ocr"` \| `"barcode"` \| `"ocr"` \| `""` |
| `reason` | `str` | human-readable explanation (free text, shown as is) |

The rules are applied in this order, and the first one that matches wins:

| # | Condition | status | source |
|---|---|---|---|
| 1 | A barcode candidate equals an OCR candidate (exactly, or up to the look-alikes O/Q/D→0, I/L→1, Z→2, A→4, S→5, G→6, B→8) | accepted | barcode+ocr |
| 2 | More than one distinct valid EAN-13 among the barcode candidates | needs_review | barcode |
| 3 | One EAN-13, and `conflicts` is not empty | needs_review | barcode |
| 4 | One EAN-13, and `agreeing[value] < 2` | needs_review | barcode |
| 5 | One EAN-13 otherwise | accepted | barcode |
| 6 | Barcode and OCR candidates both present but disagree | needs_review | barcode |
| 7 | Barcode candidates only | needs_review | barcode |
| 8 | Labelled OCR only | needs_review | ocr |
| 9 | Unlabelled OCR only | needs_review | ocr |
| 10 | Nothing | needs_review | "" (serial "") |

When a photo is accepted, `serial` is always the **barcode's** characters, not OCR's.

### 1.6 `baseline.valid_ean13`

```python
valid_ean13(s: str) -> bool
```

`True` if `s` is exactly 13 digits and its check digit is correct (weights
1,3,1,3,… on the first 12 digits). `register.item_type_of()` uses it to set `item_type`, and `app.format_problem()` to
check values typed on the Review screen.

### 1.7 `validate.looks_like_serial`

```python
looks_like_serial(s: str) -> bool
```

`True` if `s` matches `^(?=.*[A-Z])(?=(?:.*\d){2})[A-Z0-9]{10,20}$`. Used only to
warn on the Review screen when a typed value doesn't look like a serial.

---

## 2. Result object inside the app

`app.scan_photo(img: np.ndarray) -> dict`. This is the app's own wrapper, kept in
`st.session_state["result_<file_id>"]` so a photo is read only once per upload.

| Key | Type | From |
|---|---|---|
| `status` | `str` | `decide()` |
| `serial` | `str` | `decide()` |
| `source` | `str` | `decide()` |
| `reason` | `str` | `decide()` |
| `barcode_values` | `list[str]` | `BarcodeRead.values` |
| `conflicts` | `list[str]` | `BarcodeRead.conflicts` |
| `ocr_labelled` | `list[str]` | `ocr_candidates()[0]` |
| `ocr_unlabelled` | `list[str]` | `ocr_candidates()[1]` |
| `ocr_text` | `str` | `read_ocr()` |
| `t_barcode` | `float` | seconds spent in `read_barcode()` |
| `t_ocr` | `float` | seconds spent in `read_ocr()` |

---

## 3. Storage (`register.py`)

### 3.1 Files

| Path | Contents | In git? |
|---|---|---|
| `data/assets.db` | The register (SQLite) | No: `data/` is git-ignored (real serials) |
| `data/scans/<YYYYMMDD_HHMMSS>_<original name>` | A copy of each saved photo, bytes unchanged | No |
| `data/erp_outbox.sqlite` | The app's ERP outbox (`post_to_erp.Outbox`) | No |
| env `ASSET_DATA_DIR` | Replaces `data/` for all three (tests use a throwaway folder) | — |

All of them are created on first use.

### 3.2 Table `items`

```sql
CREATE TABLE IF NOT EXISTS items (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    scanned_at  TEXT NOT NULL,     -- local time, ISO 8601 to the second: 2026-10-21T21:20:09
    photo_path  TEXT NOT NULL,     -- relative path under data/scans/; also the ERP capture id
    serial      TEXT NOT NULL,     -- decide()'s serial, or a person's correction (never "")
    item_type   TEXT NOT NULL,     -- "product" if valid_ean13(serial) else "device"
    status      TEXT NOT NULL,     -- "accepted" | "needs_review" | "confirmed"
    source      TEXT,              -- decide()'s source
    reason      TEXT,              -- decide()'s reason
    assigned_to TEXT DEFAULT '',
    notes       TEXT DEFAULT ''
)
-- added by db() with ALTER TABLE when missing (registers made before the ERP link):
--   erp_job     TEXT DEFAULT ''    the job the item was last sent to
--   erp_status  TEXT DEFAULT ''    see 4.3
--   erp_detail  TEXT DEFAULT ''    the ERP step's explanation, shown as is
```

- **`status` changes:**
  - Save writes `accepted` or `needs_review`, as the pipeline decided.
  - Review changes `needs_review` to `confirmed`.
  - Nothing changes `accepted`.
- **What the Assets screen can edit:** only `assigned_to` and `notes`.
- **What can change `serial` after saving:** Review's Confirm, or a near-match fix on the ERP screen.
- **`serial` is not unique.** Scanning the same item again adds a new row. The Scan
  screen warns, and the ERP then answers `already_received`.

### 3.3 Functions and their SQL

All values are passed as `?` parameters, never pasted into the SQL text.

| Function | SQL |
|---|---|
| `save_item(bytes, name, result) -> int` | `INSERT INTO items (scanned_at, photo_path, serial, item_type, status, source, reason) VALUES (?, ?, ?, ?, ?, ?, ?)` |
| `get_item(id)` | `SELECT * FROM items WHERE id = ?` |
| `already_saved(serial)` | `SELECT id, scanned_at, status FROM items WHERE serial = ?` |
| `review_items()` | `SELECT * FROM items WHERE status = 'needs_review' ORDER BY id` |
| `confirm_item(id, serial, notes)` | `UPDATE items SET serial = ?, item_type = ?, status = 'confirmed', notes = ? WHERE id = ?` |
| `correct_serial(id, serial)` | `UPDATE items SET serial = ?, item_type = ? WHERE id = ?` |
| `all_items() -> DataFrame` | `SELECT id, serial, item_type, status, erp_status, assigned_to, notes, scanned_at, source FROM items ORDER BY id DESC`; NULL in `assigned_to`, `notes`, `source` or `erp_status` becomes `""` |
| `update_details(changes)` | `UPDATE items SET assigned_to = ?, notes = ? WHERE id = ?`, once per changed row, in one transaction |
| `delete_item(id)` | `SELECT photo_path FROM items WHERE id = ?`, then `DELETE FROM items WHERE id = ?`, then deletes the photo file |
| `set_erp(id, job, status, detail)` | `UPDATE items SET erp_job = ?, erp_status = ?, erp_detail = ? WHERE id = ?` |
| `unsent_items(job)` | `SELECT * FROM items WHERE status IN ('accepted', 'confirmed') AND (erp_job IS NULL OR erp_job != ? OR erp_status IN ('', 'offline')) ORDER BY id` |
| `attention_items(job)` | `SELECT * FROM items WHERE erp_job = ? AND erp_status NOT IN ('', 'sent', 'pending', 'offline') ORDER BY id` |
| `pending_count(job)` | `SELECT COUNT(*) FROM items WHERE erp_job = ? AND erp_status = 'pending'` |

**Values typed on the Review screen** are stripped of surrounding spaces and
uppercased. `app.format_problem()` then warns if the value is:
- empty;
- digits, but not 13 of them;
- 13 digits with a wrong EAN-13 check digit;
- not serial-shaped.

These are warnings only. Only an empty value blocks Confirm.

---

## 4. ERP (`register.py` → `erp_client`, `post_to_erp`)

### 4.1 Calls made

| Caller | Calls (Pratham's code) | HTTP on the mock ERP |
|---|---|---|
| `register.erp()` | `ErpClient.from_env(timeout=5, max_attempts=2)`, created once per process | — |
| `register.erp_online()` | `erp().http.get(base_url + "/health", timeout=2)` | `GET /health` |
| sidebar (`app.py`) | `erp().list_jobs()` | `GET /api/v1/jobs` |
| `register.match_preview(serial, job)` | `match_row({"status": "accepted", "serial": serial, ...}, erp().get_lines(job), {})` | `GET /api/v1/jobs/{job}/lines` |
| `register.post_item(id, job)` | `job_key_for`, `Outbox.find/pending/add`, `match_row`, `send_pending` | `GET /api/v1/jobs`, `GET /api/v1/jobs/{job}/lines`, `POST /api/v1/receipts` (with `Idempotency-Key`) |
| `register.retry_pending(job)` | `job_key_for`, `send_pending`, `Outbox.find` | `GET /api/v1/jobs`, `POST /api/v1/receipts` |
| ERP screen (`app.py`) | `erp().get_lines(job)` | `GET /api/v1/jobs/{job}/lines` |

Connection settings come from `ERP_URL`, `ERP_CLIENT_ID` and `ERP_CLIENT_SECRET`.
Without them, the development defaults are used, with a printed warning (`ErpClient.from_env`).

### 4.2 `post_item(item_id, job_id) -> (erp_status, detail, candidates)`

1. Compute `job_key_for(job)`, which is the job id plus the ERP's `created_at`.
2. Look up the outbox by `(job_key, photo_path)`. If an entry already exists, report its
   state and **don't post again**.
3. Otherwise build a Stage 9 row:
   - `status` is `"accepted"` if the item is `accepted` **or `confirmed`**, otherwise the item's own status;
   - `serial` and `reason` are the item's.
4. Run `match_row(row, lines, planned)`. `planned` holds the outbox's pending posts per
   line, so they count against the line's quantity.
5. If `m.action != "post"`, store `m.category` and `m.reason`, and return them with
   `m.candidates`. Nothing is posted.
6. Otherwise:
   - `Outbox.add` saves the post as pending first;
   - `send_pending` sends it;
   - the item's status is set from the outbox state (`sent`, `pending` or `rejected`).
7. Exceptions:
   - `ErpUnavailable` gives `offline`;
   - `ErpRejected` gives `rejected`.

`candidates` is only non-empty in step 5.

### 4.3 `items.erp_status` values

| Value | Shown as | Meaning | Where it's handled |
|---|---|---|---|
| `""` | Not sent | Never sent to a job | ERP screen → "Not sent yet" |
| `sent` | Received | The ERP has a receipt | — |
| `pending` | Waiting to send | In the outbox; the ERP didn't answer | ERP screen → "Retry sending" |
| `offline` | ERP was offline | The ERP was down before matching | ERP screen → "Not sent yet" |
| `near_match` | Near match | Look-alike or 1 character off a line | Review "Use line ..." / ERP screen "It's line ..., send it" |
| `not_on_manifest` | Not on manifest | No line is like it | ERP screen → "Needs a decision" |
| `already_received` | Already received | The line is full (e.g. a 2nd photo) | ERP screen → "Needs a decision" |
| `several_lines` | Several lines match | More than one line has this identifier | ERP screen → "Needs a decision" |
| `not_confident` | Not confirmed yet | Sent while still `needs_review` | Review → Confirm |
| `rejected` | Rejected by ERP | A 4xx from the ERP | ERP screen → "Needs a decision" |

### 4.4 When the app sends

| Screen | Trigger | Condition |
|---|---|---|
| Scan | "Save and send to ERP" | A job is selected **and** the pipeline accepted the photo. Otherwise the button is "Save to register" |
| Review | "Confirm and send" | A job is selected |
| Review | "Use line N: ..." | Only fills in the identifier (`on_click` callback); the person still presses Confirm |
| ERP | "Send these N to JOB" | Every item from `unsent_items(job)` |
| ERP | "It's line N: ..., send it" | `correct_serial`, then `post_item`, for a `near_match` item |
| ERP | "Retry sending (N)" | `retry_pending(job)` |

---

## 5. Constraints the frontend must respect

1. **Image format:** `read_ocr` expects BGR. Uploads are decoded with
   `cv2.imdecode(np.frombuffer(bytes, np.uint8), cv2.IMREAD_COLOR)`, which returns
   `None` for a non-image; the app checks for that. A saved photo that can't be shown
   becomes a warning on its item, not a crash (`app.show_photo`).
2. **Slow, blocking calls:**
   - The pipeline takes 1–15 s per photo. Its result is kept in `st.session_state`,
     because Streamlit re-runs the whole script on every click.
   - An ERP call can block for up to about 10 s while the ERP is down (2 attempts,
     5 s timeout). The outbox catches anything that fails.
3. **Widget state:**
   - A text box's value can't be set after it has been drawn, so "Use line" sets it
     in an `on_click` callback.
   - The sidebar screen picker keeps fixed option values (`key="page"`), so a
     changing count doesn't reset the page.
4. **One Tesseract engine per process:** `ocr_reader.ENGINE` is shared and isn't safe
   to use from two threads at once. That's fine for one user, but serving several
   would need a lock or a queue.
5. **One sqlite connection per call:** `register` opens a new connection (and a new
   `Outbox`) for each function, because a sqlite3 connection belongs to the thread
   that made it.
6. **Two outboxes:** the app's outbox (`data/erp_outbox.sqlite`) and the CLI's
   (`output/erp/outbox.sqlite`) don't know about each other. Their capture ids differ
   too: the app uses the photo path, the CLI the file name. So the ERP's idempotency
   doesn't link them. Don't post the same photos through both.
7. **Pipeline and backend code are read-only from the frontend:** the app only calls
   the functions listed here. It never changes `barcode_reader`, `ocr_reader`,
   `validate`, `mock_erp`, `erp_client` or `post_to_erp`.
8. **No secrets or serials in code:** serials exist only in `data/`, `output/` and the
   browser session. Tests use fake serials.

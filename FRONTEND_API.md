# Frontend API

What `app.py` (the Streamlit asset register) is built against.

**There is no HTTP/REST API.** The frontend imports the pipeline's Python
functions and calls them in the same process. The only other interface is a
local SQLite database that `app.py` owns.

Branch `erp-ui` (built on `product-barcodes`). Python 3.12, Streamlit 1.66.0.

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
1,3,1,3,… on the first 12 digits). `app.py` uses it to set `item_type` and to
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

## 3. Storage

### 3.1 Files

| Path | Contents | In git? |
|---|---|---|
| `data/assets.db` | SQLite database | No (`data/` is git-ignored: real serials) |
| env `ASSET_DATA_DIR` | Overrides the `data/` folder (used to test on a throwaway database) | — |
| `data/scans/<YYYYMMDD_HHMMSS>_<original name>` | Copy of each saved photo, bytes unchanged | No |

Both are created on first use.

### 3.2 Table `items`

```sql
CREATE TABLE IF NOT EXISTS items (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    scanned_at  TEXT NOT NULL,     -- local time, ISO 8601 to the second: 2026-10-21T21:20:09
    photo_path  TEXT NOT NULL,     -- relative path under data/scans/
    serial      TEXT NOT NULL,     -- decide()'s serial (never "": saving is disabled then)
    item_type   TEXT NOT NULL,     -- "product" if valid_ean13(serial) else "device"
    status      TEXT NOT NULL,     -- "accepted" | "needs_review" | "confirmed"
    source      TEXT,              -- decide()'s source
    reason      TEXT,              -- decide()'s reason
    assigned_to TEXT DEFAULT '',
    notes       TEXT DEFAULT ''
)
```

- Status changes: the pipeline writes `accepted` or `needs_review` on Save; the
  Review screen changes `needs_review` to `confirmed`. Nothing changes
  `accepted` rows. The Assets screen edits only `assigned_to` and `notes`;
  `serial`, `status` and the other columns are read-only there.
- `serial` is **not unique**: scanning the same item again adds a new row. The
  Scan screen warns about earlier rows with the same `serial`.

### 3.3 Queries the app runs

| Function | SQL |
|---|---|
| `save_item()` | `INSERT INTO items (scanned_at, photo_path, serial, item_type, status, source, reason) VALUES (?, ?, ?, ?, ?, ?, ?)` |
| `already_saved(serial)` | `SELECT id, scanned_at, status FROM items WHERE serial = ?` |
| `review_items()` | `SELECT * FROM items WHERE status = 'needs_review' ORDER BY id` |
| `confirm_item(id, serial, notes)` | `UPDATE items SET serial = ?, item_type = ?, status = 'confirmed', notes = ? WHERE id = ?` |
| `all_items()` | `SELECT id, serial, item_type, status, assigned_to, notes, scanned_at, source FROM items ORDER BY id DESC` (read into a pandas DataFrame; NULL in `assigned_to`/`notes`/`source` becomes `""`) |
| `update_details(changes)` | `UPDATE items SET assigned_to = ?, notes = ? WHERE id = ?`, once per changed row, in one transaction |
| `delete_item(id)` | `SELECT photo_path FROM items WHERE id = ?`, then `DELETE FROM items WHERE id = ?`; then deletes the file at `photo_path` if it exists |

Values typed on the Review screen are stripped of surrounding spaces and
uppercased before `confirm_item()`. They are checked by `format_problem()`
(empty; digits but not 13; 13 digits with a wrong EAN-13 check digit; not
serial-shaped). A problem is shown as a warning but doesn't block Confirm, because
real serials from other manufacturers may not fit the generic pattern. Only an
empty value blocks it.

All values are passed as `?` parameters, never pasted into the SQL text.

---

## 4. Constraints the frontend must respect

1. **Image format:** `read_ocr` expects BGR. Uploads are decoded with
   `cv2.imdecode(np.frombuffer(bytes, np.uint8), cv2.IMREAD_COLOR)`, which
   returns `None` for a non-image; the app checks for that.
2. **Slow, blocking calls:** the pipeline takes 1–15 s per photo and runs on
   Streamlit's script thread. The result must be cached in `st.session_state`,
   because Streamlit re-runs the whole script on every click.
3. **One Tesseract engine per process:** `ocr_reader.ENGINE` is loaded once at
   import and shared. It is not safe to use from two threads at once, so two
   browser tabs scanning at the same moment could collide. This is fine for one
   user on one laptop, but would need a lock or a worker queue to serve several.
4. **Pipeline code is read-only from the frontend:** `app.py` only calls the
   functions above. It never changes thresholds or rules in `barcode_reader`,
   `ocr_reader` or `validate`.
5. **No secrets or serials in code:** serials exist only in `data/` and in the
   browser session.

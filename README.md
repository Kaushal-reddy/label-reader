# Label Reader

[![CI](https://github.com/Kaushal-reddy/label-reader/actions/workflows/ci.yml/badge.svg)](https://github.com/Kaushal-reddy/label-reader/actions/workflows/ci.yml)

Reads serial numbers from phone photos of IT equipment labels (laptops, mice),
and product barcode numbers (EAN-13) from retail packaging, using classical computer vision only: **OpenCV**, **zxing-cpp** and **ZBar** (barcodes, via pyzbar)
and **Tesseract** (OCR, via pytesseract). No deep-learning models. Accepted
reads are then received into an asset-recording system (a mock ERP), and a small
Streamlit app puts it all behind a browser screen.

The design goal is **never to save a wrong serial**. A serial is accepted only
when the barcode and the printed text agree. Everything else is flagged
"needs review" with a reason and a best guess, so a person can check it quickly.

| | |
|---|---|
| **[BENCHMARKS.md](BENCHMARKS.md)** | the test data and every measured number |
| **[HOW_TO_RUN.md](HOW_TO_RUN.md)** | Docker, setup, every command, the app, the mock ERP, tests |

## Results

On 26 photos with hand-typed true values (12 serial-label photos of 5 devices,
14 photos of 7 retail products):

| Measure | Serial labels | Product barcodes | All |
|---|---|---|---|
| Accepted without a human | 6/12 | 13/14 | 19/26 (73%) |
| Accepted and **wrong** | **0** | **0** | **0** |
| Sent to review, best guess already correct | 4 of 6 | 1 of 1 | 5 of 7 |
| Time per photo, mean / max (MacBook Air) | 1.4 s / 2.9 s | 0.5 s / 1.7 s | 0.9 s / 2.9 s |

On a public set of 403 product-barcode images, 244 are accepted, again with 0
wrong. Per-stage results, the zxing-cpp comparison and the ERP demo run are in
[BENCHMARKS.md](BENCHMARKS.md).

## Try it

```bash
docker compose up --build         # then open http://localhost:8501
```

This starts the asset register and the mock ERP with a demo job, and the image
ships synthetic label photos to scan. Running without Docker, on your own
photos: [HOW_TO_RUN.md](HOW_TO_RUN.md).

## How it works

The numbers behind each design decision below are collected in [BENCHMARKS.md](BENCHMARKS.md#stage-level-measurements).

```
photo ─┬─► barcode reader (Stage 3) ───────────────────────┐
       │                                                   ├─► validate (Stage 6) ─► accepted / needs review
       └─► find label crops (4) ─► OCR the crops (5) ──────┘
```

### Stage 3 – Barcode reading (`barcode_reader.py`)
- **Two decoders: zxing-cpp first, ZBar only if zxing-cpp reads nothing.** A learned
  barcode detector (branch `barcode-detection`, measured with grouped 5-fold
  cross-validation) found the barcode in all 403 photos of the EAN set, yet ZBar read
  nothing in 178 of them. So ZBar's misses are **decoding** failures, not failures to
  find the barcode, and the detector's crops didn't fix them. zxing-cpp (Apache-2.0),
  an independent decoder with its own binarizer, read the right number on 279 of the
  403 photos (ZBar: 216) in about a tenth of the time. It missed 14 of the EAN photos
  ZBar reads, and the Code128 on `Mouse_1`, which is why ZBar stays as the backup.
  zxing-cpp reads the same 3 image variants as the confirmation step below and is
  counted the same way. Like ZBar, it sometimes misreads a barcode a second way at the
  same spot (a wrong EAN-13 on 8 of the 429 photos, from a box only 2–3 pixels thick).
  Those reads count as conflicts, so the photo goes to review. UPC-A, which zxing-cpp
  returns as 12 digits, is padded to the 13-digit form ZBar returns.
- ZBar decodes the photo; if it finds nothing, the photo is **upscaled 3×** and tried again.
  Upscaling was the only retry that rescued a photo. Rotation, sharpening and
  contrast boosting were tested on every photo (the "variant matrix") and helped none.
- ZBar is restricted to **Code128, Code39, QR and EAN-13**. When it tried every
  barcode type, it read the mousepad texture as an Interleaved-2-of-5 barcode
  (`132342`). That type has no mandatory checksum, so random stripes can pass as a
  valid code.
- **A checksum is not enough.** On a curved drink can ZBar read one EAN-13 barcode
  as two numbers from the same spot, and the wrong one also had a valid check digit.
  What separated them was ZBar's *quality*, which for bar codes counts the scan lines
  that agreed on a value: 122 votes for the true number, 14 for the wrong one (3.6×
  to 168× across every image it appeared in). So results at overlapping positions are
  treated as one barcode, and a value is kept only with **at least 3× the votes** of
  any rival. Either way the rival is reported as a conflict. A minimum-quality
  cut-off wouldn't work: correct reads went as low as 1–11 votes.
- **One read is not enough either.** On a 403-image set of product barcodes
  (`photos/ean_dataset`, values checked by hand), accepting the first clean read
  let **8 wrong EAN-13s through out of 216**. Each had a valid check digit and no
  rival at the same spot. Most swapped 1↔7 or 2↔8, digits whose bar patterns differ by
  one bar width, on small or blurry barcodes. Every one came from a single image
  variant. So when the reader finds a product barcode, it also reads it with
  `upscale2x` and `clahe_2x` (`CONFIRM_VARIANTS`). It counts how many variants read
  the same number, and any other product number they read counts as a conflict. Stage 6
  needs **2 agreeing variants**. On that set, with ZBar alone, this accepts 178 with
  0 wrong and sends 44 to review. Adding more variants without requiring agreement only added wrong
  reads (all 9 variants, stop at first read: 226 accepted, 14 wrong).

### Stage 4 – Finding the label (`label_finder.py`)
Labels come in both polarities (white label on a black mouse, dark label with
white text on a dark laptop), so "find the bright rectangle" can't work. What every
label has is **dense, sharp, high-contrast detail**:

1. Shrink to 1000 px tall (speed, and every kernel size below means the same thing at any resolution).
2. **Morphological gradient** (dilate − erode): bright at sharp edges, whatever the text colour.
3. Keep only **strong edges** (> 90).
4. **Edge density**: the fraction of edge pixels in each 21×21 window. Text is dense; texture specks are sparse.
5. **Closing** joins neighbouring text lines into one blob per label.
6. **Contours → filter**: drop blobs that are tiny, touch 2+ frame edges (background), or have few edges inside; keep the 3 biggest.
7. Cut each box out of the **full-resolution** photo, rotated upright.

The thresholds were chosen by measuring them on the photos. Otsu's automatic
threshold picked 32–66 here, because most of each photo is texture. Label text
measured 0.44–0.64 edge density and textures about 0.3, so the cut-off is 0.40.

### Stage 5 – OCR (`ocr_reader.py`)
Three methods run on every crop and all outputs are kept:
- **Whole crop**: the crop as one text block (`--psm 6`).
- **Line by line**: OpenCV finds each text line, and Tesseract reads each one alone (`--psm 7`).
- **Barcode digits**: for product barcodes only, the digits printed next to the bars (below).

Line by line exists because **a barcode inside a crop breaks Tesseract's layout
analysis**. A clean `S/N: PM2130H…` crop read as `aa` whole, but exactly
right once the line was cut out on its own. Per line:
- blobs taller than 2.5× the character height are skipped (barcodes, QR codes, logos), and so are lines too short to hold a 10-character serial;
- polarity is fixed per line (one crop can contain both dark-on-light and light-on-dark text);
- each line is resized so the text is ~30 px tall, the size Tesseract is tuned for;
- pieces on the same row with a small gap (≤ 1.5 character heights) are joined first, so a number printed in groups (`4 006381 333931`) becomes one line instead of short pieces that get thrown away;
- whether the crop is upside down is decided **once per crop**, from the summed Tesseract confidence of its 3 widest lines read both ways.

**Barcode digits.** The digits under a product barcode touch its long guard
bars, so the line finder joined digits and bars into one tall blob and skipped it
as "a barcode". Whole-crop OCR was confused by the bars as well. So for every
EAN-13 barcode ZBar finds in a crop, the barcode is used as a guide instead:
1. Find exactly where the bars start and end: walk outward from the middle of the
   barcode, row by row, while each row still has the same bar pattern as a row a
   few pixels further in.
2. Cut a thin strip just past **both** ends, because a crop can be upside down.
3. Erase the thin guard bars that reach into the strip.
4. OCR the strip with Tesseract limited to the digits 0–9, the right way up and upside down.
5. Keep only a 13-digit number with a valid EAN-13 check digit.

The barcode only says *where* to look; Tesseract reads the digits from the pixels
without seeing ZBar's value, so OCR and barcode are still two independent
readings when Stage 6 compares them. This method read 8 of the 12 product photos
exactly, with no wrong numbers.

**Calling Tesseract directly (`tess_direct.py`).** pytesseract starts a new
`tesseract.exe` for every call: it saves the image to a temp file, loads the
English model, reads, and exits. Profiling showed ~0.15 s per call, mostly that
overhead, and a photo needs dozens of calls. `tess_direct.py` uses Python's
`ctypes` to load Tesseract's own library (`libtesseract-5.dll` next to
`tesseract.exe` on Windows, `libtesseract.dylib` / `libtesseract.so.5` from
Homebrew / apt) into the Python process **once**, and passes images from memory
through Tesseract's C API. That's ~0.006 s per small image, about 28× faster.
It gave **identical text** to pytesseract on all 228 images the pipeline OCRs,
and identical results on every photo; average OCR time per photo fell from
3.3 s to 1.1 s. If the library can't be loaded, the code falls back to
pytesseract automatically (`check_setup.py` reports which path is in use).

### Stage 6 – Validation (`validate.py`)
1. **Barcode candidates**: serial-shaped pieces of the decoded values (the ASUS QR codes hold a URL ending in the serial), or a valid EAN-13 number.
2. **OCR candidates**: text right after `SN` / `S/N:` first, then any serial-shaped word, then any valid EAN-13 number (digit groups separated by spaces or dashes are joined first).
3. **Pattern**: a serial is 10–20 letters/digits, at least one letter and two digits. Deliberately generic, not built from the test serials. A product number must be exactly 13 digits whose last digit is the correct EAN-13 check digit; that catches any single wrong digit (but not every combination of several).
4. **Decision**: accept when a barcode candidate and an OCR candidate agree, allowing for OCR look-alikes (O/0, I/1, A/4, …). The **barcode's** characters are saved, because barcodes carry a checksum and OCR doesn't.
5. **Product barcodes** (EAN-13, on retail packaging): the only matching text is the same digits printed small under the bars. The barcode-digit OCR (Stage 5) reads them on many photos, and then step 4 accepts as usual. When it can't (curved packaging, glare), an EAN-13 is accepted **on its own**, but only if its check digit is valid, it was read **one way only**, and **at least 2 image variants** read the same number (Stage 3). Anything else sends it to review, with the majority value as the best guess.

Running OCR twice does not count as two sources: both methods use the same engine
on the same crop and make the same mistakes. Accepting OCR-only results would have
saved one wrong serial (`Mouse_1_Sideangle`: OCR read `SCBMNVO00…` instead of `SCBMNV00…`, an extra `O`).

`truth.csv` is never used to make a decision, only to measure afterwards.

## ERP integration

Stage 9 sends the pipeline's decisions to an asset-recording system: an accepted
serial that matches a line on the job's manifest is received automatically,
and everything else goes to a review queue. The real target system's API isn't
public, so `mock_erp.py` is a **generic stand-in written for this project**. It
is not any vendor's API; it only has the parts an integration has to cope with.
Commands: [HOW_TO_RUN.md](HOW_TO_RUN.md#the-mock-erp); a demo run under injected failures: [BENCHMARKS.md](BENCHMARKS.md#erp-demo-run).

```
validation.csv ─► post_to_erp.py ─┬─► outbox ─► erp_client.py ─► mock_erp.py (HTTP)
                                  └─► review_queue.csv
```

- **Mock ERP** (`mock_erp.py`, FastAPI + SQLite): jobs, manifest lines (identifier,
  `serial` or `ean`, quantity, received) and receipts. Access needs a bearer token
  from `POST /oauth/token` (client id + secret, read from `ERP_CLIENT_ID` /
  `ERP_CLIENT_SECRET`). `POST /api/v1/receipts` needs an `Idempotency-Key`: the
  same key and body again returns the saved receipt (200), the same key with a
  different body is refused (422), and receiving more than the line's quantity is
  refused (409). The schema is in `docs/erp_openapi.json` (`python -m label_reader.erp.mock_erp openapi`).
- **Failure injection**, to test error recovery: `--fail-rate` (random 503),
  `--lost-rate` (receipt saved, but the reply is a 500), `--slow-rate` /
  `--slow-seconds`, `--rate-limit` (429 with `Retry-After`), `--token-ttl`, `--seed`.
- **Client** (`erp_client.py`): renews the token before it expires and once on a
  401, retries network errors, timeouts, 5xx and 429 with exponential backoff and
  jitter (or the server's `Retry-After`), and raises `ErpRejected` for other 4xx
  without retrying. Retrying a POST is safe because of the idempotency key.
- **Matching** (`post_to_erp.py`): identifiers are compared as uppercase letters
  and digits. A photo is posted only if the pipeline **accepted** it, **exactly one**
  line has **exactly** that identifier, and the line still has quantity left.
  Otherwise it goes to `review_queue.csv` with a category: `not_confident`
  (with a hint when the best guess matches a line), `near_match` (OCR look-alikes
  or one character different: never posted), `not_on_manifest`, `several_lines`,
  `already_received` (e.g. the second photo of the same item) or `erp_rejected`.
- **Outbox** (`output/erp/outbox.sqlite`): each post is saved as pending with its
  idempotency key **before** it is sent. If the ERP is down, posts stay pending
  with the last error, and the next run sends them again with the same keys.

## CI/CD

GitHub Actions ([`.github/workflows/`](.github/workflows/)) runs on every push to `main` and every pull request:

| Job | What it checks |
|---|---|
| Lint, privacy guard, API schema | `ruff`; that no photos, `truth.csv`, outputs, databases or model weights are tracked; that `docs/erp_openapi.json` matches the code |
| Tests | `check_setup` and all 63 tests on Ubuntu (Python 3.12, 3.13, 3.14), macOS and Windows, each with its own Tesseract and ZBar install. Coverage in the job summary |
| Pipeline + ERP end to end | Every stage's command (`baseline` to `analyze`, and the decoder experiment) on the synthetic photos, then `scripts/erp_e2e.py` against a mock ERP failing 30% of requests. The evaluation summary goes in the job summary and the outputs are kept as an artifact |
| Package | Builds the wheel and sdist and installs the wheel in a clean venv |
| Docker | Builds both images, checks they contain no private files, runs `check_setup` and the pipeline inside the app image, then `docker compose up` and checks both services answer and the demo job exists |

The synthetic photos can't show real-world accuracy (that's what the private
photos are for), but they test the properties that matter on every OS: nothing
is accepted wrongly, a label whose printed serial disagrees with its barcode
always goes to review, and every accepted photo reaches the ERP exactly once.

**Releases:** a version tag (`v0.2.0`) runs the whole CI, then publishes a GitHub Release with the
wheel and both Docker images (amd64 + arm64) to `ghcr.io` ([how](HOW_TO_RUN.md#releasing)).
Dependabot opens weekly pull requests for outdated Python packages, actions and base images.

## Limitations

- **Small test set**: 12 serial-label photos of 5 devices (3 laptops, 2 mice) and 14 product photos of 7 products. Within a group, one photo moves a rate by 7–8 points.
- **Tuned on the test set**: thresholds (edge strength 90, density 0.40, line sizes, the digit-strip size) were chosen by looking at these same photos, the look-alike list includes `A/4` because it showed up here, and the 3× vote margin in Stage 3 was set from the one phantom read seen. There is no held-out set, so real-world accuracy is probably lower.
- **One hand-edited photo**: `Mouse_2_Sideangle.jpeg` was cropped by hand to remove books from the background. Its label is still not found.
- **Known failures** (details in [BENCHMARKS.md](BENCHMARKS.md#results-on-the-26-photos)): dim or low-contrast labels that aren't found, OCR too far off to confirm a barcode, and steep side angles, because the crop corrects rotation but not perspective.
- **Barcode-only acceptance for products**: product barcodes are accepted without a second source. The check digit, vote margin and 2-variant agreement stopped every wrong read seen (ZBar: 1 in the 14 product photos and 8 in the 403-image EAN set; zxing-cpp: 3 and 5), but the agreement rule was chosen on that same EAN set, and the zxing-cpp-first order was chosen after measuring on the same photos, so neither has been tested on unseen photos. A wrong number read the same way by two variants would still be accepted. Confirming also costs time: zxing-cpp always reads 3 image variants, and ZBar re-reads a product barcode it found.
- **Image quality matters most**: replacing downscaled photos with the camera originals did more for barcode reading than any code change, and very large photos only cost time ([measured](BENCHMARKS.md#stage-level-measurements)).
- **Speed**: about 0.9 s per photo on a MacBook Air (2.9 s worst case). On serial labels the biggest parts are OCR and, when zxing-cpp reads nothing, ZBar's retry, which decodes a 3× upscale of the full photo.

## What I would do next

1. **More photos and a held-out set**: tune on one half, measure on the other.
2. **Perspective correction**: find the label's 4 corners and use `cv2.getPerspectiveTransform` so side-angle barcodes become flat.
3. **Per-manufacturer serial formats**: e.g. if a format says position 4 is always a digit, an OCR `O` there can be corrected to `0` safely.
4. **Faster barcode step**: decode the small label crops first (0.2 s instead of 1.2 s in testing) and fall back to the full photo only when that fails. Crops alone missed one serial-label photo whose label wasn't found, but read 13 of 14 product photos.

## Project layout

```
app.py                     Streamlit asset register: Scan, Review, Assets, ERP screens
label_reader/              the pipeline (each stage runs with python -m label_reader.<stage>)
  native_libs.py   setup    finds Tesseract and ZBar on Windows / macOS / Linux
  baseline.py      Stage 2  untuned barcode + OCR on raw photos; shared helpers normalize(), found(), valid_ean13(), load_truth()
  barcode_reader.py Stage 3 read_barcode(): zxing-cpp, then ZBar with retries + barcode-type filter
  label_finder.py  Stage 4  find_label_boxes(), crop_rotated()
  ocr_reader.py    Stage 5  read_ocr(): whole-crop + line-by-line + barcode-digit OCR
  tess_direct.py   Stage 5  TessEngine: Tesseract's C API via ctypes (no process per call)
  validate.py      Stage 6  the pipeline: candidates, pattern check, agreement, CSVs
  evaluate.py      Stage 7  accuracy per stage + timing → output/evaluation/
  analyze.py       Stage 8  figures, confusion matrices, tables → output/analysis/
  register.py      app      the asset register's storage (SQLite) and ERP link, no Streamlit
  erp/             Stage 9
    mock_erp.py             mock ERP server (generic stand-in): seed / serve / openapi
    erp_client.py           ErpClient: tokens, retries with backoff, idempotent receipts
    post_to_erp.py          match results to the manifest, outbox, post + review queue
scripts/                   setup check, dataset tools, experiments, CI checks (python -m scripts.<name>)
  check_setup.py   Stage 0  checks OpenCV, pyzbar, zxing-cpp, Tesseract (both paths), pandas, matplotlib
  shrink_photos.py          shrinks photos over 4032 px; originals → photos_original/
  prepare_ean_dataset.py    downloaded barcode set → photos/ean_dataset/ + review sheets
  zxing_experiment.py       zxing-cpp vs ZBar: decoder policies measured through Stage 6
  make_synthetic_dataset.py synthetic label photos with fake serials → photos/synthetic/
  erp_e2e.py                Stage 9 end to end against a failing mock ERP (CI)
tests/                     pytest tests: barcode reader, pipeline end to end, Stage 9, asset register, app (fake serials only)
README.md, BENCHMARKS.md, HOW_TO_RUN.md   this overview, the measurements, the commands
docs/
  asset_register.md         how app.py uses the pipeline, its database and the ERP
  erp_openapi.json          the mock ERP's API schema
pyproject.toml             package metadata, ruff and pytest settings
requirements.txt           exact versions (requirements-erp.txt: ERP + tests; requirements-dev.txt: ruff, coverage, build)
Dockerfile                 two images: the app (with Tesseract + ZBar) and the mock ERP
docker-compose.yml         both together, with a demo job
.github/                   CI and release workflows, Dependabot
photos/                    input photos (not committed; see BENCHMARKS.md, Test data)
truth.csv                  true serials (not committed); format in truth.example.csv
output/                    everything generated (git-ignored, re-created by the scripts)
data/                      the asset register's database and scans (git-ignored)
```

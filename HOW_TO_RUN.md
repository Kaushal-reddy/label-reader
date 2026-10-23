# How to run

- [Quick start with Docker](#quick-start-with-docker)
- [Setup without Docker](#setup-without-docker)
- [Your own photos](#your-own-photos)
- [Running the pipeline](#running-the-pipeline)
- [Other datasets](#other-datasets)
- [Reproducing the benchmarks](#reproducing-the-benchmarks)
- [The asset register app](#the-asset-register-app)
- [The mock ERP](#the-mock-erp)
- [Tests and checks](#tests-and-checks)
- [Releasing](#releasing)

## Quick start with Docker

The asset register and the mock ERP run together with one command, with
Tesseract and ZBar already installed in the image:

```bash
docker compose up --build         # then open http://localhost:8501
```

The ERP starts with a demo job (`JOB-DEMO`) whose manifest matches the synthetic
photos built into the app image. Copy them out with
`docker compose cp app:/app/demo ./demo` and upload them on the Scan screen.
The pipeline also runs inside the image:

```bash
docker compose run --rm app python -m label_reader.validate demo
```

The images hold no real photos or serials: `.dockerignore` only lets the code
in, and CI checks the built images.

Released images are on GitHub's container registry (`ghcr.io/kaushal-reddy/label-reader-app`
and `label-reader-mock-erp`, for amd64 and arm64).
`IMAGE_PREFIX=ghcr.io/kaushal-reddy docker compose pull` uses them instead of building.

## Setup without Docker

Works on Windows, macOS and Linux. You need Python 3.12 or newer, plus two
native programs, Tesseract 5 and ZBar, which pip can't install.

1. **Native dependencies**

   | OS | Command |
   |---|---|
   | Windows | `winget install UB-Mannheim.TesseractOCR` (ZBar ships inside the `pyzbar` package) |
   | macOS | `brew install tesseract zbar` |
   | Ubuntu / Debian | `sudo apt install tesseract-ocr libzbar0` |

   The Windows installer doesn't add Tesseract to PATH, and apps started outside a
   terminal may not see Homebrew's PATH. `native_libs.py` handles both: it checks
   PATH first, then the standard install folders, and on Apple Silicon Macs
   points pyzbar at Homebrew's `/opt/homebrew/lib`.
   On Windows, if importing pyzbar fails, install the Microsoft Visual C++ 2013 redistributable.

2. **Python packages** in a virtual environment:

   ```bash
   python -m venv .venv
   # Windows:        .venv\Scripts\activate
   # macOS / Linux:  source .venv/bin/activate
   pip install -r requirements.txt
   pip install -r requirements-erp.txt   # the mock ERP, the app's ERP link and the tests
   pip install -r requirements-dev.txt   # optional: ruff, coverage, build
   ```

   This also installs zxing-cpp, the second barcode decoder. Its decoder ships
   inside the pip package, so it needs no separate install.

3. **Check everything works** (all checks should say `[OK]`):

   ```bash
   python -m scripts.check_setup
   ```

All commands below assume the venv is active and you're in the repo folder
(the stages run as modules, `python -m label_reader.<stage>`, so they find each
other without installing anything; `pip install -e .` also works).

## Your own photos

Put the photos in `photos/` and create `truth.csv` next to this file, with one
row per photo (format in `truth.example.csv`: `filename,true_serial`). The true
values are only used to measure the pipeline, never to decide.

- Save `truth.csv` as plain text. Opened and saved in Excel, long numbers become
  e.g. `9.34E+12` and the real digits are lost; `load_truth()` refuses such
  values. Rows whose photo is missing are skipped with a warning.
- Photos larger than 4032 px only make the pipeline slower
  ([measured](BENCHMARKS.md#stage-level-measurements)). Shrink them once after
  adding them; the originals are moved to `photos_original/`:

  ```bash
  python -m scripts.shrink_photos
  ```

No photos of your own? `python -m scripts.make_synthetic_dataset` draws 14
synthetic label photos with fake serials into `photos/synthetic/`; see
[Other datasets](#other-datasets).

## Running the pipeline

With `photos/` and `truth.csv` in place:

| Command | What it does | Main output |
|---|---|---|
| `python -m label_reader.validate` | **The pipeline**: reads every photo and decides | `output/results.csv`, `output/needs_review.csv` |
| `python -m label_reader.evaluate` | Measures every stage against `truth.csv`, and times it | `output/evaluation/summary.md` |
| `python -m label_reader.analyze` | Figures, confusion matrices and tables from `evaluate.py`'s results ([list](BENCHMARKS.md#figures-and-tables)) | `output/analysis/report.md` |
| `python -m label_reader.baseline` | Stage 2: untuned barcode + OCR on raw photos | `output/baseline/` |
| `python -m label_reader.barcode_reader` | Stage 3: barcode variant matrix (ZBar) | `output/barcode/variant_matrix.csv` |
| `python -m scripts.zxing_experiment` | Stage 3: zxing-cpp vs ZBar, 6 decoder policies through Stage 6 (run `validate.py` first; reuses its OCR) | `output/zxing/summary.md` |
| `python -m label_reader.label_finder` | Stage 4: label boxes, crops and debug images | `output/labels/boxes/`, `crops/`, `debug/` |
| `python -m label_reader.ocr_reader` | Stage 5: OCR method comparison | `output/ocr/variant_matrix.csv`, `text/`, `lines/` |

Every stage saves its intermediate images to `output/` so failures can be inspected by eye.

## Other datasets

Every command above also takes a dataset folder holding its own photos and
`truth.csv`. Results go to `output/<folder name>/` instead, and `truth.csv` rows
whose `checked` column isn't `yes` are skipped. To keep the output manageable, a
dataset run skips the per-variant and per-step debug images, and `analyze.py`
draws no one-row-per-photo figure above 60 photos (the CSV tables are still written).

**Synthetic photos** (fake serials, nothing to download):

```bash
python -m scripts.make_synthetic_dataset          # → photos/synthetic/
python -m label_reader.validate photos/synthetic
```

**The 403-image EAN set**: `python -m scripts.prepare_ean_dataset` builds
`photos/ean_dataset/` (410 single-barcode images) from the downloaded
`photos/Barcode_Dataset`, runs the variant matrix, and writes a pre-filled
`truth.csv` + review sheets to check by hand. Then:

```bash
python -m label_reader.baseline photos/ean_dataset
python -m label_reader.barcode_reader photos/ean_dataset   # also prints what Stage 6 accepts on the barcode alone
python -m label_reader.label_finder photos/ean_dataset
python -m label_reader.ocr_reader photos/ean_dataset
python -m label_reader.validate photos/ean_dataset
python -m label_reader.evaluate photos/ean_dataset
python -m label_reader.analyze photos/ean_dataset          # after evaluate.py
```

## Reproducing the benchmarks

The numbers in [BENCHMARKS.md](BENCHMARKS.md) come from:

| Numbers | Commands |
|---|---|
| Results on the 26 photos | `python -m label_reader.evaluate` (tables in `output/evaluation/summary.md`), then `python -m label_reader.analyze` |
| Adding zxing-cpp | `python -m label_reader.validate`, then `python -m scripts.zxing_experiment` (same with `photos/ean_dataset`) |
| The EAN set | the [EAN set commands](#other-datasets) above |
| Synthetic photos | `python -m pytest tests/test_pipeline_e2e.py`, or the synthetic commands above |
| ERP demo run | the [demo](#demo-run-with-failures) below |

The 26 photos themselves are private (real serial numbers), so those numbers can
only be re-run on your own photos.

## The asset register app

A small "mini-ERP" in the browser: scan a photo, review uncertain reads, search
the register, assign items to people, and send items to the ERP. How it's built
is in [docs/asset_register.md](docs/asset_register.md).

```bash
streamlit run app.py              # opens http://localhost:8501
```

It needs `requirements-erp.txt` too. Its database and copies of the scanned
photos go to `data/` (git-ignored, because they hold real serials). For the ERP
screens, also start the mock ERP (below); without it, scans are still saved and
can be sent later.

## The mock ERP

`mock_erp.py` is a generic stand-in for an asset-recording system, written for
this project (the design is in the [README](README.md#erp-integration)).

| Command | What it does | Main output |
|---|---|---|
| `python -m label_reader.erp.mock_erp seed --job JOB-0001 --from truth.csv` | Creates (or resets) a job with one manifest line per item; `--leave-out Mouse_2` drops an item | `output/erp/erp.sqlite` |
| `python -m label_reader.erp.mock_erp serve` | Runs the mock ERP on `localhost:8000` (API docs at `/docs`) | – |
| `python -m label_reader.erp.post_to_erp --job JOB-0001` | Stage 9: posts accepted, matched photos to the ERP and queues the rest for review | `output/erp/posted.csv`, `review_queue.csv` |
| `python -m label_reader.erp.mock_erp openapi` | Writes the API schema | `docs/erp_openapi.json` |

Credentials come from `ERP_CLIENT_ID` / `ERP_CLIENT_SECRET` (and the address
from `ERP_URL`); without them, development defaults are used with a warning.

**Failure injection**, to test error recovery: `serve` takes `--fail-rate`
(random 503), `--lost-rate` (receipt saved, but the reply is a 500),
`--slow-rate` / `--slow-seconds`, `--rate-limit` (429 with `Retry-After`),
`--token-ttl` and `--seed`.

### Demo run with failures

```bash
python -m label_reader.erp.mock_erp seed --job JOB-0001 --from truth.csv --leave-out Mouse_2
python -m label_reader.erp.mock_erp serve --fail-rate 0.3 --lost-rate 0.3 --rate-limit 5 --seed 7
python -m label_reader.erp.post_to_erp --job JOB-0001        # in a second terminal
```

The result is in [BENCHMARKS.md](BENCHMARKS.md#erp-demo-run). Run `post_to_erp`
again and it posts nothing new.

`serve --host 0.0.0.0` makes the mock reachable from other computers on your
network. Its database holds real serials when seeded from `truth.csv`, so only
do that on a network you trust.

## Tests and checks

| Command | What it checks |
|---|---|
| `python -m pytest` | 63 tests: the barcode reader (synthetic barcodes), the whole pipeline on synthetic label photos, the mock ERP, the client, Stage 9, the asset register and the Streamlit app's screens (fake serials only) |
| `python -m scripts.erp_e2e` | Stage 9 end to end with real processes: the mock ERP failing on purpose, `post_to_erp` until nothing is pending, then checks every accepted photo was received exactly once (run `validate.py photos/synthetic` first) |
| `ruff check .` | Lint (needs `requirements-dev.txt`) |
| `python -m scripts.check_setup` | The native dependencies and both Tesseract paths |

CI runs all of these on every push and pull request; see the
[README](README.md#cicd).

## Releasing

Set `version` in `pyproject.toml`, commit, then push a matching tag:

```bash
git tag v0.2.0 && git push origin v0.2.0
```

The release workflow runs the whole CI, then creates a GitHub Release with the
wheel, sdist and generated notes, and pushes both Docker images (amd64 + arm64,
with SBOM and a signed build-provenance attestation) to `ghcr.io`. A tag that
doesn't match `pyproject.toml` fails before anything is published.

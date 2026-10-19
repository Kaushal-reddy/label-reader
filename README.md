# Label Reader

Reads serial numbers from phone photos of IT equipment labels (laptops, mice),
using classical computer vision only: **OpenCV**, **ZBar** (barcodes, via pyzbar)
and **Tesseract** (OCR, via pytesseract). No deep-learning models.

The design goal is **never to save a wrong serial**. A serial is accepted only
when the barcode and the printed text agree. Everything else is flagged
"needs review" with a reason and a best guess, so a person can check it quickly.

## Results

Measured on 12 photos of 5 items, against hand-typed true serials in `truth.csv`.
Run `evaluate.py` to reproduce; the full tables are written to `output/evaluation/summary.md`.

**Final decisions**

| Measure | Result |
|---|---|
| Accepted without a human | 6/12 (50%) |
| Accepted and **wrong** | **0** |
| Precision (correct / accepted) | 6/6 (100%) |
| Sent to review | 6/12, of which 4 already have the correct best guess |
| Time per photo | 4.5 s mean, 7.9 s max (median of 3 runs, Windows 11 laptop) |

**What each stage added**

| Stage | Measure | Result |
|---|---|---|
| 2 – baseline | Barcode on the raw photo | 7/12 |
| 2 – baseline | OCR on the raw photo (serial anywhere in the text) | 2/12 |
| 3 – barcode | Barcode with retries + barcode-type filter | 8/12 |
| 4 – label crop | Photos where a label crop was found | 10/12 |
| 5 – OCR | OCR on the crops, exact | 6/12 |
| 5 – OCR | OCR on the crops, within 1 character | 9/12 |

Cropping the label and reading it line by line **tripled** exact OCR reads
(2 → 6). The near misses are almost all look-alike characters (letter `O` vs
digit `0`, `A` vs `4`), which is what the barcode cross-check in Stage 6 catches.
For example, on `Laptop_3` OCR read `TCNOCV…` (letter O); the barcode confirmed it
and supplied the correct `TCN0CV…` (digit 0). (Serials are shortened in this README; see [Data](#data).)

## How it works

```
photo ─┬─► barcode reader (Stage 3) ───────────────────────┐
       │                                                   ├─► validate (Stage 6) ─► accepted / needs review
       └─► find label crops (4) ─► OCR the crops (5) ──────┘
```

### Stage 3 – Barcode reading (`barcode_reader.py`)
- ZBar decodes the photo; if it finds nothing, the photo is **upscaled 3×** and tried again.
  Upscaling was the only retry that rescued a photo. Rotation, sharpening and
  contrast boosting were tested on every photo (the "variant matrix") and helped none.
- ZBar is restricted to **Code128, Code39, QR and EAN-13**. When it tried every
  barcode type, it read the mousepad texture as an Interleaved-2-of-5 barcode
  (`132342`). That type has no mandatory checksum, so random stripes can pass as a
  valid code.

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
Two methods run on every crop and both outputs are kept:
- **Whole crop**: the crop as one text block (`--psm 6`).
- **Line by line**: OpenCV finds each text line, and Tesseract reads each one alone (`--psm 7`).

Line by line exists because **a barcode inside a crop breaks Tesseract's layout
analysis**. A clean `S/N: PM2130H…` crop read as `aa` whole, but exactly
right once the line was cut out on its own. Per line:
- blobs taller than 2.5× the character height are skipped (barcodes, QR codes, logos), and so are lines too short to hold a 10-character serial;
- polarity is fixed per line (one crop can contain both dark-on-light and light-on-dark text);
- each line is resized so the text is ~30 px tall, the size Tesseract is tuned for;
- whether the crop is upside down is decided **once per crop**, from the summed Tesseract confidence of its 3 widest lines read both ways.

### Stage 6 – Validation (`validate.py`)
1. **Barcode candidates**: serial-shaped pieces of the decoded values (the ASUS QR codes hold a URL ending in the serial).
2. **OCR candidates**: text right after `SN` / `S/N:` first, then any serial-shaped word.
3. **Pattern**: 10–20 letters/digits, at least one letter and two digits. Deliberately generic, not built from the test serials.
4. **Decision**: accept only when a barcode candidate and an OCR candidate agree, allowing for OCR look-alikes (O/0, I/1, A/4, …). The **barcode's** characters are saved, because barcodes carry a checksum and OCR doesn't.

Running OCR twice does not count as two sources: both methods use the same engine
on the same crop and make the same mistakes. Accepting OCR-only results would have
saved one wrong serial (`Mouse_1_Sideangle`: OCR read `SCBMNVO00…` instead of `SCBMNV00…`, an extra `O`).

`truth.csv` is never used to make a decision, only to measure afterwards.

## Setup (Windows)

1. **Python 3.12** and a virtual environment:
   ```powershell
   python -m venv .venv
   .venv\Scripts\pip install -r requirements.txt
   ```
2. **Tesseract 5**: install the Windows build from UB Mannheim (this project used 5.4.0):
   ```powershell
   winget install UB-Mannheim.TesseractOCR
   ```
   The installer doesn't add Tesseract to PATH. The scripts look for `tesseract`
   on PATH first, then fall back to `C:\Program Files\Tesseract-OCR\tesseract.exe`.
3. **ZBar**: nothing to install on Windows; the DLL ships inside the `pyzbar` package.
   (If importing pyzbar fails, install the Microsoft Visual C++ 2013 redistributable.)
   On macOS: `brew install tesseract zbar`. On Ubuntu: `sudo apt install tesseract-ocr libzbar0`.
4. Check everything works:
   ```powershell
   .venv\Scripts\python.exe check_setup.py
   ```

## Data

The photos and true serials are **not in this repository**, because they contain
the real serial numbers of my own devices. The test set was:

| Item | Photos | Label | Barcode |
|---|---|---|---|
| Laptop 1 (ASUS) | 2: straight-on (label cut off at the frame edge), rotated + dim | Dark, white text | QR (URL ending in the serial), visible in one photo |
| Laptop 2 (ASUS) | 3: straight-on, rotated 90°, glare | Dark, white text | QR (URL ending in the serial) |
| Laptop 3 (ASUS) | 1: straight-on | Dark, white text, on a silver laptop | Code128 + QR |
| Mouse 1 (ASUS) | 3: straight-on, side angle, variant | White, black text | Code128 |
| Mouse 2 (Razer) | 3: straight-on, side angle, variant | Dark blue + silver strip, light text | Code128 |

All photos were taken on a phone and used at the camera's original resolution
(4032×2268), except `Mouse_1_variant` (3309×1861) and `Mouse_2_Sideangle`
(hand-cropped to 2268×2448, see Limitations).

To run the project on your own photos, put them in `photos/` and create
`truth.csv` (see `truth.example.csv` for the format: `filename,true_serial`, one
row per photo).

## Usage

With `photos/` and `truth.csv` in place:

| Command | What it does | Main output |
|---|---|---|
| `python validate.py` | **The pipeline**: reads every photo and decides | `output/results.csv`, `output/needs_review.csv` |
| `python evaluate.py` | Measures every stage against `truth.csv`, and times it | `output/evaluation/summary.md` |
| `python baseline.py` | Stage 2: untuned barcode + OCR on raw photos | `output/baseline/` |
| `python barcode_reader.py` | Stage 3: barcode variant matrix | `output/barcode/variant_matrix.csv` |
| `python label_finder.py` | Stage 4: label boxes, crops and debug images | `output/labels/boxes/`, `crops/`, `debug/` |
| `python ocr_reader.py` | Stage 5: OCR method comparison | `output/ocr/variant_matrix.csv`, `text/`, `lines/` |

(Use `.venv\Scripts\python.exe` in place of `python` if the venv isn't activated.)
Every stage saves its intermediate images to `output/` so failures can be inspected by eye.

## Limitations

- **Small test set**: 12 photos of 5 items (3 laptops, 2 mice). Percentages move by 8 points per photo.
- **Tuned on the test set**: thresholds (edge strength 90, density 0.40, line sizes) were chosen by looking at these same 12 photos, and the look-alike list includes `A/4` because it showed up here. There is no held-out set, so real-world accuracy is probably lower.
- **One hand-edited photo**: `Mouse_2_Sideangle.jpeg` was cropped by hand to remove books from the background. Its label is still not found.
- **Known failures**:
  - `Laptop_1_variant`, `Mouse_2_Sideangle`: label not found (dim, small, or low-contrast print on a dark label).
  - `Laptop_2_Vertical`: OCR too far off (4 characters) to confirm the barcode.
  - Steep side angles defeat the barcode reader: the crop corrects rotation but not perspective.
- **Image quality matters most**: the first set of photos had been downscaled to 900×1600 in transfer. Replacing them with the camera originals (4032×2268) did more for barcode reading than any code change.
- **Speed**: about 4.5 s per photo, mostly OCR. Each pytesseract call starts a new `tesseract.exe` process (~0.15 s).

## What I would do next

1. **More photos and a held-out set**: tune on one half, measure on the other.
2. **Perspective correction**: find the label's 4 corners and use `cv2.getPerspectiveTransform` so side-angle barcodes become flat.
3. **Per-manufacturer serial formats**: e.g. if a format says position 4 is always a digit, an OCR `O` there can be corrected to `0` safely.
4. **Faster OCR**: call the Tesseract API directly (e.g. `tesserocr`) to avoid starting a process per line.

## Project layout

```
check_setup.py     Stage 0  checks OpenCV, pyzbar, pytesseract, pandas
baseline.py        Stage 2  untuned barcode + OCR on raw photos; shared helpers normalize()/found()
barcode_reader.py  Stage 3  read_barcode(): retries + barcode-type filter
label_finder.py    Stage 4  find_label_boxes(), crop_rotated()
ocr_reader.py      Stage 5  read_ocr(): whole-crop + line-by-line OCR
validate.py        Stage 6  the pipeline: candidates, pattern check, agreement, CSVs
evaluate.py        Stage 7  accuracy per stage + timing → output/evaluation/
photos/            input photos (not committed; see Data)
truth.csv          true serials (not committed); format in truth.example.csv
output/            everything generated (git-ignored, re-created by the scripts)
```

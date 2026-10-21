# Label Reader

Reads serial numbers from phone photos of IT equipment labels (laptops, mice),
and product barcode numbers (EAN-13) from retail packaging, using classical computer vision only: **OpenCV**, **ZBar** (barcodes, via pyzbar)
and **Tesseract** (OCR, via pytesseract). No deep-learning models.

The design goal is **never to save a wrong serial**. A serial is accepted only
when the barcode and the printed text agree. Everything else is flagged
"needs review" with a reason and a best guess, so a person can check it quickly.

## Results

Measured on 24 photos against hand-typed true values in `truth.csv`: 12 photos of
5 IT devices (serial numbers) and 12 photos of 6 retail products (EAN-13 numbers).
Run `evaluate.py` to reproduce; the full tables are written to `output/evaluation/summary.md`,
and `analyze.py` turns them into figures and confusion matrices (`output/analysis/report.md`).

**Final decisions**

| Measure | Result |
|---|---|
| Accepted without a human | 15/24 (62%): devices 6/12, products 9/12 |
| Accepted and **wrong** | **0** |
| Precision (correct / accepted) | 15/15 (100%) |
| Sent to review | 9/24, of which 6 already have the correct best guess |
| Time per photo | 1.8 s mean, 6.5 s max on a Windows 11 laptop (median of 3 runs) |

**What each stage added**

| Stage | Measure | Result |
|---|---|---|
| 2 – baseline | Barcode on the raw photo | 18/24 |
| 2 – baseline | OCR on the raw photo (serial anywhere in the text) | 3/24 |
| 3 – barcode | Barcode with retries + barcode-type filter | 19/24 |
| 4 – label crop | Photos where a label crop was found | 22/24 |
| 5 – OCR | OCR on the crops: whole crop, exact | 3/24 |
| 5 – OCR | OCR on the crops: line by line, exact | 7/24 |
| 5 – OCR | OCR on the crops: digits next to a product barcode, exact | 8/24 |
| 5 – OCR | OCR on the crops: all methods combined, exact | 15/24 |
| 5 – OCR | OCR on the crops: combined, within 1 character | 18/24 |

On the 12 device photos, cropping the label and reading it line by line
**tripled** exact OCR reads (2 → 6). The near misses are almost all look-alike characters (letter `O` vs
digit `0`, `A` vs `4`), which is what the barcode cross-check in Stage 6 catches.
For example, on `Laptop_3` OCR read `TCNOCV…` (letter O); the barcode confirmed it
and supplied the correct `TCN0CV…` (digit 0). (Serials are shortened in this README; see [Data](#data).)

On the product photos, the first version of the pipeline accepted only 2 of 12:
the digits under a product barcode touch its bars, so OCR never saw them as text.
Reading them in strips guided by the barcode (see Stage 5) raised that to 9 of 12,
still with no wrong acceptances.

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
4. **Decision**: accept only when a barcode candidate and an OCR candidate agree, allowing for OCR look-alikes (O/0, I/1, A/4, …). The **barcode's** characters are saved, because barcodes carry a checksum and OCR doesn't.

Running OCR twice does not count as two sources: both methods use the same engine
on the same crop and make the same mistakes. Accepting OCR-only results would have
saved one wrong serial (`Mouse_1_Sideangle`: OCR read `SCBMNVO00…` instead of `SCBMNV00…`, an extra `O`).

`truth.csv` is never used to make a decision, only to measure afterwards.

## Setup

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
   ```

3. **Check everything works** (all checks should say `[OK]`):

   ```bash
   python check_setup.py
   ```

All commands below assume the venv is active and you're in the repo folder.

## Data

The photos and true serials are **not in this repository**, because they contain
the real serial numbers of my own devices. The test set was:

**IT devices** (serial numbers):

| Item | Photos | Label | Barcode |
|---|---|---|---|
| Laptop 1 (ASUS) | 2: straight-on (label cut off at the frame edge), rotated + dim | Dark, white text | QR (URL ending in the serial), visible in one photo |
| Laptop 2 (ASUS) | 3: straight-on, rotated 90°, glare | Dark, white text | QR (URL ending in the serial) |
| Laptop 3 (ASUS) | 1: straight-on | Dark, white text, on a silver laptop | Code128 + QR |
| Mouse 1 (ASUS) | 3: straight-on, side angle, variant | White, black text | Code128 |
| Mouse 2 (Razer) | 3: straight-on, side angle, variant | Dark blue + silver strip, light text | Code128 |

**Retail products** (EAN-13 barcode numbers): 6 items, each photographed
horizontal and vertical: an energy-drink can (`Monster`), a notebook, a box
(`Box1`), a supplement container, and two bottles. 12 photos.

All photos were taken on a phone and used at the camera's original resolution:
4032×2268 for the devices, except `Mouse_1_variant` (3309×1861) and
`Mouse_2_Sideangle` (hand-cropped to 2268×2448, see Limitations), and 8064×4536
for the products.

`truth.csv` must be saved as plain text. Opened and saved in Excel, long numbers
become e.g. `9.34E+12` and the real digits are lost; `load_truth()` refuses such
values. Rows whose photo is missing are skipped with a warning.

To run the project on your own photos, put them in `photos/` and create
`truth.csv` (see `truth.example.csv` for the format: `filename,true_serial`, one
row per photo).

## Usage

With `photos/` and `truth.csv` in place:

| Command | What it does | Main output |
|---|---|---|
| `python validate.py` | **The pipeline**: reads every photo and decides | `output/results.csv`, `output/needs_review.csv` |
| `python evaluate.py` | Measures every stage against `truth.csv`, and times it | `output/evaluation/summary.md` |
| `python analyze.py` | Figures, confusion matrices and tables from `evaluate.py`'s results (see below) | `output/analysis/report.md` |
| `python baseline.py` | Stage 2: untuned barcode + OCR on raw photos | `output/baseline/` |
| `python barcode_reader.py` | Stage 3: barcode variant matrix | `output/barcode/variant_matrix.csv` |
| `python label_finder.py` | Stage 4: label boxes, crops and debug images | `output/labels/boxes/`, `crops/`, `debug/` |
| `python ocr_reader.py` | Stage 5: OCR method comparison | `output/ocr/variant_matrix.csv`, `text/`, `lines/` |

Every stage saves its intermediate images to `output/` so failures can be inspected by eye.

### Analysis (`analyze.py`)

Reads `output/evaluation/per_photo.csv` (run `evaluate.py` first; `barcode_reader.py`
adds the variant figure), prints every table, and writes a PNG + CSV for each to
`output/analysis/`, all collected in `report.md`. Takes about 2 seconds.

| Figure | Shows |
|---|---|
| `stage_accuracy` | Accuracy per stage with 95% Wilson intervals (with 24 photos, they're wide) |
| `photo_outcomes` | Photo × stage pass/fail grid: where each photo is lost |
| `decision_confusion` | Accepted / needs review × correct / wrong, plus precision, recall, F1 |
| `acceptance_policies` | What looser rules would accept, and how many wrong serials they'd let through |
| `source_agreement` | Barcode vs OCR on the same photo: how often one rescues the other |
| `ocr_chars_wrong`, `ocr_tolerance` | Characters wrong per OCR method, and photos read within *k* characters |
| `char_confusion` | Which characters OCR misreads, and as what (from an edit-distance alignment) |
| `barcode_variants` | Which image variant let ZBar read each photo |
| `by_condition` | Success by capture condition (glare, side angle, …) parsed from file names |
| `timing` | Time per photo, split into label finding, barcode and OCR |

## Limitations

- **Small test set**: 24 photos of 11 items (3 laptops, 2 mice, 6 products). Percentages move by 4 points per photo.
- **Tuned on the test set**: thresholds (edge strength 90, density 0.40, line sizes, the digit-strip size) were chosen by looking at these same photos, and the look-alike list includes `A/4` because it showed up here. There is no held-out set, so real-world accuracy is probably lower.
- **One hand-edited photo**: `Mouse_2_Sideangle.jpeg` was cropped by hand to remove books from the background. Its label is still not found.
- **Known failures**:
  - `Laptop_1_variant`, `Mouse_2_Sideangle`: label not found (dim, small, or low-contrast print on a dark label).
  - `Laptop_2_Vertical`: OCR too far off (4 characters) to confirm the barcode.
  - `Monster_Horizontal` (curved can) and `Bottle1_Horizontal` (glare): the barcode reads, but the digits don't come out as a valid EAN-13, so there's nothing to confirm it.
  - `Supplement_Horizontal`: the barcode isn't read, so the digit reader has nothing to go on.
  - Steep side angles defeat the barcode reader: the crop corrects rotation but not perspective.
- **Image quality matters most**: the first set of photos had been downscaled to 900×1600 in transfer. Replacing them with the camera originals (4032×2268) did more for barcode reading than any code change.
- **Speed**: about 1.8 s per photo. The biggest part is the barcode step (1.2 s mean, 6.3 s max), which decodes the full photo (up to 36 megapixels) and, when that fails, a 3× upscale of it. OCR of a large junk crop (e.g. mousepad texture) can still take several seconds.

## What I would do next

1. **More photos and a held-out set**: tune on one half, measure on the other.
2. **Perspective correction**: find the label's 4 corners and use `cv2.getPerspectiveTransform` so side-angle barcodes become flat.
3. **Per-manufacturer serial formats**: e.g. if a format says position 4 is always a digit, an OCR `O` there can be corrected to `0` safely.
4. **Faster barcode step**: decode the small label crops first (0.2 s instead of 1.2 s in testing) and fall back to the full photo only when that fails. Crops alone missed one photo whose label wasn't found.

## Project layout

```
check_setup.py     Stage 0  checks OpenCV, pyzbar, Tesseract (both paths), pandas, matplotlib
native_libs.py     setup    finds Tesseract and ZBar on Windows / macOS / Linux
baseline.py        Stage 2  untuned barcode + OCR on raw photos; shared helpers normalize(), found(), valid_ean13(), load_truth()
barcode_reader.py  Stage 3  read_barcode(): retries + barcode-type filter
label_finder.py    Stage 4  find_label_boxes(), crop_rotated()
ocr_reader.py      Stage 5  read_ocr(): whole-crop + line-by-line + barcode-digit OCR
tess_direct.py     Stage 5  TessEngine: Tesseract's C API via ctypes (no process per call)
validate.py        Stage 6  the pipeline: candidates, pattern check, agreement, CSVs
evaluate.py        Stage 7  accuracy per stage + timing → output/evaluation/
analyze.py         Stage 8  figures, confusion matrices, tables → output/analysis/
photos/            input photos (not committed; see Data)
truth.csv          true serials (not committed); format in truth.example.csv
output/            everything generated (git-ignored, re-created by the scripts)
```

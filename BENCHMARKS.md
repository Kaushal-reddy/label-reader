# Benchmarks

Every number measured for the label reader, and what it was measured on. How
the pipeline works is in the [README](README.md); the commands that reproduce
these numbers are in [HOW_TO_RUN.md](HOW_TO_RUN.md#reproducing-the-benchmarks).

- [Test data](#test-data)
- [Results on the 26 photos](#results-on-the-26-photos)
- [Adding zxing-cpp](#adding-zxing-cpp)
- [The 403-image EAN set](#the-403-image-ean-set)
- [Stage-level measurements](#stage-level-measurements)
- [Synthetic photos (CI)](#synthetic-photos-ci)
- [ERP demo run](#erp-demo-run)
- [Figures and tables](#figures-and-tables)

## Test data

The photos and true serials are **not in this repository**, because they contain
the real serial numbers of my own devices (serials are shortened in these
documents). The test set was:

| Item | Photos | Label | Barcode |
|---|---|---|---|
| Laptop 1 (ASUS) | 2: straight-on (label cut off at the frame edge), rotated + dim | Dark, white text | QR (URL ending in the serial), visible in one photo |
| Laptop 2 (ASUS) | 3: straight-on, rotated 90°, glare | Dark, white text | QR (URL ending in the serial) |
| Laptop 3 (ASUS) | 1: straight-on | Dark, white text, on a silver laptop | Code128 + QR |
| Mouse 1 (ASUS) | 3: straight-on, side angle, variant | White, black text | Code128 |
| Mouse 2 (Razer) | 3: straight-on, side angle, variant | Dark blue + silver strip, light text | Code128 |
| 7 retail products (energy-drink can, notebook, 2 boxes, supplement, 2 bottles) | 2 each: horizontal, vertical | Printed packaging | EAN-13; the bottles also carry a QR code (a web link) |

All photos were taken on a phone and used at the camera's original resolution
(4032×2268), except `Mouse_1_variant` (3309×1861), `Mouse_2_Sideangle`
(hand-cropped to 2268×2448, see [Limitations](README.md#limitations)) and the
product photos. Those were taken at 8064×4536 and shrunk to 4032×2268 by
`shrink_photos.py`.

For products the "serial" is the 13-digit EAN-13 number. It identifies the
product, not the individual item, so both photos of a product share it.

`truth.csv` holds the hand-typed true values. It is never used to make a
decision, only to measure afterwards.

## Results on the 26 photos

12 photos of 5 IT devices with **serial labels**, and 14 photos of 7 retail
products whose **product barcode** (EAN-13) is the value to read.

**Final decisions**

| Measure | Serial labels | Product barcodes | All |
|---|---|---|---|
| Accepted without a human | 6/12 | 13/14 | 19/26 (73%) |
| Accepted and **wrong** | **0** | **0** | **0** |
| Precision (correct / accepted) | 6/6 | 13/13 | 19/19 (100%) |
| Sent to review, best guess already correct | 4 of 6 | 1 of 1 | 5 of 7 |
| Time per photo, mean / max (MacBook Air, median of 3 runs) | 1.4 s / 2.9 s | 0.5 s / 1.7 s | 0.9 s / 2.9 s |

An earlier run of just the 12 serial-label photos on a Windows 11 laptop took
2.4 s mean, 5.3 s max.

**What each stage added** (photos where the value was read correctly)

| Stage | Measure | Serial labels | Product barcodes |
|---|---|---|---|
| 2 – baseline | Barcode on the raw photo | 7/12 | 12/14 |
| 2 – baseline | OCR on the raw photo (value anywhere in the text) | 3/12 | 4/14 |
| 3 – barcode | zxing-cpp, then ZBar with retries + barcode-type filter | 8/12 | 14/14 |
| 4 – label crop | Photos where a label crop was found | 10/12 | 14/14 |
| 4 – label crop | Barcode on the crops | 7/12 | 14/14 |
| 5 – OCR | OCR on the crops, exact | 6/12 | 8/14 |
| 5 – OCR | OCR on the crops, within 1 character | 9/12 | 8/14 |

On serial labels, cropping the label and reading it line by line **doubled**
exact OCR reads (3 → 6). The near misses are almost all look-alike characters (letter `O` vs
digit `0`, `A` vs `4`), which is what the barcode cross-check in Stage 6 catches.
For example, on `Laptop_3` OCR read `TCNOCV…` (letter O); the barcode confirmed it
and supplied the correct `TCN0CV…` (digit 0).

Product barcodes are the opposite case: plain OCR misses the small digits under
the bars, but the barcode reader gets 14 of 14. The barcode-guided digit OCR
(Stage 5) reads those digits exactly on 8 of 14, and then barcode and OCR agree.
Otherwise a product barcode is accepted on the barcode alone, guarded by the check
digit, a check for rival reads and 2 agreeing image variants (Stage 3). That guard
sent the one photo with a phantom read (`Supplement_Horizontal`) to review.

Accepting OCR-only results would have saved one wrong serial
(`Mouse_1_Sideangle`: OCR read `SCBMNVO00…` instead of `SCBMNV00…`, an extra `O`),
which is why OCR alone never counts as confirmation.

**Known failures**

- `Laptop_1_variant`, `Mouse_2_Sideangle`: label not found (dim, small, or low-contrast print on a dark label).
- `Laptop_2_Vertical`: OCR too far off (4 characters) to confirm the barcode.
- Steep side angles defeat the barcode reader: the crop corrects rotation but not perspective.
- `Supplement_Horizontal`: sent to review because zxing-cpp also read a wrong number at the same spot (Stage 3); the best guess is correct.

## Adding zxing-cpp

zxing-cpp as the first decoder (Stage 3), with ZBar as the backup, was measured
with the same OCR on both sides (`zxing_experiment.py`):

| Photos | Accepted, ZBar only | Accepted, zxing-cpp first | Accepted and wrong | Barcode step, mean / max |
|---|---|---|---|---|
| These 26 | 18 | 19 | 0 | 1.4 s / 3.4 s → 0.4 s / 2.2 s |
| 403-image EAN set | 179 | 244 | 0 | 0.43 s / 1.4 s → 0.12 s / 1.3 s |

(Barcode-step times: the ZBar-only times are from the experiment, the zxing-cpp
times from `evaluate.py`, so they come from different runs.) On the EAN set the
barcode was read on 291 photos instead of 216, and 47 photos sent to review had
the right best guess (37 with ZBar only). The rule for accepting a product barcode on its
own was not changed.

Raw reads on the EAN set, any image variant: zxing-cpp read the right number on
279 of the 403 photos, ZBar on 216, in about a tenth of the time. zxing-cpp
missed 14 photos ZBar reads (and the Code128 on `Mouse_1`), which is why ZBar
stays as the backup. Both decoders sometimes misread a barcode a second way at
the same spot (zxing-cpp: a wrong EAN-13 on 8 of 429 photos, from a box only
2–3 pixels thick); those count as conflicts and go to review.

## The 403-image EAN set

`photos/ean_dataset`: 403 single-barcode product images built from a public
barcode dataset by `prepare_ean_dataset.py`, values checked by hand.

| Reader | Accepted | Accepted and wrong |
|---|---|---|
| ZBar, first clean read accepted (no confirmation) | 216 | **8** |
| ZBar, all 9 variants, stop at first read | 226 | **14** |
| ZBar + 2 agreeing variants (`CONFIRM_VARIANTS`) | 178 | 0 (44 sent to review) |
| Full pipeline, ZBar only | 179 | 0 |
| Full pipeline, zxing-cpp first (current) | 244 | 0 |

The 8 wrong reads each had a valid check digit and no rival at the same spot.
Most swapped 1↔7 or 2↔8, digits whose bar patterns differ by one bar width, on
small or blurry barcodes, and every one came from a single image variant: hence
the 2-variant rule. A learned barcode detector (branch `barcode-detection`,
grouped 5-fold cross-validation) found the barcode in all 403 photos, while
ZBar read nothing in 178, so ZBar's misses are decoding failures, not failures
to find the barcode.

## Stage-level measurements

- **Barcode retries:** the "variant matrix" ran every image variant (upscaling,
  rotation, sharpening, contrast) on every photo. Only the 3× upscale rescued a
  photo; the others helped none.
- **Barcode types:** with every type enabled, ZBar read the mousepad texture as an
  Interleaved-2-of-5 barcode (`132342`), so the reader is restricted to Code128,
  Code39, QR and EAN-13.
- **Vote margin:** on a curved drink can ZBar read one EAN-13 two ways from the
  same spot, both with valid check digits: 122 scan-line votes for the true number,
  14 for the wrong one (3.6× to 168× across every image it appeared in). Correct
  reads went as low as 1–11 votes, so a minimum-quality cut-off couldn't work.
- **Label finding:** Otsu's automatic threshold picked 32–66 on these photos
  (most of each photo is texture). Label text measured 0.44–0.64 edge density and
  textures about 0.3, so the cut-off is 0.40.
- **Barcode-digit OCR:** read the digits under the bars exactly on 8 of the 12
  product photos, with no wrong numbers.
- **Calling Tesseract directly (`tess_direct.py`):** ~0.006 s per small image
  instead of ~0.15 s through pytesseract (about 28× faster), identical text on all
  228 images the pipeline OCRs, and average OCR time per photo down from 3.3 s to 1.1 s.
- **Image resolution:** the first set of photos had been downscaled to 900×1600 in
  transfer; the camera originals (4032×2268) did more for barcode reading than any
  code change. Shrinking the 36-megapixel product photos to 4032×2268 made the
  pipeline 1.7× faster (worst case 9.8 s → 3.3 s) and let one more barcode decode;
  3024 px or JPEG quality 75 each lost a serial-label photo, and 2016 px lost
  several (measured once; the table is in `shrink_photos.py`).
- **Faster barcode step (not adopted yet):** decoding the label crops first took
  0.2 s instead of 1.2 s, but missed one serial-label photo whose label wasn't
  found (13 of 14 product photos read).

## Synthetic photos (CI)

`make_synthetic_dataset.py` draws 14 label photos with fake serials, so CI can
run every stage on every OS. They show nothing about real-world accuracy, only
that the pipeline's safety properties hold. On a MacBook Air:

| Kind | Photos | Accepted | Accepted and wrong |
|---|---|---|---|
| Serial labels | 6 | 5 | 0 |
| Product barcodes | 6 | 6 | 0 |
| Printed serial disagrees with the barcode | 2 | 0 (both sent to review) | 0 |

The tests require 0 wrong, both mismatches in review, every product accepted
and at least 4 of the 6 serial labels (Tesseract versions differ between OSes).

## ERP demo run

On the 26 photos (Mouse_2 left off the manifest), with the mock ERP failing 30%
of requests with a 503 and losing the reply on 30% of new receipts:

- 10 photos were posted and 16 went to review (8 not confident, 6 already
  received, 2 not on the manifest); 10 of the 11 manifest lines were received
  (Laptop_1's photos were both "needs review").
- The server answered 15 requests with 503, 2 with a lost-response 500 and 4 with
  429, and the database still held exactly one receipt per photo posted.
- Running it again posted nothing new.

This is one run with one seed. The ERP tests repeat the same checks with fake
serials, including every line received exactly once under 50% 503s plus 50% lost
responses, and pending posts sent after an outage. On the synthetic photos,
`scripts/erp_e2e.py` (also run in CI) saw 16 injected 503s and 2 lost replies in
a local run and still ended with exactly one receipt per accepted photo.

## Figures and tables

`analyze.py` reads `output/evaluation/per_photo.csv` (written by `evaluate.py`;
`barcode_reader.py` adds the variant figure), prints every table, and writes a
PNG + CSV for each to `output/analysis/`, all collected in `report.md`. Takes
about 2 seconds.

| Figure | Shows |
|---|---|
| `stage_accuracy` | Accuracy per stage with 95% Wilson intervals (with this few photos, they're wide); split by serial labels vs product barcodes |
| `photo_outcomes` | Photo × stage pass/fail grid: where each photo is lost |
| `decision_confusion` | Accepted / needs review × correct / wrong, plus precision, recall, F1 |
| `acceptance_policies` | What looser rules would accept, and how many wrong serials they'd let through |
| `source_agreement` | Barcode vs OCR on the same photo: how often one rescues the other |
| `ocr_chars_wrong`, `ocr_tolerance` | Characters wrong per OCR method, and photos read within *k* characters |
| `char_confusion` | Which characters OCR misreads, and as what (from an edit-distance alignment) |
| `barcode_variants` | Which image variant let ZBar read each photo |
| `by_condition` | Success by capture condition (glare, side angle, …) parsed from file names |
| `timing` | Time per photo, split into label finding, barcode and OCR |

With so few photos, one photo moves a rate by 7–8 points within a group, and
the thresholds were tuned on these same photos with no held-out set (see
[Limitations](README.md#limitations)).

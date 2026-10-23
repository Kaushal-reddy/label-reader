"""Read serial numbers and product barcodes from photos of equipment labels.

The pipeline stages, each runnable on its own from the repo root:
  python -m label_reader.baseline        Stage 2  untuned barcode + OCR
  python -m label_reader.barcode_reader  Stage 3  barcode reading
  python -m label_reader.label_finder    Stage 4  finding the label
  python -m label_reader.ocr_reader      Stage 5  OCR
  python -m label_reader.validate        Stage 6  the full pipeline: read and decide
  python -m label_reader.evaluate        Stage 7  accuracy and timing per stage
  python -m label_reader.analyze         Stage 8  figures and tables
The ERP integration (Stage 9) is in label_reader.erp.
"""

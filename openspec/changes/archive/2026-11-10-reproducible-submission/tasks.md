## 1. CSV and validation

- [x] 1.1 `write_csv` asserts unique ids, stems in the test index, non-empty 2048 masks. Verify:
  `filseg.predict` on any run. Result: every submission (1-8) written through it; submission 8 has 1,065
  filaments on 177 of 180 images.
- [x] 1.2 No overlapping masks. Verify: overlap check on the CSV. Result: 0 overlapping pixels in submissions 5-8.

## 2. Notebook

- [x] 2.1 `pipeline.ipynb` from `scripts/make_notebook.py`, run on CPU from raw JPEGs and
  `weights/unet_r34_all.pt`. Result: MEASURED 2026-11-04 byte-identical CSV to submission 4 (made on MPS from
  the cache; 1,306 rows; 8.1 s/image on CPU under load).
- [x] 2.2 Notebook regenerated for the tree scorer and refiners. Result: README states the notebook output is
  byte-identical to the final submission (README.md line 29); the byte-for-byte check for submission 8 is
  stated there, not re-measured in DESIGN.md (see README).

## 3. Runtime and size

- [x] 3.1 Runtime per image. Result: submission 6 8.6 s, submission 7 6.6 s, submission 8 15.9 s on the M3 Max
  (shared, loaded machine).
- [x] 3.2 float16 refiner weights. Result: 49 MB (U-Net) / 52 MB (UNet++) each; about 1 crop pixel in 10^5
  differs from float32 at threshold 0.40.

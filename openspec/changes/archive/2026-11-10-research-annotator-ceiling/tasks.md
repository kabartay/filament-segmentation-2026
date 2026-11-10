## 1. Measurement

- [x] 1.1 Pairwise annotator PQ. Verify: `scripts/experiments/oof_analysis.py human`. Result: MEASURED, 0.3399 (SQ 0.634, RQ 0.536), 296 images.
- [x] 1.2 Model on the same images. Result: MEASURED, U-Net 0.405 (1-annotator 0.414, 2-annotator 0.414, 3-annotator 0.400); SQ 0.667, RQ 0.616 on all OOF images.
- [x] 1.3 Re-measurement. Result: 0.340 holds (2026-11-04).

## 2. Records

- [x] 2.1 Recorded in docs/DESIGN.md section 2 and the entries "Annotator ceiling", "Annotator ceiling, re-measured", "Where the ceiling is".

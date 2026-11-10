## 1. Measured results (fold 0, tuned unless stated)

- [x] 1.1 Mask R-CNN alone. Verify: `scripts/experiments/detector_tune_disjoint.py`. Result: val PQ at score >= 0.7
  by epoch e2 0.337, e6 0.350, e10 0.366, e12 0.366 (overlaps merged); with disjoint masks best 0.377 at score
  >= 0.75 (SQ 0.642, TP 944, FP 614, FN 711) vs U-Net 0.389 (SQ 0.657). Raw overlapping masks scored 0.381 only by
  double counting.
- [x] 1.2 Combinations (`unet_detector_combo*.py`) with the epoch-10 U-Net (0.389). Result: U-Net plus detector
  filaments touching no U-Net component 0.398 (score >= 0.75) / 0.399 (>= 0.85); detector grouping with U-Net
  pixels 0.396; splitting U-Net components by detector instances 0.388; grouping plus leftover components
  0.383-0.388.
- [x] 1.3 Same combination on the 40-epoch U-Net (0.415). Result: extras add 0.002 (0.417 at score >= 0.85), inside
  fold noise. Detector not used.
- [x] 1.4 Axis-aligned crop refiner at native 2048 px (crop 3x the box, second channel marking the box, 256 px
  input), epoch-10 proposals. Result: crop Dice 0.829; PQ by mask threshold 0.4 0.395, 0.5 0.399, 0.6 0.402
  (SQ 0.663, TP 954, FP 539, FN 701), 0.7 0.399, 0.8 0.390, 0.9 0.372, from 0.389.
- [x] 1.5 Same refiner on the 40-epoch U-Net (0.415). Result: 0.413 at threshold 0.6, 0.409 at 0.7. No gain.
- [x] 1.6 DenseCRF. Result: about 6.5 s per 2K image and it fits image edges rather than annotators' polygons;
  judged on cost and not trained or scored.

## 2. Records

- [x] 2.1 All results are in DESIGN.md sections 4 and 5; the detector and refiner code stay in the repo as
  tested and not used.

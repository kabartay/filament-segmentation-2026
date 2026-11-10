## 1. Measured results (5-fold OOF, 707 images unless stated)

- [x] 1.1 Size-dependent confidence (0.85 below an area split of 800/1500/3000 px, 0.7-0.8 above, min area
  150-300). Result: best 0.4114 vs reference 0.4112 (0.45, 31, area 300, conf 0.85); noise.
- [x] 1.2 Hysteresis (mask at 0.30/0.35/0.40, kept if its maximum reaches 0.85-0.95). Result: no setting beats a
  single threshold of 0.45 (0.4079 at 0.40).
- [x] 1.3 Lower minimum area (150-200). Result: no gain; extra small components cost more FP than they add TP.
- [x] 1.4 Eroding/dilating masks, and growing into nearby pixels (`grow_shrink_masks.py`). Result: erode 1 px
  0.3986, 2 px 0.2841; dilate 1 px 0.3811, 2 px 0.3112, 3 px 0.2438; grow into pixels above 0.35 / 0.30 / 0.20:
  0.4074 / 0.4038 / 0.3877; reference 0.4112.
- [x] 1.5 Limb filter (70 deg) with the corrected Hough disk. Result: no change (0.41032).
- [x] 1.6 Twelve tree levels without spread features vs six. Result: 0.4252 (+0.0034 [-0.0005, +0.0070]) vs six
  levels 0.4260; more levels alone do not help (the spread features carry the gain).
- [x] 1.7 Nested isotonic calibration of the scorer (2026-11-09). Result: 0.4264 at tau 0.26 vs 0.4267; no gain.
- [x] 1.8 Per-annotator-count tau (2026-11-09). Result: 0.4277, inside noise, and the number of annotators is
  unknown at test time. Tau per image load: no gain.
- [x] 1.9 Post-stage-2 keep/drop rescoring, nested fold-held-out (236-image subsample, 2026-11-09). Result: score
  only 0.4247 (-0.0014), refiner features only 0.4260 (0), both 0.4275 (+0.0014; per fold +0.0013, -0.0003,
  -0.0002, +0.0028, +0.0031); at most about +0.001 in full, not adopted. The refiner output barely separates
  TP from FP (AUC 0.63 for mean refined probability vs 0.71 for the stage-1 score).

## 2. Records

- [x] 2.1 All results are in DESIGN.md section 5. A per-instance threshold was never run (DESIGN.md section 8
  lists it as untried), so no number is recorded for it.

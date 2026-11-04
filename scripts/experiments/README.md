# Experiments

One-off analyses whose results are recorded in `docs/DESIGN.md` (section 5). Each reads the saved
out-of-fold probabilities in `runs/unet_r34full_f{0..4}/val_probs.npy` and runs on CPU.
Run from the project root with `PYTHONPATH=. .venv/bin/python scripts/experiments/<script> [args]`.

| Script | What it measures |
|---|---|
| `oof_analysis.py human` | annotator-vs-annotator PQ (the ceiling) |
| `oof_analysis.py res` | IoU lost by the 1024 px round trip |
| `oof_analysis.py errors` | FN/FP breakdown by size and cause on all 707 OOF images |
| `oof_analysis.py ens` | averaging the 40-epoch and epoch-20 fold-0 models |
| `pq_by_annotator_count.py` | PQ on images with 1, 2 and 3 annotators |
| `postprocess_variants.py` | size-dependent confidence, hysteresis, near-miss diagnosis |
| `small_filaments.py` | agreement by size, why small filaments are missed, `docs/figures/missed_small.png` |
| `grow_shrink_masks.py` | dilating/eroding predicted masks, growing them into lower-probability pixels |

These scripts were written for speed of iteration, not as library code; the pipeline lives in `filseg/`.

Earlier detector experiments (2026-11-03, fold 0, read `runs/mrcnn_f0/val_dets.pkl`):

| Script | What it measures |
|---|---|
| `detector_tune.py` | Mask R-CNN score threshold, overlapping masks (inflated by double counting) |
| `detector_tune_disjoint.py` | the same with disjoint masks painted in score order |
| `unet_detector_combo*.py` | U-Net plus detector extras, detector grouping of U-Net pixels, splits |

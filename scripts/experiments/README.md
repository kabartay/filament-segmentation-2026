# Experiments

Analyses and experiments whose results are recorded in `docs/DESIGN.md` (section 5, the decision log; entries
below are named by their title). Some are steps of the submitted pipeline (submission 8), some are
analyses behind figures and numbers in the report, and some are rejected or historical experiments kept
because DESIGN cites their numbers. They were written for speed of iteration, not as library code; the pipeline
itself lives in `filseg/`.

Run from the project root:

```bash
PYTHONPATH=. FILSEG_CACHE=data/cache1536 .venv/bin/python scripts/experiments/<script> [args]
```

`FILSEG_CACHE` selects the image cache (`filseg/cache.py`). Leave it unset for the 1024 px cache
(`data/cache`), which the older scripts below were written for.

## Reproducibility caveat

Local `runs/` and the 1024 px cache were lost on 2026-11-10 and only the 1536 px pipeline was rebuilt
(DESIGN: "Lost local artefacts", "Rebuild after the loss"). The submitted-path scripts and everything that
reads `runs/unet_r34_1536_f*` or `runs/oof_tree12s_1536_v4.pkl` run on the rebuilt files. The analyses and
experiments marked "needs 1024 px runs" read `runs/unet_r34full_f{0..4}/val_probs.npy` (the 1024 px U-Net of
submissions 3 and 4) and would first need that model retrained with `scripts/train_folds.sh` on the 1024 px
cache. The detector scripts need `runs/mrcnn_f0/val_dets.pkl`, which `filseg.detector` wrote and which is
not in the repository.

## Submitted path

These run in this order (the commands are in the top-level `README.md`, "Train from scratch").

| Script | What it does | DESIGN.md entry |
|---|---|---|
| `oof_probs_views.py` | Re-predicts each fold's validation images with 4 flipped views from `runs/<name>_f{0..4}/best.pt` and saves `val_probs_v4.npy` (the mean). Checks the first three images against the saved 2-view `val_probs.npy` when it exists. `--views`, `--name` (default `unet_r34full`; the submission used `--name unet_r34_1536`). GPU or MPS; `FILSEG_DEVICE` overrides. | "Scorer on 4-flip probabilities, 5-fold OOF" |
| `oof_view_spread.py` | Per-pixel standard deviation of the four flipped views of each fold model, saved as float16 `val_spread_v4.npy`: the uncertainty feature of the tree scorer. `--name`, `--folds`, `--save-mean`, `--limit` (smoke test). | "Scorer on 4-flip probabilities", "Full 1536 px pipeline, 5-fold OOF" |
| `component_tree.py` | `build`: thresholds the OOF maps at several levels, nests the components into a forest and writes one record per image (features, level, parent, IoU against every annotator entry) to a node pickle. `eval`: cross-validated scorer (fitted on four folds, applied to the fifth), pooled PQ per selection threshold tau, and the single-level control. The submission used `build --spread --levels 0.25,...,0.80 --out runs/oof_tree12s_1536_v4.pkl` (12 levels) and `eval --context --modes hurdle`; the script's own defaults (6 levels, `runs/oof_tree_1536_v4.pkl`) are not the submitted setting. The selection and scorer code is `filseg/tree.py`. | "Full 1536 px pipeline, 5-fold OOF", tree and tau entries |
| `stage2_oof.py` | Out-of-fold training and evaluation of the stage-2 refiners (`filseg/stage2.py`). Steps below. | "Stage 2, rotated-crop refiner", "Second refiner architecture and blend", "Jitter refiner", "Refiner blends with the jitter model", "Submission 8 candidate" |
| `bootstrap_pq.py` | Uncertainty of the OOF PQ: per-fold spread and a bootstrap over images (whole images resampled, all annotator entries together), plus paired bootstraps between pipelines (hand rule, single-threshold scorer, tree). Writes `runs/bootstrap_pq.json`. The "rule" and "final" arms need the 1024 px runs and `runs/oof_components_1536_v4.pkl`; the "tree" arm needs `runs/oof_tree12s_1536_v4.pkl`. | "Uncertainty of the final OOF PQ" |

### `stage2_oof.py`

Every step reads and writes under `runs/<FILSEG_STAGE2_OUT>/` (default `runs/stage2/`). It uses the node pickle
`runs/oof_tree12s_1536_v4.pkl` and the probability maps of run `unet_r34_1536`.

```bash
PYTHONPATH=. FILSEG_CACHE=data/cache1536 .venv/bin/python scripts/experiments/stage2_oof.py <step> [flags]
```

| Step | What it does | Output |
|---|---|---|
| `crops` | Applies the stage-1 selection out of fold (hurdle scorer fitted on the other four folds, selection threshold `FILSEG_TAU`), rebuilds each selected seed from the tree levels, and builds the rotated 384 x 128 crop inputs and soft targets. | `crops.npy`, `targets.npy`, `meta.pkl` |
| `jitter` | Extra training crops from tree nodes at other levels than the selected seed (fragments, thinner or wider versions), matched to an annotator mask by IoU or by containment, near-duplicates skipped. Training only; evaluation stays on the selected seeds. With `--negatives` it writes the nodes nobody drew to `jitneg_*` instead. | `jit_crops.npy`, `jit_targets.npy`, `jit_meta.pkl` |
| `train` | One refiner per fold, trained on the other folds' crops (only seeds some annotator drew) and predicting its own fold's crops. | `<tag>_f{k}.pt`, `<tag>_pred.npy` |
| `predict` | Predicts the crops of the output directory with already trained fold refiners `<tag>_f{k}.pt` (each crop by its own fold's model), without retraining. | `<tag>_pred.npy` |
| `blend` | Averages the OOF predictions of several tags (optionally weighted) into one tag. | `<tag>_pred.npy` |
| `eval` | Pastes the refined crops back at several pixel thresholds, pooled PQ against every annotator entry, paired bootstrap against stage 1 alone. | `<tag>_<name>.json` |
| `compare` | Paired bootstrap of two evaluated variants over the same images. | printed |
| `export` | Writes the fold refiners as float16 state dicts for inference. | `weights/stage2_<tag>_f{k}.pt` |

Key flags:

| Flag | Applies to | Meaning |
|---|---|---|
| `--tag` (default `unet`) | train, predict, blend, eval, export | Name of the refiner set. The submission used `upp` and `jit`, blended as `ensj`. |
| `--arch` (default `Unet`), `--encoder` (default `resnet34`) | train | Network of `segmentation_models_pytorch`; `UnetPlusPlus` for the `upp` refiners. |
| `--epochs` (30), `--batch` (32), `--lr` (5e-4), `--folds` (`0,1,2,3,4`) | train, predict, export | Training settings and the folds to process. The `jit` refiners were trained for 15 epochs. |
| `--extra jit` | train | Adds the jitter crops to the training set. |
| `--aug` | train | Per-sample flips and small affine jitter. The `jit` refiners used it. |
| `--negatives`, `--neg-weight` (0.5), `--aux-weight` (0.0) | train, jitter | Also train on seeds nobody drew (all-zero targets), with this loss weight; `--aux-weight` above 0 adds a head for the share of annotators who drew the crop. Tried and not adopted (DESIGN: refiners trained with undrawn seeds, tag `negj`). |
| `--limit` | train | Smoke test: at most this many crops per kind. |
| `--blend`, `--weights` | blend, predict | Comma-separated tags to average, with optional weights (the submission used equal weights). |
| `--thresholds` (`0.30,...,0.55`) | eval | Pixel thresholds of the paste-back. The submission uses 0.40. |
| `--name` (`eval`) | eval | The report is written to `<tag>_<name>.json`. |
| `--erase`, `--drop-grid`, `--single`, `--cap` | eval | Variants that drop erased or low-drew-fraction seeds, or force each pasted instance into one piece. None is in the submission. |
| `--new`, `--old` | compare | `<tag>_<name>:<key>`, for example `ens_pi:t0.40`. |
| `--workers` (4) | crops, jitter, eval | Processes. |

Environment overrides:

| Variable | Default | Meaning |
|---|---|---|
| `FILSEG_CACHE` | `data/cache` (1024 px) | Image cache. Set to `data/cache1536` for the submitted pipeline. |
| `FILSEG_STAGE2_OUT` | `stage2` | Output directory name under `runs/`. Set `stage2_<slug>` to leave `runs/stage2` untouched. |
| `FILSEG_TAU` | `0.26` | Selection threshold of the out-of-fold selection used by `crops`. |
| `FILSEG_DEVICE` | cuda, else mps, else cpu | Torch device (steps that train or predict). |

## Analyses

Measurements behind the figures in the report and in DESIGN. None changes the submitted pipeline. Those
marked "needs 1024 px runs" read the 1024 px fold models' saved probabilities.

| Script | What it measures | DESIGN.md entry |
|---|---|---|
| `oof_analysis.py human` | Annotator against annotator PQ (the ceiling). Needs 1024 px runs. | "Annotator ceiling", "Where the ceiling is" |
| `oof_analysis.py res` | IoU lost by the 1024 px round trip of a ground-truth mask. | "Resolution" |
| `oof_analysis.py errors` | FN and FP breakdown by size and cause on all 707 OOF images. Needs 1024 px runs. | "Where PQ is lost", "Near misses" |
| `oof_analysis.py ens [runA runB]` | Averaging two fold-0 models (default the 40-epoch and epoch-20 models). Needs 1024 px runs. | "Two-model average, fold 0" |
| `pq_by_annotator_count.py` | PQ on images with 1, 2 and 3 annotators. Needs 1024 px runs. | "Annotator ceiling, re-measured" |
| `small_filaments.py` | Annotator agreement by filament size, why small filaments are missed, `docs/figures/missed_small.png`. Needs 1024 px runs. | "Small filaments" |
| `qualitative.py` | Report figure: fold-0 OOF predictions next to every annotator's masks, hand-rule pipeline, `docs/figures/qualitative.png`. Needs 1024 px runs. | none (report figure) |
| `qualitative_final.py` | The same figure for the later pipelines: single-threshold scorer (default) or the tree pipeline (`--tree`, tau 0.26), at the 50th, 75th and 90th percentile of image PQ. Writes `docs/figures/qualitative_final.jpg` or `qualitative_tree.jpg`. | none (report figure) |
| `adopt_scorer_features.py` | Scorer variants on the cached tree nodes (lower capacity, observing station, neighbour features), cross-validated, paired bootstrap against the baseline. Steps `centroids`, `score`; `--taus`. Result: none beat the baseline. | "Adoption round from the public notebooks" (adopt-scorer-features) |
| `adopt_spread_parity.py` | Whether the view-spread features of the all-data model on test images match the out-of-fold spread the scorer was fitted on (50 training images; GPU). Result: no systematic scale mismatch. | "Adoption round from the public notebooks" (adopt-scorer-features, spread parity) |
| `adopt_tau_sparse_labels.py` | PQ against the selection threshold tau when the test labels are sparser than the training labels (ground truth thinned at random), plus a closed form. Result: keep tau 0.26. | "Selection threshold under sparser test labels" |

`adopt_scorer_features.py`, `adopt_spread_parity.py` and `adopt_tau_sparse_labels.py` come from the
2026-11-10 adoption round, in which every change proposed from public notebooks was measured and none
adopted. They need the rebuilt 1536 px files and are the only analyses here that do.

## Rejected and historical experiments

| Script | What it measures | Result | DESIGN.md entry |
|---|---|---|---|
| `postprocess_variants.py` | Hysteresis thresholds, size-dependent confidence and a near-miss diagnosis on the 1024 px OOF probabilities. Needs 1024 px runs. | All failed. | "Post-processing variants on 5-fold OOF" |
| `grow_shrink_masks.py` | Dilating or eroding the kept masks by k px, or growing them into pixels above a lower probability. Needs 1024 px runs. | Failed. | "Growing or shrinking the masks" |
| `tta_views.py` | Test-time augmentation with 2, 4 and 8 views on fold 0. Needs `runs/unet_r34full_f0`. | 4 flips kept. | "More TTA views" |
| `component_features.py` | Caches candidate components of the OOF maps (threshold 0.45, closing 31, area at least 100 px) with features and the IoU matrix against each annotator entry. Feeds `component_scorer.py` and `filseg/scorer.py`. | Superseded by the tree. | "Second-stage component scorer" |
| `component_scorer.py` | Cross-validated gradient-boosted scorer on those candidates, keeping components whose predicted PQ contribution exceeds tau (the pipeline of submission 5). | Beat the hand rule (0.4162 against 0.4086 on 1024 px), then superseded by the tree. | "Second-stage component scorer", "Scorer on 4-flip probabilities" |
| `detector_tune.py` | Mask R-CNN score threshold, merging and area filter on fold 0, with overlapping masks. Reads `runs/mrcnn_f0/val_dets.pkl`. | The overlap inflates the score by double counting. | "Mask R-CNN detector, fold 0", "Overlapping predictions" |
| `detector_tune_disjoint.py` | The same with disjoint masks painted in score order. Best 0.377 at score 0.75, against 0.389 for the U-Net alone. | Detector weaker than the U-Net. | "Mask R-CNN detector, fold 0" |
| `unet_detector_combo.py`, `unet_detector_combo2.py`, `unet_detector_combo3.py` | U-Net components combined with Mask R-CNN instances on fold 0: extra detector filaments, detector grouping of U-Net pixels, splitting U-Net components. All outputs disjoint. `combo` and `combo2` read `runs/sub1`; `combo3` reads `runs/unet_r34full_f0`. | At most +0.01, inside fold noise. | "Mask R-CNN detector, fold 0" |
| `duplicate_exploit.py` | Diagnostic of the organisers' scorer lacking one-to-one matching: it simulates submitting each confident OOF mask k times and reports the resulting PQ. It was never used in a submission and never will be: our masks are disjoint, the organisers warned against exploiting the metric, and duplicates would also distort the 1:n counts of the final evaluation. Needs 1024 px runs. | Shows the metric can be inflated (about 0.50 to 0.61 against 0.409 honest); not used. | "Duplicate-mask inflation, simulated" |

`component_features.py` and `component_scorer.py` default to the run name `unet_r34full` and write
`runs/oof_components.pkl`; for the 1536 px version pass `--name unet_r34_1536 --probs val_probs_v4.npy --out
runs/oof_components_1536_v4.pkl` (as `scripts/pipeline_1536.sh` does).

## Other rejected experiments without a script here

Mask R-CNN, the 2026-11-03 crop refiner, ResNet-50, limb-darkening input, Lovasz loss, ConvNeXt-T, 2048 px
input and the 15-level tree were trained or evaluated with `filseg/detector.py`, `filseg/refiner.py`,
`filseg/flatten.py`, `filseg.train` options, `scripts/run_convnext.sh`, `scripts/hf/run_2048.sh` and
`component_tree.py` respectively; their results are in the same section of `docs/DESIGN.md`.

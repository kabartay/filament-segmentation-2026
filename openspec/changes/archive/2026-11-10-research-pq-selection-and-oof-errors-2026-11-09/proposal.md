# RESEARCH: PQ selection maths and OOF error analysis, research round of 2026-11-09

## Why

With two days to the deadline, one round of five read-only agents (literature, new Kaggle notebooks, stage-2
code review, PQ-optimal selection, OOF error analysis) checked where the next gain could be. Recorded in
DESIGN.md section 5, "Research round (2026-11-09...)". All numbers are estimates unless marked MEASURED.

## What Changes

Nothing in the pipeline; this change records the findings and the plan that followed.

Questions and findings:

- Kaggle since 2026-11-05: no new technique. New notebooks are empty, write-ups of the scorer's missing
  one-to-one matching, or re-hosts of the "LB 0.70" Mask R-CNN + crop refiner from frozen checkpoints (it forbids
  public-MAGFiLO paths, which hints at label leakage). Public LB top 0.70 / 0.64 / 0.62 and a crowd at 0.56-0.60;
  no public method shown above about 0.39. Forum pages did not load through the fetch tool (unread).
- Selection maths (MEASURED on OOF, refit of the hurdle CV): the PQ denominator is exactly 0.5(K + G) per entry,
  so the marginal rule is keep a node when y > PQ/2 = 0.213, independent of the number of annotators. tau 0.20
  0.4240, 0.2135 0.4250, 0.24 0.4265, 0.26 0.4267, 0.28 0.4269, 0.30 0.4262. The best tau sits above PQ/2 because
  y_hat is optimistic on the selected antichain (mean y_hat 0.523 vs y 0.456; winner's curse). Nested isotonic
  calibration: no gain (0.4264 at 0.26). Per-annotator-count tau 0.4277, inside noise and n is unknown at test
  time. Tau per image load: no gain. Images with no ground truth: none in the training set. Selection is not where
  the next gain is (at most +0.001).
- Stage-2 code review: no hard bug. Train/test gaps: the refiner trains only on selected seeds matched by IoU > 0.3
  (602 of 4,429 seeds have no match and are dropped), so it never sees fragments of a longer filament or seeds cut
  at other thresholds; one flip decision per batch and no other augmentation for about 3,000 crops; the stage-1
  channel is out of fold from fold models in training and from the all-data model at test. Checked fine: no seed
  touches the image border, unwarp/flip TTA consistent, OOF crops recomputed with the test-time `components` call,
  crop scale < 1 for only 13% of seeds (43 below MIN_SCALE).
- OOF error analysis after stage 2 (MEASURED on every 3rd image, SUBSAMPLE of 236 images, 1,485 instances;
  subsample PQ 0.4260 vs 0.4330 in full, intervals about +-0.007): TP 1,629, FP 769, FN 1,141. 24% of instances match
  no entry. FN: 661 overlap no prediction, 139 at IoU 0-0.3, 341 near misses at 0.3-0.5 (the same 341 before and after
  stage 2; stage 2 gains elsewhere: seed TP 1,619 -> 1,629 and better IoU of matched ones). 26% of the FN had an
  unselected tree node at IoU > 0.5; 40% have no node above 0.3. Recall by size: < 300 px 0.16, 300-500 0.39,
  500-1,000 0.57, 1,000-2,000 0.70, >= 2,000 0.64. Every stage-1 score bucket from 0.30 up is net positive
  (precision per entry 0.41 at 0.30-0.35 to 0.86 at 0.60-0.80), so tau is fine.
- Refiner output barely separates TP from FP: AUC for "matches something" 0.71 for the stage-1 score, 0.63 for the
  mean refined probability (1,453 of 1,472 instances have it >= 0.8), 0.43 area ratio, 0.40 share outside the
  seed. Nested, fold-held-out keep/drop rescoring: score only 0.4247 (-0.0014), refiner features only 0.4260 (0),
  both 0.4275 (+0.0014; per fold +0.0013, -0.0003, -0.0002, +0.0028, +0.0031): at most about +0.001 in full, not
  adopted. An oracle dropping net-negative instances would reach 0.474, but that information is annotator
  disagreement. Value of one change in full data: an FN turned into a TP at IoU 0.55 +4.3e-5 PQ, one FP removed +2.8e-5.
- Agents' ranked ideas: per-instance choice among seed and refined masks at 0.30 / 0.40 / 0.50 (CPU, est.
  +0.001-0.003); refiner retrained with undrawn seeds as zero targets plus an agreement-fraction head (est.
  +0.003); refining more tree nodes than the chosen antichain (est. +0.004, about 3x the crops).

Decisions (ranked plan of the round, and what acted on it):

1. A second refiner architecture (UNet++ ResNet-34): `stage2-refiner-ensemble` (OOF 0.4360 blend, +0.0031 [0.0016, 0.0044] over the submission-7 refiner).
2. Refiner trained also on tree nodes at other levels, with per-sample flips and small affine jitter: `stage2-refiner-ensemble` (jitter U-Net 0.4383; UNet++ + jitter 0.4399, +0.0070 [0.0050, 0.0090], 5/5 folds).
3. Post-stage-2 keep/drop rescoring: tested, not adopted (+0.0014 at best, nested).
4. Selection tau left at 0.26 (`component-tree-selection`).
- Not done: undrawn-seed zero targets with an agreement head; refining all tree nodes; per-instance mask choice (see `research-ideas-list`).

## Non-goals

No GPU work; no change to the submitted pipeline in this change.

## Falsified by

The keep/drop rescoring or the tau variants exceeding +0.003 in the nested per-fold OOF evaluation; measured
at most +0.0014 and +0.0007 (0.4267 -> 0.4269 over tau).

## Cost

CPU only; five read-only agents, about one session. No GPU.

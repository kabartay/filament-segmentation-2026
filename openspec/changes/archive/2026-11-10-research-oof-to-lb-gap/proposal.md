# RESEARCH: OOF-to-LB gap analysis

## Why

Our out-of-fold PQ is 0.41-0.44 while the public LB shows 0.32-0.37. We need to know whether the gap is a
sign of a bug or of leakage in our validation, and whether the LB can be used to choose between pipelines.
Recorded in DESIGN.md sections 4-6 (submission entries, bootstrap entry, survey, snapshot).

## What Changes

Nothing in the pipeline; this change records the analysis.

Our pairs (local value / public LB; two decimals shown by the LB):

| Submission | Local | LB |
|---|---|---|
| 1 (fold-0 epoch 10, 2026-11-03) | val 0.389 | 0.32 |
| 2 (40-epoch U-Net) | val 0.415 | 0.35 |
| 3 (5-fold average), 4 (all data) | 0.4086 OOF (rule) | 0.35, 0.35 |
| 5 (1536 px, 4 flips, scorer, 2026-11-05) | OOF 0.4219 | 0.35 |
| 6 (tree + spread) | OOF 0.4267 | 0.36 |
| 7 (stage 2, 2026-11-08) | OOF 0.4330 | 0.37 |
| 8 (ten refiners, 2026-11-10) | OOF 0.4399 | 0.37 |

Findings:

- The local-to-LB gap is about 0.065-0.07 for sub 1 and 2, and 0.06-0.07 for the later ones; the same as other
  U-Net entries (two independent U-Nets: 0.39-0.40 local, 0.32-0.35 LB).
- The public notebook family has an OOF-to-LB gap of 0.05-0.06 on every pipeline (0.4331/0.37, 0.4413/0.37,
  0.4429/0.38, 0.4486/0.39); they report adversarial AUC 0.496 between test and training frames and 5 of 24
  annotation groups absent from training, so they attribute the gap to the labels (test labels sparser)
  (`research-public-notebook-survey`, `research-leaderboard-snapshot`). Not verified by us.
- Uncertainty (bootstrap over images, 2,000 resamples, 2026-11-05): final pipeline of the day 0.4219 [0.412, 0.431];
  paired gain over submission 4 +0.0133 [0.008, 0.018]; per-fold spread about 0.01 at 141 images (mean 0.422 +- 0.012
  s.d.). Scaled to about 90 public images the interval is about +-0.027, twice the gain.
- So LB steps of 0.00-0.01 between our submissions cannot be resolved: a gain of +0.013 OOF showed 0.35 again; the
  sub-8 gain of +0.007 was expected to show 0.37 again (it did).
- Validation grouping (5-fold GroupKFold by calendar month, filaments persist for days) keeps same-week images from
  leaking; Hough disk and the limb handling are label-free.

Decisions:

- Do not choose by the LB; paired OOF with bootstrap over images, per-fold results (rule in openspec/config.yaml).
- Final selection: submissions 8, 7 and 6 by OOF order (DESIGN.md: "Final selection: submissions 7 and 6" before sub 8 existed); `reproducible-submission`.
- No attempt to close the gap by tuning to the LB.

## Non-goals

No access to the private split; no LB probing beyond the one empty-mask diagnostic.

## Falsified by

An OOF-validated pipeline whose public LB is within 0.01 of its OOF PQ would show our gap is pipeline-specific.

## Cost

CPU minutes for the bootstrap; no GPU.

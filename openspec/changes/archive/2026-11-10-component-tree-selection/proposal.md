## Why

A hand rule (area >= 300, mean probability >= 0.8) throws away good components and keeps bad ones, and a single
threshold either merges or fragments filaments. A learned scorer over a component tree chooses the best
threshold per filament. Landed 2026-11-05 as submission 6 (public LB 0.36; submissions 4 and 5: 0.35).

Source: DESIGN.md section 5 entries "Second-stage component scorer" (2026-11-04), "Component tree over
thresholds", "View-spread maps", "12-level tree with view-spread features" (2026-11-05), "Submission 5
candidate", "Research round" (selection maths, 2026-11-09). Idea cited from the public notebook
tushhaaarrrr "solar-filament-segmentation-2026-0-40" (component tree + hurdle scorer + exact antichain DP,
tau 0.26; their claimed gain about +0.065 over a fixed threshold, against a baseline with no scorer). Our
own measured gain is much smaller.

## What Changes

- Candidates: components at 12 probability thresholds 0.25-0.80 (closing 31 px for grouping only, >= 100 px),
  nested into a forest.
- 32 features per node: the 20 of `filseg/scorer.py`, the level, tree context, view spread.
- Hurdle scorer P(match) x E[IoU | match] (HistGradientBoosting), fitted on the 72,712 out-of-fold nodes with
  rows weighted by annotator count; `weights/tree_scorer_1536.pkl`.
- Exact antichain maximising sum(score - tau) by dynamic programming, tau 0.26; masks disjoint by construction.

## Non-goals

- Learned grouping (CondInst, DBSCAN), a detector for extra filaments, post-selection rescoring (see
  `rejected-selection-variants`, `rejected-detector-and-refiner-variants`).

## Falsified by

A paired OOF gain over the single-threshold + scorer pipeline (submission 5, 0.4219) whose bootstrap
interval includes zero. At landing: +0.0049 [0.0012, 0.0085]; the six-level tree without spread features
was +0.0034 [-0.0005, +0.0070], so the spread features carry the claim.

## Cost

CPU: tree build 59 min with 4 workers on a shared machine (72,712 nodes, 707 images); spread maps on HF Jobs
about $0.25. At test time 8.6 s per image on the M3 Max (loaded).

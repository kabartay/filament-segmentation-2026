# RESEARCH: where PQ is lost, near misses and small filaments (2026-11-04)

## Why

After the baseline U-Net, the next step had to be chosen by what the errors look like. Recorded in DESIGN.md
section 5 on 2026-11-04 (entries "Where PQ is lost", "Near misses", "Small filaments", "When does a doubtful
prediction pay?").

## What Changes

Nothing in the pipeline; this change records the analysis.

Question: are the losses grouping errors, extent errors, small filaments or annotator disagreement?

Method (MEASURED, 5-fold OOF, 707 images, setting 0.45 / 31 / 300 / 0.8): classify FN and FP by best-overlap
IoU, by GT area and by whether another annotator marked the same filament; marginal-value algebra for a
doubtful prediction.

Findings:

- PQ 0.4086, TP 5025, FP 3155, FN 3174. FN: 45% touch no prediction, 18% overlap one at IoU < 0.3, 37% at IoU 0.3-0.5.
- Recall by GT area: < 500 px 0.36, 500-1000 0.61, 1000-2000 0.68, 2000-5000 0.70, > 5000 0.64; small filaments are 812 of the FN.
- FP: 28% match another annotator's filament above IoU 0.5, 39% touch no annotation, 37% near misses.
- Turning every near-miss FN into a TP at IoU 0.55 would give at most PQ 0.487.
- Of 1,169 near-miss FN (IoU 0.3-0.5): prediction smaller than GT (area ratio < 0.67) 46%, similar size 20%, larger (> 1.5) 34%; only 17% also touch another GT (a merge), 16% of those GT are touched by several predictions (a split). The loss is mostly extent.
- Small filaments: share of other annotators marking the same filament (IoU > 0.5) is 0.28 below 300 px (65% marked by nobody else), 0.39 at 300-500, 0.49 at 500-1000, about 0.60 above 1000. Of 1,263 small GT (< 500 px): 451 TP, 400 inside a larger or wrongly shaped prediction, 295 with probability below 0.45 (31% below 0.2), 100 removed by the area filter, 17 by the confidence filter. Missed ones are mostly faint low-contrast darkenings.
- A prediction matching k of n annotators pays when k/n exceeds about PQ / (2 x SQ) = 0.41 / 1.33 = 0.31; one annotator of three is break-even, so the 28% of FP that match another annotator are roughly neutral.

Decisions:

- Small filaments are mostly annotator-specific: no small-object work.
- Extent, not grouping, is the loss: learned grouping (CondInst, DBSCAN) is low priority (`research-ideas-list`); the extent route went to resolution (`resolution-1536-four-views`) and a per-instance refiner (`stage2-rotated-crop-refiner`).
- A per-component keep rule from expected PQ contribution led to the scorer (`component-tree-selection`).
- Rule-based extent fixes were tested and failed (`rejected-*`: size-dependent confidence, hysteresis, lower minimum area, grow/shrink masks).

## Non-goals

No new model; no external labels.

## Falsified by

The same classification on other OOF probabilities giving a majority of grouping (merge/split) errors.

## Cost

CPU minutes on saved OOF probabilities. No GPU.

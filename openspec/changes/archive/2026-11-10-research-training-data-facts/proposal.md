# RESEARCH: facts measured on the training data

## Why

Every design choice (consensus target, component-based post-processing, limb handling, minimum-area filter)
depends on a few properties of the 707 training images and their annotations. They were measured once,
2026-11-03/04, and recorded in DESIGN.md section 2 and the 2026-11-04 decision-log entries (disk fit, resolution).

## What Changes

Nothing in the pipeline; this change records the question, method, numbers and the decisions they led to.

Question: what do the annotations look like (agreement, grouping, location on the disk, size), and what
does that allow the pipeline to assume?

Method (all MEASURED on the training set unless stated): inter-annotator PQ with `filseg/metric.py` on the 296
images with 2-3 annotators; own union mask split into 8-connected components and scored against itself; disk
position of annotations; Hough circle fit to the limb on all 887 images; per-image instance counts and areas;
GT mask round trip through 1024 and 1536 px.

Findings:

| Fact | Value |
|---|---|
| Training images / with 2-3 annotators | 707 / 296 |
| Inter-annotator PQ (296 images) | 0.3399 (SQ 0.634) |
| Own union mask as 8-connected components, PQ | 0.9972 |
| Annotations reaching beyond 70 deg from the central meridian | 2.0% (0.6% beyond 80 deg) |
| Disk centre / radius at 2048 px (Hough, all 887 images) | centre 1010-1034 px; radius 889-912 px |
| Instances per image (median) | 7 |
| Instance area, 5th-95th percentile | 316 to 6963 px |
| Bright-pixel disk fit (rejected) | overestimated the radius by up to 14% (516 vs 452 px at 1024 on train image 613) because it included the scattered-light halo |
| GT mask down to 1024 px and back up, mean IoU (400 sampled masks) | 0.975 (1536 px: 0.996) |
| Limb filter re-tuned with the Hough disk, 5-fold OOF | best 0.41032, same as the old fit; the 70-degree cut changes no TP and at most one FP |

Decisions:

- Train on the annotator consensus, not on one annotator (`stage1-consensus-unet`).
- A semantic mask plus connected components loses almost nothing (0.9972), so the instance step is a selection
  problem (`component-tree-selection`).
- Limb filter stays at 90 degrees (off): the model already learned that annotators skip the limb.
- The Hough disk replaced the bright-pixel fit (limb results re-run, no change).
- 1024 px is cheap in SQ, but the later fold-wise comparison chose 1536 px (`resolution-1536-four-views`).

## Non-goals

No new model or experiment; no external data.

## Falsified by

Re-running the same measurements on the training set giving other numbers (they are deterministic).

## Cost

Seconds to minutes of CPU per measurement; the Hough fit 0.1 s/image. No GPU.

# RESEARCH: annotator-versus-annotator ceiling

## Why

If annotators disagree as much as the model disagrees with them, further model work only fights label noise.
The ceiling was measured 2026-11-04 (DESIGN.md section 5, "Annotator ceiling", "Where the ceiling is").

## What Changes

Nothing in the pipeline; this change records the analysis.

Question: what PQ does one human annotator get when scored against another?

Method (MEASURED): on the 296 training images with 2-3 annotators, each annotator's masks scored as
predictions against each other annotator's masks with `filseg/metric.py`; compared with the U-Net on the same images.

Findings:

- Annotator against annotator: PQ 0.340 (0.3399), SQ 0.634, RQ = PQ / SQ = 0.536.
- U-Net (OOF, submission-4 model) on the same 296 images: 0.405; 0.414 on 1-annotator images, 0.414 on
  2-annotator, 0.400 on 3-annotator. On all OOF images SQ 0.667, RQ 0.616. The model is above a single
  annotator on both boundary quality and detection because it learns the consensus.
- Four model changes on fold 0 (ResNet-50, limb-flattened input, Lovasz loss, averaging) moved PQ by -0.026 to +0.000.
- The organisers write that a PQ above 0.35 is of great value to them (announcements 20 Aug and 23 Oct 2026).
- Re-measured 2026-11-04, same result.

Decisions:

- Train on the consensus (`stage1-consensus-unet`).
- Stop stage-1 architecture and loss work (`rejected-stage1-variants`) and spend the remaining time on selection and
  refinement (`component-tree-selection`, `stage2-rotated-crop-refiner`).
- Treat 0.55-0.70 public scores as implausible against a 0.34 human ceiling (see `research-leaderboard-snapshot`).

## Non-goals

No attempt to model individual annotators (identity codes exist only in training ids; see `research-ideas-list`).

## Falsified by

A re-measurement giving an annotator-versus-annotator PQ near or above the model's OOF PQ on the same images.

## Cost

CPU minutes. No GPU.

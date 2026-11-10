# REJECTED: selection and post-processing variants

## Why

Between 2026-11-03 and 2026-11-09 we tried rules and recalibrations on the saved OOF probabilities and
selections (CPU only) to cut false positives or fix extent. None beat the adopted pipeline (hand rule, later the
tree + hurdle scorer at tau 0.26, `component-tree-selection`). Kept so the question is not asked twice.

Source: DESIGN.md section 5 entries "Post-processing variants on 5-fold OOF", "Growing or shrinking the masks",
"Limb filter with the Hough disk", "Near misses", "Small filaments", "12-level tree", "Research round
(2026-11-09)".

## What Changes

Nothing ships.

## Non-goals

Re-opening selection: the research round measured that selection offers at most +0.001.

## Falsified by

A variant had to beat the reference on pooled OOF PQ, per fold. All were inside noise or worse.

## Cost

CPU only; the post-stage-2 rescoring was a nested, fold-held-out refit on 236 images (every 3rd).

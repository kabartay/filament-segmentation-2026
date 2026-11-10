## Why

The public "LB 0.70" notebook ([public notebook B], from nomannic19: Mask R-CNN R50-FPN 1024 + ResNet-18 crop U-Net refiner,
flip TTA) claims 0.70 against 0.37-0.39 for our OOF-validated pipelines (DESIGN section 6). The gap is unexplained.
Knowing whether the frozen pipeline reproduces on our data, out of fold, would tell us whether a detector + refiner
design carries signal we lack. Source: the notebook's own claim (0.70, twice on 2026-08-05, on what it calls a
preliminary board with a known empty-prediction defect).

## What Changes

STATUS: PARKED by the owner's decision (2026-11-10); no work is scheduled before the deadline. Nothing changes in
the pipeline. Checked so far:
- Notebook read as text only (never executed); no leakage visible in code; "magfilo" paths are blocked.
- Frozen weights downloaded and MD5-matched to the notebook's log: detector `dcd3053561e40ba78718546c31525d26`,
  refiner `74991b8d95f7922f0ca6fd75f8ab4f5f`.

What would be needed to proceed: run the frozen detector + refiner on our 707 training images (their weights are
trained on unknown data, so this is NOT out-of-fold evidence), score with our PQ, and look at whether PQ is near 0.70
(suggesting training on the labels) or near 0.4; about a day of GPU for 707 images at detector + refiner cost.

## Non-goals

- Not a candidate submission, no use of external weights in our final pipeline (config rule: no external data/labels).
- Not run on the test images beyond what the rules allow; the public leaderboard is not evidence.

## Falsified by

Not applicable while parked. If resumed: PQ on our 707 images near 0.7 means the weights saw these labels (no
conclusion about the design); near 0.4 means the design is not special.

## Cost

Parked: 0 h. If resumed: about 8-12 h GPU on the shared M3 Max, one heavy job at a time, plus inference setup.

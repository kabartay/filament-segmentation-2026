# SUPERSEDED: superseded-sub1-unet-fold0-epoch10

## Why

SUPERSEDED: submission 1 (2026-11-03), the first upload, made only to calibrate local PQ against the public
leaderboard. Recorded so the reasoning stays readable; it is not the current pipeline.

Source: docs/DESIGN.md section 5 ("Post-processing tuning, epoch-5 fold-0 probabilities", "Submission 1")
and section 4 step 2.

## What Changes (what it did)

- One U-Net, ResNet-34, fold-0 model at epoch 10 (a snapshot of a 40-epoch run), 1024 px (DESIGN.md section 3
  names no resolution for it; 1024 px is the pipeline of the early submissions, see submission 4).
- Hand-tuned post-processing: threshold 0.45, closing 31 px, area >= 400, mean-probability confidence >= 0.8.
- Result: 1,177 filaments on 175 of 180 test images, 1.23 s per image on MPS.
- Numbers (date 2026-11-03): local val PQ 0.389 (fold 0, 141 images); public LB 0.32. The local-to-LB gap
  (about 0.07) matched the two other U-Net entries, so local CV was judged a usable guide.

Not stated in DESIGN.md: the file name of the submission CSV and the exact augmentation of this snapshot.

## Replaced by

`superseded-sub2-unet-fold0-40-epochs`: the same model trained for the full 40 epochs (the epoch-10 snapshot
came from a run stopped early by mistake while the U-Net was still improving; DESIGN.md section 4 step 1).
The DESIGN entry's own decision ("the plain U-Net looks to level off around 0.32-0.35 LB; move to Mask R-CNN")
was partly mistaken for the same reason.

## Non-goals

- Not evidence for any later choice: one fold, one epoch snapshot, hand rule; the LB is never evidence.

## Falsified by

Local val PQ (0.389) not tracking the LB. It was not falsified: the 0.07 gap matched other entries, and
submission 2 later moved both together (0.415 local, 0.35 LB).

## Cost

About 1.23 s/image inference on the M3 Max (MPS); training cost of the epoch-10 snapshot not recorded as a
separate figure in DESIGN.md.

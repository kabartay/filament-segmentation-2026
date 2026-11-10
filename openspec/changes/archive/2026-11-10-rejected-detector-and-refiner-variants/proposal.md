# REJECTED: Mask R-CNN detector, detector + U-Net combinations, axis-aligned crop refiner

## Why

On 2026-11-03 the plain U-Net looked to level off around 0.32-0.35 on the leaderboard (DESIGN.md "Submission 1"
entry), and a public notebook family used Mask R-CNN plus a crop refiner. Both ideas were tried on fold 0 and
dropped; the stage-1 U-Net with tuned post-processing was stronger. Kept so the question is not asked twice.

Source: DESIGN.md section 4 steps 3-4 and section 5 entries "Mask R-CNN detector, fold 0", "Crop refiner on
U-Net proposals", "U-Net, full 40 epochs, fold 0", "The LB 0.70 entry".

## What Changes

Nothing ships. The successor of the refiner idea is `stage2-rotated-crop-refiner` (rotated crops, selected
instances as seeds), which did pass.

## Non-goals

CondInst, DBSCAN grouping and YOLO (not run; licence, build or time reasons are recorded in DESIGN.md).

## Falsified by

Accepted only if it beat the U-Net on the same fold-0 images by more than fold noise. It did not.

## Cost

Mask R-CNN (torchvision) 12 epochs, SGD lr 0.005, batch 2, about 6 min/epoch; crop refiner 15 epochs at about
1.6 min/epoch.

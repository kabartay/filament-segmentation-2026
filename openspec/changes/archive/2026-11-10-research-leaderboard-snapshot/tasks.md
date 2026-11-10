## 1. Snapshot and reading

- [x] 1.1 Download the LB. Verify: `kaggle competitions leaderboard --download`. Result: 2026-11-10 00:40 UTC, 1,460 teams, we are rank 477 at 0.37; 104 teams at 0.55, 322 at 0.38-0.42.
- [x] 1.2 Read the 0.38-0.42 notebooks as text. Result: dariushafshar OOF 0.4502 / LB 0.39; tushhaaarrrr OOF 0.4597 / LB 0.37-0.39.
- [x] 1.3 Decode the 0.55 literal as data. Result: 1,342 masks on 178 of 180 images, no overlaps or duplicates, 59% matched by submission 8 at IoU > 0.5; origin unknown.
- [x] 1.4 Read the 0.70 notebook. Result: Mask R-CNN + crop refiner, frozen checkpoints, claim dated 2026-08-05; unexplained.
- [x] 1.5 Empty-mask probe. Result: LB 0.00 (2026-11-04), defect fixed.
- [x] 1.6 Duplicate-mask simulation. Verify: `scripts/experiments/duplicate_exploit.py`. Result: honest 0.409 vs 0.506 (c 0.8, k=100), 0.611 (c 0.95, k=100); never submitted.

## 2. Records

- [x] 2.1 Recorded in docs/DESIGN.md sections 5 and 6.

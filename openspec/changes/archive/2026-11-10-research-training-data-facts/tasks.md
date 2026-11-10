## 1. Measurements

- [x] 1.1 Inter-annotator PQ. Verify: `scripts/experiments/oof_analysis.py human`. Result: MEASURED, 0.3399 (SQ 0.634), 296 images.
- [x] 1.2 Components of the own union mask. Result: MEASURED, PQ 0.9972.
- [x] 1.3 Limb reach of annotations. Result: MEASURED, 2.0% beyond 70 deg, 0.6% beyond 80 deg.
- [x] 1.4 Disk fit. Verify: Hough circle at 512 px on all 887 images, 12 random images checked by eye. Result: radius 889-912 px, centre 1010-1034 px at 2048, 0.1 s/image; bright-pixel fit overestimated by up to 14%.
- [x] 1.5 Instance count and area. Result: MEASURED, median 7 per image; area 5th-95th percentile 316 to 6963 px.
- [x] 1.6 Resolution round trip of the GT. Result: 400 sampled masks, mean IoU 0.975 at 1024 px, 0.996 at 1536 px.
- [x] 1.7 Limb filter with the Hough disk. Result: best 0.41032, no change from the old fit; kept off (90).

## 2. Records

- [x] 2.1 Recorded in docs/DESIGN.md section 2 and the 2026-11-04 entries "Disk fit", "Limb filter with the Hough disk", "Resolution".

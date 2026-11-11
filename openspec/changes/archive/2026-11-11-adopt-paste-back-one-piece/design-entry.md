2026-11-10, stage 2 paste-back, one connected piece per instance (change `adopt-paste-back-one-piece`). Idea from
tushhaaarrrr's notebook (6% of refined instances in 2-4 pieces; largest-piece only -0.0023; corridor join 0.4600 vs
0.4597). MEASURED on the submission 8 blend (`ensj`, threshold 0.40, OOF, 707 images): after paste-back 329 of 4,417
instances (7.45%) are more than one 8-connected piece (259 with 2, 52 with 3, 13 with 4, 5 with 5 or more). The share
rises with size: 0.3% under 500 px (1 of 310), 5.5% for 500-2,000 px (147 of 2,653), 12.1% for 2,000-8,000 px (162 of
1,336), 16.1% above 8,000 px (19 of 118). Three variants against `ensj_pi:t0.40` (PQ 0.43993, SQ 0.6803, RQ 0.6467,
multi-annotator subset 0.4406 on 296 images), paired bootstrap over images, 2,000 resamples:
`largest` (keep the largest piece) PQ 0.44027, gain +0.00035 (95% CI -0.00157 to +0.00236), per fold +0.0003, -0.0002,
-0.0000, +0.0017, -0.0002, SQ 0.6811, RQ 0.6464, multi-annotator PQ 0.4410; `join` (3 px corridors on a minimum spanning
tree of nearest-pixel gaps, free pixels only, gaps over 40 px left unbridged) PQ 0.43943, gain -0.00050 (-0.00094 to
-0.00013), all five folds negative (-0.0006, -0.0006, -0.0006, -0.0005, -0.0003), SQ 0.6802, RQ 0.6461, multi-annotator
PQ 0.4400, 12 instances stay multi-piece; `join_drop` (as `join`, then keep the largest piece) PQ 0.43974, gain
-0.00019 (-0.00081 to +0.00055), per fold +0.0007, -0.0003, -0.0006, -0.0005, -0.0003, SQ 0.6802, RQ 0.6464,
multi-annotator PQ 0.4403. Verdict: not adopted. Joining costs 0.0005 PQ (interval excludes 0, all folds negative);
`largest` is +0.00035, inside noise (interval spans -0.0016 to +0.0024), so it does not justify changing the submitted
pipeline. Default stays `single=""`; the flag remains in the code.

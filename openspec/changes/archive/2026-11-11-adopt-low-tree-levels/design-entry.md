**15-level tree, thresholds 0.10-0.80 (2026-11-10, runs/adopt-low-tree-levels_tree15s.pkl; REJECTED)**: tushhaaarrrr's
notebook states "15 levels 0.05-0.80" with "finer levels +0.0025" (step 0.05 would give 16 values; their exact grid is
not stated). We built levels 0.10, 0.15, 0.20, 0.25, ..., 0.80 (15 levels, view-spread features; 707 images, 92,376
nodes, 15,038 s on 3 workers under heavy machine load) and ran the cross-validated hurdle with context features
(`component_tree.py eval --context --modes hurdle`, MEASURED, OOF on all 707 images). Variants tried: one (the 14-level
0.15-0.80 variant was not run because the first result was negative at every tau). PQ by tau: 0.22 0.4199, 0.24 0.4208,
0.26 0.4217, 0.28 0.4221, 0.30 0.4230, 0.32 0.4224, against the 12-level tree at 0.4253 / 0.4265 / 0.4267 / 0.4269 /
0.4262 (tau 0.22 / 0.24 / 0.26 / 0.28 / 0.30). At the submitted tau 0.26 the change is -0.0050 PQ, and it is negative
on 5/5 folds (15-level vs 12-level, fold 0-4: 0.4301/0.4184/0.4029/0.4250/0.4325 vs 0.4340/0.4247/0.4107/0.4304/0.4341).
Even at its best tau (0.30) the 15-level tree is 0.0032 below the 12-level best. At tau 0.26 TP 4,833 / FP 2,440 / FN
3,366 against 4,872 / 2,341 / 3,327: the extra low-level candidates cost true positives and add false positives. The
selection does use the low levels (tau 0.22: 697 of 5,709 selected nodes at 0.10-0.20: 277, 152, 268) but they do not
pay. Against the level-0.45 control the 15-level hurdle is -0.0002 at tau 0.26 (95 % CI -0.0042 to +0.0036), while the 12-level
tree is +0.0049 (CI +0.0012 to +0.0085): the gain of the 12-level tree over the control is lost. Verdict: low levels
hurt; keep 12 levels 0.25-0.80. Note: `component_tree.py eval` shadows `levels` inside the tau loop, so level counts
in the json are correct only for the first tau (0.22); later taus print zeros (not fixed, unrelated to the verdict).

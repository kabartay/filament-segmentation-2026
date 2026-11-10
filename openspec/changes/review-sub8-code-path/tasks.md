## Verdict

No error found that changes submission 8's masks or invalidates its OOF numbers. Six confirmed issues, all low or
medium severity; none needs a fix before the deadline. Scripts for every check live outside the repo
(scratchpad `rsub8/`) and use only read-only paths in `runs/` and `data/`. Python: `$MAIN/.venv/bin/python -I`,
`OMP_NUM_THREADS=1`, `nice -n 10`; set `FILSEG_RAW` to `$MAIN/data/raw/MAGFiLO_1.0_Kaggle_2026` when running from a worktree.

## 1. Confirmed issues (ranked)

- [x] 1.1 MEDIUM, silent failure mode: `filseg.predict` does not check the cache resolution. With `FILSEG_CACHE` unset,
  `CACHE` is `data/cache` (1024 px, `filseg/cache.py:34`; `data/cache/index.json` says `res: 1024`), the network sees
  1024 px inputs and the tree scorer (fitted on 1536 px maps) gets a different feature distribution, with no error.
  `submissions/sub8_*.json` does not record the resolution, so the CSV cannot be proved 1536 px from its metadata
  (the notebook's byte-identical rebuild sets `FILSEG_CACHE` itself, `scripts/make_notebook.py:204`).
  Evidence: `filseg/predict.py:84-86,107`; `filseg/cache.py:34`. Fix (applied, output-neutral): print a warning when
  the cache is not 1536 px with the tree scorer, and record `cache_res` in the metadata JSON. Verify:
  `FILSEG_CACHE=data/cache1536 .venv/bin/python -m filseg.predict ...` gives `"cache_res": 1536` and an identical CSV;
  without the variable it prints the warning. Result: applied in `filseg/predict.py`; not re-run end to end (GPU/CPU
  budget); the CSV path is untouched.
- [x] 1.2 LOW-MEDIUM, mild optimism in OOF: fold checkpoints are `best.pt`, chosen by the fold's own validation PQ among
  epochs 5, 10, ..., 40 (`filseg/train.py:236-240,269-273`; `scripts/experiments/oof_view_spread.py:55`), while the
  submitted model is `last.pt` of the all-data run (epoch 40, no selection). From `runs/unet_r34_1536_f{0..4}/log.txt`
  the best epoch is the last one on folds 0, 2, 3, 4 (0.4029, 0.3786, 0.3872, 0.3997 on the 2-view hand rule) and
  epoch 35 on fold 1 (0.3852 against 0.3834 at epoch 40), so the selection is worth at most 0.0018 on one fold of
  five on the stage-1 hand-rule PQ. Confirmed by reading the logs; size of the effect on the final 0.4399 not measured
  (it would need the epoch-40 fold maps). Fix: none needed; state in the report that the fold checkpoints were selected on
  their validation fold. Verify: `grep -E "val PQ|done" runs/unet_r34_1536_f1/log.txt | tail -4`.
- [x] 1.3 LOW, second-order leakage in the stacked OOF design. (a) The refiner of fold f is trained on crops of folds
  g != f whose seeds were chosen by a hurdle scorer fitted on all folds except g, that is including fold f's labels
  (`stage2_oof.py:78-82`); the stage-1 map of fold g also comes from a model trained on fold f's images. (b) Choices were
  made on the same OOF data: tau 0.26 (7 values), pixel threshold 0.40 (3-6 values), the refiner blend (5 candidates).
  Neither is a direct label leak into the evaluated fold. Size of (b): PQ at t = 0.35 / 0.40 / 0.45 is 0.43907 /
  0.43984 / 0.43971 for the weighted blend (`runs/stage2/w12_pi.json`) and 0.43453 / 0.43546 / 0.43507 for UNet++
  (`upp_eval.json`), so the threshold is chosen on a plateau (spread 0.0009); the blend choice margin over the runner-up
  is 0.0009 [-0.0001, 0.0019] (DESIGN.md). Fix: none; report the gain over stage 1 (paired, same selection) rather than
  the absolute 0.4399 as the evidence. Verify: `python3 -c` over the JSON files above.
- [x] 1.4 LOW, train/test mismatch in the jitter refiner only: `augment` (`stage2_oof.py:233-255`) resamples the
  normalised input with `padding_mode="zeros"`, so the rotated or shifted border strips of the image channel are 0
  (raw grey 0.5) while a real crop padded by `warpAffine` would be -2 (raw black) and, in practice, never has borders
  (0.000 of 32 sampled crops' image pixels equal -2). Measured on 32 crops: 10.9% of augmented image-channel pixels
  are exactly 0.0 against 0.0% in unaugmented crops. The evaluation (`step_eval`) uses unaugmented crops, so OOF is
  not optimistic; the effect is a small train/test mismatch for the five `stage2_jit` models, which already
  top the OOF table. Geometry of `augment` itself is correct (identity parameters reproduce the input to 2e-5).
  Fix if ever retrained: pad the image channel with its own border value, or use `padding_mode="reflection"` for it.
  Verify: `rsub8/augcheck.py`.
- [x] 1.5 LOW, cosmetic: 105 of 1,065 submitted masks have more than one 8-connected piece (checked on the CSV).
  `paste_back` keeps only the pieces touching the seed (`stage2.py:257-260`), but the later arg-max ownership step
  (`stage2.py:278-283`) can cut a mask into pieces. The scorer sees one RLE per instance, so IoU is unaffected; the
  pieces are not errors. Verify: `rsub8/csvcheck.py`.
- [x] 1.6 INFO, ensemble-size mismatch between OOF and test for stage 2: OOF crop probabilities of a tag come from one
  fold model (`stage2_oof.py:315-316`), the test crop probabilities average ten models and four flips
  (`stage2.py:171-183`). An average of five fold models is smoother than one model, so the pixel threshold 0.40
  sits at a slightly different point of the distribution. The plateau in 1.3 bounds the effect; expected test PQ is
  not lower than OOF on this account. No fix.
- [x] 1.7 INFO, stage-1 map shift between OOF and test: the hurdle scorer and refiners were fitted on maps of fold
  models (80% of the data, `best.pt`) and applied at test to the all-data model's maps (`predict.py:99-103`). This is the
  deployment the docs describe; the test probabilities come from a model that never saw the test image, as the OOF
  ones did, so there is no leak. Whether the all-data maps are slightly more confident was not measured on test (a
  180-image CPU run was out of budget); counts per image give a weak check: 1,065 masks on 177 of 180 test images
  (5.9 per image) against 6.3 stage-1 seeds per image in OOF (4,429 over 707 images).

## 2. Refuted suspicions (checked, no defect)

- [x] 2.1 Crop rotation/unwarp axes. `frame_of`/`warp`/`unwarp` (`stage2.py:54-83`): rotation matrix has positive
  determinant (no mirror), `WARP_INVERSE_MAP` with the forward matrix is the correct inverse, no off-by-one. Checks
  (`rsub8/geomcheck.py`, 157 stored crops from `runs/stage2/meta.pkl`): recomputed `M` equals stored `M` exactly; the
  seed channel equals `warp(seed)` exactly; seed -> crop -> seed IoU min 0.934, median 0.986 (loss from the downscale
  of the 0.48-1.0 scale crops); principal axis |cos| with the crop x-axis >= 0.9999; a marked 4x4 dot at
  (881.5, 691.5) returns at (881.6, 691.4); synthetic tilted bar lands on y = 64 (the crop centre row).
- [x] 2.2 Flip views and prob/logit. `predict_with_spread` and `predict_crops` flip the input, apply sigmoid and flip the
  output back on the same dims; mean and population std over the four views match `scripts/experiments/oof_view_spread.py`
  (the 2-view `predict` is not on the submission path). Normalisation (x/255 - 0.5)/0.25 is the same in `train.normalise`
  and `stage2.to_tensor`; channel order `CHANNELS` is the same in `crop_one`, training and inference.
- [x] 2.3 float16. Stage-1 mean and spread are float16 in OOF (`val_probs_v4.npy`) and at test (`predict_with_spread`);
  both are cast to float32 before `upsample`. Refiner weights are float16 on disk and loaded as float32
  (`load_refiners`); DESIGN records 1 crop pixel in 10^5 changing at 0.40.
- [x] 2.4 RLE format. Masks are encoded with `np.asfortranarray` (`stage2.py:230`, `postprocess.py:78`), the same path
  the ground-truth polygons and our scorer use; all 1,065 submitted RLEs decode to the 2048 x 2048 size and
  re-encode to byte-identical counts (`rsub8/csvcheck.py`; 0 non-canonical). `counts.decode()` of the bytes is
  ASCII-safe. The organisers' scorer was not available; "compressed COCO RLE" is what `filseg/metric.py` documents.
- [x] 2.5 Ids, empty images, instance dropping. Ids are `<stem>_<n>`, n = 1..k per image without gaps, all stems known,
  unique (asserted in `write_csv`); 3 of 180 test images have no row, as designed (the scorer counts their
  annotations as misses; they carry no false positives). In OOF the same rule leaves `len(chosen) == 0` for the images
  (17 of 707, 2.4%; task 4.2). `paste_back` cannot erase every instance of an image: the instance
  scored lowest is absorbed first, and the last alive one has no `others`.
- [x] 2.6 Overlap. 0 pixels shared between any two masks on all 177 predicted test images (`csvcheck.py`). Two disjoint
  predictions cannot both have IoU > 0.5 with one ground-truth mask, so `PQCounts.multi` stays 0 and OOF PQ is not
  inflated by double matching.
- [x] 2.7 Tree structure. Parent of a node is read from its first pixel at the previous level (`tree.py:81-83`); closing
  and thresholding are monotone, so the assertion holds and a dropped (< 100 px) lower node implies the higher one
  was dropped too. `antichain` indexes children after parents, ties resolved deterministically. `select_scored`
  returns scores aligned with the sorted node indices. Feature names are stored with the scorer, in training order.
- [x] 2.8 Seeds and nondeterminism. Stored OOF frames reproduce exactly (2.1), so `np.linalg.eigh` axis sign is stable on
  this machine; if it flipped, the crop would be rotated by 180 degrees, which the four-flip average and flip training
  absorb. Gradient boosting at inference is deterministic; `np.argsort(score)` in `paste_back` is unstable only for
  exactly tied scores. Training seeds are fixed (`torch.manual_seed(f)`, `default_rng(f)`). MPS against CPU/CUDA: see task 4.
- [x] 2.9 Fold leakage in the splits. `fold_split` groups by YYYYMM; the OOF maps, the hurdle CV (`component_tree.py:144-155`,
  `stage2_oof.py:79-82`) and the stage-2 train/eval (`stage2_oof.py:276,315`) all use the same fold ids, and the jitter
  crops of fold f are excluded from fold f's refiner (`stage2_oof.py:279`). `compare` asserts identical image lists and
  resamples whole images; per-fold deltas are reported.
- [x] 2.10 Threshold chosen on the same data: refuted as a material issue, see 1.3 (plateau).

## 3. Code touched

- [x] 3.1 `filseg/predict.py`: warning and `cache_res` in the metadata JSON (task 1.1). Verify: `git diff --stat`; the CSV
  code path is unchanged.

## 4. Re-run and paste-back checks

- [x] 4.1 CPU re-run (single process, 1 thread, 667 s on the loaded machine) of test image `20110214175414Mh` through the
  all-data model's four views, the tree scorer and the ten refiners (`rsub8/rerun.py`): 3 masks, each IoU 1.000 with
  the submitted mask and identical area (1444, 1996, 975 px). So the CSV is reproducible from the weights, CPU and MPS
  agree on this image, and the pipeline is deterministic here. Only one of the two planned images finished (the second
  was stopped at the time budget).
- [x] 4.2 Paste-back on every 20th OOF image with the blended refiner output (`rsub8/pastecheck.py`): 35 images, 213
  seeds, 212 masks out (one seed absorbed), 0 overlapping pixels, 2 seeds kept as stage 1 for scale < 0.5. OOF images
  with no selected seed: 17 of 707 (2.4%), against 3 of 180 test images (1.7%).

## Why

The best submission-8 refiner pair is UNet++ (plain crops) + U-Net (jitter crops + augmentation). DESIGN section 5
(refiner blends with the jitter model) ends with "a UNet++ trained on the jitter crops was not tried". The jitter
training lifted the U-Net by +0.0054 over the sub7 refiner; the UNet++ was the stronger architecture on plain crops, so the
combination is the one untested cell. Source: our own OOF results; no outside claim.

## What Changes

- Train a UNet++ (ResNet-34) on plain + jitter crops with augmentation, 15 epochs, five folds:
  `stage2_oof.py train --tag uppj --arch UnetPlusPlus --extra jit --aug --epochs 15`. Status 2026-11-10: RUNNING, log
  `runs/stage2/uppj_train.out`. Status 2026-11-11: REJECTED by the rule below (see tasks.md 2.3).
- Evaluate it (`eval`/`pi`), then blends: uppj alone, uppj + jit, uppj + upp + jit, compared paired against `ensj_pi:t0.40`
  and against any adopted weighted blend (`stage2-weighted-blends`).
- If adopted: ship `weights/stage2_uppj_f{0..4}.pt` (float16, about 50 MB each) and build a new submission.
- Capability note: the delta is `## ADDED Requirements` under `stage2-refinement` (absent from `openspec/specs/` until
  another agent's archive creates it); it applies only on adoption.

## Non-goals

- No other architecture or encoder, no change to stage 1, tree or paste-back.
- No retraining of upp/jit.
- No leaderboard-driven choice.

## Falsified by

The best blend containing uppj does not beat `ensj_pi:t0.40` with a paired interval excluding 0 and 5/5 folds
positive (the sub8-to-sub7 gain was +0.0070 [0.0050, 0.0090]; a gain below about +0.001 is not worth 15 more weight files).

## Cost

GPU: five folds x 15 epochs on the shared M3 Max, one heavy job at a time (about as long as the jit run, see `runs/stage2/jit_train.log`);
CPU: evaluation and blends under an hour. Inference adds five 50 MB refiners (about +50% refine time per image) if adopted.

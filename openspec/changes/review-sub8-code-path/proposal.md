## Why

Submission 8 (OOF PQ 0.4399, public LB 0.37) is the final entry. Its code path was written over several weeks by
several sessions, and its out-of-fold numbers are what we chose it on. This is an independent read of the code
(not of the docs' conclusions) for train/test mismatches, geometry errors, paste-back and RLE faults,
nondeterminism and optimism in the OOF evaluation, each suspicion confirmed or refuted with a small CPU check
on files in `runs/` and `data/` (read-only). Source: our own code; no outside claim.

## What Changes

- This change records findings only (`tasks.md`). No pipeline behaviour changes.
- One trivial, output-neutral edit in `filseg/predict.py`: the metadata JSON gains `cache_res` and the script prints
  a warning when the tree scorer runs on a cache that is not 1536 px. The CSV is byte-identical.

## Non-goals

- No retraining, no new submission, no change to thresholds, tau, refiner set or the metric.
- No Kaggle upload, no push.

## Falsified by

The review is wrong if a confirmed issue below does not reproduce with the commands in `tasks.md`, or if a
"refuted" item is shown to change submission 8's CSV (a re-run of `filseg.predict` that differs from
`submissions/sub8_r34_1536_tree_stage2_ens.csv` beyond the CPU/MPS numerical differences recorded in task 4).

## Cost

CPU only, single process, nice 10, `OMP_NUM_THREADS=1`: about 20 min in total on the loaded machine. No GPU.

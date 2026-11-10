## Why

A public notebook (tushhaaarrrr, read as text) reports a hurdle scorer with lower capacity (15 leaves, its E24),
the observing station as a feature, neighbour features (nb_*) and "rich descriptors" (+0.0030 claimed, confounded
with a third ensemble member). Our stage-1 tree scorer (HistGradientBoosting hurdle, 31 leaves, 32 features,
OOF PQ 0.4267 at tau 0.26) might gain from the first three on the existing node table.

## What Changes

Evaluation only (scripts/experiments/adopt_scorer_features.py, adopt_spread_parity.py). No change to the pipeline
unless a variant wins (>= 4/5 folds, paired CI excluding 0); then stage 2 would need new crops.

## Non-goals

No tree rebuild, no GPU training, no stage 2 run, no "rich descriptors" (need a feature bank we do not have).

## Falsified by

No variant beating the baseline selection at tau 0.24/0.26/0.28 on >= 4/5 folds with a paired-bootstrap interval
above 0 (10 variants tried; selection optimism is reported).

## Cost

CPU only: centroid sidecar ~1 h CPU, 10 CV fits of the hurdle; 50-image GPU inference (~10 min under the lock).

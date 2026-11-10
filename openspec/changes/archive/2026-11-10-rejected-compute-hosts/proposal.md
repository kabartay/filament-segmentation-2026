# REJECTED: Kaggle as a training host

## Why

All training runs on one shared Apple M3 Max, with HF Jobs A100 for the larger runs. Kaggle's GPU and CPU
sessions were checked as extra capacity (2026-11-07 and 2026-11-10). Neither was usable. Kept so the question
is not asked twice.

Source: DESIGN.md section 5 entries "Kaggle GPU not used (2026-11-07)" and "Kaggle CPU as a training host
(2026-11-10)".

## What Changes

Nothing ships. Kaggle CPU could take CPU-only jobs (crop building, tree builds, evaluations), none of which is
the bottleneck.

## Non-goals

Moving the pipeline to Kaggle. HF Jobs use is recorded with the change that needed it
(`resolution-1536-four-views`).

## Falsified by

A refiner training step on Kaggle CPU within a small factor of the local MPS time, or a free batch GPU
session. Neither held.

## Cost

One probe kernel (`scripts/kaggle/cpu_probe/`, private kernel muhakabartay/filseg-cpu-probe).

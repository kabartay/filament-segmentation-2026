## Why

tushhaaarrrr's public notebook reports that 6% of its refined instances come out in 2-4 pieces; keeping only the
largest piece cost -0.0023 there, and joining the pieces with 3 px corridors along a minimum spanning tree of the
shortest gaps (`filseg.cascade.join_parts`) gave 0.4600 vs 0.4597 (neutral) while fixing 1:n morphology. Our stage-2
paste-back (`paste_back`, `filseg/stage2.py`) keeps every piece touching the seed, so some instances are several
pieces. This change measures how many, and whether forcing one piece changes OOF PQ.

## What Changes

Status 2026-11-11: REJECTED (run and evaluated, not adopted; sub8 stays final).

- `paste_back(..., single="")` gains a flag, default off (submitted pipeline unchanged): `largest`, `join` (corridors
  width 3 px, minimum spanning tree of nearest-pixel gaps, corridor pixels only where no instance has pixels, gaps over
  40 px not bridged so those pieces stay separate), `join_drop` (as `join`, then keep the component holding the
  largest piece).
- `stage2_oof.py eval --single ... --cap ...` passes the flag.
- Source: tushhaaarrrr's notebook (read as text only). Claimed gain: +0.0003 (0.4597 -> 0.4600), their own data.

## Non-goals

- No change to stage 1, the tree, refiners or threshold 0.40; no new training.

## Falsified by

Paired gain over `ensj_pi:t0.40` below 0, or any fold clearly negative -> reject.

## Cost

CPU only, about 5 min per variant on a free machine (3 workers); far longer on the loaded machine.

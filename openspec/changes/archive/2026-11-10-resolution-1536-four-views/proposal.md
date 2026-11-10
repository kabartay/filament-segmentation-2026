## Why

One boundary pixel moves IoU a lot (filaments are a few pixels wide at 2048 px; grow/shrink failed), so a
sharper probability map may raise SQ. Four flipped views average out flip-dependent noise and give a free
uncertainty signal. Landed 2026-11-05.

Source: DESIGN.md section 5 entries "1536 px input, fold 0", "More TTA views", "Full 1536 px pipeline",
"2048 px (native)", "View-spread maps".

## What Changes

- Input resolution 1024 -> 1536 px (area-downsampled cache; target computed at 2048 then downsampled).
- Inference averages four flipped views (identity, h, v, both): mean = probability map, per-pixel std = view
  spread; both upsampled to 2048.

## Non-goals

- 2048 px input (measured, no gain); 8 dihedral views.

## Falsified by

Paired OOF gain of 1536 over 1024 px not positive on most folds; 4 flips not better than 2 views at the
same setting. Neither was falsified: 1536 better on four of five folds, 4 flips +0.0043 on fold 0.

## Cost

About 270 s/epoch at 1536 px on the M3 Max; HF Jobs A100: 43-53 s/epoch, about $1.70 per fold (total spend
about $5.0 by 2026-11-04 plus $2.75 per 2048 px fold). View-spread maps: 63 s per fold, about $0.25.

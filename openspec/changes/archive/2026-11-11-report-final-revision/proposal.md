## Why
The report text predated submission 8 (ten refiners, public LB 0.37) and its "where PQ is lost" section used the 1024 px rule pipeline only.

## What Changes
report/main.tex: abstract and Table 1 carry the final pipeline and its LB 0.37; stage-2 description matches filseg/stage2.py (extra-crop criteria, paste-back fallbacks); OOF-to-LB gap explained as label-side with hedges; "what did not help" gains the post-stage-2 negatives; "where PQ is lost" gains the post-stage-2 subsample analysis (SUBSAMPLE, 236 images); selection-optimism note; qualitative figure shrunk to 0.78 of the text width to keep the body on page 4.

## Non-goals
No new experiments, no change to the submission, no new numbers outside docs/DESIGN.md.

## Falsified by
Any report number that differs from docs/DESIGN.md.

## Cost
No compute.

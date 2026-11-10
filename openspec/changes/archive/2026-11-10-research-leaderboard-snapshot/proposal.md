# RESEARCH: public leaderboard snapshot and the notebooks behind its groups (2026-11-10)

## Why

Decide whether the public leaderboard (LB) contains information that should change the method, and what the
0.38-0.42, 0.55 and 0.70 groups are. Recorded in DESIGN.md section 5 (2026-11-04 entries on the LB 0.70 entry,
competition pages and duplicate-mask simulation) and section 6 (snapshot 2026-11-10 00:40 UTC, notebooks read by one read-only agent).

## What Changes

Nothing in the pipeline; this change records the snapshot and reading. Notebooks read as text, none executed.

Snapshot (full CSV via `kaggle competitions leaderboard --download`): 1,460 teams; we are rank 477 at 0.37 (9
submissions). Teams per displayed score: 0.70 1, 0.64 1, 0.62 4, 0.61 1, 0.60 3, 0.58 2, 0.57 4, 0.56 6, 0.55 104,
0.54 3, 0.53 1, 0.49 1, 0.42 1, 0.41 19, 0.40 63, 0.39 136, 0.38 103, 0.37 137, 0.36 116, 0.35 87, below 0.35 about 800.

Groups:

- 0.38-0.42 (322 teams): legitimate pipelines from competition data only, disjoint masks. dariushafshar
  "solar-filament-u-net-oof-0-45-vs-public-0-39": 10 U-Nets (5 R34 + 5 EfficientNetV2-S) at native 2048 px, 8-view D4
  TTA, keep rule on components (low threshold 0.35, peak >= 0.99, mean p >= 0.85, area >= 200); nested OOF 0.4502, LB 0.39.
  tushhaaarrrr "0-40" (v4): OOF 0.4597, LB 0.37-0.39. The teams at 0.40-0.42 are most likely these and forks (not
  mapped team by team). Selection on a noisy LB (about 90 public images, +-0.027; teams submitting 10-30 times keep their
  luckiest variant) is the rest. Their own gap 0.05-0.06, ours 0.06-0.07 (`research-oof-to-lb-gap`).
- 0.55 (104 teams, first on 2026-08-26): [public notebook A] "[notebook A]" has no model at
  inference; it decodes a hard-coded base85 + LZMA payload into the submission. Decoded as data
  (`ast.literal_eval` of the string literals, nothing executed): 1,342 masks on 178 of 180 test images, median area
  1,294 px (ours 1,390), no overlapping pixels, no duplicate masks, so no scorer exploit in the file. 59% of its
  masks are matched by submission 8 at IoU > 0.5 (median best IoU 0.64). Origin not stated; consistent with a model
  trained on labels of the test images (e.g. the public MAGFiLO release), not verified. Earlier (2026-11-04) read
  as a 5-epoch throwaway U-Net plus an embedded compressed CSV ("CHAMPION_PAYLOAD"), not decoded then.
- 0.70 (one team, [top team]): no public notebook by that user. The public "LB 0.70" notebook ([public notebook B] handoff,
  from nomannic19's Mask R-CNN R50-FPN 1024 + ResNet-18 crop U-Net refiner, flip-4 TTA on the refiner) loads frozen
  checkpoints, claims 0.70 on 2026-08-05 on a preliminary board with a known empty-prediction defect, blocks
  "magfilo" paths; masks are disjoint (positive overlaps merged), so the double-counting hypothesis is wrong. Our
  detector scored 0.377 and the annotator ceiling is 0.34 (`research-annotator-ceiling`). Unexplained.
  perseus7 "competition-metric-flaw-proof" shows repeating a mask raises PQ; none found applying it.
- Defect history: before the 12 Aug 2026 rescoring the LB gave 1.00 for five empty masks, 0.88 for one empty mask per
  image and 0.70 for seven (refs/please-fix-the-leaderboard); empty masks on 5 images scored 1.00 (artkomissar). Our
  empty-mask probe, uploaded with the owner's approval 2026-11-04: LB 0.00, so the defect is fixed and the LB computes PQ.
- Organisers (20 Aug, 23 Oct): any PQ above 0.35 is of great value; they review the top 250 teams; hacking warned
  against. The honest level is about 0.35.
- Duplicate-mask inflation (2026-11-04, `scripts/experiments/duplicate_exploit.py`; simulation only, never
  submitted): c 0.8: k=2 0.452, k=20 0.501, k=100 0.506; c 0.9: k=2 0.463, k=20 0.557, k=100 0.569; c 0.95: k=100 0.611;
  honest 0.409. PQ tends to the SQ of the duplicated masks (about 0.67-0.70), matching the 0.55-0.70 cluster.
- Legitimate techniques seen in the 0.38-0.42 group and not used, with their claimed gains: 8-view D4 TTA (our 8
  views no better than 4: 0.4187 vs 0.4192 fold 0), model-family ensemble (+0.003; our R34+R50 gave nothing),
  15 tree levels (+0.0025; our 6 -> 12 gave nothing), RF-DETR detections as scorer features (+0.0034), weight EMA,
  forcing one connected piece per refined instance. Already in ours: soft annotator-mean target, view-spread scorer
  features, tree + hurdle + antichain, rotated crop refiner with the other instances as a channel.

Decisions:

- No change of method; choose on paired OOF, never on the public LB (also a rule in openspec/config.yaml).
- Submissions stay disjoint, no duplicate masks; no use of the 0.55 file or any label from the public release
  (`reproducible-submission`).
- Submission 8 (OOF 0.4399) scored 0.37; the expected change was +0.007, a quarter of the LB interval.

## Non-goals

No chasing of 0.55-0.70; no execution of public notebooks; no use of their masks.

## Falsified by

An honest, documented, OOF-validated public method scoring above about 0.45 out of fold with competition data only.

## Cost

Reading and one CSV download; no GPU.

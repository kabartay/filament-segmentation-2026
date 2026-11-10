# RESEARCH: public notebook survey (2026-11-05)

## Why

Before spending the last days on a method, find out what the other entrants' public notebooks report, which
techniques carry documented gains, and which are not usable. Recorded in DESIGN.md sections 5 and 6
("Public notebook survey (2026-11-05)").

## What Changes

Nothing in the pipeline; this change records the survey.

Question: which published ideas have a measured gain that we can reproduce from competition data only?

Method: about 19 public notebooks read as text (none executed; notebooks pulled without saved outputs, so
numbers are their own markdown claims). Only tushhaaarrrr "solar-filament-segmentation-2026-0-40" (v4) has a
documented ablation table.

Findings (the entrants' own claims, ESTIMATED from our side):

| Source | Method | Local PQ | LB |
|---|---|---|---|
| tushhaaarrrr 0-40 (v4) | U-Net R34 + U-Net ConvNeXt-T + UNet++ R34, 5 folds, D4; component tree over 15 thresholds, LightGBM hurdle scorer (P(match) x E[IoU given match]), exact antichain DP, tau 0.26; rotation-normalised crop refiner | 0.4597 OOF | 0.39 (v3) |
| tushhaaarrrr 0-37 | residual U-Net, 3 physical channels, components + GBM scorer | 0.4331 OOF | 0.37 |
| muhammadshoaibaltaf V11 | YOLO26x 1280 detect + UNet++ EffB4 crops | 0.4347 (15% hold-out, pooling unclear) | not stated |
| timgluhkikh | YOLO11s detect + crop U-Net R34 | about 0.409 | not stated |
| pandadev0069 | YOLO26s-seg at 2048, union target | not stated | 0.36 |
| sivapnlr80 | 5-fold Mask R-CNN, no refiner | not stated | 0.34 |
| vmerckle | no ML: dark residual percentile threshold, up to 120 components | none | claims 0.46 (unverified) |

Reported gains (paired over 5 partitions; seed noise about +-0.002): component tree + hurdle + DP about +0.065
over a fixed threshold (0.372 to about 0.450); stage-2 refiner +0.006 to +0.009; ConvNeXt-T member +0.003,
UNet++ member +0.002; RF-DETR evidence features +0.003. Reported failures: clDice/skeleton loss -0.045,
Lovasz -0.003, no Dice term -0.006, FPN decoder -0.008, DeepLabV3+ and MAnet -0.002, temporal TTA -0.020,
noisy-student pseudo-labels on external frames +0.003 (noise), per-annotator / leave-group-out modelling
+0.000, isotonic calibration -0.0005. Their OOF-to-LB gap is 0.05-0.06 on every pipeline (0.4331/0.37,
0.4413/0.37, 0.4429/0.38, 0.4486/0.39); adversarial AUC test vs train frames 0.496; 5 of 24 annotation groups
absent from training (see `research-oof-to-lb-gap`).

Earlier entries (before the survey, DESIGN.md section 5, 2026-11-04): "LB 1st" notebooks (rokaiyasomapti,
foysalemonshanto) trained on empty masks and encode RLE at 512 px, not a usable baseline; Agnuxo1 repo
ResNet-34 U-Net 0.400 local (3 seeds, 1024 px), best LB 0.35; FarnooshMemari U-Net 0.389 local, LB 0.32;
jamesmills1975 (CAGE) ResNet-34 U-Net + hull head 0.402 local (optimistic split), not submitted.

Decisions:

- Implement the component tree with a hurdle scorer in our own code, first (CPU), then a hurdle target and
  view-spread features: `component-tree-selection`. Our measured gain was +0.0041 (6 levels) and about +0.005
  (12 levels with spread), far below the public +0.065 because their baseline had no scorer.
- A rotated crop refiner followed the tushhaaarrrr design: `stage2-rotated-crop-refiner`.
- Model-family ensembles only if GPU budget allowed; our R34+R50 average had given nothing (`rejected-stage1-variants`); ConvNeXt-T fold 0 gave +0.0016, not continued.
- Dice term, Lovasz, clDice, DeepLabV3+: not pursued, consistent with their failures.

## Non-goals

No notebook was executed; none of their weights or data were used.

## Falsified by

Our own paired OOF gain from the component tree being indistinguishable from zero (measured: +0.0041
[0.0006, 0.0076], 4 of 5 folds better).

## Cost

Reading only; no GPU.

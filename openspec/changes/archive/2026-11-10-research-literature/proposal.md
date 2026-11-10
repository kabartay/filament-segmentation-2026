# RESEARCH: literature used

## Why

The report and repository must cite the sources of adopted and rejected ideas. DESIGN.md section 7 lists them;
the 2026-11-09 research round added a literature agent (see `research-pq-selection-and-oof-errors-2026-11-09`).

## What Changes

Nothing in the pipeline; this change records the reading and what each source led to.

Sources and use:

- Kirillov et al. 2019, Panoptic Segmentation (CVPR): the metric (`filseg/metric.py`).
- Ahmadzadeh et al. 2024, MAGFiLO (Scientific Data): the data; annotator kappa 0.47 (0.66 excluding Unidentifiable).
- Ahmadzadeh et al. 2019, Toward filament segmentation using deep neural networks (IEEE BigData).
- Reche & Cid 2026 (Acta Astronautica 249, doi:10.1016/j.actaastro.2026.07.010): 70-degree annotation limit, crop segmenter with a box channel, limb-darkening correction. Led to the crop refiner (fold 0: crop Dice 0.829, lifted the epoch-10 U-Net val PQ 0.389 to 0.402, no gain on the 40-epoch U-Net 0.415 to 0.413), the limb filter (no change) and limb-darkening input (0.4112 vs 0.4160, no gain).
- Guo et al. 2022 (Solar Physics 297:104): CondInst instance segmentation against fragmentation. Idea kept, code not used (2022 detectron2/AdelaiDet fork, non-commercial licence, hard to build on MPS).
- Liu et al. 2021 (Solar Physics 296:176): ASPP U-Net best on good-quality images.
- Zhu et al. 2025 (ApJ 980:176): Flat U-Net, under 1 MB; supports a small model for the efficiency criterion.
- Shang et al. 2024 (Applied Sciences 14:3745): CNN-Transformer; masks outside the disk.
- Hu et al. 2026 (arXiv 2607.24525): MORDEN; Instance Norm, gamma/zoom augmentation, DBSCAN grouping. Architecture too slow to train here; DenseCRF costs about 6.5 s per 2K image and fits image edges rather than annotators' polygons.
- Chalmers and Ahmadzadeh 2025 (arXiv 2509.18214): chirality classification on MAGFiLO; not used for segmentation.
- 2026-11-09 literature agent: nothing filament-specific changes the plan. Ideas: predicted-IoU head or scorer refit on refined-mask features (Mask Scoring R-CNN, Huang et al. 2019), refining more tree nodes, an ordinal annotator-agreement target, a boundary-band loss, per-instance expected-IoU extents (RankSEG).
- Not used for licence reasons: Ultralytics YOLO-seg (AGPL-3.0); Mask R-CNN from torchvision (BSD) was tested (fold 0 val PQ 0.377, below the U-Net; as extra filaments +0.002 on the 40-epoch U-Net; `rejected-*`).

Decisions: small efficient U-Net family with soft consensus target; no CRF, no transformer; the rest of the
ideas in `research-ideas-list`.

## Non-goals

No external data or labels from these papers' releases (public MAGFiLO, SWEFil, HSOS).

## Falsified by

n/a (a reading record); the adopted ideas were each tested out of fold in their own change.

## Cost

Reading only.

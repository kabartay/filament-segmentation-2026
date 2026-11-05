# Design and decision log

Kaggle Filament Segmentation Challenge 2026 (MAGFiLO, GONG H-alpha, 2048 x 2048).
Deadline 2026-11-15 23:59 CDT. This file records what we built, why, and what each
experiment showed. Numbers are on our own validation folds unless marked "LB".

## 1. What is scored

- Panoptic Quality, one class, pooled over every annotator-image entry:
  PQ = sum(IoU of TP) / (TP + 0.5 FP + 0.5 FN), TP when IoU > 0.5 (strict).
- 296 of 707 training images have 2-3 annotators; each annotator's set is scored
  separately against the same prediction.
- Final evaluation adds Dice and IoU distributions and 1:n / n:1 counts (all at
  IoU > 0), plus 30% qualitative: pipeline description, morphology, code quality.
- `filseg/metric.py` reproduces the organisers' self-evaluation notebook exactly
  (0.314922 from both on 15 check pairs).

## 2. Facts measured on the training data

| Fact | Value | Consequence |
|---|---|---|
| Inter-annotator PQ (one annotator scored against another, 296 images) | 0.3399 (SQ 0.634) | Train on the consensus, not on one annotator |
| Own union mask split into 8-connected components | PQ 0.9972 | Annotated filaments almost never touch; a good semantic mask plus components loses almost nothing |
| Annotations reaching beyond 70 deg from the central meridian | 2.0% (0.6% beyond 80 deg) | Predictions near the east/west limb are mostly FP |
| Disk centre / radius at 2048 px (Hough fit to the limb, all 887 images) | centre 1010-1034 px; radius 889-912 px (the annual change of the apparent solar radius is about +-1.7%) | The earlier bright-pixel fit (906-1010 px) included the scattered-light halo; see the decision log |
| Instances per image (median) | 7 | |
| Instance area, 5th-95th percentile | 316 to 6963 px | Sets the range for the minimum-area filter |

## 3. Pipeline (current, submission 6 candidate, 2026-11-05)

1. **Cache** (`filseg/cache.py`): images area-downsampled to 1536 px; target = fraction
   of annotators marking each pixel, computed at 2048 then downsampled.
2. **Model** (`filseg/train.py`): `smp.Unet` with ImageNet ResNet-34, 1 input channel,
   BCE + soft Dice on the soft target, AdamW 5e-4 with warm-up and cosine decay,
   batch 4, 40 epochs, flips, 90-degree rotations, +-30 deg affine, brightness/gamma.
   Final model on all 707 images (`weights/unet_r34_1536_all.pt`).
3. **Folds**: 5-fold `GroupKFold` grouped by calendar month (filaments persist for days,
   so same-week images would leak).
4. **Inference** (`filseg.train.predict_with_spread`): four flipped views; mean = probability map
   (identical to `predict(views=4)`), per-pixel std = view spread; both upsampled to 2048.
5. **Candidate tree** (`filseg/tree.py`): components at 12 thresholds 0.25-0.80 (closing 31 px for
   grouping only, >= 100 px), nested into a forest; 32 features per node (the 20 of `filseg/scorer.py`,
   level, tree context, view spread).
6. **Selection**: hurdle scorer P(match) x E[IoU | match] (`weights/tree_scorer_1536.pkl`, fitted on the
   72,712 out-of-fold nodes, rows weighted by annotator count); exact antichain maximising sum(score - 0.26).
   Disjoint by construction (checked on the submission: 0 overlapping pairs).
7. **Submission** (`filseg/predict.py --tree-scorer`): writes and validates the CSV (unique ids, known
   stems, non-empty 2048 masks), records runtime per image. `pipeline.ipynb` reproduces it.

Out of fold: PQ 0.4267 [0.417, 0.436]; +0.0049 [0.0012, 0.0085] over submission 5 (single threshold +
scorer, 0.4219) and +0.0182 [0.0134, 0.0233] over submission 4 (1024 px hand rule, 0.4086); per fold
0.434 / 0.425 / 0.411 / 0.430 / 0.434 (mean 0.427 +- 0.010). Multi-annotator images 0.4275 (SQ 0.672,
RQ 0.636). Mean IoU 0.575, mean Dice 0.708; merges 3.1%, splits 3.8%, predictions overlapping nothing
10.6%, annotations overlapping nothing 22.7%. Test submission `submissions/sub6_r34_1536_tree_spread.csv`:
1,067 filaments on 177 of 180 images, public LB 0.36 (2026-11-05; sub4/sub5 0.35), 8.6 s per image on the M3 Max (shared, loaded machine). View spread on
MPS vs the HF (CUDA) OOF maps: max per-pixel difference 0.008, mean spread about 0.0002-0.0004.

## 4. Plan

| Step | What | Status |
|---|---|---|
| 1 | U-Net baseline, fold 0, tuned post-processing | done: val PQ 0.389 (epoch 10), 0.396 (epoch 20), 0.415 (full 40 epochs, run unet_r34full_f0); stopped by mistake at epoch 22 of 40 while still improving (default-setting val PQ: e5 0.308, e10 0.357, e15 0.355, e20 0.379); best.pt is e20; re-run the full schedule later |
| 2 | First submission, to calibrate local PQ against LB | done: LB 0.32 public (local 0.389); submission 2 (40-epoch U-Net) LB 0.35 public (local 0.415); submission 3 (5-fold average) 0.35; submission 4 (all data) 0.35; empty-mask probe 0.00; submission 5 (1536 px, 4 flips, scorer; OOF 0.422) 0.35 |
| 3 | Per-filament crop refiner at native 2048 px (crop 3x the box, second channel marking the target box, 256 px input; after Reche & Cid 2026 and the nomannic19 notebook) | done, fold 0: crop Dice 0.829 (15 epochs, about 1.6 min/epoch); lifts U-Net val PQ 0.389 to 0.402 at threshold 0.6; no gain on the 40-epoch U-Net (0.415 to 0.413), not used |
| 4 | Mask R-CNN detector (torchvision, BSD) for learned instance grouping | done, fold 0, 12 epochs (SGD lr 0.005, batch 2, about 6 min/epoch): val PQ 0.377 with disjoint masks, below the U-Net; as a source of extra filaments it lifts the epoch-10 U-Net to 0.399 but the 40-epoch U-Net only 0.415 to 0.417; not used |
| 5 | Model variants, fold 0, 40 epochs each, one at a time: (a) ResNet-50 encoder; (b) limb-darkening correction (`filseg/flatten.py`); later stronger gamma/zoom augmentation, DeepLabV3+ (ASPP) decoder | (a) done: tuned 0.4139 vs 0.4160 for ResNet-34, and averaging both gives 0.4159, so no gain; (b) done: tuned 0.4112 vs 0.4160, no gain; (c) Lovasz hinge loss: tuned 0.3901, failed |
| 5d | 1536 px input, fold 0 (`scripts/run_1536.sh`, separate cache via `FILSEG_CACHE=data/cache1536`) | tuned 1536 vs 1024: f0 0.4245/0.4153, f1 0.4070/0.4091, f3 0.4192/0.4140, f4 0.4263/0.4190; fold 2 on MPS; all-data 1536 model done (HF Jobs); 2048 px folds 0 and 3 next |
| 5e | Second-stage component scorer and 4-flip TTA | scorer: OOF 0.4162 vs 0.4086, wins every fold; 4 flips: fold 0 0.4192 vs 0.4149; 8 views no better; combination pending (`scripts/after_1536.sh`) |
| 5c | Post-processing on the saved 5-fold OOF probabilities (CPU only): size-dependent confidence, hysteresis, growing/shrinking masks | all failed: size-dependent confidence, hysteresis, growing/shrinking masks (see log); limb filter re-tuned with the Hough disk: no change |
| 5b | 5-fold U-Net ensemble: folds 1-4 at 40 epochs (`scripts/train_folds.sh 1 2 3 4`), then tune post-processing on all 707 out-of-fold images (`filseg.tune --run unet_r34full_f0 ... f4`) | done: default-setting best val PQ f0 0.393, f1 0.389, f2 0.375, f3 0.376, f4 0.400; pooled 5-fold OOF PQ 0.410 (best setting) / 0.409 (fold-0 setting); ensemble CSV `submissions/sub3_unet_r34_5fold.csv` (1,224 rows, 1.17 s/image), not uploaded yet |
| 6 | Final: train on all folds, report figures (organisers' three plots: `filseg/report.py`), notebook, repo | 2026-11-04: final model in `weights/unet_r34_all.pt`; OOF figures `docs/figures/oof5_*` and `qualitative.png`; `pipeline.ipynb` (inference from raw images + pre-computed validation + training commands); `README.md`, `.gitignore`; report draft `report/report.tex` (IEEEtran; to move into the organisers' Overleaf template). 2026-11-05: author e-mail filled (no affiliation), Reche & Cid title found on Crossref (doi:10.1016/j.actaastro.2026.07.010), GONG acknowledgment and competition citation added, MIT licence; compiled with Tectonic to 4 pages (half of page 4 free). Open: move into the organisers' Overleaf template (needs an Overleaf login), Google form, repo public at the close |

## 5. Decision log

Each entry: what we tried, the result, and what we decided.

- **Public "LB 1st" notebooks (rokaiyasomapti, foysalemonshanto)**: trained on empty masks
  (lookup bug) and encode RLE at 512 px. Not a usable baseline. Their scores come from
  a known empty-prediction defect in the public leaderboard ([public notebook B] notebook).
- **"Unet Segmentation 0.55+" notebook ([public notebook A])**: not a model. It trains a 5-epoch
  throwaway U-Net, then submits a compressed CSV embedded in the code ("CHAMPION_PAYLOAD").
  The cluster of 0.55-0.56 public scores is most likely copies of that file. Not used, not
  decoded; such entries cannot pass the final reproducibility review.
- **Old Dice evaluator (before the 12 Aug rescoring)**: empty masks on 5 images scored 1.00
  (artkomissar disclosure). Our submissions always contain real, non-empty masks.
- **External MAGFiLO release and SWEFil**: not used. They may contain test-image labels, and
  the rules forbid other ground truth.
- **Spines and chirality labels**: not used for training (rules: no other ground-truth
  metadata). Chirality classes are collapsed into one filament class.
- **Ultralytics YOLO-seg**: not used (AGPL-3.0, no reported scores in the public notebook).
- **CondInst-SFDS (Guo et al. 2022)**: idea kept (instance model to stop fragmentation), code
  not used (2022 detectron2/AdelaiDet fork, non-commercial licence, hard to build on MPS).
- **Post-processing tuning, epoch-5 fold-0 probabilities (2026-11-03)**: defaults (0.5, no
  closing, area 200) PQ 0.308; area 400 and threshold 0.65 alone 0.338; best 0.355 with
  threshold 0.45, closing 31 px, area 400, mean probability >= 0.8. Closing is the largest
  gain (0 / 15 / 31 px: 0.345 / 0.351 / 0.355) and 31 was the grid edge, so the grid now
  goes to 75 px. Confidence adds +0.003. The 70-degree limb filter changed nothing (the
  model already ignores the limb); kept in the code, off by default.
- **Submission 1 (2026-11-03)**: U-Net R34 fold-0 epoch 10, threshold 0.45, closing 31,
  area 400, confidence 0.8. Local val PQ 0.389, public LB 0.32. 1,177 filaments on 175 of 180
  test images, 1.23 s per image on MPS. The local-to-LB gap (about 0.07) matches the two other
  U-Net entries, so local CV is a usable guide. Decision (partly mistaken: the U-Net was still improving at e20): the plain U-Net looks to level off around
  0.32-0.35 LB; move to the Mask R-CNN detector.
- **Overlapping predictions (2026-11-03)**: the organisers' scorer counts every pair with
  IoU > 0.5 as a TP and never enforces one-to-one matching, so two overlapping predictions
  of one filament score two TPs. Our raw detector scored 0.381 that way (TP + FN exceeded
  the number of annotations). Decision: every submission is disjoint (pixels painted in
  score order), and `PQCounts.summary()` flags any annotation hit more than once.
- **Mask R-CNN detector, fold 0 (2026-11-03)**: val PQ at score >= 0.7 by epoch: e2 0.337,
  e6 0.350, e10 0.366, e12 0.366 (merged overlaps). With disjoint masks (score order, drop a
  mask left with < 50% of its area): best 0.377 at score >= 0.75 (SQ 0.642, TP 944, FP 614,
  FN 711). The U-Net alone is 0.389 (SQ 0.657). Combinations on the same images: U-Net plus
  detector filaments touching no U-Net component 0.398 (score >= 0.75) / 0.399 (>= 0.85);
  detector grouping with U-Net pixels 0.396; splitting U-Net components by detector instances
  0.388; detector grouping plus leftover U-Net components 0.383-0.388. Decision: the detector
  is a weak model on its own here (one fold, 12 epochs); keep only the "extra filaments" use
  (+0.01, inside fold noise until confirmed on a second fold).
- **Crop refiner on U-Net proposals, fold 0 (2026-11-03)**: proposals from submission 1
  (val PQ 0.389, SQ 0.657). Refined, by mask threshold: 0.4 0.395, 0.5 0.399, 0.6 0.402
  (SQ 0.663, TP 954, FP 539, FN 701), 0.7 0.399, 0.8 0.390, 0.9 0.372. The gain comes both
  from sharper masks (SQ +0.006) and from 22 proposals moving above IoU 0.5. Decision: keep
  the refiner, threshold 0.6; next test it on the epoch-40 U-Net and the detector extras.
- **Pooled tuning, folds 0-2 (2026-11-03, 425 images)**: best threshold 0.45, closing 31,
  area 200, confidence 0.85: val PQ 0.406; the fold-0 choice (closing 31, confidence 0.8,
  area 400) 0.404; closing 51 0.401; no closing, confidence or limb filter 0.397. The
  fold-0 setting carries over to the other months within 0.002. Fold 3 was cut at epoch 6
  when the session ended; folds 3 and 4 restarted 22:21, detached (`nohup`).
- **U-Net, full 40 epochs, fold 0 (2026-11-03)**: default-setting val PQ by epoch 5-40:
  0.331, 0.359, 0.366, 0.372, 0.382, 0.391, 0.392, 0.393. Tuned 0.415 (threshold 0.45,
  closing 31, area 200, confidence 0.85). The confidence filter has a cliff just above:
  0.88 gives 0.403, 0.90 gives 0.366. Chosen: area 300, confidence 0.8, val PQ 0.415
  (TP 1030, FP 625, FN 625), kept back from that edge. On this model the refiner no longer
  helps (0.413 at threshold 0.6, 0.409 at 0.7) and detector extras add 0.002 (0.417 at score
  >= 0.85), inside fold noise. Decision: the plain U-Net with tuned post-processing is the
  pipeline; the refiner and detector stay in the repo as tested and not used. Submission 2
  candidate: `submissions/sub2_unet_r34_f0e40.csv`, 1,313 filaments on 177 of 180 test images,
  0.20 s per image on MPS (flip TTA, no refiner). Uploaded: public LB 0.35 (from 0.32); the
  local-to-LB gap held at about 0.065.
- **U-Net epoch 20 (2026-11-03)**: tuned val PQ 0.396 (threshold 0.45, no closing, area 400,
  confidence 0.8; 0.391 without the confidence filter). Closing no longer helps: the longer
  trained model already joins fragments.
- **MORDEN (Hu et al. 2026), DenseCRF**: architecture too slow to train here; CRF costs
  about 6.5 s per 2K image and fits image edges rather than annotators' polygons.

- **Pooled tuning, all 5 folds (2026-11-04, 707 images)**: best 0.41034 at (0.45, 31, 200, 0.85, limb 70);
  the fold-0 setting (0.45, 31, 300, 0.8, 90) gives 0.4086. Closing 51 0.4077; no closing, confidence
  or limb filter 0.3977. Submission 3 keeps the fold-0 setting: averaged probabilities are lower, and
  in the two-model test below confidence 0.8 beat 0.85.
- **Where PQ is lost (2026-11-04, 5-fold OOF, setting 0.45/31/300/0.8)**: PQ 0.4086, TP 5025, FP 3155,
  FN 3174. Of the FN, 45% touch no prediction, 18% overlap one at IoU < 0.3 and 37% at IoU 0.3-0.5
  (near misses: wrong extent, split or merge). Recall by GT area: < 500 px 0.36, 500-1000 0.61,
  1000-2000 0.68, 2000-5000 0.70, > 5000 0.64; small filaments are 812 of the FN. Of the FP, 28% match
  another annotator's filament (above IoU 0.5) on the same image, 39% touch no annotation, 37% are
  near misses. Turning every near-miss FN into a TP at IoU 0.55 would give at most PQ 0.487.
- **Annotator ceiling (2026-11-04)**: on the 296 train images with 2-3 annotators, each annotator's
  masks scored as predictions against each other annotator give PQ 0.340 (SQ 0.634). The U-Net scores
  0.405 on the same images (0.414 on 1-annotator images, 0.414 on 2-annotator, 0.400 on 3-annotator).
  The model already beats a single annotator because it learns the consensus; most of the remaining
  loss is annotator disagreement.
- **Resolution (2026-11-04)**: a GT mask taken down to 1024 px and back up keeps mean IoU 0.975
  (1536 px: 0.996; 400 sampled masks). Training at 1024 px costs little SQ; not a priority.
- **Two-model average, fold 0 (2026-11-04)**: the 40-epoch model (best 0.4160) averaged with the
  epoch-20 model (best 0.3954) gives best 0.4099, between the two. Averaging with a weaker model does
  not help; the gain from five equal-strength fold models cannot be measured out of fold and will be
  read from the LB (submission 3 against submission 2, LB 0.35).

- **Submission 3 (2026-11-04)**: average of the five fold models, flip TTA, setting
  (0.45, 31, 300, 0.8, 90): 1,224 rows, 1.17 s/image. Public LB 0.35, the same as the single
  fold-0 model (submission 2). The ensemble brings no visible LB gain at two decimals.
- **Post-processing variants on 5-fold OOF (2026-11-04, `scripts/experiments/postprocess_variants.py`), all failed**:
  reference (0.45, 31, area 300, conf 0.85) 0.4112. Size-dependent confidence (0.85 below an area
  split of 800/1500/3000 px, 0.7-0.8 above, min area 150-300): best 0.4114, noise. Hysteresis (mask at
  0.30/0.35/0.40, kept only if its maximum reaches 0.85-0.95, with or without the mean filter): no
  setting beats a single threshold of 0.45 (0.4079 at 0.40). Lowering the minimum area to 150-200
  does not help: the extra small components cost more FP than they add TP.
- **Near misses, what they are (2026-11-04)**: of the 1,169 FN whose best prediction has IoU 0.3-0.5,
  the prediction is smaller than the GT (area ratio < 0.67) in 46%, about the same size in 20% and
  larger (> 1.5) in 34%. Only 17% of those predictions also touch another GT (a merge) and 16% of
  those GT are touched by several predictions (a split). So the loss is mostly extent (too short or
  too thin, or overgrown), not grouping; closing/splitting rules cannot fix it.
- **Small filaments (2026-11-04)**: the share of the other annotators who mark the same filament
  (IoU > 0.5) falls with size: < 300 px 0.28 (65% marked by nobody else), 300-500 px 0.39,
  500-1000 px 0.49, > 1000 px about 0.60. Of 1,263 small GT (< 500 px) in the OOF set: 451 TP, 400 lie
  inside a larger or wrongly shaped prediction, 295 get probability below 0.45 (31% below 0.2), 100 are
  found but removed by the area filter, 17 by the confidence filter. `docs/figures/missed_small.png`
  shows missed small filaments with low probability: mostly faint low-contrast darkenings.
  Conclusion: small filaments are mostly annotator-specific; little to gain.
- **When does a doubtful prediction pay? (2026-11-04)**: a prediction matching k of the n annotators of
  an image adds about k x IoU to the numerator and 0.5 n to the denominator (k TP replace k FN at
  half weight, n - k FP), so it pays when k/n exceeds about PQ / (2 x SQ) = 0.41 / 1.33 = 0.31.
  A filament marked by one annotator of three is break-even. The 28% of FP that match another
  annotator's filament are therefore roughly neutral, not errors to fix.
- **The LB 0.70 entry (2026-11-04, refs/lb070)**: Mask R-CNN + crop U-Net refiner (after nomannic19)
  with flip-4 TTA on the refiner; it merges positive overlaps, so its masks are disjoint and the
  double-counting hypothesis is wrong. The notebook is dated 5 August 2026 and says the public LB
  then had an empty-prediction defect; refs/please-fix-the-leaderboard reports that the same LB gave
  1.00 for five empty masks, 0.88 for one empty mask per image and 0.70 for seven empty masks per
  image. The top LB score today is 0.70 (submitted 2026-11-03), most likely a rerun of that public
  notebook. Our detector scored 0.377 and the annotator ceiling is 0.34, so a true PQ of 0.70 is not
  plausible. Open question: does the current LB still use the defective scorer? A one-submission probe
  (one empty mask per image: about 0.88 means defective, about 0 means PQ) would settle it; it awaits
  the owner's approval. The final evaluation uses the organisers' PQ notebook, which `filseg/metric.py`
  reproduces, so local OOF PQ stays the target.
- **Disk fit (2026-11-04)**: the bright-pixel fit (`fit_disk_bright`, pixels above 0.3 of the 99th
  percentile) includes the scattered-light halo outside the limb and overestimated the radius by up to
  14% (516 vs 452 px at 1024 on train image 613). Replaced by a Hough circle on the limb at 512 px
  (radius 0.40-0.47 of the width): found on all 887 images, radius 889-912 px at 2048, 0.1 s/image,
  checked by eye on 12 random images. All earlier limb-filter results used the old fit; the setting
  needs re-tuning with the new one.
- **Growing or shrinking the masks (2026-11-04, `scripts/experiments/grow_shrink_masks.py`), failed**:
  reference 0.4112. Eroding by 1 px 0.3986, by 2 px 0.2841; dilating by 1 px 0.3811, 2 px 0.3112,
  3 px 0.2438. Growing each mask into nearby pixels above 0.35 / 0.30 / 0.20: 0.4074 / 0.4038 / 0.3877.
  The 0.45 threshold already sits at the best average extent. Filaments are only a few pixels wide at
  2048 px, so one pixel of boundary moves IoU a lot; the extent errors are per-instance, not a bias.
  Sharper boundaries need a better model (higher input resolution or a boundary loss), not a rule.
- **Limb filter with the Hough disk (2026-11-04, 5-fold OOF)**: the full grid re-run with the corrected
  disk gives the same numbers as with the old fit (best 0.41032; the 70-degree cut changes no TP and at
  most one FP). The model has already learned that annotators skip the limb, so the filter is
  redundant either way; it stays at 90 (off) for submissions. Old grid kept in
  `runs/tune_unet_r34full_f0+4/tune_brightfit.json`.
- **ResNet-50 encoder, fold 0 (2026-11-04, `runs/unet_r50_f0`), no gain**: 40 epochs at about 210 s/epoch
  (1.7x ResNet-34). Default-setting val PQ e5 0.338, e10 0.341, e30 0.383, e35 0.392, e40 0.393 (best
  0.3934, ResNet-34 0.393). Tuned: 0.4139 at (0.45, 31, 300, 0.85), ResNet-34 0.4160.
- **Two equal-strength models averaged, fold 0 (2026-11-04), no gain**: ResNet-34 (0.4160) + ResNet-50
  (0.4139) gives best 0.4159 (0.5, 31, 300, 0.85); SQ rises slightly (0.6693 vs 0.6680) but TP falls.
  With the weaker-model test and the unchanged LB of submission 3, this settles it: the two encoders make
  the same mistakes, because the mistakes come from the annotators, so averaging does not pay. The
  final model will be a single network (trained on all images), not an ensemble.
- **Limb-darkening correction as input, fold 0 (2026-11-04, `runs/unet_r34flat_f0`), no gain**: each cached
  image divided by its per-ring median around the Hough disk, quiet Sun mapped to 128, off-disk (the
  halo) set to 0 (`filseg/flatten.py`, `--images flat`). About 110 s/epoch. Default-setting val PQ
  e5 0.260, e10 0.335, e20 0.362, e25 0.381, e30 0.387, e35 0.3956 (best), e40 0.3939; plain input best
  0.3926. Tuned: 0.4112 at (0.65, 31, 400, 0.85), plain 0.4160. The best threshold moves from 0.45 to
  0.65 and the confidence filter stops mattering (0.4111 without it): the model is more confident, not
  more accurate. The network already normalises brightness by itself.
  Averaged with the plain-input model: best 0.4140 at threshold 0.55, below the plain model alone.
- **Lovasz hinge loss, fold 0 (2026-11-04, `runs/unet_r34lovasz_f0`)**: BCE on the soft target plus
  Lovasz hinge (per image) on the majority mask (`--loss bce_lovasz`), aimed at the extent errors.
  About 113 s/epoch. Default-setting val PQ e5 0.187, e10 0.147, e15 0.274, e20 0.311, e25 0.306,
  e30 0.319, e35 0.364, e40 0.380 (plain loss 0.393); tuned 0.3901 at (0.55, 51, 400, no confidence
  filter), plain loss 0.4160. SQ is lower (0.651-0.656 vs 0.668), so it does not even help the extent. Failed.
- **Where the ceiling is (2026-11-04)**: annotator against annotator gives SQ 0.634 and RQ = PQ / SQ =
  0.536; the U-Net gets SQ 0.667 and RQ 0.616 on all OOF images. The model is already above a single
  annotator on both boundary quality and detection, and four model changes (ResNet-50, flat input,
  Lovasz, averaging) moved fold-0 PQ by -0.026 to +0.000. Further model work is fighting annotator noise;
  the remaining time goes to the final model (all 707 images) and the deliverables.
- **Submission 4 and the LB probe (2026-11-04)**: the all-data model (`runs/unet_r34full_f-1/last.pt`,
  setting 0.45/31/300/0.8, 1,306 rows) scores public LB 0.35, the same as submissions 2 and 3. With the
  owner's approval a diagnostic file with one empty mask per test image was uploaded: LB 0.00, so the
  empty-prediction defect is fixed and the LB now computes PQ as documented.
- **Competition pages (2026-11-04, `refs/competition_pages/pages.md`, fetched with the Kaggle API)**:
  the LB was re-scored with PQ on 12 Aug 2026. Organisers' announcements: 20 Aug, "some participants are
  spending their time and energy on hacking the system ... Any PQ score of greater than 0.35 is of great
  value to us"; 23 Oct, "After reviewing top 250 teams' submissions, we remain confident that a PQ greater
  than [0.35] has significant merit for us". So the organisers themselves treat about 0.35 as the honest
  level and the 0.55-0.70 LB scores as suspect. Our 0.35 sits on that line. The likely source of the
  high scores, not verified: the public MAGFiLO release contains the test images' labels, which the
  rules forbid using (the lb070 notebook mentions "public MAGFiLO overlap data" and "contamination-audit
  assets"). We do not chase those scores. The LB is also only part of the 70% quantitative criterion,
  next to the Dice/IoU distributions and the 1:n / n:1 counts.
- **Duplicate-mask inflation, simulated (2026-11-04, `scripts/experiments/duplicate_exploit.py`; analysis only, never
  submitted)**: the scorer has no one-to-one matching, so k copies of a mask that matches a GT count as k TPs.
  Duplicating our kept OOF masks with confidence >= c, k times: c 0.8: k=2 0.452, k=20 0.501, k=100 0.506;
  c 0.9: k=2 0.463, k=20 0.557, k=100 0.569; c 0.95: k=100 0.611; honest 0.409. PQ tends to the SQ of the
  duplicated masks (about 0.67-0.70), which matches the 0.55-0.70 cluster at the top of the public LB with no
  better model. Our submissions stay disjoint: the organisers warned against hacking (20 Aug), review the top
  submissions, and duplicates would also wreck the 1:n counts of the final evaluation.
- **Notebook reproducibility (2026-11-04)**: `pipeline.ipynb` (generated by `scripts/make_notebook.py`), run on CPU from
  the raw test JPEGs and `weights/unet_r34_all.pt`, writes a CSV byte-identical to submission 4, which was made on MPS
  from the cache (1,306 rows; 8.1 s/image on CPU while a training job shared the machine).
- **Annotator ceiling, re-measured (2026-11-04)**: the inter-annotator PQ in section 2 (0.340) holds;
  the U-Net gets 0.405 on the same 296 images, 0.414 on single-annotator images.
- **Second-stage component scorer (2026-11-04, `scripts/experiments/component_features.py`, `component_scorer.py`)**:
  instead of the hand rule (area >= 300, mean probability >= 0.8), a gradient-boosted regressor (scikit-learn
  HistGradientBoosting) predicts each candidate component's expected PQ contribution, y = sum over entries of
  IoU * [IoU > 0.5] / number of entries, from 20 shape, probability, contrast and disk-position features.
  Candidates: threshold 0.45, closing 31, area >= 100 (6,161 OOF components). Trained on four folds, applied to the
  fifth; keep components with predicted y > tau. Pooled OOF PQ by tau: 0.12 0.4116, 0.15 0.4144, 0.20 0.4159,
  **0.22 0.4162**, 0.25 0.4135, 0.32 0.4031 (tau is picked on the same pooled OOF, as the rule's thresholds were; the curve is flat over 0.18-0.22, so the selection bias is small).
  Rule 0.4086 (submission setting) / 0.4112 (best rule). Per fold, rule vs scorer: f0 0.4149/0.4175, f1 0.4087/0.4180,
  f2 0.3942/0.4042, f3 0.4087/0.4134, f4 0.4171/0.4288, so it wins on every fold. Mostly it drops false positives
  (3,155 to 2,289) at the cost of 724 more FNs. First approach change that helps.
- **More TTA views (2026-11-04, `scripts/experiments/tta_views.py`, fold 0, submission setting)**: identity+hflip
  (current) 0.4149; 4 flips (identity, h, v, both) **0.4192**, at the same setting, so no re-tune is needed.
  8 dihedral views 0.4187 (best re-tuned 0.4193), so no gain over 4 flips at twice the cost. 4 flips are now
  `predict(..., views=4)` and `filseg.predict --views 4`.
- **Scorer on 4-flip probabilities, 5-fold OOF (2026-11-04, `scripts/experiments/oof_probs_views.py`, then
  component_features/scorer on `val_probs_v4.npy`)**: rule (0.45, 31, 300, 0.8) 0.4117 (vs 0.4086 with 2 views);
  scorer tau 0.22 **0.4168** (vs 0.4162 with 2 views). Per fold, rule vs scorer: f0 0.4192/0.4215, f1 0.4078/0.4137,
  f2 0.3997/0.4067, f3 0.4123/0.4169, f4 0.4201/0.4254. The two gains mostly overlap (both remove false positives),
  so together they give +0.008 over the current pipeline, not +0.011. The re-predicted 2-view check matched the
  saved probabilities exactly (max difference 0.0000).
- **1536 px input, fold 0 (2026-11-04, run `unet_r34_1536_f0`, 40 epochs, about 270 s/epoch)**: default-setting best
  val PQ 0.4029 vs 0.3926 at 1024 px; tuned (same tuner, per run) 0.4245 (threshold 0.45, closing 31, area 200,
  confidence 0.85) vs 0.4153 (an earlier note said 0.4213: that was a lower line of the tune output, not the best).
  Fold 1 (run `unet_r34_1536_f1`): default-setting best 0.3852 vs 0.3893, tuned 0.4070 vs 0.4091, so no gain there.
  Folds 3 and 4 (trained on HF Jobs A100s): tuned f3 0.4192 vs 0.4140, f4 0.4263 vs 0.4190. Four folds: +0.009,
  -0.002, +0.005, +0.007, mean +0.005, better on three of four. Fold 2 (MPS): tuned 0.4059 vs 0.3965. All five folds: +0.009, -0.002,
  +0.009, +0.005, +0.007, mean +0.006, better on four of five, so 1536 px is adopted; the all-data 1536 model is `runs/unet_r34_1536_f-1/last.pt` (HF Jobs, 40 epochs, 53 s/epoch).
- **Full 1536 px pipeline, 5-fold OOF (2026-11-05, `scripts/pipeline_1536.sh`)**: 1536 px folds, 4 flips, candidates and
  scorer as above. Rule (0.45, 31, 300, 0.8) 0.4165, rule with confidence 0.85 0.4196; scorer tau 0.15 0.4213,
  0.18 0.4220, 0.20 0.4219, 0.22 0.4219, 0.25 0.4208, so flat over 0.15-0.25; tau 0.20 is used. Per fold, rule vs scorer:
  f0 0.4238/0.4327, f1 0.4101/0.4166, f2 0.4030/0.4083, f3 0.4169/0.4190, f4 0.4297/0.4342 (scorer better on every
  fold). Against the submission-4 pipeline (1024 px, 2 views, rule; 0.4086) this is +0.013. Folds 3 and 4 were trained on
  CUDA and re-predicted on MPS: probabilities differ by up to 0.015 from the CUDA ones, device rounding only.
- **2048 px (native), folds 0 and 3 (2026-11-05, HF Jobs A100, 76 s/epoch, about 66 min and $2.75 per fold)**:
  default-setting best val PQ f0 0.4035 (1536: 0.4029), f3 0.3876 (1536: 0.3872). No gain over 1536 px, so the
  resolution gain stops at 1536 and no all-data 2048 model is trained. Tuned figures: `runs/tune_unet_r34_2048_f*.out`.
- **Submission 5 candidate (2026-11-05, `submissions/sub5_r34_1536_v4_scorer.csv`)**: all-data 1536 model
  (`runs/unet_r34_1536_f-1/last.pt`), 4 flips, scorer `weights/component_scorer_1536.pkl` (fitted on all 6,182 OOF
  candidates, tau 0.20); 1,096 rows (submission 4: 1,306, the scorer drops low-value components), 176 of 180 images,
  no overlapping masks; 6.4 s/image on MPS including features. 5-fold OOF of this pipeline 0.4220 vs 0.4086.
  The scorer is saved as a plain dict (a pickled class from `python -m filseg.scorer` resolved to `__main__` and
  failed to load).
  Uploaded 2026-11-05: **public LB 0.35**, the same as submissions 2-4 at the two decimals the LB shows. With about
  90 public images (per-fold PQ spread about 0.01 at 141 images) a +0.013 OOF gain cannot be resolved there; the
  choice rests on the 5-fold OOF, where the pipeline wins on every fold. OOF evaluation figures for it:
  `docs/figures/oof1536_*` (PQ 0.4219, SQ 0.674, mean IoU 0.573, mean Dice 0.707 over pairs with IoU > 0; GT
  degree {0: 1911, 1: 5981, 2: 267, 3: 38, 4: 2}, prediction degree {0: 847, 1: 6127, 2: 226, 3: 18, 4: 1}; against
  the 1024 px rule pipeline: fewer false positives, splits and merges, more missed GT).
- **HF Jobs for training (2026-11-04, `scripts/hf/`)**: code and an image cache go to a private HF dataset repo
  (`launch.py upload`, which refuses a public repo, since the competition data is not ours to share); `launch.py run`
  starts `job.sh` on a GPU flavour with a timeout; each job uploads its run folder, `launch.py fetch` pulls it.
  A100-large ($2.50/h): 43 s/epoch at 1536 px on 566 images vs about 270-300 s on the M3 Max, so a 40-epoch fold
  costs about 41 min, $1.70. An L40S request waited 25 min without hardware and was cancelled. Spend so far about
  $5.0 (folds 3, 4 and the all-data model). Next: 2048 px (native) folds 0 and 3, cache `data/cache2048`.
  The first resolution gain, consistent with the grow/shrink entry (one boundary pixel matters). One fold of 141
  images is too noisy to decide on, so folds 1-4 are training at 1536 (`NAME=unet_r34_1536 FILSEG_CACHE=data/cache1536
  scripts/train_folds.sh 1 2 3 4`, about 3 h per fold) for a 5-fold comparison and a 1536-based scorer.

- **Uncertainty of the final OOF PQ (2026-11-05, `scripts/experiments/bootstrap_pq.py`, runs/bootstrap_pq.json)**:
  bootstrap over images (all entries of an image resampled together, 2,000 resamples). Final pipeline 0.4219,
  95% interval [0.412, 0.431]; 1024 px hand rule (submission 4) 0.4086 [0.399, 0.418]; paired gain +0.0133
  [0.008, 0.018], no resample at or below zero. Per fold, final vs rule: 0.432/0.415, 0.415/0.409, 0.407/0.394,
  0.421/0.409, 0.435/0.417 (mean 0.422 +- 0.012 s.d.). Scaled to the ~90 public images the interval of one
  pipeline is about +-0.027, twice the gain, which is why the public LB shows 0.35 for both.
- **Report moved into the organisers' template (2026-11-05)**: `report/main.tex` (ACM sigconf, from the Overleaf
  link on the Overview page), black template text kept verbatim, pipeline diagram (TikZ), bootstrap intervals,
  final-pipeline example figure (`scripts/experiments/qualitative_final.py`; 50th/75th/90th percentile of image PQ
  among fold-0 images with three annotators and at least five filaments each). Four pages with acknowledgment
  and references; the limit excludes those. The IEEEtran draft `report/report.tex` is removed (in git history).

## 6. Reference points from other entrants

| Source | Method | Local PQ | LB |
|---|---|---|---|
| [public notebook B] (from nomannic19) | Mask R-CNN 1024 + crop U-Net refiner + flip TTA | not stated | 0.70 |
| Agnuxo1 repo | ResNet-34 U-Net, temporal split | 0.400 (3 seeds, 1024 px) | best 0.35 |
| FarnooshMemari repo | U-Net from scratch, 512 px tiles at 1024, soft targets, month split | 0.389 | 0.32 |
| jamesmills1975 (CAGE) | ResNet-34 U-Net + hull head | 0.402 (optimistic split) | not submitted |

Two independent U-Net entries score 0.39-0.40 locally and 0.32-0.35 on the LB, so a
plain semantic U-Net sits around 0.35 on the LB; the detector + refiner design is
the one with a public score near 0.70.

- **Component tree over thresholds (2026-11-05, `scripts/experiments/component_tree.py`, runs/oof_tree_1536_v4.pkl)**:
  candidates are the components at levels 0.30, 0.38, 0.45, 0.55, 0.65, 0.75 (closing 31 px, >= 100 px), which
  nest into a forest (each higher-level component lies inside one lower-level component; asserted). Same 20
  features plus the level and the area ratio to the parent; the scorer is cross-validated as before; the
  submitted set is the exact antichain maximising sum(y_hat - tau) (dynamic programming on the forest, so the
  masks stay disjoint). 707 images, 36,406 nodes (about 5 per level-0.45 candidate), build 34 min with 4 workers.
  Control: level 0.45 only reproduces the submitted pipeline (0.4219 at tau 0.20). Results (best tau of 0.16-0.28):
  plain regressor 0.4247 (tau 0.26), rows weighted by annotator count 0.4248 (0.24), hurdle P(match) x E[IoU |
  match] 0.4252 (0.24). With six tree-context features (n_children, child_area_frac, max_child_frac, n_siblings,
  stable_up, stable_down): weighted 0.4254 (0.24-0.26), hurdle 0.4260 (0.26), paired gain over the control
  +0.0041 [0.0006, 0.0076] (bootstrap over images, p(gain <= 0) = 0.012), per fold 0.429/0.421/0.412/0.430/0.438
  vs 0.432/0.415/0.407/0.421/0.435 (4 of 5 better). Gain is real but small and the best of about 30 settings
  carries some selection optimism; far below the public notebook's +0.08 because their baseline had no scorer.
  Next: view-spread (uncertainty) features from an HF inference job.

- **View-spread maps (2026-11-05, `scripts/experiments/oof_view_spread.py`, `scripts/hf/job_spread.sh`)**: per-pixel
  std of the four flipped views for every OOF image, from each fold's best.pt, as float16
  (runs/unet_r34_1536_f{k}/val_spread_v4.npy, 0.67 GB per fold). One HF Job on an A10G: 63 s per fold, about
  $0.25 including a failed first launch (the `hf` CLI takes one pattern per `--include`; with several values only
  the first is a pattern and code was not downloaded). Local check on CPU: the mean of the four views matches
  val_probs_v4.npy to 0.0002. Used as uncertainty features of tree nodes (spread_mean, spread_q90, spread_max,
  spread_rel).

- **12-level tree with view-spread features (2026-11-05, runs/oof_tree12s_1536_v4.pkl)**: levels 0.25-0.80 in steps
  of 0.05, 72,712 nodes, build 59 min (4 workers, shared machine), context features on, paired bootstrap against
  the submitted pipeline (0.4219). Without spread features (ablation `--drop spread`): hurdle best 0.4252 (tau
  0.26, +0.0034 [-0.0005, +0.0070]), so twelve levels add nothing over six (0.4260). With spread features
  (spread_mean, spread_q90, spread_max, spread_rel): hurdle 0.4265 / 0.4267 / 0.4269 at tau 0.24 / 0.26 / 0.28,
  gain +0.0046 / +0.0049 / +0.0051, 95% interval about [+0.001, +0.009], p(gain <= 0) 0.003-0.006; per fold at
  tau 0.26: 0.434 / 0.425 / 0.411 / 0.430 / 0.434 vs 0.432 / 0.415 / 0.407 / 0.421 / 0.435 (4 of 5 better).
  Weighted regressor with spread: 0.4245 (worse than hurdle). Decision: adopt the hurdle scorer + context +
  spread on the tree, tau 0.26 (middle of the plateau), as the next pipeline candidate; view spread is free at
  test time because inference already computes the four views.

### Public notebook survey (2026-11-05)

About 19 public notebooks read (none executed; notebooks pulled without saved outputs, so numbers are their
own markdown claims). The only one with a documented ablation table is tushhaaarrrr
"solar-filament-segmentation-2026-0-40" (v4): OOF host-exact pooled PQ 0.4597, public LB 0.39 (v3).

| Source | Method | Local PQ | LB |
|---|---|---|---|
| tushhaaarrrr 0-40 (v4) | U-Net R34 + U-Net ConvNeXt-T + UNet++ R34, 5 folds, D4; component tree over 15 thresholds, LightGBM hurdle scorer (P(match) x E[IoU given match]), exact antichain DP, tau 0.26; rotation-normalised crop refiner | 0.4597 OOF | 0.39 (v3) |
| tushhaaarrrr 0-37 | residual U-Net, 3 physical channels, components + GBM scorer | 0.4331 OOF | 0.37 |
| muhammadshoaibaltaf V11 | YOLO26x 1280 detect + UNet++ EffB4 crops | 0.4347 (15% hold-out, pooling unclear) | not stated |
| timgluhkikh | YOLO11s detect + crop U-Net R34 | about 0.409 | not stated |
| pandadev0069 | YOLO26s-seg at 2048, union target | not stated | 0.36 |
| sivapnlr80 | 5-fold Mask R-CNN, no refiner | not stated | 0.34 |
| vmerckle | no ML: dark residual percentile threshold, up to 120 components | none | claims 0.46 (unverified) |

Their reported gains (paired over 5 partitions; seed noise about +-0.002): component tree + hurdle + DP
about +0.065 over a fixed threshold (0.372 to about 0.450); stage-2 refiner +0.006 to +0.009; ConvNeXt-T
member +0.003, UNet++ member +0.002; RF-DETR evidence features +0.003. Reported failures: clDice/skeleton
loss -0.045, Lovasz -0.003, no Dice term -0.006, FPN decoder -0.008, DeepLabV3+ and MAnet -0.002, temporal TTA
-0.020, noisy-student pseudo-labels on external frames +0.003 (noise), per-annotator / leave-group-out
modelling +0.000, isotonic calibration -0.0005. Their OOF-to-LB gap is 0.05-0.06 on every pipeline
(0.4331/0.37, 0.4413/0.37, 0.4429/0.38, 0.4486/0.39), the same as ours (0.07); they find test frames
indistinguishable from training frames (adversarial AUC 0.496) and report that 5 of 24 annotation groups
are absent from training, so the gap is on the label side. Decision: try the component tree first (CPU,
our own implementation: `scripts/experiments/component_tree.py`), then a hurdle target and spread features;
model diversity only if GPU budget allows.

Annotator identity: image ids carry a code such as `010401` (37 codes; the last two digits are the slot
01-03 within an image, the first four look like an annotation batch). The test set has no such codes, so
per-annotator heads could only be averaged at test time; not pursued.

## 7. Literature used

- Kirillov et al. 2019, Panoptic Segmentation (CVPR): the metric.
- Ahmadzadeh et al. 2024, MAGFiLO (Scientific Data): the data; annotator kappa 0.47
  (0.66 excluding Unidentifiable).
- Ahmadzadeh et al. 2019, Toward filament segmentation using deep neural networks (IEEE BigData).
- Reche & Cid 2026 (Acta Astronautica 249): 70-degree annotation limit, crop segmenter with a
  box channel, limb-darkening correction.
- Guo et al. 2022 (Solar Physics 297:104): CondInst instance segmentation against fragmentation.
- Liu et al. 2021 (Solar Physics 296:176): ASPP U-Net best on good-quality images.
- Zhu et al. 2025 (ApJ 980:176): Flat U-Net, under 1 MB; supports a small model for the efficiency criterion.
- Shang et al. 2024 (Applied Sciences 14:3745): CNN-Transformer; masks outside the disk.
- Hu et al. 2026 (arXiv 2607.24525): MORDEN; Instance Norm, gamma/zoom augmentation, DBSCAN grouping.
- Chalmers and Ahmadzadeh 2025 (arXiv 2509.18214): chirality classification on MAGFiLO; not used for segmentation.

## 8. Ideas not yet tried

Ranked by expected value per hour, given that the remaining loss is mostly extent and annotator
disagreement:

1. ~~Re-tune the limb filter with the Hough disk~~: done, no change (the model already skips the limb).
2. ~~Limb-darkening correction as input~~: tried, no gain (0.4112 vs 0.4160).
3. ~~Heavier encoder~~: ResNet-50 tried, no gain alone or averaged.
4. ~~Boundary-aware loss~~: Lovasz hinge tried, worse (0.390); a boundary/distance term not tried. (Sampling one annotator's mask per step instead of the consensus is not worth a run: with
   BCE its expected gradient equals training on the consensus.)
5. **Per-instance threshold** (relative to the instance's own peak) for the extent problem; CPU test.
   Global growing/shrinking already failed, so expect little.
5b. **Higher input resolution** (1536 px, or 2048 px tiles): one pixel of boundary matters a lot (see the
   grow/shrink entry), so a sharper probability map may raise SQ even though a 1024 px round trip of the
   GT itself keeps IoU 0.975.
6. **Train on all 707 images** for the final model instead of the 5-fold average (no LB gain from the
   ensemble, 5x cheaper inference): done 2026-11-04 09:08 (`runs/unet_r34full_f-1/last.pt`, 40 epochs, 140 s/epoch, final loss 0.3436);
   CSV `submissions/sub4_unet_r34_alldata.csv` with the submission-2/3 setting (0.45, 31, 300, 0.8), 0.39 s/image;
   public LB 0.35. This is the final model unless something beats it.
7. **Learned grouping** (CondInst, Guo et al. 2022; DBSCAN on fragments, Hu et al. 2026): low priority,
   because merges and splits are only about a sixth of the near misses.

Tried and rejected so far: Mask R-CNN detector, crop refiner, detector + U-Net combinations, ensembling
with a weaker model, size-dependent confidence, hysteresis, a lower minimum area (all in section 5).

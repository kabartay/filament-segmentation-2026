## 1. Secrets and personal data

- [x] 1.1 Author and committer addresses. Verify: `git log --all --format='%ae|%ce' | sort -u`. Result: one
  address, mukharbek.organokov@gmail.com, on all 28 commits; no other address in any diff.
- [x] 1.2 Credentials. Verify: grep of `git log --all -p` for `hf_...`, `AKIA`, `sk-`, `ghp_`, `api_key`, `secret`,
  `password`, `BEGIN ... PRIVATE`. Result: none. Only matches are `$HF_TOKEN` references in `scripts/hf/job.sh`,
  `job_spread.sh`, `launch.py` (read from the environment, never a value) and base64 PNG data.
- [x] 1.3 Absolute local paths at HEAD. Verify: `grep -rE '/Users/|/home/' --exclude-dir=.git .`. Result: none.
- [x] 1.4 Absolute local paths in history (owner decision, low severity). Result: commits 6505dea, b0ff882, 5d1c5c7
  once tracked the symlinks `data` and `runs` whose targets are `...`,
  and older `pipeline.ipynb` versions carry outputs with the same prefix (test-image and submission paths, 2
  commits). The local user name `user` is exposed, nothing else. Fix if wanted: rewrite history before the
  flip (`git filter-repo`), or accept. Verify after: the history grep above returns nothing.
  Done 2026-11-10: history rewritten with git filter-repo (local user name and the symlink paths replaced), force-pushed; the history grep returns nothing.
- [x] 1.5 Other projects and companies. Verify: grep of history and HEAD for temelion, finkele, jile, kestrel, fintfm,
  bwa, aava, siemens. Result: none.
- [x] 1.6 Account names are public by design but note them: Kaggle `muhakabartay` (`scripts/kaggle/push.sh`,
  `cpu_probe/kernel-metadata.json`, DESIGN.md:459), Hugging Face `kabartay/filseg-2026` (`scripts/hf/run_2048.sh`).
  The private dataset and kernel are named. Owner: keep or drop `scripts/kaggle/` and `scripts/hf/` from the
  public tree (they are training-host helpers the judges do not need).
  Owner decision 2026-11-10: keep the own account names; scripts/README.md marks scripts/hf and scripts/kaggle as host helpers and the HF dataset as private.
- [x] 1.7 DESIGN.md names competitors' Kaggle handles and scores (lines about muhammadshoaibaltaf, timgluhkikh,
  pandadev0069, around 548-550) and says "the owner" in several places (213, 260, 489). Owner: confirm the
  naming of other participants is acceptable in a public file; they are public notebooks, read as text only.
  Owner decision 2026-11-10: suspected exploits are anonymised in DESIGN.md ("[public notebook B]", "[top team]"); handles of public notebooks credited as reference points are kept.

## 2. Licensing and data

- [x] 2.1 LICENSE present: MIT, "Copyright (c) 2026 Mukharbek Organokov". The owner decides; not changed.
- [x] 2.2 Dependency licences. Verify: `importlib.metadata` in the project venv. Result: torch Apache-2.0/BSD,
  torchvision BSD, segmentation-models-pytorch MIT, timm Apache-2.0, numpy BSD-3, opencv-python-headless Apache-2.0
  (bundles FFmpeg under LGPL in some wheels; headless wheel used), scikit-learn BSD-3, albumentations MIT,
  pycocotools BSD (FreeBSD), matplotlib PSF-style, tqdm MPL-2.0 + MIT (file-level copyleft, no obligation for
  unmodified use), jupyterlab/nbconvert/ipykernel BSD. No GPL or AGPL; Ultralytics is absent (DESIGN.md:101
  records the decision) and `grep -ri ultralytics` finds only that line.
- [x] 2.3 Competition data not tracked. Verify: `git ls-files | grep -iE '\.(jpe?g|tif|fits|zip|csv)$'`. Result:
  only three figures (`docs/figures/qualitative_final.jpg`, `qualitative_tree.jpg`, `report/figures/qualitative_tree.jpg`).
  `data/` is gitignored and not in any commit.
- [x] 2.4 Owner check: the three qualitative figures show crops of GONG H-alpha images with our masks. Confirm the
  competition rules and the MAGFiLO licence allow derived figures in a public repository and report (usual for a
  report; I could not read the terms from here). If not, drop them from the repo.
  Checked 2026-11-10 against the competition rules (section 4, non-commercial use): README "Data and credits" states that figures and weights are provided for evaluation, research and education, with the MAGFiLO citation and the GONG/NSO acknowledgement.
- [x] 2.5 Weights are derived from competition data; the same check as 2.4 applies to publishing them (the
  deliverable list requires them).
  Same check as 2.4; the weights stay in the repository as the deliverable list requires.
- [x] 2.6 README step 7 and DESIGN credit the public notebook "Solar Filament Segmentation 2026 (0.40)" as the
  source of the stage-2 idea; no code was copied as far as the text says. Owner: confirm that.
  Owner decision 2026-11-10: keep and credit. 2026-11-11: README and report now also credit the notebook for the stage-1 threshold tree, hurdle target and exact selection (DESIGN, public notebook survey), not only stage 2.

## 3. Size

- [x] 3.1 Measured. Verify: `git ls-files | xargs du -ck | tail -1` gives 938,664 KiB (about 917 MiB) tracked, of
  which `weights/` is 929,512 KiB in 19 files; `git count-objects -vH` gives 500 loose objects, 852.89 MiB (not
  packed; the pack will be of similar size, .pt files do not compress). 211 tracked files.
- [x] 3.2 Per-file limit. Largest files: `unet_r34_1536_all.pt` 95,588 KiB (93.3 MiB), `unet_r34_all.pt` 93.3 MiB,
  stage-2 UNet++ 49.9 MiB each, stage-2 U-Net 46.7 MiB each. All under GitHub's 100 MiB push limit (the repo is
  already pushed at 1753f31). Two files exceed GitHub's 50 MiB warning size.
- [x] 3.3 Superseded weights kept "for comparison": `stage2_unet_f0..4.pt` (233.4 MiB), `unet_r34_all.pt`
  (93.3 MiB), `component_scorer_1536.pkl` (small). About 327 MiB the notebook does not load. Removing them from
  HEAD does not shrink a clone (they stay in history) and needs a history rewrite to help; the clone is about
  0.9 GB either way. Owner decision: keep (simplest, nothing breaks; README documents them) or rewrite.
  Verify if done: `git count-objects -vH` after `git gc`, and the notebook still runs.
  Done 2026-11-10: stage2_unet_f*.pt and unet_r34_all.pt removed from HEAD and history; component_scorer_1536.pkl (1.4 MB) kept and documented.

## 4. Reproducibility from a fresh clone

- [x] 4.1 README CLI flags. Verify: compared every `python -m filseg...` and `scripts/experiments/...` command in
  README.md and `scripts/make_notebook.py` against the argparse definitions (`--res`, `--fold`, `--name`,
  `--epochs`, `--views`, `--levels`, `--spread`, `--out`, `--context`, `--modes`, `--nodes`, `--tau`, `--tag`,
  `--arch`, `--extra`, `--aug`, `--blend`, `--workers`, `--weights`, `--tree-scorer`, `--refiner`; steps
  build/eval and crops/jitter/train/blend/eval/export). Result: all exist; every path in the README layout table exists.
- [x] 4.2 Imports against requirements. Verify: grep of `import` in `filseg/`, `scripts/` and the notebook.
  Result: third-party imports are numpy, cv2, torch, torchvision, segmentation_models_pytorch, sklearn, pycocotools,
  matplotlib, tqdm, albumentations (all pinned; timm is pinned as smp's dependency), plus `huggingface_hub`
  (only `scripts/hf/`, not needed to reproduce) and `nbformat` (only `scripts/make_notebook.py`). Fixed: the
  README's `jupyter lab` / `nbconvert` and `nbformat` were not listed; added `jupyterlab==4.6.5`,
  `nbconvert==7.17.2`, `nbformat==5.11.1`, `ipykernel==7.4.0` (venv versions). `huggingface_hub` deliberately left out.
- [x] 4.3 Notebook needs only weights/ and test images. Verify: its code cells never reference `runs/`, `docs/`
  or `sys.path`; weights come from `weights/`, test images from `data/raw/MAGFiLO_1.0_Kaggle_2026/test/test_images`
  (override: `FILSEG_RAW`), `submissions/` is created by the notebook (`mkdir`). `build_model(..., pretrained=False)`
  so no ImageNet download is needed. Smoke test, one process, nice 10: `import filseg.predict, stage2, tree, train,
  report, metric` works from the worktree; `load_tree_scorer`, `torch.load` of the 1536 px U-Net and
  `load_refiners` for one UNet++ and one U-Net refiner all load on CPU in 45 s.
- [ ] 4.4 Not done: a fresh-venv `pip install -r requirements.txt` (torch is several GB; machine load) and the
  full 180-image run (about 16 s per image on the M3 Max, so about 48 min there; slower on CPU-only machines).
  Owner: run both once on a clean machine or the Kaggle CPU kernel before the flip.
- [x] 4.5 Risk: the pinned versions (torch 2.15.0, numpy 2.5.4, ...) must exist on PyPI for the judges' Python;
  README says Python 3.12. Owner: confirm with `pip install --dry-run -r requirements.txt` on a clean 3.12.
  Done 2026-11-11: `uv pip install --dry-run -r requirements-dev.txt` on a fresh Python 3.12 venv (macOS arm64) and `uv pip compile --python-platform x86_64-unknown-linux-gnu --python-version 3.12` both resolve every pin (torch 2.15.0, numpy 2.5.4, scikit-learn 1.9.1, ...).
- [x] 4.6 Risk: `tree_scorer_1536.pkl` and `component_scorer_1536.pkl` are pickles of scikit-learn models: they load
  only with scikit-learn 1.9.1 (pinned) and pickle executes code on load, which judges may dislike. Fix if wanted:
  note it in the README.
  Done 2026-11-11: README notes that the two .pkl files are scikit-learn pickles that load with the pinned 1.9.1 and run code when loaded.

## 5. Notebook outputs

- [x] 5.1 Verify: parsed `pipeline.ipynb` (15 cells, 719 KiB, 7 code cells, 8 with outputs) for `/Users/`, `/home/`,
  `hf_`, email patterns in text outputs and for error outputs. Result: none at HEAD (the printed test-image path is
  relative, `data/raw/...`). Images are base64 PNG figures. Old versions in history are the exception, see 1.4.

## 6. Documentation

- [x] 6.1 Record in docs/DESIGN.md: not needed, no pipeline change; this change only touches requirements.txt.

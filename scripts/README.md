# Scripts

Everything under `scripts/` is either a step of the submitted pipeline, a helper around it, or a record of
an earlier experiment. The submitted pipeline (submission 8) is described in `docs/DESIGN.md` section 3, and
its commands are in the "Train from scratch" section of the top-level `README.md`. Entries in the last column
refer to the decision log in `docs/DESIGN.md` section 5 by their title.

Status words:

- **submitted path**: run to produce submission 8 (or generates a file the submission ships with).
- **helper**: a tool around the pipeline that does not change its output.
- **historical**: written for an earlier stage or a rejected experiment; kept because DESIGN cites its numbers.
  Some of these need run folders or caches that no longer exist locally (see the notes below the table).

Run Python scripts from the repository root as
`PYTHONPATH=. FILSEG_CACHE=data/cache1536 .venv/bin/python scripts/<script>`. The 24 Python scripts in
`scripts/experiments/` are listed in `scripts/experiments/README.md`.

## Files

| File | What it does | Status | DESIGN.md entry |
|---|---|---|---|
| `make_notebook.py` | Generates `pipeline.ipynb` from the cells it contains: the end-to-end inference notebook (test images to CSV with the stored weights) plus the training and validation commands. Execute afterwards with `jupyter nbconvert --to notebook --execute --inplace pipeline.ipynb`. | submitted path | "Notebook reproducibility" |
| `analyse_gt.py` | Checks on the training annotations alone: the RLE-based PQ against the organisers' dense-tensor code, inter-annotator PQ, and the connected-components ceiling of one annotator's own union mask. | helper (section 2 figures) | "Annotator ceiling", "Where the ceiling is" |
| `train_folds.sh` | Trains `filseg.train` on the given folds one after another (`NAME`, `FILSEG_CACHE` select run and cache), keeps `best.pt` and `val_probs.npy`, deletes `last.pt`, stops when disk is low. Used for the 1536 px folds on the Mac; the README loop does the same without the guard. | helper | "Full 1536 px pipeline, 5-fold OOF", "HF Jobs for training" |
| `run_1536.sh` | Builds the 1536 px cache, then trains fold 0 of `unet_r34_1536` (40 epochs). The first 1536 px experiment. | historical | "1536 px input, fold 0" |
| `pipeline_1536.sh` | 1536 px out-of-fold chain for the single-threshold pipeline: `oof_probs_views.py`, then `component_features.py` and `component_scorer.py`. Superseded by the tree scorer (`component_tree.py`). | historical | "Full 1536 px pipeline, 5-fold OOF" |
| `after_1536.sh` | Follow-ups queued after the first 1536 px training: `filseg.tune`, 4-flip probabilities, component features and scorer on fold-0-era run names. Superseded by `pipeline_1536.sh`. | historical | "1536 px input, fold 0" |
| `after.sh` | Waits for a process id to exit, then runs a command, so that only one GPU job runs at a time on the Mac. | helper | none (machine housekeeping) |
| `run_convnext.sh` | Waits for an idle GPU, then trains ConvNeXt-T U-Nets (`tu-convnext_tiny`, lr 3e-4, 1536 px) on folds 0-4 and all data, to ensemble with ResNet-34. Only fold 0 was trained (interrupted at epoch 31): the gain over ResNet-34 was below what one fold resolves and the cost was about 4.3 times higher, so the chain was not continued. Rejected. | historical | "ConvNeXt-T U-Net, fold 0" |
| `hf/launch.py` | Uploads code and an image cache to a private Hugging Face dataset repo (refuses a public one) and starts, fetches or runs training jobs on HF Jobs. | helper (training host) | "HF Jobs for training" |
| `hf/job.sh` | Runs inside an HF Job: downloads code and cache from the dataset repo, trains one fold with `filseg.train`, uploads the run folder. | helper (training host) | "HF Jobs for training" |
| `hf/job_spread.sh` | Runs inside an HF Job: computes `oof_view_spread.py` for the five fold models and uploads `val_spread_v4.npy` per fold. Produced the OOF spread maps the submitted scorer was fitted on. | helper (training host) | "Scorer on 4-flip probabilities", "Full 1536 px pipeline, 5-fold OOF" |
| `hf/run_2048.sh` | Waits for the 2048 px cache, uploads it and starts two 2048 px test folds (0 and 3) on A100s. | historical | "2048 px (native), folds 0 and 3" |
| `hf/watch.py` | Polls an HF job and prints its stage and new training-log lines until it ends. | helper | none |
| `kaggle/push.sh` | Pushes the code as a private Kaggle dataset and the training kernel as a private kernel (`kaggle` CLI, `muhakabartay/...`). | helper (training host) | "Kaggle GPU not used" |
| `kaggle/train_kernel.py` | Kaggle kernel for two T4 GPUs: builds the 1536 px cache, trains one chain of folds per GPU with `--amp`, saves the 4-view mean and spread. Ready but never run, because the account's GPU sessions were taken by another project. | historical | "Kaggle GPU not used" |
| `kaggle/kernel_config.json` | Configuration of `train_kernel.py`: the ConvNeXt-T run name, encoder, learning rate, epochs and the fold chains per GPU. | historical | "Kaggle GPU not used" |
| `kaggle/cpu_probe/cpu_probe.py` | Kaggle CPU kernel that times stage-2 refiner training steps (U-Net and UNet++, ResNet-34, batch 32 of 4 x 128 x 384) on random data. Result: 13.7 s and 32.7 s per step, so 25 to 40 times slower than the Mac and over the 12 hour session limit. | historical | "Kaggle CPU as a training host" |
| `kaggle/cpu_probe/kernel-metadata.json` | Kaggle metadata of the probe kernel (private, CPU, internet on, no data attached). | historical | "Kaggle CPU as a training host" |

## Notes

- `scripts/hf/` and `scripts/kaggle/` are training-host helpers. They are not needed to reproduce the
  submission: the commands in the top-level `README.md` train on any CUDA or MPS machine. The Hugging Face
  dataset repo that `scripts/hf/` names is private, because it holds the competition data, which is not ours
  to redistribute. The Kaggle code dataset and kernels are private for the same reason. Neither is available to
  readers, so these scripts are published as a record of how the folds were trained, not as something to run.
- The three `*_1536.sh` chains and `run_convnext.sh` write `.out` logs into `runs/` and assume the shared
  Mac used for the project; `run_convnext.sh` reads macOS `ioreg` and `vm_stat`.
- Of the top-level scripts only `make_notebook.py` belongs to the submitted path. The training that the
  submission used was started by hand or by `train_folds.sh` and `hf/job.sh`, with the arguments recorded in
  each run's `args.json`.

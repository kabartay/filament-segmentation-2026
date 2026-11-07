"""Kaggle training kernel: fold models of one architecture at 1536 px on 2 x T4, one chain of folds per GPU.

Runs inside a private Kaggle kernel with the competition data and the private code dataset (``filseg-code``,
built by scripts/kaggle/push.sh) attached and internet on (pip, pretrained weights). Steps:

1. install the pinned packages the Kaggle image lacks;
2. build the 1536 px cache from the competition data (filseg.cache);
3. per GPU, train its folds one after another (``--amp``, a wall-clock budget so everything ends before the
   12-hour session limit), then predict the fold's validation images with four flipped views and save their mean
   and spread (val_probs_v4.npy, val_spread_v4.npy), as the tree scorer needs;
4. copy best.pt, the maps, val_stems.json, args.json and log.txt of every run to /kaggle/working/runs.

Configuration: scripts/kaggle/kernel_config.json in the code dataset.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

START = time.time()
HOURS = 11.3  # leave margin under Kaggle's 12-hour limit


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} [{(time.time() - START) / 60:6.1f} min] {msg}", flush=True)


def find(suffix: str) -> Path:
    for d, _, files in os.walk("/kaggle/input"):
        for f in files:
            p = Path(d) / f
            if p.as_posix().endswith(suffix):
                return p
    raise FileNotFoundError(suffix)


def main() -> None:
    cfg = {"name": "unet_cnxt_1536", "encoder": "tu-convnext_tiny", "lr": 3e-4, "epochs": 40,
           "chains": [[1, 3], [2, 4]], **json.loads(find("kaggle/kernel_config.json").read_text())}
    log(f"config {cfg}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "segmentation-models-pytorch==0.6.0",
                    "pycocotools==2.0.11"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "albumentations==2.0.8"], check=True)
    code_src = find("filseg/train.py").parent.parent
    code = Path("/tmp/code")
    shutil.copytree(code_src, code, dirs_exist_ok=True)
    raw = find("train/MAGFiLO_1.0_Annotations_kaggle2026_train.json").parent.parent
    env = {**os.environ, "PYTHONPATH": str(code), "FILSEG_RAW": str(raw), "FILSEG_CACHE": "/tmp/cache1536",
           "OMP_NUM_THREADS": "2"}
    log(f"code {code_src}, data {raw}")
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], check=False)
    subprocess.run([sys.executable, "-m", "filseg.cache", "--res", "1536"], cwd=code, env=env, check=True)
    log("cache built")
    deadline = START + HOURS * 3600
    out = Path("/kaggle/working/runs")
    out.mkdir(parents=True, exist_ok=True)

    def chain(gpu: int, folds: list[int]) -> None:
        genv = {**env, "CUDA_VISIBLE_DEVICES": str(gpu), "FILSEG_DEVICE": "cuda"}
        for i, f in enumerate(folds):
            post_min = 12.0  # 4-view maps of the fold plus copying
            budget = ((deadline - time.time()) / 60) / (len(folds) - i) - post_min
            if budget < 30:
                log(f"gpu {gpu}: no time left for fold {f}")
                return
            log(f"gpu {gpu}: fold {f} start, budget {budget:.0f} min")
            name = cfg["name"]
            with open(f"/kaggle/working/train_{name}_f{f}.log", "w") as fh:
                subprocess.run([sys.executable, "-m", "filseg.train", "--fold", str(f), "--name", name,
                                "--encoder", cfg["encoder"], "--lr", str(cfg["lr"]), "--epochs", str(cfg["epochs"]),
                                "--amp", "--max-minutes", f"{budget:.1f}", "--workers", "2"],
                               cwd=code, env=genv, stdout=fh, stderr=subprocess.STDOUT, check=False)
            run = code / "runs" / f"{name}_f{f}"
            if f >= 0 and (run / "best.pt").exists():
                with open(f"/kaggle/working/spread_{name}_f{f}.log", "w") as fh:
                    subprocess.run([sys.executable, "scripts/experiments/oof_view_spread.py", "--name", name,
                                    "--folds", str(f), "--save-mean"],
                                   cwd=code, env=genv, stdout=fh, stderr=subprocess.STDOUT, check=False)
            dst = out / run.name
            dst.mkdir(exist_ok=True)
            keep = ["best.pt", "val_probs_v4.npy", "val_spread_v4.npy", "val_stems.json", "args.json", "log.txt"]
            if f < 0:
                keep.append("last.pt")
            for k in keep:
                if (run / k).exists():
                    shutil.copy(run / k, dst / k)
            log(f"gpu {gpu}: fold {f} done -> {dst} ({sorted(p.name for p in dst.iterdir())})")

    import torch
    chains = cfg["chains"]
    if torch.cuda.device_count() < len(chains):  # one GPU: run every fold on it, one after another
        log(f"{torch.cuda.device_count()} GPU(s); running the chains one after another")
        chains = [[f for c in chains for f in c]]
    threads = [threading.Thread(target=chain, args=(g, folds)) for g, folds in enumerate(chains)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log("all chains finished")


if __name__ == "__main__":
    main()

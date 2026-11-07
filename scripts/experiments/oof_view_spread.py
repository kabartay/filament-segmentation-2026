"""Per-pixel disagreement between the four flipped views, for each fold's validation images.

The component scorer judges how likely a candidate is to match an annotator; where the flipped views of the
same model disagree, the model is unsure, so the spread is a natural uncertainty feature. For every fold
model (runs/<name>_f{k}/best.pt, the checkpoint behind val_probs_v4.npy) this predicts the four views
(identity, horizontal, vertical, both flips) separately and saves their standard deviation per pixel,
float16, in the order of val_stems.json: runs/<name>_f{k}/val_spread_v4.npy. The inference pipeline computes
the same four views, so the feature costs nothing extra at test time.

Run (GPU): ``FILSEG_CACHE=data/cache1536 PYTHONPATH=. python scripts/experiments/oof_view_spread.py --name unet_r34_1536``
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from filseg.cache import CACHE
from filseg.train import FLIP_VIEWS, RUNS, build_model, fold_split, normalise, run_encoder


@torch.no_grad()
def view_probs(model: torch.nn.Module, image: np.ndarray, device: str) -> np.ndarray:
    """(4, R, R) float32 sigmoid probabilities of the four flipped views, each flipped back."""
    x = torch.from_numpy(normalise(np.array(image)))[None, None].to(device)
    out = [torch.sigmoid(model(x.flip(d))).flip(d) if d else torch.sigmoid(model(x)) for d in FLIP_VIEWS[4]]
    return torch.cat(out)[:, 0].float().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="unet_r34_1536")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--save-mean", action="store_true", help="also save the 4-view mean as val_probs_v4.npy")
    ap.add_argument("--limit", type=int, default=0, help="first N validation images only (smoke test; nothing saved)")
    a = ap.parse_args()
    device = os.environ.get("FILSEG_DEVICE", "cuda" if torch.cuda.is_available()
                            else "mps" if torch.backends.mps.is_available() else "cpu")
    index = json.loads((CACHE / "index.json").read_text())["train"]
    images = np.load(CACHE / "train_images.npy", mmap_mode="r")
    for f in (int(k) for k in a.folds.split(",")):
        run = RUNS / f"{a.name}_f{f}"
        t0 = time.time()
        _, val_idx = fold_split(index, f)
        stems = json.loads((run / "val_stems.json").read_text())
        assert [index[k] for k in val_idx] == stems
        if a.limit:
            val_idx = val_idx[:a.limit]
        model = build_model(run_encoder(run), pretrained=False).to(device)
        model.load_state_dict(torch.load(run / "best.pt", map_location=device))
        model.eval()
        spread = np.empty((len(val_idx),) + images.shape[1:], np.float16)
        mean = np.empty_like(spread) if a.save_mean else None
        mean_check = []
        for j, k in enumerate(val_idx):
            v = view_probs(model, images[k], device)
            spread[j] = v.std(axis=0)
            if mean is not None:
                mean[j] = v.mean(axis=0)
            if j < 3:
                mean_check.append(v.mean(axis=0))
        if not a.limit:
            np.save(run / "val_spread_v4.npy", spread)
            if mean is not None:
                np.save(run / "val_probs_v4.npy", mean)
        if (run / "val_probs_v4.npy").exists():  # same model and views: the mean must match the saved 4-view map
            saved = np.load(run / "val_probs_v4.npy", mmap_mode="r")[:len(mean_check)].astype(np.float32)
            print(f"fold {f}: 4-view mean vs saved max |diff| {np.abs(np.stack(mean_check) - saved).max():.4f}")
        print(f"fold {f}: {len(val_idx)} images, {time.time() - t0:.0f} s, mean spread {float(spread.mean()):.4f}",
              flush=True)


if __name__ == "__main__":
    main()

"""Train/test parity of the view-spread features.

OOF spread (the scorer's training features) comes from fold models that did not see the image; test spread
comes from the all-data model. On training images the all-data model is in-sample, the fold model is
out-of-sample. This runs the all-data model on a sample of training images (10 per fold) and compares the
per-pixel spread and the node-level spread features (spread_mean/q90/max/rel over the tree's nodes at the
selected levels) with the saved OOF ones. Test images' spread from the all-data model is also summarised
(whole-image statistics only) for reference.

Run (GPU, under the lock): ``lockf -k runs/gpu.lock python scripts/experiments/adopt_spread_parity.py``
Writes runs/adopt-scorer-features_parity.json.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from filseg.cache import CACHE
from filseg.train import FLIP_VIEWS, RUNS, build_model, image_cache, normalise

NAME = "unet_r34_1536"
PER_FOLD = 10


@torch.no_grad()
def views(model: torch.nn.Module, img: np.ndarray, device: str) -> np.ndarray:
    x = torch.from_numpy(normalise(np.array(img)))[None, None].to(device)
    out = [torch.sigmoid(model(x.flip(d))).flip(d) if d else torch.sigmoid(model(x)) for d in FLIP_VIEWS[4]]
    return torch.cat(out)[:, 0].float().cpu().numpy()


def summary(a: np.ndarray) -> dict:
    return {"mean": float(a.mean()), "p50": float(np.median(a)), "p90": float(np.quantile(a, 0.9)),
            "p99": float(np.quantile(a, 0.99))}


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    index = json.loads((CACHE / "index.json").read_text())
    images = np.load(CACHE / "train_images.npy", mmap_mode="r")
    model = build_model("resnet34", pretrained=False).to(device)
    model.load_state_dict(torch.load("weights/unet_r34_1536_all.pt", map_location=device))
    model.eval()
    rows = []
    for f in range(5):
        run = RUNS / f"{NAME}_f{f}"
        stems = json.loads((run / "val_stems.json").read_text())
        sp_oof = np.load(run / "val_spread_v4.npy", mmap_mode="r")
        pr_oof = np.load(run / "val_probs_v4.npy", mmap_mode="r")
        for i in np.random.default_rng(f).choice(len(stems), PER_FOLD, replace=False):
            v = views(model, images[index["train"].index(stems[i])], device)
            sp_all, pr_all = v.std(axis=0), v.mean(axis=0)
            so, po = np.asarray(sp_oof[i], np.float32), np.asarray(pr_oof[i], np.float32)
            fg = (po > 0.3) | (pr_all > 0.3)  # where the model thinks there is a filament
            rows.append(dict(
                stem=stems[i], fold=f,
                spread_all_mean=float(sp_all.mean()), spread_oof_mean=float(so.mean()),
                spread_all_fg=float(sp_all[fg].mean()) if fg.any() else np.nan,
                spread_oof_fg=float(so[fg].mean()) if fg.any() else np.nan,
                prob_all_mean=float(pr_all.mean()), prob_oof_mean=float(po.mean()),
                corr_spread=float(np.corrcoef(sp_all.ravel(), so.ravel())[0, 1]),
                fg_frac=float(fg.mean())))
            print(rows[-1]["stem"], f"all {rows[-1]['spread_all_fg']:.4f} oof {rows[-1]['spread_oof_fg']:.4f}", flush=True)
    # test images: whole-image stats from the all-data model (same code path as inference)
    tim = np.load(CACHE / image_cache("test", "plain"), mmap_mode="r")
    test = [float(views(model, tim[k], device).std(axis=0).mean()) for k in range(0, len(tim), 6)]
    out = dict(rows=rows, test_all_model_spread_mean_sample=test)
    (RUNS / "adopt-scorer-features_parity.json").write_text(json.dumps(out, indent=1))
    a = lambda k: np.array([r[k] for r in rows])
    print("n", len(rows))
    for k in ("spread_mean", "spread_fg"):
        x, y = a(k.replace("spread", "spread_all")), a(k.replace("spread", "spread_oof"))
        print(f"{k}: all {np.nanmean(x):.5f}  oof {np.nanmean(y):.5f}  ratio all/oof {np.nanmean(x) / np.nanmean(y):.3f}")
    print("test-image mean spread (all-data model) ", np.mean(test), "train OOF mean", a("spread_oof_mean").mean(),
          "train all-model mean", a("spread_all_mean").mean())


if __name__ == "__main__":
    main()

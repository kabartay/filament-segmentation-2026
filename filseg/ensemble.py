"""Average several models' probability maps, with the view spread pooled exactly over all their views.

Each model contributes the mean m_i and the population std s_i of its four flipped views. Over all 4k views
of k models the mean is M = mean_i(m_i) and the variance is mean_i(s_i^2 + m_i^2) - M^2 (every model has the
same number of views), so the pooled spread needs no new inference and includes the disagreement between
the models as well as within each one.

``average_runs`` writes the result as a run folder (runs/<out>_f{k}) with the same file names as a trained run,
so ``filseg.tune`` and ``scripts/experiments/component_tree.py`` read it unchanged.

Run (fold 0, the 2-view maps saved during training):
``.venv/bin/python -m filseg.ensemble --runs unet_r34_1536 unet_cnxt_1536 --out ens_r34_cnxt_1536 --folds 0 --probs val_probs.npy``

Status: only ``pool_views`` is on the submitted path, and only when ``filseg.predict`` is given more than one
checkpoint with ``--tree-scorer``; submission 8 uses one model, so it is not exercised. ``average_runs`` and the
command line were for the ConvNeXt-T plus ResNet-34 test, which was not continued. docs/DESIGN.md,
"ConvNeXt-T U-Net, fold 0".
"""

from __future__ import annotations

import argparse
import json
import shutil

import numpy as np

from filseg.train import RUNS


def pool_views(means: list[np.ndarray], spreads: list[np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Mean and pooled std over all models' views, float32.

    Args:
        means: per model, the mean of its views, any equal shape.
        spreads: per model, the population std of its views (same shapes), or None to skip the spread.

    Returns:
        (mean, spread); spread is None when ``spreads`` is None.
    """
    mean = np.mean([np.asarray(m, np.float32) for m in means], axis=0)
    if spreads is None:
        return mean, None
    second = np.mean([np.asarray(s, np.float32) ** 2 + np.asarray(m, np.float32) ** 2
                      for m, s in zip(means, spreads)], axis=0)
    return mean, np.sqrt(np.maximum(second - mean ** 2, 0.0))


def average_runs(runs: list[str], out: str, folds: list[int], probs: str, spread: str | None) -> None:
    """Write runs/<out>_f{k}/<probs> (and <spread>) averaged over ``runs``, image by image."""
    for f in folds:
        dirs = [RUNS / f"{r}_f{f}" for r in runs]
        stems = [json.loads((d / "val_stems.json").read_text()) for d in dirs]
        assert all(s == stems[0] for s in stems), f"fold {f}: the runs validate on different images"
        target = RUNS / f"{out}_f{f}"
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy(dirs[0] / "val_stems.json", target / "val_stems.json")
        m_in = [np.load(d / probs, mmap_mode="r") for d in dirs]
        s_in = [np.load(d / spread, mmap_mode="r") for d in dirs] if spread else None
        m_out = np.empty(m_in[0].shape, np.float16)
        s_out = np.empty(m_in[0].shape, np.float16) if spread else None
        for i in range(len(m_out)):
            mean, sd = pool_views([m[i] for m in m_in], [s[i] for s in s_in] if s_in else None)
            m_out[i] = mean
            if s_out is not None:
                s_out[i] = sd
        np.save(target / probs, m_out)
        if s_out is not None:
            np.save(target / spread, s_out)
        (target / "args.json").write_text(json.dumps({"ensemble_of": runs, "probs": probs, "spread": spread}))
        print(f"fold {f}: {len(m_out)} images -> {target}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="run names; runs/<name>_f{k}")
    parser.add_argument("--out", required=True)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--probs", default="val_probs_v4.npy")
    parser.add_argument("--spread", default=None, help="e.g. val_spread_v4.npy; pooled over all views")
    args = parser.parse_args()
    average_runs(args.runs, args.out, [int(k) for k in args.folds.split(",")], args.probs, args.spread)


if __name__ == "__main__":
    main()

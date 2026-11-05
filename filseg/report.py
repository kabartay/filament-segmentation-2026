"""The organisers' three evaluation plots, computed on out-of-fold validation predictions.

The final evaluation asks for the IoU distribution, the Dice distribution (both over
GT/prediction pairs with IoU > 0) and the many-to-one / one-to-many counts (links at
IoU > 0, per annotator-image entry). This module reproduces the self-evaluation
notebook's definitions on our own predictions, so the report figures match what the
organisers will compute on the test set.

Run: ``.venv/bin/python -m filseg.report --run unet_r34full_f0 --post runs/unet_r34full_f0/post.json``
or, for the component-scorer pipeline, from cached OOF candidates with a cross-validated scorer:
``.venv/bin/python -m filseg.report --components runs/oof_components_1536_v4.pkl --tau 0.20 --name oof1536``
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from filseg.cache import CACHE
from filseg.data import ROOT, group_by_stem, load_train
from filseg.geometry import fit_disk
from filseg.metric import PQCounts, iou_matrix
from filseg.postprocess import instances
from filseg.predict import load_post
from filseg.scorer import make_regressor, training_table
from filseg.train import RUNS

FIGURES = ROOT / "docs" / "figures"


def collect(runs: list[str], post: Path) -> tuple[list[np.ndarray], PQCounts]:
    """IoU matrix (GT x prediction) for every annotator-image entry, and the pooled PQ."""
    cfg = load_post(post)
    index = json.loads((CACHE / "index.json").read_text())["train"]
    images = np.load(CACHE / "train_images.npy", mmap_mode="r")
    groups = group_by_stem(load_train())
    matrices, total = [], PQCounts()
    for run in runs:
        stems = json.loads((RUNS / run / "val_stems.json").read_text())
        probs = np.load(RUNS / run / "val_probs.npy", mmap_mode="r")
        for prob, stem in zip(probs, stems):
            disk = fit_disk(np.array(images[index.index(stem)]))
            pred = instances(np.asarray(prob, np.float32), cfg, disk)
            for entry in groups[stem]:
                iou = iou_matrix(entry.rles, pred)
                total.add(iou, len(entry.rles), len(pred))
                matrices.append(iou)
    return matrices, total


def collect_scored(components: Path, tau: float) -> tuple[list[np.ndarray], PQCounts]:
    """As ``collect``, for the scorer pipeline: candidates kept by a scorer trained on the other four folds."""
    data = pickle.load(open(components, "rb"))
    _, X, y, fold = training_table(data)
    score = np.zeros_like(y)
    for f in np.unique(fold):
        score[fold == f] = make_regressor().fit(X[fold != f], y[fold != f]).predict(X[fold == f])
    matrices, total, start = [], PQCounts(), 0
    for d in data:
        n = len(d["feats"])
        kept = np.nonzero(score[start:start + n] > tau)[0]
        start += n
        for n_gt, iou in d["entries"]:
            iou = iou[:, kept] if iou.shape[1] else iou
            total.add(iou, n_gt, len(kept))
            matrices.append(iou)
    return matrices, total


def distribution(values: np.ndarray, title: str, label: str, path: Path) -> None:
    """Histogram of pair scores above zero, as probabilities, with the mean marked."""
    counts, edges = np.histogram(values, bins=np.linspace(0, 1, 51))
    plt.figure(figsize=(5, 3.5))
    plt.bar((edges[:-1] + edges[1:]) / 2, counts / max(counts.sum(), 1), width=0.02,
            edgecolor="black", alpha=0.7, color="navy")
    plt.axvline(values.mean(), color="crimson", linestyle="--", label=f"Mean = {values.mean():.3f}")
    plt.grid(axis="y", linestyle="--")
    plt.xlabel(label)
    plt.ylabel("Probability")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def degrees(matrices: list[np.ndarray]) -> tuple[Counter, Counter]:
    """Per GT, the number of predictions it overlaps (1:n); per prediction, GTs it overlaps (n:1)."""
    gt, pred = Counter(), Counter()
    for iou in matrices:
        hit = iou > 0
        gt.update(hit.sum(axis=1).tolist())
        pred.update(hit.sum(axis=0).tolist())
    return gt, pred


def many_to_one(gt: Counter, pred: Counter, path: Path) -> None:
    """Split bar chart: GT degrees to the right (1:n), prediction degrees to the left (n:1)."""
    plt.figure(figsize=(6, 3.5))
    for degree, count in gt.items():
        plt.bar(degree + 0.5, count, width=0.8, edgecolor="black", alpha=0.7,
                color="crimson" if degree == 0 else "forestgreen")
    for degree, count in pred.items():
        plt.bar(-(degree + 0.5), count, width=0.8, edgecolor="black", alpha=0.7,
                color="darkorange" if degree == 0 else "steelblue")
    ticks = [d + 0.5 for d in gt] + [-(d + 0.5) for d in pred]
    labels = [f"1:{d}" for d in gt] + [f"{d}:1" for d in pred]
    plt.xticks(ticks, labels, rotation=45, fontsize=8)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.grid(axis="y", linestyle="--")
    plt.xlabel("Matching case")
    plt.ylabel("Frequency")
    plt.title("Many to one and one to many")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", nargs="+", help="one run, or one per fold to pool")
    parser.add_argument("--post", type=Path, help="post.json or tune.json to apply")
    parser.add_argument("--components", type=Path, help="cached OOF candidates (scorer pipeline) instead of --run")
    parser.add_argument("--tau", type=float, default=0.20)
    parser.add_argument("--name", default="oof")
    args = parser.parse_args()

    if args.components:
        matrices, total = collect_scored(args.components, args.tau)
    else:
        matrices, total = collect(args.run, args.post)
    pairs = np.concatenate([m.ravel() for m in matrices if m.size])
    iou = pairs[pairs > 0]
    dice = 2 * iou / (1 + iou)
    gt, pred = degrees(matrices)

    FIGURES.mkdir(parents=True, exist_ok=True)
    distribution(iou, "IoU distribution", "IoU", FIGURES / f"{args.name}_iou.png")
    distribution(dice, "Dice distribution", "Dice", FIGURES / f"{args.name}_dice.png")
    many_to_one(gt, pred, FIGURES / f"{args.name}_m2n.png")
    stats = {"runs": args.run, "post": str(args.post) if args.post else f"scorer tau {args.tau} on {args.components}", "entries": len(matrices), "pq": round(total.score(), 4),
             "summary": total.summary(), "pairs_iou_gt0": int(iou.size),
             "mean_iou": round(float(iou.mean()), 4), "mean_dice": round(float(dice.mean()), 4),
             "gt_degree": dict(sorted(gt.items())), "pred_degree": dict(sorted(pred.items()))}
    (FIGURES / f"{args.name}_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()

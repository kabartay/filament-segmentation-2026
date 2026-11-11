"""Grid-search post-processing on a run's saved validation probabilities.

Only post-processing is tuned; the network is fixed. For each (threshold, closing)
pair the components of every validation image, and their IoU against every annotator
entry, are computed once. The cheap filters (area, confidence, longitude) then just
select columns of those IoU matrices, so the full grid costs little more than the
number of (threshold, closing) pairs.

Several runs (one per fold) are tuned together on their pooled out-of-fold images.

Run: ``.venv/bin/python -m filseg.tune --run unet_r34_f0`` or ``--run unet_r34full_f{0..4}``

Status: not used by the submitted pipeline (submission 8). It tunes the hand rule (threshold, closing, area,
confidence, limb filter) of submissions 1 to 4, which the tree scorer replaced. docs/DESIGN.md, "Pooled tuning,
all 5 folds" and "Second-stage component scorer".
"""

from __future__ import annotations

import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from filseg.cache import CACHE
from filseg.data import group_by_stem, load_train
from filseg.geometry import fit_disk
from filseg.metric import PQCounts, iou_matrix
from filseg.postprocess import PostConfig, components, keep
from filseg.train import RUNS

THRESHOLDS = (0.35, 0.45, 0.55, 0.65)
CLOSINGS = (0, 31, 51, 75)
MIN_AREAS = (200, 400, 800, 1500)
MIN_SCORES = (0.0, 0.7, 0.8, 0.85)
MAX_LONS = (90.0, 70.0)

_STATE: dict = {}


def _init(runs: list[str]) -> None:
    index = json.loads((CACHE / "index.json").read_text())["train"]
    images = np.load(CACHE / "train_images.npy", mmap_mode="r")
    probs, stems = [], []
    for run in runs:
        run_stems = json.loads((RUNS / run / "val_stems.json").read_text())
        run_probs = np.load(RUNS / run / "val_probs.npy", mmap_mode="r")
        probs += [run_probs[i] for i in range(len(run_stems))]
        stems += run_stems
    assert len(stems) == len(set(stems)), "runs share validation images"
    _STATE.update(
        probs=probs,
        stems=stems,
        groups=group_by_stem(load_train()),
        disks=[fit_disk(np.array(images[index.index(s)])) for s in stems],
    )


def _grid_for(pair: tuple[float, int]) -> list[tuple[PostConfig, PQCounts]]:
    """Score every cheap-filter setting for one (threshold, closing) pair."""
    threshold, close_px = pair
    cached = []  # per image: (components, [IoU matrix per annotator entry])
    for prob, stem, disk in zip(_STATE["probs"], _STATE["stems"], _STATE["disks"]):
        comps = components(np.asarray(prob, np.float32), threshold, close_px, disk, min_area=min(MIN_AREAS))
        rles = [c.rle for c in comps]
        cached.append((comps, [(len(e.rles), iou_matrix(e.rles, rles)) for e in _STATE["groups"][stem]]))
    results = []
    for area, score, lon in itertools.product(MIN_AREAS, MIN_SCORES, MAX_LONS):
        cfg = PostConfig(threshold, close_px, area, score, lon)
        total = PQCounts()
        for comps, entries in cached:
            cols = keep(comps, cfg)
            for n_gt, iou in entries:
                total.add(iou[:, cols], n_gt, len(cols))
        results.append((cfg, total))
    return results


def _score_one(cfg: PostConfig) -> tuple[PostConfig, PQCounts]:
    """Score a single setting end to end."""
    total = PQCounts()
    for prob, stem, disk in zip(_STATE["probs"], _STATE["stems"], _STATE["disks"]):
        comps = components(np.asarray(prob, np.float32), cfg.threshold, cfg.close_px, disk, cfg.min_area)
        pred = [comps[i].rle for i in keep(comps, cfg)]
        for entry in _STATE["groups"][stem]:
            total.add(iou_matrix(entry.rles, pred), len(entry.rles), len(pred))
    return cfg, total


def _pooled_stems(runs: list[str]) -> list[str]:
    return [s for run in runs for s in json.loads((RUNS / run / "val_stems.json").read_text())]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", nargs="+", required=True, help="one run, or one per fold to pool")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--check", nargs="+", metavar="T,C,A,S,L",
                        help="score only these settings (threshold,close,area,score,lon) and exit")
    args = parser.parse_args()

    if args.check:
        cfgs = [PostConfig(float(t), int(c), int(a), float(s), float(lon))
                for t, c, a, s, lon in (x.split(",") for x in args.check)]
        with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.run,)) as pool:
            for cfg, counts in pool.map(_score_one, cfgs):
                print(cfg, counts.summary())
        return

    pairs = list(itertools.product(THRESHOLDS, CLOSINGS))
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.run,)) as pool:
        results = [r for batch in pool.map(_grid_for, pairs) for r in batch]
    results.sort(key=lambda r: -r[1].score())
    rows = [{**cfg.__dict__, "pq": round(c.score(), 5), "tp": c.tp, "fp": c.fp, "fn": c.fn}
            for cfg, c in results]
    out = RUNS / args.run[0] if len(args.run) == 1 else RUNS / f"tune_{args.run[0]}+{len(args.run) - 1}"
    out.mkdir(exist_ok=True)
    (out / "tune.json").write_text(json.dumps(rows, indent=1))
    print("wrote", out / "tune.json", "on", len(_pooled_stems(args.run)), "images")
    for row in rows[:8]:
        print(row)
    plain = [r for r in rows if r["close_px"] == 0 and r["min_score"] == 0.0 and r["max_lon"] == 90.0]
    print("best without closing, confidence or limb filter:", plain[0])


if __name__ == "__main__":
    main()

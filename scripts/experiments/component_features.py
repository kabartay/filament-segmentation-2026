"""Cache candidate components of the 5-fold OOF probabilities with features and per-entry IoU.

Candidates and features come from ``filseg.scorer.candidates`` (threshold 0.45, closing 31 px, area
>= 100 px, no confidence filter). For each annotator entry we store the IoU matrix (GT x candidates).
Output is used by component_scorer.py and ``filseg.scorer``.
Run: PYTHONPATH=. .venv/bin/python scripts/experiments/component_features.py [--probs val_probs.npy] [--out runs/oof_components.pkl]
"""
import argparse, json, pickle
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from filseg.cache import CACHE
from filseg.data import TRAIN_IMAGES, group_by_stem, load_train, read_image
from filseg.geometry import fit_disk
from filseg.metric import iou_matrix
from filseg.postprocess import upsample
from filseg.scorer import candidates
from filseg.train import RUNS

G = {}
def init(probs_file, name):
    G["groups"] = group_by_stem(load_train()); G["index"] = json.loads((CACHE / "index.json").read_text())["train"]
    G["images"] = np.load(CACHE / "train_images.npy", mmap_mode="r")
    G["probs"] = {f: np.load(RUNS / f"{name}_f{f}" / probs_file, mmap_mode="r") for f in range(5)}

def work(item):
    f, i, s = item
    full = upsample(np.asarray(G["probs"][f][i], np.float32))
    disk = fit_disk(np.array(G["images"][G["index"].index(s)]))
    comps, feats = candidates(full, read_image(TRAIN_IMAGES / f"{s}.jpeg"), disk)
    rles = [c.rle for c in comps]
    entries = [(len(e.rles), iou_matrix(e.rles, rles)) for e in G["groups"][s]]
    return dict(fold=f, stem=s, feats=feats, entries=entries)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--probs", default="val_probs.npy")
    ap.add_argument("--out", default=str(RUNS / "oof_components.pkl")); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--name", default="unet_r34full")
    a = ap.parse_args()
    items = [(f, i, s) for f in range(5) for i, s in enumerate(json.loads((RUNS / f"{a.name}_f{f}" / "val_stems.json").read_text()))]
    if a.limit: items = items[:a.limit]
    with ProcessPoolExecutor(4, initializer=init, initargs=(a.probs, a.name)) as ex:
        out = list(ex.map(work, items, chunksize=4))
    pickle.dump(out, open(a.out, "wb"))
    print(len(out), "images,", sum(len(o["feats"]) for o in out), "candidate components ->", a.out)

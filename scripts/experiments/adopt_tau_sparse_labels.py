"""Stage 1: OOF PQ and sparse-label PQ' for the selection threshold tau (slug adopt-tau-sparse-labels).

Cross-validated hurdle scores are computed once (cached), the antichain selection is made for each tau, and the
same selections are re-scored against ground truth thinned at random (each GT instance kept with prob r), plus
the closed form PQ' = r*SumIoU / (0.5|P| + 0.5 r|G|) for comparison.  Writes runs/adopt-tau-sparse-labels_stage1.json.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import component_tree as ct  # noqa: E402
from filseg.tree import antichain  # noqa: E402

SLUG = "adopt-tau-sparse-labels"
TAUS = [round(0.22 + 0.02 * i, 2) for i in range(8)]
RS = [1.0, 0.85, 0.77, 0.65]
SEEDS = 20
RUNS = Path("runs")


def score_thinned(data, picks, r: float, masks) -> float:
    """Pooled PQ with each GT row kept when its uniform draw is below r."""
    tot = ct.PQCounts()
    for d, kept, mk in zip(data, picks, masks):
        kept = np.array(kept, dtype=np.int64)
        for (n_gt, iou), u in zip(d["entries"], mk):
            keep = u < r
            sub = iou[keep][:, kept] if iou.shape[1] else iou[keep]
            tot.add(sub, int(keep.sum()), len(kept))
    return tot.score()


def main() -> None:
    data = pickle.load(open(RUNS / "oof_tree12s_1536_v4.pkl", "rb"))
    ct.add_context(data)
    cache = RUNS / f"{SLUG}_scores.pkl"
    if cache.exists():
        s_all = pickle.load(open(cache, "rb"))
    else:
        s_all, _ = ct.cv_scores(data, lambda ft: True, "hurdle")
        pickle.dump(s_all, open(cache, "wb"))
    picks = {t: [antichain(s, d["parent"], t) for s, d in zip(s_all, data)] for t in TAUS}
    res = {"taus": TAUS, "rs": RS, "closed": {}, "sim": {}, "counts": {}}
    for t in TAUS:
        c = ct.pooled(data, picks[t])
        res["counts"][t] = dict(pq=c.score(), tp=c.tp, fp=c.fp, fn=c.fn, iou=c.tp_iou_sum)
        P, G = c.tp + c.fp, c.tp + c.fn
        res["closed"][t] = {r: r * c.tp_iou_sum / (0.5 * P + 0.5 * r * G) for r in RS}
        print(t, c.summary(), flush=True)
    sim = {t: {r: [] for r in RS} for t in TAUS}
    for seed in range(SEEDS):
        rng = np.random.default_rng(seed)
        masks = [[rng.random(n) for n, _ in d["entries"]] for d in data]  # same draws for every r and tau
        for r in RS:
            for t in TAUS:
                sim[t][r].append(score_thinned(data, picks[t], r, masks))
    res["sim"] = {t: {r: [float(np.mean(v)), float(np.std(v))] for r, v in sim[t].items()} for t in TAUS}
    for name, getter in (("closed", lambda t, r: res["closed"][t][r]), ("sim", lambda t, r: res["sim"][t][r][0])):
        reg = {t: max(max(getter(u, r) for u in TAUS) - getter(t, r) for r in RS) for t in TAUS}
        res[f"maxregret_{name}"] = reg
        print(name, "max regret", {t: round(v, 5) for t, v in reg.items()})
    for r in RS:
        print("r", r, "sim mean", {t: round(res["sim"][t][r][0], 4) for t in TAUS}, "closed",
              {t: round(res["closed"][t][r], 4) for t in TAUS})
    arr = {t: np.array([sim[t][r] for r in RS]) for t in TAUS}  # (r, seeds)
    best = np.max([arr[u] for u in TAUS], axis=0)
    res["seed_maxregret"] = {t: [float((best - arr[t]).max(0).mean()), float((best - arr[t]).max(0).std())] for t in TAUS}
    print("per-seed max regret mean/std", {t: [round(a, 5), round(b, 5)] for t, (a, b) in res["seed_maxregret"].items()})
    (RUNS / f"{SLUG}_stage1.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()

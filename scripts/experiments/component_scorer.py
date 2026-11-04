"""Second-stage component scorer: keep the components whose expected PQ contribution is positive.

A kept component matching k of the n annotator entries of its image adds sum(IoU of those k) to the PQ
numerator and about 0.5 n to the denominator, so it pays when E[sum IoU_TP] / n > PQ / 2 (about 0.2).
We regress y = sum_entries IoU * [IoU > 0.5] / n from component features with gradient boosting,
cross-validated by fold (trained on four folds' components, applied to the fifth), then keep components
with y_hat > tau and score the pooled OOF PQ exactly. Needs runs/oof_components.pkl (or the pickle given as the first argument).
Run: PYTHONPATH=. .venv/bin/python scripts/experiments/component_scorer.py [components.pkl]
"""
import pickle, sys
import numpy as np
from filseg.metric import PQCounts
from filseg.scorer import make_regressor, training_table
from filseg.train import RUNS

data = pickle.load(open(sys.argv[1] if len(sys.argv) > 1 else RUNS / "oof_components.pkl", "rb"))
names, X, y, fold = training_table(data)
print(f"{len(y)} candidates, {len(names)} features; mean target {y.mean():.3f}")

def pooled_pq(keep_mask, only_fold=None):
    t = PQCounts(); start = 0
    for k, d in enumerate(data):
        nc = len(d["feats"]); sel = np.nonzero(keep_mask[start:start + nc])[0]; start += nc
        if only_fold is not None and d["fold"] != only_fold:
            continue
        for n_gt, m in d["entries"]:
            t.add(m[:, sel] if m.shape[1] else m, n_gt, len(sel))
    return t

col = {c: i for i, c in enumerate(names)}
base = (X[:, col["area"]] >= 300) & (X[:, col["mean"]] >= 0.8)
print("baseline rule (area >= 300, mean >= 0.8):", pooled_pq(base).summary())
base85 = (X[:, col["area"]] >= 300) & (X[:, col["mean"]] >= 0.85)
print("baseline rule (area >= 300, mean >= 0.85):", pooled_pq(base85).summary())

pred = np.zeros_like(y)
for f in range(5):
    tr, te = fold != f, fold == f
    pred[te] = make_regressor().fit(X[tr], y[tr]).predict(X[te])
for tau in (0.12, 0.15, 0.18, 0.2, 0.22, 0.25, 0.28, 0.32):
    print(f"scorer tau {tau:.2f}:", pooled_pq(pred > tau).summary())
# per-fold check of the best tau, so a gain is not one lucky fold
best_tau = max((0.15, 0.18, 0.2, 0.22, 0.25), key=lambda t: pooled_pq(pred > t).score())
print("best tau", best_tau)
for f in range(5):
    print(f"fold {f}: rule {pooled_pq(base, f).score():.4f}  scorer {pooled_pq(pred > best_tau, f).score():.4f}")

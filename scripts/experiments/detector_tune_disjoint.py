import itertools, pickle, sys
import numpy as np
from pycocotools import mask as mask_util
sys.path.insert(0, ".")
from filseg.data import SIZE, group_by_stem, load_train
from filseg.metric import PQCounts, entry_counts
from filseg.train import RUNS

dets = pickle.loads((RUNS / "mrcnn_f0" / "val_dets.pkl").read_bytes())
groups = group_by_stem(load_train())
n_gt = sum(len(e.rles) for s in dets for e in groups[s])

def paint(found, thr, keep_frac):
    """Disjoint masks: higher score claims pixels first; drop masks left with < keep_frac of their area."""
    found = sorted([d for d in found if d[0] >= thr], key=lambda d: -d[0])
    taken = np.zeros((SIZE, SIZE), bool)
    out = []
    for s, rle in found:
        m = mask_util.decode(rle).astype(bool)
        own = m & ~taken
        if own.sum() >= keep_frac * m.sum() and own.any():
            out.append(mask_util.encode(np.asfortranarray(own.astype(np.uint8))))
            taken |= own
    return out

for thr in (0.6, 0.7, 0.75, 0.8, 0.85):
    for frac in (0.0, 0.5):
        total = PQCounts()
        for stem, found in dets.items():
            pred = paint(found, thr, frac)
            for e in groups[stem]:
                total += entry_counts(e.rles, pred)
        assert total.tp + total.fn == n_gt, (total.tp + total.fn, n_gt)
        print(thr, frac, total.summary(), flush=True)

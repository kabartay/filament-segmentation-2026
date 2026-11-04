import itertools, pickle, sys
import numpy as np
from pycocotools import mask as mask_util
sys.path.insert(0, ".")
from filseg.data import group_by_stem, load_train
from filseg.detector import merge_overlapping
from filseg.metric import PQCounts, entry_counts
from filseg.train import RUNS

dets = pickle.loads((RUNS / "mrcnn_f0" / "val_dets.pkl").read_bytes())
groups = group_by_stem(load_train())
rows = []
for thr, merge, area in itertools.product((0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95), (True, False), (0, 200, 400, 800)):
    total = PQCounts()
    for stem, found in dets.items():
        rles = [r for s, r in found if s >= thr]
        if merge:
            rles = merge_overlapping(rles)
        rles = [r for r in rles if mask_util.area(r) >= area]
        for e in groups[stem]:
            total += entry_counts(e.rles, rles)
    rows.append((total.score(), thr, merge, area, total.summary()))
rows.sort(reverse=True)
for r in rows[:12]:
    print(r[1:])

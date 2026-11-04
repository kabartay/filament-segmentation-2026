import json, numpy as np
from filseg.cache import CACHE
from filseg.data import group_by_stem, load_train
from filseg.geometry import fit_disk
from filseg.metric import PQCounts, iou_matrix
from filseg.postprocess import PostConfig, instances
from filseg.train import RUNS
groups = group_by_stem(load_train()); index = json.loads((CACHE/"index.json").read_text())["train"]
images = np.load(CACHE/"train_images.npy", mmap_mode="r"); cfg = PostConfig(0.45, 31, 300, 0.8, 90.0)
t = {1: PQCounts(), 2: PQCounts(), 3: PQCounts()}
for f in range(5):
    stems = json.loads((RUNS/f"unet_r34full_f{f}"/"val_stems.json").read_text()); probs = np.load(RUNS/f"unet_r34full_f{f}"/"val_probs.npy", mmap_mode="r")
    for i, s in enumerate(stems):
        g = groups[s]
        preds = instances(np.asarray(probs[i], np.float32), cfg, fit_disk(np.array(images[index.index(s)])))
        for e in g: t[len(g)].add(iou_matrix(e.rles, preds), len(e.rles), len(preds))
for k, v in t.items(): print(f"images with {k} annotator(s):", v.summary())
m = PQCounts(); m += t[2]; m += t[3]; print("images with >=2 annotators:", m.summary())

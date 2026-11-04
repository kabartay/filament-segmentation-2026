"""Test-time augmentation with more views: identity+hflip (current), 4 flips, 8 dihedral views.

Re-predicts fold-0 validation with runs/unet_r34full_f0/best.pt on the device given by FILSEG_DEVICE
(default cpu), scores each variant at the submission setting and at its own best threshold/confidence.
Run: PYTHONPATH=. .venv/bin/python scripts/experiments/tta_views.py
"""
import itertools, json, os
import numpy as np, torch
from filseg.cache import CACHE
from filseg.data import group_by_stem, load_train
from filseg.geometry import fit_disk
from filseg.metric import PQCounts, iou_matrix
from filseg.postprocess import PostConfig, components, keep
from filseg.train import RUNS, build_model, fold_split, normalise

device = os.environ.get("FILSEG_DEVICE", "cpu")
torch.set_num_threads(8)
index = json.loads((CACHE / "index.json").read_text())["train"]
images = np.load(CACHE / "train_images.npy", mmap_mode="r")
_, val_idx = fold_split(index, 0)
stems = [index[k] for k in val_idx]
groups = group_by_stem(load_train())
model = build_model("resnet34", pretrained=False).to(device)
model.load_state_dict(torch.load(RUNS / "unet_r34full_f0" / "best.pt", map_location=device)); model.eval()

def views(x):  # (transform, inverse) pairs over the dihedral group
    out = []
    for k in range(4):
        for flip in (False, True):
            out.append((k, flip))
    return out
probs = {"h2": [], "flip4": [], "d8": []}
with torch.no_grad():
    for k in val_idx:
        x = torch.from_numpy(normalise(np.array(images[k])))[None, None].to(device)
        p = {}
        for rot, flip in views(x):
            y = torch.rot90(x, rot, (-2, -1)); y = y.flip(-1) if flip else y
            q = torch.sigmoid(model(y)); q = q.flip(-1) if flip else q
            p[(rot, flip)] = torch.rot90(q, -rot, (-2, -1))[0, 0].cpu().numpy()
        probs["h2"].append((p[(0, False)] + p[(0, True)]) / 2)
        vflip = np.flip(p[(2, True)], (0, 1))  # not needed: build 4-flip from rot/flip equivalents
        probs["flip4"].append((p[(0, False)] + p[(0, True)] + p[(2, False)] + p[(2, True)]) / 4)  # id, h, rot180 (=h+v), v
        probs["d8"].append(sum(p.values()) / 8)
disks = [fit_disk(np.array(images[k])) for k in val_idx]
for name, ps in probs.items():
    best = []
    for thr in (0.4, 0.45, 0.5, 0.55):
        cached = []
        for prob, s, d in zip(ps, stems, disks):
            cs = components(prob.astype(np.float32), thr, 31, d, 200)
            cached.append((cs, [(len(e.rles), iou_matrix(e.rles, [c.rle for c in cs])) for e in groups[s]]))
        for area, score in itertools.product((200, 300, 400), (0.75, 0.8, 0.85)):
            cfg = PostConfig(thr, 31, area, score, 90.0); t = PQCounts()
            for cs, ents in cached:
                cols = keep(cs, cfg)
                for n, m in ents: t.add(m[:, cols], n, len(cols))
            best.append((t.score(), cfg, t.summary()))
    best.sort(key=lambda r: -r[0])
    sub = [r for r in best if r[1] == PostConfig(0.45, 31, 300, 0.8, 90.0)][0]
    print(f"{name}: submission setting {sub[2]} | best {best[0][2]} at thr {best[0][1].threshold} area {best[0][1].min_area} conf {best[0][1].min_score}", flush=True)

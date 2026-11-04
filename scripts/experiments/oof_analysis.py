"""Where does PQ go? Human ceiling, resolution ceiling, error breakdown, 2-model ensemble on fold 0."""
import json, sys, itertools
from collections import Counter
import numpy as np, cv2
from pycocotools import mask as mu
from filseg.cache import CACHE
from filseg.data import group_by_stem, load_train
from filseg.geometry import fit_disk
from filseg.metric import PQCounts, iou_matrix
from filseg.postprocess import PostConfig, components, keep
from filseg.train import RUNS

part = sys.argv[1]
groups = group_by_stem(load_train())
index = json.loads((CACHE / "index.json").read_text())["train"]
images = np.load(CACHE / "train_images.npy", mmap_mode="r")
CFG = PostConfig(0.45, 31, 300, 0.8, 90.0)

def load(run):
    return json.loads((RUNS / run / "val_stems.json").read_text()), np.load(RUNS / run / "val_probs.npy", mmap_mode="r")

if part == "human":
    multi = {s: g for s, g in groups.items() if len(g) >= 2}
    print("images with >=2 annotators:", len(multi), "of", len(groups), Counter(len(g) for g in groups.values()))
    total = PQCounts()
    pairs = Counter()
    for s, g in multi.items():
        for a, b in itertools.permutations(g, 2):
            c = PQCounts(); c.add(iou_matrix(b.rles, a.rles), len(b.rles), len(a.rles)); total += c
    print("annotator-as-prediction vs other annotators:", total.summary())

if part == "res":
    rng = np.random.default_rng(0)
    entries = [r for g in groups.values() for e in g for r in e.rles]
    sample = [entries[i] for i in rng.choice(len(entries), 400, replace=False)]
    for res in (1024, 1536):
        ious = []
        for r in sample:
            m = mu.decode(r).astype(np.float32)
            back = cv2.resize(cv2.resize(m, (res, res), interpolation=cv2.INTER_AREA), (2048, 2048), interpolation=cv2.INTER_LINEAR) > 0.5
            ious.append((back & (m > 0)).sum() / max((back | (m > 0)).sum(), 1))
        ious = np.array(ious)
        print(f"res {res}: mean IoU of GT round trip {ious.mean():.4f}, p10 {np.percentile(ious,10):.3f}, share<0.5 {(ious<0.5).mean():.3f}")

if part == "errors":
    stems, probs = [], []
    for f in range(5):
        s, p = load(f"unet_r34full_f{f}")
        stems += s; probs += [p[i] for i in range(len(s))]
    total = PQCounts()
    fn_maxiou, fn_area, fp_other, fp_maxiou, fp_area, fp_score, tp_area = [], [], [], [], [], [], []
    for prob, stem in zip(probs, stems):
        disk = fit_disk(np.array(images[index.index(stem)]))
        comps = components(np.asarray(prob, np.float32), CFG.threshold, CFG.close_px, disk, CFG.min_area)
        cols = keep(comps, CFG)
        preds = [comps[i].rle for i in cols]
        ent = groups[stem]
        all_gt = [r for e in ent for r in e.rles]
        iou_all = iou_matrix(all_gt, preds)
        for e in ent:
            iou = iou_matrix(e.rles, preds)
            total.add(iou, len(e.rles), len(preds))
            if len(e.rles) == 0: 
                for j, i in enumerate(cols):
                    fp_maxiou.append(0.0); fp_other.append(iou_all[:, j].max() > 0.5 if len(all_gt) else False); fp_area.append(comps[i].area); fp_score.append(comps[i].score)
                continue
            areas = mu.area(e.rles)
            for k in range(len(e.rles)):
                if not len(preds) or iou[k].max() <= 0.5:
                    fn_maxiou.append(iou[k].max() if len(preds) else 0.0); fn_area.append(areas[k])
                else: tp_area.append(areas[k])
            for j, i in enumerate(cols):
                if iou[:, j].max() <= 0.5:
                    fp_maxiou.append(iou[:, j].max()); fp_other.append(iou_all[:, j].max() > 0.5)
                    fp_area.append(comps[i].area); fp_score.append(comps[i].score)
    print("5-fold OOF, fixed setting", CFG, "\n", total.summary(), "images", len(stems))
    fn_maxiou, fn_area, tp_area = map(np.array, (fn_maxiou, fn_area, tp_area))
    fp_maxiou, fp_other, fp_area, fp_score = map(np.array, (fp_maxiou, fp_other, fp_area, fp_score))
    print(f"FN {len(fn_maxiou)}: untouched(IoU=0) {(fn_maxiou==0).mean():.2f}, weak(0-0.3) {((fn_maxiou>0)&(fn_maxiou<0.3)).mean():.2f}, near-miss(0.3-0.5] {(fn_maxiou>=0.3).mean():.2f}")
    bins = [0, 500, 1000, 2000, 5000, 10**9]
    h_fn = np.histogram(fn_area, bins)[0]; h_tp = np.histogram(tp_area, bins)[0]
    print("GT area bins", bins[:-1], "FN", h_fn.tolist(), "TP", h_tp.tolist(), "recall", np.round(h_tp/np.maximum(h_tp+h_fn,1),2).tolist())
    print(f"FP {len(fp_maxiou)}: matches another annotator's GT {fp_other.mean():.2f}, untouched {(fp_maxiou==0).mean():.2f}, near-miss(0.3-0.5] {(fp_maxiou>=0.3).mean():.2f}")
    h_fp = np.histogram(fp_area, bins)[0]
    print("FP area bins", h_fp.tolist(), " FP score quartiles", np.round(np.percentile(fp_score,[25,50,75]),3).tolist())
    # PQ if every near miss became a TP at IoU 0.55 (upper bound on better boundaries/grouping)
    n_near = int((fn_maxiou >= 0.3).sum())
    tp, fp, fn = total.tp + n_near, total.fp - n_near, total.fn - n_near
    print(f"if near-miss FNs became TPs (paired with one FP each): PQ <= {(total.tp_iou_sum + 0.55*n_near)/(tp+0.5*fp+0.5*fn):.4f}")

if part == "ens":  # usage: ens [runA runB], default the 40-epoch and epoch-20 fold-0 models
    run_a, run_b = sys.argv[2:4] if len(sys.argv) >= 4 else ("unet_r34full_f0", "unet_r34_f0")
    s1, p1 = load(run_a); s2, p2 = load(run_b)
    assert s1 == s2
    disks = [fit_disk(np.array(images[index.index(s)])) for s in s1]
    for name, get in ((run_a, lambda i: p1[i]), (run_b, lambda i: p2[i]), ("mean", lambda i: (p1[i].astype(np.float32) + p2[i].astype(np.float32)) / 2)):
        for thr in (0.4, 0.45, 0.5, 0.55, 0.65):
            cached = []
            for i, s in enumerate(s1):
                comps = components(np.asarray(get(i), np.float32), thr, 31, disks[i], 200)
                cached.append((comps, [(len(e.rles), iou_matrix(e.rles, [c.rle for c in comps])) for e in groups[s]]))
            for area, score in itertools.product((200, 300, 400), (0.75, 0.8, 0.85)):
                cfg = PostConfig(thr, 31, area, score, 90.0); t = PQCounts()
                for comps, ents in cached:
                    cols = keep(comps, cfg)
                    for n, iou in ents: t.add(iou[:, cols], n, len(cols))
                print(name, thr, area, score, t.summary(), flush=True)

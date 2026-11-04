"""Re-predict each fold's validation images with more flip views and save the OOF probabilities.

Uses runs/unet_r34full_f{0..4}/best.pt (the checkpoints that wrote val_probs.npy) and writes
val_probs_v{views}.npy next to them, in the order of val_stems.json. As a check it also re-predicts the
first three images with 2 views and reports the largest difference from the saved val_probs.npy.
Run: PYTHONPATH=. .venv/bin/python scripts/experiments/oof_probs_views.py --views 4   (GPU; FILSEG_DEVICE overrides)
"""
import argparse, json, os, time
import numpy as np, torch
from filseg.cache import CACHE
from filseg.train import RUNS, build_model, fold_split, predict

ap = argparse.ArgumentParser(); ap.add_argument("--views", type=int, default=4); a = ap.parse_args()
device = os.environ.get("FILSEG_DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
index = json.loads((CACHE / "index.json").read_text())["train"]
images = np.load(CACHE / "train_images.npy", mmap_mode="r")
for f in range(5):
    run = RUNS / f"unet_r34full_f{f}"; t0 = time.time()
    _, val_idx = fold_split(index, f)
    assert [index[k] for k in val_idx] == json.loads((run / "val_stems.json").read_text())
    model = build_model("resnet34", pretrained=False).to(device)
    model.load_state_dict(torch.load(run / "best.pt", map_location=device))
    val = np.stack([np.array(images[k]) for k in val_idx])
    check = predict(model, val[:3], device, views=2).astype(np.float32)
    saved = np.load(run / "val_probs.npy", mmap_mode="r")[:3].astype(np.float32)
    probs = predict(model, val, device, views=a.views)
    np.save(run / f"val_probs_v{a.views}.npy", probs)
    print(f"fold {f}: {len(val_idx)} images, {time.time() - t0:.0f} s, 2-view check max |diff| {np.abs(check - saved).max():.4f}", flush=True)

"""Upload code + an image cache (default the 1536 px one) to a private HF dataset repo, and launch training as HF Jobs.

  .venv/bin/python scripts/hf/launch.py upload --repo <user>/filseg-2026
  .venv/bin/python scripts/hf/launch.py run --repo <user>/filseg-2026 --fold 3 [--flavor l40sx1] [--timeout 2h]
  .venv/bin/python scripts/hf/launch.py fetch --repo <user>/filseg-2026 --run unet_r34_1536_f3

The repo stays private: it holds competition data, which is not ours to redistribute.
"""
import argparse
from pathlib import Path
from huggingface_hub import HfApi, get_token

ROOT = Path(__file__).resolve().parents[2]
api = HfApi()
ap = argparse.ArgumentParser()
ap.add_argument("action", choices=["upload", "run", "fetch"])
ap.add_argument("--repo", required=True)
ap.add_argument("--fold", type=int, default=0)
ap.add_argument("--name", default="unet_r34_1536")
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--flavor", default="l40sx1")
ap.add_argument("--timeout", default="2h")
ap.add_argument("--run", default=None)
ap.add_argument("--cache", default="data/cache1536", help="cache folder to upload / train from")
a = ap.parse_args()

if a.action == "upload":
    api.create_repo(a.repo, repo_type="dataset", private=True, exist_ok=True)
    assert api.repo_info(a.repo, repo_type="dataset").private, "repo must be private"
    raw = "data/raw/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    files = ["requirements.txt", raw, f"{a.cache}/index.json", f"{a.cache}/train_images.npy",
             f"{a.cache}/train_targets.npy", "scripts/hf/job.sh"] + [str(p.relative_to(ROOT)) for p in (ROOT / "filseg").glob("*.py")]
    for f in files:
        api.upload_file(path_or_fileobj=str(ROOT / f), path_in_repo=f, repo_id=a.repo, repo_type="dataset")
        print("uploaded", f)
elif a.action == "run":
    script = (ROOT / "scripts/hf/job.sh").read_text()
    job = api.run_job(image="python:3.12", command=["bash", "-c", script], flavor=a.flavor, timeout=a.timeout,
                      env={"REPO": a.repo, "FOLD": str(a.fold), "NAME": a.name, "EPOCHS": str(a.epochs), "CACHE_DIR": a.cache},
                      secrets={"HF_TOKEN": get_token()})
    print("job", job.id, job.url if hasattr(job, "url") else "")
else:
    api.snapshot_download(a.repo, repo_type="dataset", allow_patterns=[f"runs/{a.run}/*"], local_dir=str(ROOT))
    print("fetched runs/" + a.run)

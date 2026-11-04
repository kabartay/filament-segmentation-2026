"""Print an HF job's stage and its new training-log lines (epoch, val, errors) until it ends."""
import sys, time
from huggingface_hub import HfApi
api, jid, seen = HfApi(), sys.argv[1], 0
keys = ("epoch", "val PQ", "done", "device=", "Error", "error", "Traceback", "uploaded", "NVIDIA", "Killed")
while True:
    try:
        stage = api.inspect_job(job_id=jid).status.stage
        lines = list(api.fetch_job_logs(job_id=jid, follow=False))
        for l in lines[seen:]:
            if any(k in l for k in keys): print(l.strip()[:200], flush=True)
        seen = len(lines)
        if stage not in ("RUNNING", "SCHEDULING", "PENDING", "QUEUED"):
            print("job stage:", stage, flush=True); break
    except Exception as e:
        print("poll error:", type(e).__name__, str(e)[:120], flush=True)
    time.sleep(60)

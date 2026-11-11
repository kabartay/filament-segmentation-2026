"""Shared pytest setup: single-threaded CPU runs and an importable ``filseg``.

The environment variables must be set before numpy, OpenCV or torch are imported, so they are set at
module level here. None of the tests read ``data/`` or ``runs/`` or need a GPU.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)
cv2.setNumThreads(1)

WEIGHTS = ROOT / "weights"

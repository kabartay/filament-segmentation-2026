## 1. Analysis

- [x] 1.1 FN/FP breakdown. Verify: `scripts/experiments/oof_analysis.py` on the saved OOF probabilities. Result: MEASURED, PQ 0.4086, TP 5025, FP 3155, FN 3174; FN 45% no prediction, 18% IoU < 0.3, 37% IoU 0.3-0.5.
- [x] 1.2 Recall by size. Result: < 500 px 0.36, 500-1000 0.61, 1000-2000 0.68, 2000-5000 0.70, > 5000 0.64.
- [x] 1.3 Near-miss anatomy. Result: 1,169 FN at IoU 0.3-0.5; smaller 46%, similar 20%, larger 34%; merge 17%, split 16%.
- [x] 1.4 Small-filament agreement. Verify: `scripts/experiments/small_filaments.py`. Result: other-annotator share 0.28 / 0.39 / 0.49 / about 0.60 by size; 1,263 small GT, 451 TP.
- [x] 1.5 Break-even algebra. Result: pays when k/n > PQ / (2 SQ) = 0.31.
- [x] 1.6 Upper bound for near misses. Result: at most PQ 0.487 at IoU 0.55.

## 2. Records

- [x] 2.1 Recorded in docs/DESIGN.md section 5 (2026-11-04 entries).

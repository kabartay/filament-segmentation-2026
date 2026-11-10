## 1. Analysis

- [x] 1.1 Tabulate OOF against LB for submissions 1-8. Result: MEASURED/LB pairs in the proposal; gap 0.06-0.07.
- [x] 1.2 Bootstrap over images. Verify: `scripts/experiments/bootstrap_pq.py`. Result: 0.4219 [0.412, 0.431]; paired gain +0.0133 [0.008, 0.018]; scaled to about 90 images +-0.027.
- [x] 1.3 Compare with other entrants' gap. Result: 0.05-0.06 on four pipelines, adversarial AUC 0.496, 5 of 24 annotation groups absent from training (their claims, not verified).
- [x] 1.4 Decide the selection rule. Result: paired OOF only.

## 2. Records

- [x] 2.1 Recorded in docs/DESIGN.md sections 4, 5 (bootstrap entry, submission entries), 6.

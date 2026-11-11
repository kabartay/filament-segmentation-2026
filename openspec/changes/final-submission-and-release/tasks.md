## 1. Final selection

- [x] 1.1 Verify sub7 and sub8 CSVs are intact and pixel-disjoint; verify: overlap check script on `submissions/sub8_*.csv` and `submissions/sub7_*.csv` (0 shared pixels). Result 2026-11-11: sub8 177 images, 1,065 masks, 0 shared pixels; sub7 177 images, 1,066 masks, 0 shared pixels; notebook_submission.csv byte-identical to sub8.
- [x] 1.2 Owner-gated: owner selects the ONE final submission on Kaggle (sub8, the 2026-11-10 upload with its description); verify: the Submissions page no longer shows "Auto-selection candidates" and sub8 is ticked. Result: selected by the owner on 2026-11-10.
- [x] 1.3 Report LB rows in DESIGN and the report; verify: `grep -n "LB" docs/DESIGN.md report/main.tex`. Result 2026-11-11: submission table (sub1-sub8, local PQ and public LB) at the top of DESIGN.md and in README.md; the report gives the LB steps 0.35/0.36/0.37 and the +-0.027 resolution.

## 2. Release

- [ ] 2.1 Final README, report PDF (4 pages) and notebook consistent; verify: `.venv/bin/python scripts/make_notebook.py` and compile `report/main.tex`
- [ ] 2.2 Owner-gated: owner asks, then at or after 2026-11-15 23:59 CDT run `gh repo edit kabartay/filament-segmentation-2026 --visibility public --accept-visibility-change-consequences`; verify: `gh repo view kabartay/filament-segmentation-2026 --json visibility`
- [ ] 2.3 Owner-gated: owner fills the Google form with the repo URL; verify: owner confirms
- [ ] 2.4 After going public, check CI and README links; verify: `gh run list -R kabartay/filament-segmentation-2026 --limit 3` and a link check of README.md

## 3. Record

- [ ] 3.1 DESIGN final entry; verify: `grep -n "final" docs/DESIGN.md`
- [ ] 3.2 Archive; verify: `openspec validate final-submission-and-release --strict`

# release-readiness-audit

## Why
The repository becomes public at the deadline (2026-11-15 23:59 CDT; the owner flips it) and judges will clone
it and try to reproduce the submission. This audit looks for secrets, personal data, licence problems, size
problems and reproduction breaks before that happens. Read-only checks plus small CPU checks at HEAD 1753f31.

## What Changes
- `requirements.txt`: pins `jupyterlab`, `nbconvert`, `nbformat`, `ipykernel` (the README runs `jupyter lab` /
  `jupyter nbconvert`, `scripts/make_notebook.py` imports `nbformat`; none were listed). Versions are the ones in
  the project venv.
- Everything else is reported in tasks.md for the owner; nothing is removed, no history is rewritten.

## Non-goals
No full inference run, no fresh-venv install of torch (too heavy on the shared machine), no history rewrite,
no licence choice, no GitHub settings.

## Falsified by
A judge cloning the repo cannot run the README commands, or finds a credential or personal path in HEAD or history.

## Cost
CPU only, a few minutes, one process.

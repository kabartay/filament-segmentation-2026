## 1. Measured results

- [x] 1.1 Kaggle GPU (`scripts/kaggle/` training kernel and private code dataset, 2026-11-07). Result: ready but
  never run; the account's two batch GPU sessions were taken by another project with priority.
- [x] 1.2 Kaggle CPU probe (2026-11-10): 4 cores of a Xeon at 2.20 GHz, 31 GB RAM, torch 2.11 CPU with 2 threads.
  Result: MEASURED one refiner training step (batch 32, 4 x 128 x 384, ResNet-34): U-Net 13.7 s, UNet++ 32.7 s,
  i.e. about 15 h (U-Net) and 35 h (UNet++) per 15-epoch fold against 28-36 min / about 55 min on local MPS;
  25-40x slower and over the 12 h session limit. Not used for training.

## 2. Records

- [x] 2.1 DESIGN.md section 5 entries as above.

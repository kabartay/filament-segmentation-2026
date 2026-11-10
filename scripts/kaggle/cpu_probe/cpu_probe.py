"""Kaggle CPU probe: machine facts and the time of stage-2 refiner training steps.

Random uint8 crops of the real shape (batch 32, 4 x 128 x 384) stand in for the data, so no dataset is
attached; timing does not depend on the content. Prints seconds per step for a U-Net and a UNet++ with
a ResNet-34 encoder, and the implied minutes per epoch and per fold.
"""
import os
import platform
import subprocess
import sys
import time

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch==0.6.0"], check=True)
import torch  # noqa: E402
import segmentation_models_pytorch as smp  # noqa: E402

print("cores", os.cpu_count(), "| torch threads", torch.get_num_threads(), "| torch", torch.__version__)
print("cpu", platform.processor(), subprocess.run("lscpu | grep 'Model name'", shell=True, capture_output=True,
                                                  text=True).stdout.strip())
print(subprocess.run("free -g | head -2", shell=True, capture_output=True, text=True).stdout)

STEPS_PER_EPOCH = 8300 // 32  # one fold of the jitter training set
for arch in ("Unet", "UnetPlusPlus"):
    m = getattr(smp, arch)("resnet34", encoder_weights="imagenet", in_channels=4, classes=1)
    opt = torch.optim.AdamW(m.parameters(), 5e-4)
    x = torch.rand(32, 4, 128, 384)
    y = (torch.rand(32, 1, 128, 384) > 0.9).float()
    m.train()
    times = []
    for i in range(8):
        t0 = time.time()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(m(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        times.append(time.time() - t0)
    s = sorted(times[2:])[len(times[2:]) // 2]
    print(f"{arch}: {s:.2f} s/step (median of 6 after 2 warm-up) -> {s * STEPS_PER_EPOCH / 60:.1f} min/epoch, "
          f"{s * STEPS_PER_EPOCH * 15 / 3600:.1f} h per 15-epoch fold", flush=True)

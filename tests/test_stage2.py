"""Stage 2 (``filseg.stage2``): paste-back disjointness, defaults of ``refine``, architecture detection."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from pycocotools import mask as mask_util

from filseg import stage2
from filseg.data import SIZE
from filseg.stage2 import build_refiner, paste_back, refine, refiner_arch


def box(r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    """Boolean SIZE x SIZE rectangle."""
    m = np.zeros((SIZE, SIZE), bool)
    m[r0:r1, c0:c1] = True
    return m


def pairwise_disjoint(masks: list[np.ndarray]) -> bool:
    total = np.zeros((SIZE, SIZE), np.int32)
    for m in masks:
        total += m
    return int(total.max(initial=0)) <= 1


class ConstantSeedModel(torch.nn.Module):
    """Stand-in refiner: confident logit where the seed channel is set, so the output follows the seed."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 12.0 * (x[:, 2:3] - 0.5)


def test_paste_back_output_is_disjoint_when_refined_masks_overlap() -> None:
    seeds = [box(100, 140, 100, 400), box(120, 160, 300, 600)]
    # both refined maps cover the overlap strip with different probabilities
    refined = [np.where(box(90, 150, 90, 450), 0.9, 0.0).astype(np.float32),
               np.where(box(110, 170, 280, 620), 0.7, 0.0).astype(np.float32)]
    out = paste_back(seeds, refined, np.array([0.8, 0.6]))
    assert len(out) == 2
    assert pairwise_disjoint(out)
    strip = box(120, 150, 300, 450)
    assert out[0][strip].all() and not out[1][strip].any()  # higher refined probability wins


def test_paste_back_none_keeps_the_seed() -> None:
    seeds = [box(10, 30, 10, 200)]
    out = paste_back(seeds, [None], np.array([0.5]))
    assert len(out) == 1 and (out[0] == seeds[0]).all()


def test_paste_back_erased_seed_kept_unless_drop_erased() -> None:
    seeds = [box(10, 30, 10, 200)]
    empty = [np.zeros((SIZE, SIZE), np.float32)]
    assert len(paste_back(seeds, empty, np.array([0.5]))) == 1
    assert paste_back(seeds, empty, np.array([0.5]), drop_erased=True) == []


def test_paste_back_keeps_only_pieces_touching_the_seed() -> None:
    seed = box(10, 30, 10, 200)
    q = np.where(seed | box(500, 520, 500, 520), 0.9, 0.0).astype(np.float32)  # a stray blob far away
    out = paste_back([seed], [q], np.array([0.5]))
    assert len(out) == 1 and (out[0] == seed).all()


def test_paste_back_absorbs_the_weaker_covered_seed() -> None:
    big = box(100, 200, 100, 600)
    small = box(120, 140, 150, 250)  # fully inside the big seed's refined mask
    q_big = np.where(big, 0.9, 0.0).astype(np.float32)
    q_small = np.where(small, 0.9, 0.0).astype(np.float32)
    out = paste_back([big, small], [q_big, q_small], np.array([0.9, 0.3]))
    assert len(out) == 1 and out[0].sum() == big.sum()


@pytest.mark.parametrize("mode", ["largest", "join", "join_drop"])
def test_one_piece_modes_stay_disjoint(mode: str) -> None:
    seeds = [box(100, 120, 100, 300), box(100, 120, 400, 600)]
    # the first mask has two pieces separated by a 30 px gap
    q0 = np.where(box(100, 120, 100, 300) | box(100, 120, 330, 380), 0.9, 0.0).astype(np.float32)
    q1 = np.where(seeds[1], 0.9, 0.0).astype(np.float32)
    out = paste_back(seeds, [q0, q1], np.array([0.8, 0.7]), single=mode)
    assert pairwise_disjoint(out)


def test_refine_defaults_equal_the_manual_pipeline() -> None:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (SIZE, SIZE), dtype=np.uint8)
    seeds = [box(300, 340, 200, 500), box(800, 850, 900, 1200)]
    prob = np.zeros((SIZE, SIZE), np.float32)
    for s in seeds:
        prob[s] = 0.8
    rles = [mask_util.encode(np.asfortranarray(s.astype(np.uint8))) for s in seeds]
    score = np.array([0.7, 0.5])
    models = [ConstantSeedModel()]

    got = refine(models, image, prob, rles, score, "cpu")

    x, frames = stage2.crop_inputs(image, prob, seeds)
    pred = stage2.predict_crops(models, x, "cpu")
    refined = [None if fr.scale < stage2.MIN_SCALE else stage2.unwarp(q, fr) for q, fr in zip(pred, frames)]
    want = paste_back(seeds, refined, score, stage2.T_PIX)

    assert len(got) == len(want) == 2
    for g, w in zip(got, want):
        assert (mask_util.decode(g).astype(bool) == w).all()
    # and the stand-in model follows its seeds closely
    for g, s in zip(got, seeds):
        m = mask_util.decode(g).astype(bool)
        assert (m & s).sum() / (m | s).sum() > 0.8
    assert pairwise_disjoint([mask_util.decode(g).astype(bool) for g in got])


def test_refine_without_instances_returns_empty() -> None:
    image = np.zeros((SIZE, SIZE), np.uint8)
    assert refine([ConstantSeedModel()], image, np.zeros((SIZE, SIZE), np.float32), [], np.zeros(0), "cpu") == []


def test_frame_scale_never_exceeds_one() -> None:
    assert stage2.frame_of(box(10, 20, 10, 40)).scale == 1.0
    assert stage2.frame_of(box(10, 20, 10, 1500)).scale < 1.0


@pytest.mark.parametrize("arch, expected", [("Unet", "Unet"), ("UnetPlusPlus", "UnetPlusPlus")])
def test_refiner_arch_from_state_dict_keys(arch: str, expected: str) -> None:
    state = build_refiner(arch, "resnet18", pretrained=False).state_dict()
    assert refiner_arch(state) == expected


def test_refiner_arch_on_bare_keys() -> None:
    assert refiner_arch({"decoder.blocks.x_0_0.conv1.weight": 0}) == "UnetPlusPlus"
    assert refiner_arch({"decoder.blocks.0.conv1.weight": 0}) == "Unet"

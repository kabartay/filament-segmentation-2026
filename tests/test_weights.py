"""Optional checks on the shipped weights; each test is skipped when its file is absent."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from conftest import WEIGHTS
from filseg.stage2 import CROP_H, CROP_W, load_refiners, predict_crops, refiner_arch
from filseg.tree import TREE_LEVELS, load_tree_scorer

TREE = WEIGHTS / "tree_scorer_1536.pkl"


@pytest.mark.skipif(not TREE.exists(), reason="weights/tree_scorer_1536.pkl not present")
def test_tree_scorer_loads_and_predicts_for_its_own_features() -> None:
    scorer = load_tree_scorer(TREE)
    assert len(scorer.names) > 0 and 0.0 < scorer.tau < 1.0
    assert tuple(scorer.levels) == TREE_LEVELS
    feats = [{name: 0.5 for name in scorer.names}]
    out = scorer.predict(feats)
    assert out.shape == (1,) and np.isfinite(out).all()
    assert scorer.predict([]).shape == (0,)


@pytest.mark.parametrize("name, arch", [("stage2_upp_f0.pt", "UnetPlusPlus"), ("stage2_jit_f0.pt", None)])
def test_stage2_checkpoint_loads_on_cpu(name: str, arch: str | None) -> None:
    path = WEIGHTS / name
    if not path.exists():
        pytest.skip(f"weights/{name} not present")
    state = torch.load(path, map_location="cpu")
    if arch is not None:
        assert refiner_arch(state) == arch
    (model,) = load_refiners([path], "cpu")
    assert not model.training
    x = np.zeros((1, 4, CROP_H, CROP_W), np.uint8)
    out = predict_crops([model], x, "cpu")
    assert out.shape == (1, CROP_H, CROP_W) and np.isfinite(out).all()

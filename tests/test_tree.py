"""Exact instance selection on a component forest (``filseg.tree.antichain``)."""

from __future__ import annotations

import itertools
import random

import numpy as np

from filseg.tree import add_context, antichain, children_of


def is_antichain(nodes: list[int], parent: np.ndarray) -> bool:
    """True if no chosen node is an ancestor of another chosen node."""
    chosen = set(nodes)
    for j in nodes:
        p = int(parent[j])
        while p >= 0:
            if p in chosen:
                return False
            p = int(parent[p])
    return True


def brute_force_best(score: np.ndarray, parent: np.ndarray, tau: float) -> float:
    """Best sum(score - tau) over every antichain (the empty set included), by enumeration."""
    best = 0.0
    n = len(score)
    for r in range(1, n + 1):
        for subset in itertools.combinations(range(n), r):
            if is_antichain(list(subset), parent):
                best = max(best, float(sum(score[j] - tau for j in subset)))
    return best


def test_hand_made_forest() -> None:
    #   0 (0.9)           3 (0.30)
    #   |- 1 (0.5)        4 (0.10)  (root, below tau)
    #   |- 2 (0.6)
    # tau 0.2: root 0 alone gives 0.7; its children give 0.3 + 0.4 = 0.7 (tie, parent kept);
    # node 3 gives 0.1; node 4 is negative.
    parent = np.array([-1, 0, 0, -1, -1])
    score = np.array([0.9, 0.5, 0.6, 0.30, 0.10])
    assert sorted(antichain(score, parent, 0.2)) == [0, 3]


def test_children_preferred_when_they_sum_higher() -> None:
    parent = np.array([-1, 0, 0])
    score = np.array([0.5, 0.45, 0.45])
    # parent: 0.5 - 0.2 = 0.3; children: 0.25 + 0.25 = 0.5
    assert sorted(antichain(score, parent, 0.2)) == [1, 2]


def test_nothing_above_tau_selects_nothing() -> None:
    parent = np.array([-1, 0, -1])
    assert antichain(np.array([0.1, 0.1, 0.05]), parent, 0.26) == []


def test_empty_forest() -> None:
    assert antichain(np.zeros(0), np.zeros(0, dtype=np.int64), 0.26) == []


def test_deep_chain_picks_one_node() -> None:
    parent = np.array([-1, 0, 1, 2])
    score = np.array([0.4, 0.5, 0.9, 0.6])
    assert antichain(score, parent, 0.26) == [2]


def test_matches_brute_force_on_random_forests() -> None:
    rng = random.Random(0)
    for _ in range(60):
        n = rng.randint(1, 10)
        # a parent always has a smaller index than its children, as in tree_candidates
        parent = np.array([-1 if i == 0 or rng.random() < 0.25 else rng.randrange(i) for i in range(n)])
        score = np.array([rng.random() for _ in range(n)])
        tau = rng.choice([0.1, 0.26, 0.4])
        chosen = antichain(score, parent, tau)
        assert is_antichain(chosen, parent)
        got = float(sum(score[j] - tau for j in chosen))
        assert abs(got - brute_force_best(score, parent, tau)) < 1e-9


def test_children_of() -> None:
    assert children_of(np.array([-1, 0, 0, 1, -1])) == [[1, 2], [3], [], [], []]


def test_add_context_features() -> None:
    parent = np.array([-1, 0, 0])
    feats = [{"area": 100}, {"area": 90}, {"area": 10}]
    add_context(feats, parent)
    assert feats[0]["n_children"] == 2
    assert feats[0]["child_area_frac"] == 1.0
    assert abs(feats[0]["max_child_frac"] - 0.9) < 1e-12
    assert feats[1]["n_siblings"] == 1 and feats[0]["n_siblings"] == 0
    assert feats[1]["stable_up"] == 1.0   # keeps 90% >= 85% of its parent's area
    assert feats[2]["stable_up"] == 0.0
    assert feats[0]["stable_down"] == 1.0

"""Phase3 F3: within-group balanced accuracy (GBA)."""

from __future__ import annotations

import numpy as np

from prime.config import FitnessCfg
from prime.fitness.metrics import compute_metrics, group_balanced_accuracies
from prime.fitness.objective import compute_fitness


def _balanced_toy(n_per_cell: int = 12):
    """8 identity groups × 2 labels, equal cells."""
    preds, gold, groups, users = [], [], [], []
    uid = 0
    for g in range(1, 9):
        for y in (0, 1):
            for _ in range(n_per_cell):
                gold.append(y)
                groups.append(g)
                users.append(uid)
                uid += 1
                # Perfect on even groups; always predict 0 on odd groups.
                preds.append(y if g % 2 == 0 else 0)
    return (
        np.asarray(preds),
        np.asarray(gold),
        np.asarray(users),
        np.asarray(groups),
    )


def test_all_zero_gba_is_half():
    # Build equal pos/neg per group so cells are eligible.
    gold, groups, preds = [], [], []
    for g in range(1, 9):
        gold.extend([0] * 10 + [1] * 10)
        groups.extend([g] * 20)
        preds.extend([0] * 20)
    gold = np.asarray(gold)
    groups = np.asarray(groups)
    preds = np.asarray(preds)
    gba = group_balanced_accuracies(preds, gold, groups, min_pos=5, min_neg=5)
    assert gba
    assert all(abs(v - 0.5) < 1e-9 for v in gba.values())


def test_perfect_predictor_gba_one():
    gold, groups = [], []
    for g in range(1, 9):
        gold.extend([0] * 10 + [1] * 10)
        groups.extend([g] * 20)
    gold = np.asarray(gold)
    groups = np.asarray(groups)
    preds = gold.copy()
    gba = group_balanced_accuracies(preds, gold, groups, min_pos=5, min_neg=5)
    assert all(abs(v - 1.0) < 1e-9 for v in gba.values())


def test_compute_metrics_exposes_worst_gba():
    preds, gold, users, groups = _balanced_toy(12)
    m = compute_metrics(
        preds, gold, users, cluster_ids=groups, gba_min_pos=10, gba_min_neg=10
    )
    assert "R_worst_gba" in m
    assert "R_gba_mean" in m
    assert m["R_worst_gba"] < m["R_gba_mean"]
    # Odd groups are silence → GBA=0.5; even are perfect → 1.0
    assert abs(m["R_worst_gba"] - 0.5) < 1e-9


def test_soft_min_lex_on_gba():
    preds, gold, users, groups = _balanced_toy(12)
    cfg = FitnessCfg(
        mode="soft_min_lex",
        group_acc="balanced_within",
        soft_min_tau=0.10,
        epsilon_global=0.01,
        shrink_prior_weight=0.0,
        class_balanced=True,
        gba_min_pos=10,
        gba_min_neg=10,
        fail_closed=False,
        len_penalty_start=10_000,
    )
    out = compute_fitness(preds, gold, users, "word " * 50, cfg, cluster_ids=groups)
    assert "R_soft_min_gba" in out or "R_soft_min_group" in out
    assert out["fitness"] > 0.4
    # Lifting odd groups from silence to perfect should raise fitness.
    preds2 = gold.copy()
    out2 = compute_fitness(preds2, gold, users, "word " * 50, cfg, cluster_ids=groups)
    assert out2["fitness"] > out["fitness"]

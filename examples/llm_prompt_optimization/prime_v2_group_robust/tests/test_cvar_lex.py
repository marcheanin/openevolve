"""cvar_lex fitness mode: lexicographic approximation, no kappa (SPEC v3 §4.4)."""

from __future__ import annotations

import numpy as np
import pytest

from prime.config import FitnessCfg
from prime.fitness.objective import compute_fitness


def _fixture():
    # 3 clusters, cluster 2 broken
    preds = np.array([1, 2, 3, 4, 1, 2, 3, 4, 5, 5, 5, 5])
    gold = np.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 1, 1, 1])
    users = np.arange(12)
    cids = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    return preds, gold, users, cids


def _legacy_cfg(**kw) -> FitnessCfg:
    """Pre-M14 objective: Beta(1,1) smoothing, no class balancing."""
    return FitnessCfg(shrink_prior_weight=0.0, class_balanced=False, **kw)


def test_cvar_lex_uses_shrunk_cvar_plus_epsilon_global():
    preds, gold, users, cids = _fixture()
    cfg = _legacy_cfg(mode="cvar_lex", epsilon_global=0.01, beta_a=1.0, beta_b=1.0)
    res = compute_fitness(preds, gold, users, "p", cfg, cluster_ids=cids)
    # cluster 2 acc: (0+1)/(4+2) = 1/6 -> CVaR (worst 1 of 3) = 1/6
    expected = 1 / 6 + 0.01 * (8 / 12)
    assert res["fitness"] == pytest.approx(expected, abs=1e-6)


def test_grand_mean_shrinkage_pulls_the_tail_toward_the_mean():
    """M14: Beta(1,1) moved a cluster by ~4%; a real prior halves the spread."""
    preds, gold, users, cids = _fixture()
    legacy = compute_fitness(preds, gold, users, "p", _legacy_cfg(), cluster_ids=cids)
    shrunk = compute_fitness(
        preds,
        gold,
        users,
        "p",
        FitnessCfg(shrink_prior_weight=4.0, class_balanced=False),
        cluster_ids=cids,
    )
    raw = list(legacy["cluster_accuracies"].values())
    legacy_spread = max(legacy["cluster_accuracies_shrunk"].values()) - min(
        legacy["cluster_accuracies_shrunk"].values()
    )
    new_spread = max(shrunk["cluster_accuracies_shrunk"].values()) - min(
        shrunk["cluster_accuracies_shrunk"].values()
    )
    assert new_spread < legacy_spread < max(raw) - min(raw) + 1e-9
    # every shrunk value sits between the raw value and the grand mean
    grand = shrunk["R_global"]
    for cid, raw_acc in shrunk["cluster_accuracies"].items():
        val = shrunk["cluster_accuracies_shrunk"][cid]
        assert min(raw_acc, grand) - 1e-9 <= val <= max(raw_acc, grand) + 1e-9


def test_class_balancing_is_neutral_to_a_pure_threshold_shift():
    """C11: raw accuracy rewards moving the 4/5 boundary; balanced accuracy does not."""
    # 8 gold-5 and 2 gold-4 examples: predicting everything 5 wins on raw accuracy.
    gold = np.array([5] * 8 + [4] * 2)
    users = np.arange(10)
    cids = np.array([0] * 5 + [1] * 5)
    all_five = np.array([5] * 10)  # raw 0.80, macro 0.50
    # Lowering the boundary catches both 4s at the cost of three 5s: raw 0.70,
    # macro 0.81. Raw accuracy prefers the first, balanced accuracy the second.
    balanced_pred = np.array([5] * 5 + [4] * 5)

    cfg = FitnessCfg(mode="macro", shrink_prior_weight=0.0, class_balanced=True)
    raw_cfg = FitnessCfg(mode="global", shrink_prior_weight=0.0, class_balanced=False)

    assert compute_fitness(all_five, gold, users, "p", raw_cfg, cluster_ids=cids)["fitness"] > (
        compute_fitness(balanced_pred, gold, users, "p", raw_cfg, cluster_ids=cids)["fitness"]
    )
    assert compute_fitness(balanced_pred, gold, users, "p", cfg, cluster_ids=cids)["fitness"] > (
        compute_fitness(all_five, gold, users, "p", cfg, cluster_ids=cids)["fitness"]
    )


def test_cvar_lex_ignores_kappa():
    preds, gold, users, cids = _fixture()
    # identical vs disagreeing workers should not change fitness in cvar_lex
    wp_same = [preds.copy(), preds.copy()]
    wp_diff = [preds.copy(), np.roll(preds, 1)]
    cfg = FitnessCfg(mode="cvar_lex")
    f_same = compute_fitness(preds, gold, users, "p", cfg, worker_predictions=wp_same, cluster_ids=cids)
    f_diff = compute_fitness(preds, gold, users, "p", cfg, worker_predictions=wp_diff, cluster_ids=cids)
    assert f_same["fitness"] == pytest.approx(f_diff["fitness"])


def test_legacy_cvar_mode_has_no_kappa_term():
    preds, gold, users, cids = _fixture()
    wp_diff = [preds.copy(), np.roll(preds, 1)]
    cfg = FitnessCfg(mode="cvar", w_cvar=0.5, w_global=0.3, w_kappa=0.2)
    res = compute_fitness(preds, gold, users, "p", cfg, worker_predictions=wp_diff, cluster_ids=cids)
    expected = 0.5 * res["CVaR_cluster"] + 0.3 * res["R_global"]
    assert res["fitness"] == pytest.approx(expected, abs=1e-6)


def test_global_mode_unchanged():
    preds, gold, users, cids = _fixture()
    cfg = FitnessCfg(mode="global")
    res = compute_fitness(preds, gold, users, "p", cfg, cluster_ids=cids)
    assert res["fitness"] == pytest.approx(8 / 12)

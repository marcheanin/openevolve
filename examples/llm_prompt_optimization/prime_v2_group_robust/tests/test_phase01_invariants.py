"""Phase 0–1 invariant tests (IMPLEMENTATION.md §6)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from prime.acquisition.eval_sets import EvalSets, build_eval_sets, power_rule_max_k
from prime.config import BudgetCfg, load_config
from prime.data.splits import split_fit_heldout
from prime.data.wilds_loader import ReviewSplit
from prime.experiment.budget import BudgetExhausted, TokenTracker
from prime.fitness.metrics import smoothed_cluster_accuracies
from prime.fitness.objective import compute_fitness
from prime.config import FitnessCfg


def _toy_split(n_users: int = 20, reviews: int = 5, seed: int = 0) -> ReviewSplit:
    rng = np.random.RandomState(seed)
    texts, labels, uids = [], [], []
    for u in range(n_users):
        for _ in range(reviews):
            texts.append("x" * int(rng.randint(10, 100)))
            labels.append(int(rng.randint(1, 6)))
            uids.append(u)
    return ReviewSplit(name="toy", texts=texts, labels=labels, user_ids=uids)


def test_objective_cvar_lex_no_kappa():
    preds = np.array([1, 2, 3, 4, 5, 1, 2, 3])
    gold = np.array([1, 2, 3, 4, 5, 1, 2, 9])  # last wrong
    users = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    clusters = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    cfg = FitnessCfg(
        mode="cvar_lex",
        epsilon_global=0.01,
        beta_a=1.0,
        beta_b=1.0,
        shrink_prior_weight=0.0,
        class_balanced=False,
    )
    out = compute_fitness(preds, gold, users, "short prompt", cfg, cluster_ids=clusters)
    assert "fitness" in out
    assert "CVaR_cluster_shrunk" in out
    # kappa must not enter cvar_lex base
    assert out["base_score"] == pytest.approx(
        out["CVaR_cluster_shrunk"] + 0.01 * out["R_global"], rel=1e-6
    )


def test_class_balanced_cvar_lex_uses_the_balanced_blocks():
    preds = np.array([1, 2, 3, 4, 5, 1, 2, 3])
    gold = np.array([1, 2, 3, 4, 5, 1, 2, 9])
    users = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    clusters = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    cfg = FitnessCfg(
        mode="cvar_lex", epsilon_global=0.01, shrink_prior_weight=4.0, class_balanced=True
    )
    out = compute_fitness(preds, gold, users, "short prompt", cfg, cluster_ids=clusters)
    assert out["base_score"] == pytest.approx(
        out["CVaR_cluster_balanced_shrunk"] + 0.01 * out["R_macro"], rel=1e-6
    )


def test_metrics_smoothing():
    preds = np.array([1, 1, 1, 2])
    gold = np.array([1, 1, 2, 2])
    cids = np.array([0, 0, 0, 1])
    raw = {0: 2 / 3, 1: 1.0}
    shrunk = smoothed_cluster_accuracies(preds, gold, cids, beta_a=1, beta_b=1)
    assert shrunk[0] == pytest.approx((2 + 1) / (3 + 2))
    assert shrunk[1] == pytest.approx((1 + 1) / (1 + 2))
    assert shrunk[0] != pytest.approx(raw[0])


def test_splits_stratified_disjoint():
    split = _toy_split(24, 4, seed=1)
    ss = split_fit_heldout(split, fit_fraction=0.7, seed=7, stratify=True)
    ss.validate()
    assert ss.fit_users.isdisjoint(ss.heldout_users)
    assert ss.strata_balance.get("stratify") is True
    assert ss.strata_balance["n_fit_users"] + ss.strata_balance["n_heldout_users"] == 24


def test_eval_sets_disjoint_and_hash(tmp_path: Path):
    base = _toy_split(16, 5, seed=2)
    split = ReviewSplit(
        name=base.name,
        texts=base.texts,
        labels=base.labels,
        user_ids=base.user_ids,
        example_cluster_ids=[i % 4 for i in range(len(base))],
    )
    preds = list(split.labels)
    diss = [0.0] * len(split)
    sets = build_eval_sets(
        split,
        d_select_size=20,
        d_anchor_size=10,
        d_audit_size=8,
        predictions=preds,
        disagreements=diss,
        seed=3,
    )
    sets.validate()
    assert set(sets.d_select).isdisjoint(sets.d_anchor)
    assert set(sets.d_select).isdisjoint(sets.d_audit)
    assert set(sets.d_anchor).isdisjoint(sets.d_audit)
    path = tmp_path / "eval_sets.json"
    sets.save(path)
    loaded = EvalSets.load(path, verify_hash=True)
    assert loaded.d_select == sets.d_select
    assert loaded.content_hash == sets.content_hash


def test_budget_tracker_levels_and_exhaustion():
    tr = TokenTracker.from_cfg(BudgetCfg(total_calls=5, on_exhausted="stop"))
    tr.charge("e0_infer", n_calls=3, tokens=10)
    assert tr.remaining() == 2
    with pytest.raises(BudgetExhausted):
        tr.charge("val", n_calls=3)
    snap = tr.snapshot()
    assert snap["calls_by_level"]["e0_infer"] == 3
    assert snap["exhausted"] is True


def test_power_rule():
    assert power_rule_max_k(400, 40) == 10
    assert power_rule_max_k(200, 40) == 5


def test_base_v3_config_loads():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "base_v3.yaml")
    assert cfg.fitness.mode == "cvar_lex"
    assert cfg.ensemble.aggregation == "median"
    assert cfg.data_roles.d_audit_size == 100
    assert cfg.clusters.control == "none"
    assert cfg.budget.total_calls == 60000

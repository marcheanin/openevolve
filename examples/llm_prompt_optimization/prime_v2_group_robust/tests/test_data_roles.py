"""Tests for SPEC v3 data roles: fit/heldout split, D_select, D_anchor, smoothing, median."""

from __future__ import annotations

import numpy as np
import pytest

from prime.acquisition.eval_sets import (
    EvalSets,
    build_d_anchor,
    build_d_select,
    power_rule_max_k,
)
from prime.data.splits import split_fit_heldout
from prime.data.wilds_loader import ReviewSplit
from prime.fitness.metrics import (
    cvar_from_accuracies,
    smoothed_cluster_accuracies,
)
from prime.workers.ensemble import MedianAggregator, build_aggregator


def _make_split(n_users: int = 20, per_user: int = 5) -> ReviewSplit:
    texts, labels, users, cids = [], [], [], []
    for u in range(n_users):
        for j in range(per_user):
            texts.append(f"review u{u} #{j}")
            labels.append(1 + (u + j) % 5)
            users.append(u)
            cids.append(u % 3)
    return ReviewSplit(
        name="train", texts=texts, labels=labels, user_ids=users, example_cluster_ids=cids
    )


class TestFitHeldoutSplit:
    def test_user_disjoint_and_ratio(self):
        split = _make_split(n_users=20)
        result = split_fit_heldout(split, fit_fraction=0.7, seed=42)
        assert not (result.fit_users & result.heldout_users)
        assert len(result.fit_users) == 14
        assert len(result.heldout_users) == 6
        # examples of one user entirely on one side
        assert set(result.fit.user_ids) == result.fit_users
        assert set(result.heldout.user_ids) == result.heldout_users

    def test_deterministic_by_seed(self):
        split = _make_split()
        a = split_fit_heldout(split, 0.7, seed=1)
        b = split_fit_heldout(split, 0.7, seed=1)
        c = split_fit_heldout(split, 0.7, seed=2)
        assert a.fit_users == b.fit_users
        assert a.fit_users != c.fit_users

    def test_rejects_single_user(self):
        split = _make_split(n_users=1)
        with pytest.raises(ValueError):
            split_fit_heldout(split, 0.7, seed=0)


class TestDSelect:
    def test_stratified_and_sized(self):
        split = _make_split(n_users=30, per_user=4)
        idx = build_d_select(split, size=60, seed=0)
        assert len(idx) == 60
        assert len(set(idx)) == 60
        picked_groups = {split.example_cluster_ids[i] for i in idx}
        assert picked_groups == {0, 1, 2}  # every non-empty group represented

    def test_size_capped_by_pool(self):
        split = _make_split(n_users=4, per_user=2)
        idx = build_d_select(split, size=100, seed=0)
        assert len(idx) == 8


class TestDAnchor:
    def test_confident_cells_disjoint_from_select(self):
        split = _make_split(n_users=20, per_user=5)
        n = len(split)
        d_select = build_d_select(split, size=30, seed=0)
        predictions = list(split.labels)  # everything solved
        disagreements = [0.0] * n
        anchors = build_d_anchor(
            split, d_select, predictions, disagreements, size=20, seed=0
        )
        assert len(anchors) == 20
        assert not (set(anchors) & set(d_select))

    def test_excludes_errors_and_high_disagreement(self):
        split = _make_split(n_users=10, per_user=4)
        n = len(split)
        predictions = [-1] * n  # nothing predicted correctly
        anchors = build_d_anchor(split, [], predictions, [0.0] * n, size=10, seed=0)
        assert anchors == []

    def test_eval_sets_validate(self):
        with pytest.raises(ValueError):
            EvalSets(d_select=[1, 2], d_anchor=[2, 3]).validate()


class TestPowerRule:
    def test_power_rule(self):
        assert power_rule_max_k(400, 40) == 10
        assert power_rule_max_k(200, 40) == 5
        assert power_rule_max_k(50, 40) == 2  # never below 2


class TestSmoothing:
    def test_beta_smoothing_shrinks_small_groups(self):
        preds = np.array([1, 1, 1, 1, 2])
        gold = np.array([1, 1, 1, 1, 1])
        cids = np.array([0, 0, 0, 0, 1])
        accs = smoothed_cluster_accuracies(preds, gold, cids, beta_a=1.0, beta_b=1.0)
        # group 0: 4/4 -> (4+1)/(4+2) = 0.833; group 1: 0/1 -> (0+1)/(1+2) = 0.333
        assert accs[0] == pytest.approx(5 / 6)
        assert accs[1] == pytest.approx(1 / 3)

    def test_cvar_from_accs(self):
        accs = {0: 0.9, 1: 0.5, 2: 0.7}
        # ceil(3 * 0.33) = 1 worst group
        assert cvar_from_accuracies(accs, quantile=0.33) == pytest.approx(0.5)
        # ceil(3 * 0.34) = 2 worst groups
        assert cvar_from_accuracies(accs, quantile=0.34) == pytest.approx(0.6)


class TestMedianAggregation:
    def test_median_odd(self):
        assert MedianAggregator().aggregate([1, 5, 3]) == 3

    def test_median_even_lower(self):
        assert MedianAggregator().aggregate([2, 4]) == 2

    def test_median_resists_outlier(self):
        # majority would tie 4 vs 4; median stays near consensus
        assert MedianAggregator().aggregate([4, 4, 1]) == 4

    def test_build_aggregator_modes(self):
        assert isinstance(build_aggregator("median"), MedianAggregator)
        assert build_aggregator("majority").aggregate([1, 1, 5]) == 1

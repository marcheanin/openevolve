"""Phase3 F1/F2: fail-closed parsing and degenerate fitness guards."""

from __future__ import annotations

import numpy as np

from prime.config import FitnessCfg
from prime.fitness.objective import REJECT_FITNESS, compute_fitness
from prime.workers.ensemble import INVALID, parse_binary, parse_label


def test_parse_binary_legacy_defaults_to_zero():
    assert parse_binary("???") == 0
    assert parse_binary("garbage", fail_closed=False) == 0


def test_parse_binary_fail_closed_returns_invalid():
    assert parse_binary("???", fail_closed=True) == INVALID
    assert parse_label("no label here", "binary", fail_closed=True) == INVALID


def test_parse_binary_still_reads_label():
    assert parse_binary("Label: 1", fail_closed=True) == 1
    assert parse_binary("Label: 0", fail_closed=True) == 0


def test_all_invalid_rejected_under_fail_closed():
    n = 40
    preds = np.full(n, INVALID, dtype=np.int16)
    gold = np.zeros(n, dtype=np.int16)
    gold[:4] = 1
    users = np.arange(n)
    cids = np.array([1 + (i % 8) for i in range(n)], dtype=np.int16)
    # Ensure enough pos/neg per group for GBA eligibility is not required for reject.
    cfg = FitnessCfg(
        mode="global",
        fail_closed=True,
        max_invalid_rate=0.02,
        min_pred_pos_rate=0.02,
        len_penalty_start=10_000,
        class_balanced=False,
    )
    out = compute_fitness(preds, gold, users, "p", cfg, cluster_ids=cids)
    assert out["fitness"] == REJECT_FITNESS
    assert out["reject_reason"] == "invalid_rate"
    assert out["invalid_rate"] == 1.0


def test_all_zeros_rejected_as_degenerate():
    n = 80
    preds = np.zeros(n, dtype=np.int16)
    gold = np.zeros(n, dtype=np.int16)
    gold[::10] = 1
    users = np.arange(n)
    cids = np.array([1 + (i % 8) for i in range(n)], dtype=np.int16)
    cfg = FitnessCfg(
        mode="global",
        fail_closed=True,
        max_invalid_rate=0.02,
        min_pred_pos_rate=0.02,
        len_penalty_start=10_000,
        class_balanced=False,
    )
    out = compute_fitness(preds, gold, users, "p", cfg, cluster_ids=cids)
    assert out["fitness"] == REJECT_FITNESS
    assert out["reject_reason"] == "degenerate"


def test_honest_predictor_not_rejected():
    n = 80
    gold = np.zeros(n, dtype=np.int16)
    gold[::5] = 1
    preds = gold.copy()
    users = np.arange(n)
    cids = np.array([1 + (i % 8) for i in range(n)], dtype=np.int16)
    cfg = FitnessCfg(
        mode="global",
        fail_closed=True,
        max_invalid_rate=0.02,
        min_pred_pos_rate=0.02,
        len_penalty_start=10_000,
        class_balanced=False,
    )
    out = compute_fitness(preds, gold, users, "p", cfg, cluster_ids=cids)
    assert out["reject_reason"] is None
    assert out["fitness"] > 0.9


def test_empty_gba_rejected_under_balanced_within():
    # Tiny cells → no eligible GBA groups → must reject, not fall back to global.
    preds = np.array([0, 1, 0, 1], dtype=np.int16)
    gold = np.array([0, 1, 0, 1], dtype=np.int16)
    users = np.arange(4)
    cids = np.array([1, 1, 2, 2], dtype=np.int16)
    cfg = FitnessCfg(
        mode="soft_min_lex",
        group_acc="balanced_within",
        gba_min_pos=10,
        gba_min_neg=10,
        fail_closed=False,
        soft_min_tau=0.1,
        shrink_prior_weight=0.0,
        class_balanced=True,
        len_penalty_start=10_000,
    )
    out = compute_fitness(preds, gold, users, "p", cfg, cluster_ids=cids)
    assert out["fitness"] == REJECT_FITNESS
    assert out["reject_reason"] == "gba_empty"

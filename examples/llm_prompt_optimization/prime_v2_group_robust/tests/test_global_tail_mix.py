"""global_tail_mix: 0.5 R_global + 0.5 R_tail - length."""

from __future__ import annotations

import numpy as np
import pytest

from prime.config import FitnessCfg, load_config
from prime.fitness.metrics import tail_user_accuracy
from prime.fitness.objective import compute_fitness, length_penalty
from pathlib import Path


def test_global_tail_mix_formula():
    # 10 users, 1 review each: first 2 wrong → R_global=0.8; worst 20% tail = 0.0
    preds = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    gold = np.array([2, 2, 2, 2, 3, 3, 4, 4, 5, 5])
    users = np.arange(10)
    cfg = FitnessCfg(
        mode="global_tail_mix",
        w_global_mix=0.5,
        w_tail=0.5,
        tail_quantile=0.2,
        shrink_prior_weight=0.0,
        class_balanced=False,
        len_penalty_start=10_000,
    )
    res = compute_fitness(preds, gold, users, "short prompt", cfg)
    r_global = 0.8
    r_tail = tail_user_accuracy(preds, gold, users, quantile=0.2)
    assert r_tail == pytest.approx(0.0)
    expected = 0.5 * r_global + 0.5 * r_tail
    assert res["fitness"] == pytest.approx(expected)
    assert res["R_global"] == pytest.approx(r_global)
    assert res["R_tail"] == pytest.approx(r_tail)


def test_global_tail_mix_applies_length_penalty():
    preds = np.ones(8, dtype=int)
    gold = np.ones(8, dtype=int)
    users = np.arange(8)
    prompt = "word " * 2500  # over len_penalty_start=2000
    cfg = FitnessCfg(
        mode="global_tail_mix",
        shrink_prior_weight=0.0,
        class_balanced=False,
        len_penalty_start=2000,
        len_penalty_per_100=0.01,
    )
    res = compute_fitness(preds, gold, users, prompt, cfg)
    pen = length_penalty(prompt, cfg)
    assert pen > 0
    assert res["fitness"] == pytest.approx(1.0 - pen)


def test_e2_config_loads_global_tail_mix():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "experiments" / "E2_cheap_ensemble_global_tail" / "config.yaml")
    assert cfg.fitness.mode == "global_tail_mix"
    assert cfg.fitness.w_global_mix == 0.5
    assert cfg.fitness.w_tail == 0.5
    assert cfg.evolution.n_evolve_iterations == 15
    assert cfg.evolution.mutator_model == "deepseek/deepseek-v4-pro"
    assert [w.name for w in cfg.ensemble.workers] == [
        "openai/gpt-4o-mini",
        "google/gemini-2.5-flash-lite",
        "qwen/qwen3-32b",
    ]
    assert cfg.dataset.cluster_artifact is None
    assert cfg.consolidation.enabled is False
    assert cfg.data_roles.anchor_gate_mode == "reject"


def test_e2_category_config_loads():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "experiments" / "E2_category_shift_books" / "config.yaml")
    assert cfg.dataset.train_category_id == 0
    assert cfg.dataset.eval_exclude_category_ids == [0]
    assert cfg.fitness.mode == "global_tail_mix"

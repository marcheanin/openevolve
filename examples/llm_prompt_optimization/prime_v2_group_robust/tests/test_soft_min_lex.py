"""soft_min_lex fitness + selection-key alignment (post-M28 form)."""

from __future__ import annotations

import numpy as np

from prime.config import FitnessCfg, load_config
from prime.fitness.objective import compute_fitness, soft_min_accuracies


def _toy():
    preds = np.array([1, 1, 1, 1, 0, 1, 0, 0, 0], dtype=np.int16)
    gold = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int16)
    users = np.arange(9)
    cids = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int16)
    return preds, gold, users, cids


def test_soft_min_between_min_and_mean():
    accs = {0: 1.0, 1: 2 / 3, 2: 0.0}
    hard = soft_min_accuracies(accs, tau=0.0)
    soft = soft_min_accuracies(accs, tau=0.08)
    mean = float(np.mean(list(accs.values())))
    assert abs(hard - 0.0) < 1e-12
    assert hard < soft < mean


def test_soft_min_lex_above_hard_min_when_floor_present():
    preds, gold, users, cids = _toy()
    cfg_hard = FitnessCfg(
        mode="min_group_lex",
        epsilon_global=0.0,
        shrink_prior_weight=0.0,
        class_balanced=False,
        len_penalty_start=10_000,
    )
    cfg_soft = FitnessCfg(
        mode="soft_min_lex",
        soft_min_tau=0.08,
        epsilon_global=0.0,
        shrink_prior_weight=0.0,
        class_balanced=False,
        len_penalty_start=10_000,
    )
    f_h = compute_fitness(preds, gold, users, "p", cfg_hard, cluster_ids=cids)
    f_s = compute_fitness(preds, gold, users, "p", cfg_soft, cluster_ids=cids)
    assert abs(f_h["R_worst_group"] - 0.0) < 1e-9
    assert abs(f_s["R_worst_group"] - 0.0) < 1e-9  # headline still hard min
    assert f_s["R_soft_min_group"] > f_h["fitness"]
    assert f_s["fitness"] == f_s["R_soft_min_group"]


def test_soft_min_rewards_lifting_floor():
    preds_bad, gold, users, cids = _toy()
    preds_fix = preds_bad.copy()
    preds_fix[6:] = 1
    cfg = FitnessCfg(
        mode="soft_min_lex",
        soft_min_tau=0.08,
        epsilon_global=0.01,
        shrink_prior_weight=0.0,
        class_balanced=False,
        len_penalty_start=10_000,
    )
    f_bad = compute_fitness(preds_bad, gold, users, "p", cfg, cluster_ids=cids)
    f_fix = compute_fitness(preds_fix, gold, users, "p", cfg, cluster_ids=cids)
    assert f_fix["fitness"] > f_bad["fitness"]
    assert f_fix["R_worst_group"] > f_bad["R_worst_group"]


def test_length_penalty_fires_above_seed_budget():
    preds, gold, users, cids = _toy()
    short = "word " * 211
    long = "word " * 389
    cfg = FitnessCfg(
        mode="soft_min_lex",
        soft_min_tau=0.08,
        epsilon_global=0.0,
        shrink_prior_weight=0.0,
        class_balanced=False,
        len_penalty_start=250,
        len_penalty_per_100=0.025,
    )
    f_s = compute_fitness(preds, gold, users, short, cfg, cluster_ids=cids)
    f_l = compute_fitness(preds, gold, users, long, cfg, cluster_ids=cids)
    assert f_s["length_penalty"] == 0.0
    assert f_l["length_penalty"] > 0.0
    assert f_l["fitness"] < f_s["fitness"]


def test_soft_min_config_loads():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "experiments/E4_civilcomments/config_arm_b_soft_min_2x20.yaml")
    assert cfg.fitness.mode == "soft_min_lex"
    assert cfg.fitness.soft_min_tau == 0.08
    assert cfg.data_roles.d_select_rotate is True
    assert cfg.fitness.len_penalty_start == 250
    assert cfg.active_learning.n_cycles == 2
    assert cfg.evolution.n_evolve_iterations == 20


def test_selection_key_soft_min():
    from pathlib import Path

    from prime.controller import PrimeController

    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "experiments/E4_civilcomments/config_arm_b_soft_min_2x20.yaml")

    class Stub:
        pass

    stub = Stub()
    stub.cfg = cfg
    key = PrimeController._selection_key(
        stub,
        {
            "R_global": 0.7,
            "R_macro": 0.65,
            "R_soft_min_group": 0.58,
            "R_worst_group": 0.55,
        },
    )
    assert key == (0.58, 0.65)

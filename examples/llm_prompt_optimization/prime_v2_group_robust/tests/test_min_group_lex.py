"""min_group_lex fitness + selection-key alignment (Phase 2c arm B)."""

from __future__ import annotations

import numpy as np

from prime.config import FitnessCfg, load_config
from prime.fitness.objective import compute_fitness


def _toy():
    # 9 examples, 3 groups. Group 0 perfect, group 1 mixed, group 2 all wrong.
    preds = np.array([1, 1, 1, 1, 0, 1, 0, 0, 0], dtype=np.int16)
    gold = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int16)
    users = np.arange(9)
    cids = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int16)
    return preds, gold, users, cids


def test_min_group_lex_uses_min_not_mean():
    preds, gold, users, cids = _toy()
    cfg = FitnessCfg(mode="min_group_lex", epsilon_global=0.0, shrink_prior_weight=0.0, class_balanced=False)
    res = compute_fitness(preds, gold, users, "p", cfg, cluster_ids=cids)
    # Acc: g0=1, g1=2/3, g2=0 → min = 0
    assert abs(res["R_worst_group"] - 0.0) < 1e-9
    assert abs(res["fitness"] - 0.0) < 1e-6
    assert abs(res["R_global"] - (5 / 9)) < 1e-9


def test_min_group_lex_beats_global_when_tail_lifts():
    preds_bad, gold, users, cids = _toy()
    # Fix group 2 only (tail): three flips 0→1
    preds_fix = preds_bad.copy()
    preds_fix[6:] = 1
    cfg = FitnessCfg(mode="min_group_lex", epsilon_global=0.01, shrink_prior_weight=0.0, class_balanced=False)
    f_bad = compute_fitness(preds_bad, gold, users, "p", cfg, cluster_ids=cids)
    f_fix = compute_fitness(preds_fix, gold, users, "p", cfg, cluster_ids=cids)
    assert f_fix["R_worst_group"] > f_bad["R_worst_group"]
    assert f_fix["fitness"] > f_bad["fitness"]


def test_min_group_lex_differs_from_cvar_when_one_group_is_floor():
    preds, gold, users, cids = _toy()
    cfg_min = FitnessCfg(mode="min_group_lex", epsilon_global=0.0, shrink_prior_weight=0.0, class_balanced=False, cvar_quantile=0.33)
    cfg_cvar = FitnessCfg(mode="cvar_lex", epsilon_global=0.0, shrink_prior_weight=0.0, class_balanced=False, cvar_quantile=0.33)
    # With K=3 and q=0.33, CVaR averages ceil(0.99)=1 cluster → but validate requires >=2.
    # Use q=0.67 so CVaR averages 2 worst groups: (0 + 2/3)/2 = 1/3 > min=0.
    cfg_cvar.cvar_quantile = 0.67
    f_min = compute_fitness(preds, gold, users, "p", cfg_min, cluster_ids=cids)
    f_cvar = compute_fitness(preds, gold, users, "p", cfg_cvar, cluster_ids=cids)
    assert f_min["fitness"] < f_cvar["base_score"] - 1e-9 or f_min["R_worst_group"] < f_cvar["CVaR_cluster_shrunk"]


def test_arm_configs_load():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for name in (
        "config_arm_a_global.yaml",
        "config_arm_b_min_group.yaml",
        "config_arm_c_style.yaml",
    ):
        cfg = load_config(root / "experiments" / "E4_civilcomments" / name)
        assert cfg.dataset.name == "civilcomments"
        assert cfg.dataset.label_space == "binary"
        assert cfg.data_roles.anchor_gate_mode == "reject"
    a = load_config(root / "experiments/E4_civilcomments/config_arm_a_global.yaml")
    b = load_config(root / "experiments/E4_civilcomments/config_arm_b_min_group.yaml")
    c = load_config(root / "experiments/E4_civilcomments/config_arm_c_style.yaml")
    assert a.fitness.mode == "global"
    assert b.fitness.mode == "min_group_lex"
    assert c.fitness.mode == "min_group_lex"
    assert a.clusters.geometry == "oracle"
    assert b.clusters.geometry == "oracle"
    assert c.clusters.geometry == "style"


def test_selection_key_respects_mode():
    from dataclasses import replace
    from prime.controller import PrimeController

    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    cfg = load_config(root / "experiments/E4_civilcomments/config_arm_a_global.yaml")
    # Avoid constructing full controller filesystem — call unbound method with stub.
    class Stub:
        pass

    stub = Stub()
    stub.cfg = cfg
    key_a = PrimeController._selection_key(stub, {"R_global": 0.7, "R_macro": 0.65, "CVaR_cluster": 0.4, "CVaR_cluster_shrunk": 0.5, "R_worst_group": 0.55})
    assert key_a == (0.65,)  # class_balanced → R_macro

    stub.cfg = load_config(root / "experiments/E4_civilcomments/config_arm_b_min_group.yaml")
    key_b = PrimeController._selection_key(
        stub,
        {
            "R_global": 0.7,
            "R_macro": 0.65,
            "CVaR_cluster_shrunk": 0.5,
            "R_worst_group": 0.55,
        },
    )
    assert key_b == (0.55, 0.65)

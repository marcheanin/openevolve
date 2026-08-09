"""v1_weighted must score the Hard/Anchor active batch, not D_select."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from prime.config import FitnessCfg, load_config
from prime.evolution.openevolve_adapter import CandidateEvaluator
from prime.fitness.objective import compute_fitness


def test_v1_weighted_formula_matches_hard_anchor_kappa():
    preds = np.array([1, 1, 2, 2, 3, 3])
    gold = np.array([1, 0, 2, 0, 3, 3])  # hard: 2/3 correct? indices 0,1,2 hard
    # hard mask: first 3
    hard = np.array([True, True, True, False, False, False])
    # Acc_Hard = 2/3 (1==1, 1!=0, 2==2), Acc_Anchor = 2/3 (2!=0, 3==3, 3==3) wait gold[3]=0 pred[3]=2
    # Acc_Anchor on [3,4,5]: 2==0? no, 3==3 yes, 3==3 yes → 2/3
    users = np.arange(6)
    wp = [preds.copy(), preds.copy(), preds.copy()]
    cfg = FitnessCfg(mode="v1_weighted", class_balanced=False, shrink_prior_weight=0.0)
    res = compute_fitness(preds, gold, users, "short prompt", cfg, worker_predictions=wp, hard_mask=hard)
    assert "Acc_Hard" in res and "Acc_Anchor" in res
    assert res["Acc_Hard"] == np.mean(preds[hard] == gold[hard])
    assert res["Acc_Anchor"] == np.mean(preds[~hard] == gold[~hard])
    expected = 0.5 * res["Acc_Hard"] + 0.3 * res["Acc_Anchor"] + 0.2 * max(0.0, res["mean_kappa"])
    assert abs(res["base_score"] - expected) < 1e-9


def test_evaluate_prompt_routes_v1_weighted_to_batch(tmp_path: Path):
    cfg_path = Path(__file__).resolve().parents[1] / "experiments" / "E1_constraint_global" / "config.yaml"
    cfg = load_config(cfg_path)
    cfg.fitness.mode = "v1_weighted"
    cfg.experiment.force_mock = True

    pool_texts = [f"review {i}" for i in range(10)]
    pool_labels = [5, 4, 3, 2, 1, 5, 4, 3, 2, 1]
    pool_users = list(range(10))
    pool_clusters = [0] * 10
    batch = {
        "indices": [0, 1, 2, 3, 4, 5],
        "hard_indices": [0, 1, 2, 3],
        "anchor_indices": [4, 5],
    }
    select_data = {
        "texts": ["heldout a", "heldout b"],
        "labels": [5, 5],
        "user_ids": [100, 101],
        "cluster_ids": [0, 1],
        "example_ids": [1, 2],
    }
    ev = CandidateEvaluator(
        cfg,
        pool_texts,
        pool_labels,
        pool_users,
        pool_clusters,
        active_batch=batch,
        use_mock=True,
        select_data=select_data,
        cache_dir=tmp_path / "cache",
    )
    out = ev.evaluate_prompt("Rate the review.\n{review}")
    assert out.get("eval_set") == "active_batch_v1"
    assert out.get("n_eval_examples") == 6
    assert out.get("n_hard") == 4
    assert "Acc_Hard" in out
    assert "Acc_Anchor" in out


def test_evaluate_prompt_global_still_uses_d_select(tmp_path: Path):
    cfg_path = Path(__file__).resolve().parents[1] / "experiments" / "E1_constraint_global" / "config.yaml"
    cfg = load_config(cfg_path)
    cfg.fitness.mode = "global"
    cfg.experiment.force_mock = True

    batch = {"indices": [0, 1], "hard_indices": [0], "anchor_indices": [1]}
    select_data = {
        "texts": ["a", "b", "c"],
        "labels": [5, 4, 3],
        "user_ids": [1, 2, 3],
        "cluster_ids": [0, 0, 1],
        "example_ids": [10, 11, 12],
    }
    ev = CandidateEvaluator(
        cfg,
        ["p0", "p1"],
        [5, 4],
        [1, 2],
        [0, 0],
        active_batch=batch,
        use_mock=True,
        select_data=select_data,
        cache_dir=tmp_path / "cache",
    )
    out = ev.evaluate_prompt("Rate.\n{review}")
    assert out.get("eval_set") == "d_select"
    assert out.get("n_eval_examples") == 3

"""E5 config loads and validates."""

from __future__ import annotations

from pathlib import Path

from prime.config import load_config


def test_e5_prime_main_config():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "experiments/E5_civilcomments/config_prime_main.yaml")
    assert cfg.ensemble.mode == "single"
    assert len(cfg.ensemble.workers) == 1
    assert cfg.ensemble.workers[0].name == "google/gemma-3-12b-it"
    assert cfg.ensemble.fail_closed is True
    assert cfg.fitness.fail_closed is True
    assert cfg.fitness.mode == "soft_min_lex"
    assert cfg.fitness.group_acc == "balanced_within"
    assert cfg.fitness.soft_min_tau == 0.10
    assert cfg.data_roles.d_select_rotate is False
    assert cfg.data_roles.dev_gate_mode == "reject"
    assert cfg.data_roles.d_dev_size == 900
    assert cfg.data_roles.balanced_cells is True
    assert cfg.acquisition.expansion_policy == "uncertainty"
    assert cfg.active_learning.n_cycles == 4
    assert cfg.budget.scorer_calls == 60000
    assert cfg.openevolve_config_path.endswith("openevolve_e5_civilcomments.yaml")
    assert cfg.active_learning.selection_split == "d_dev"

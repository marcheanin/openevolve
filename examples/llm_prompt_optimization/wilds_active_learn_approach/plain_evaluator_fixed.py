"""
OpenEvolve evaluator for the "plain evolution" ablation.

Differences vs the AL evaluator:
- No active_batch.json.
- Fitness is computed on a fixed TRAIN subset using the same ensemble + unified metrics.

Inputs:
- Config YAML: env WILDS_ACTIVE_LEARN_CONFIG (same as existing evaluator.py)
- Fixed splits manifest: env WILDS_FIXED_SPLITS_PATH (JSON produced by fixed_splits.py)

Output:
EvaluationResult with metrics including: R_global, R_worst, mae, mean_kappa, combined_score.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
WILDS_EXPERIMENT = SCRIPT_DIR.parent / "wilds_experiment"
EXPERIMENTS_ROOT = WILDS_EXPERIMENT / "experiments"
OPENEVOLVE_PKG_ROOT = SCRIPT_DIR.parent.parent.parent

for p in (OPENEVOLVE_PKG_ROOT, WILDS_EXPERIMENT, EXPERIMENTS_ROOT, SCRIPT_DIR):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from experiments.metrics import compute_metrics, compute_combined_score_unified
from openevolve.evaluation_result import EvaluationResult

def _import_al_evaluator():
    """
    Import wilds_active_learn_approach/evaluator.py by absolute path to avoid
    collisions with wilds_experiment/evaluator.py (same module name).
    """
    import importlib.util

    p = (SCRIPT_DIR / "evaluator.py").resolve()
    spec = importlib.util.spec_from_file_location("wilds_active_learn_evaluator", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_AL_EVAL = _import_al_evaluator()
_build_workers = _AL_EVAL._build_workers
_parallel_predict = _AL_EVAL._parallel_predict
from fixed_splits import load_fixed_splits


def _load_config() -> dict:
    env_path = os.environ.get("WILDS_ACTIVE_LEARN_CONFIG")
    if env_path and Path(env_path).is_file():
        path = Path(env_path)
    else:
        path = SCRIPT_DIR / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_CACHE: Dict[str, Tuple[List[str], np.ndarray, np.ndarray]] = {}


def _load_train_fixed(cfg: dict) -> Tuple[List[str], np.ndarray, np.ndarray]:
    p = os.environ.get("WILDS_FIXED_SPLITS_PATH")
    if not p:
        raise RuntimeError("WILDS_FIXED_SPLITS_PATH is not set (need fixed splits manifest).")
    pth = Path(p)
    key = str(pth.resolve())
    if key in _CACHE:
        return _CACHE[key]
    fixed = load_fixed_splits(pth, config_path=os.environ.get("WILDS_ACTIVE_LEARN_CONFIG"))
    texts = fixed.texts_by_split["train"]
    labels = fixed.labels_by_split["train"]
    user_ids = fixed.user_ids_by_split["train"]
    _CACHE[key] = (texts, labels, user_ids)
    return texts, labels, user_ids


def evaluate(program: str) -> EvaluationResult:
    """
    Evaluate one prompt program on fixed TRAIN subset.
    """
    # OpenEvolve passes a *program path* to the evaluation function.
    # For our task, the program is the prompt text. Resolve to actual prompt string.
    prompt_text = program
    try:
        p = Path(str(program))
        if p.exists() and p.is_file():
            prompt_text = p.read_text(encoding="utf-8")
    except Exception:
        pass

    cfg = _load_config()
    workers = _build_workers(cfg)
    max_parallel = cfg.get("worker_defaults", {}).get("max_parallel", 8)

    texts, labels, user_ids = _load_train_fixed(cfg)
    # predict
    predictions, worker_preds = _parallel_predict(workers, list(texts), prompt_text, max_parallel=max_parallel)

    pred_arr = np.asarray(predictions, dtype=np.int64)
    lab_arr = np.asarray(labels, dtype=np.int64)
    user_arr = np.asarray(user_ids, dtype=np.int64)
    wp_arr = [np.asarray(wp, dtype=np.int64) for wp in worker_preds]

    metrics = compute_metrics(pred_arr, lab_arr, user_arr, worker_predictions=wp_arr)
    metrics["combined_score"] = compute_combined_score_unified(metrics, is_ensemble=True)

    # Feature dimensions required by config (MAP-Elites):
    # - prompt_length: raw continuous value (token estimate); database will scale/bin it.
    # - Acc_Hard: computed on the fixed train subset using the same rule as AL.
    from data_manager import disagreement_score

    uncertainty_threshold = float(cfg.get("active_learning", {}).get("uncertainty_threshold", 0.0) or 0.0)
    hard_correct = hard_total = 0
    anchor_correct = anchor_total = 0

    for i in range(len(pred_arr)):
        gold = int(lab_arr[i])
        pred = int(pred_arr[i])
        votes = [int(wp[i]) for wp in wp_arr]
        ds = float(disagreement_score(votes))
        is_hard = (pred != gold) or (ds > uncertainty_threshold)
        is_correct = (pred == gold)
        if is_hard:
            hard_total += 1
            if is_correct:
                hard_correct += 1
        else:
            anchor_total += 1
            if is_correct:
                anchor_correct += 1

    metrics["Acc_Hard"] = (hard_correct / hard_total) if hard_total else 0.0
    metrics["Acc_Anchor"] = (anchor_correct / anchor_total) if anchor_total else 0.0
    metrics["n_hard"] = float(hard_total)
    metrics["n_anchor"] = float(anchor_total)
    metrics["prompt_length"] = float(len(prompt_text) // 4)

    # OpenEvolve uses metrics["combined_score"] as the primary optimization signal.
    return EvaluationResult(metrics=metrics)


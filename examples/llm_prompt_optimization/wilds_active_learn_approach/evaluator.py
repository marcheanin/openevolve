"""
Modified evaluator for Active Prompt Evolution.

Supports:
- Active batch evaluation (reads active_batch.json)
- Weighted fitness: w1*Acc_Hard + w2*Acc_Anchor + w3*kappa - P_Len
- evaluate_fast on active batch, evaluate_full on validation
- Per-example results for DataManager
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
WILDS_EXPERIMENT = SCRIPT_DIR.parent / "wilds_experiment"
EXPERIMENTS_ROOT = WILDS_EXPERIMENT / "experiments"
if str(WILDS_EXPERIMENT) not in sys.path:
    sys.path.insert(0, str(WILDS_EXPERIMENT))
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from workers import LLMWorker
from collections import Counter
from experiments.metrics import compute_metrics, compute_combined_score_unified

# Fitness weights (from plan)
W1_ACC_HARD = 0.5
W2_ACC_ANCHOR = 0.3
W3_KAPPA = 0.2
PROMPT_LEN_LIMIT = 2000
PENALTY_PER_100_TOKENS = 0.01


class MajorityVoteAggregator:
    def aggregate(self, worker_predictions: List[int]) -> int:
        return Counter(worker_predictions).most_common(1)[0][0]


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


def _load_config() -> dict:
    with open(SCRIPT_DIR / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_active_batch() -> Optional[Dict[str, Any]]:
    """Load active batch from active_batch.json if present."""
    for base in [SCRIPT_DIR, Path.cwd()]:
        path = base / "active_batch.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


# Кэш загрузки сплитов: при каждой оценке кандидата OpenEvolve вызывается evaluate(),
# и без кэша каждый раз читался бы .pkl с диска. Ключ: (split_name, max_users) для согласованности с config.
_SPLIT_DATA_CACHE: Dict[Tuple[str, Optional[int]], Tuple[List[str], np.ndarray, np.ndarray]] = {}


def _load_split_data(config: dict, split_name: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Load data for a split. Result is cached so repeated calls (e.g. every evaluate() in evolution) don't hit disk."""
    ds_cfg = config.get("dataset", {})
    max_users = ds_cfg.get("max_train_users" if split_name == "train" else "max_val_users")
    if max_users is not None and max_users <= 0:
        max_users = None
    cache_key = (split_name, max_users)
    if cache_key in _SPLIT_DATA_CACHE:
        return _SPLIT_DATA_CACHE[cache_key]

    import importlib.util
    base_eval_path = WILDS_EXPERIMENT / "evaluator.py"
    spec = importlib.util.spec_from_file_location("wilds_base_evaluator", base_eval_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    load_preprocessed_data = mod.load_preprocessed_data
    load_wilds_dataset = mod.load_wilds_dataset
    preprocess_category_data = mod.preprocess_category_data
    create_splits_from_preprocessed = mod.create_splits_from_preprocessed
    save_preprocessed_data = mod.save_preprocessed_data
    dataset_cfg_path = SCRIPT_DIR / "dataset.yaml"
    with open(dataset_cfg_path, "r", encoding="utf-8") as f:
        dataset_cfg = yaml.safe_load(f)
    dataset_cfg.setdefault("data_root", "./data")

    preprocessed = load_preprocessed_data(dataset_cfg)
    if preprocessed is None:
        dataset, _ = load_wilds_dataset(dataset_cfg)
        preprocessed = preprocess_category_data(dataset, dataset_cfg["category_id"])
        save_preprocessed_data(dataset_cfg, preprocessed)

    splits = create_splits_from_preprocessed(
        preprocessed,
        train_ratio=dataset_cfg.get("train_ratio", 0.7),
        val_ratio=dataset_cfg.get("validation_ratio", 0.15),
        test_ratio=dataset_cfg.get("test_ratio", 0.15),
        seed=dataset_cfg.get("split_seed", 42),
    )
    split = splits[split_name]
    indices = split["indices"]
    texts = [preprocessed["texts"][i] for i in indices]
    labels = np.array([int(preprocessed["labels"][i]) for i in indices])
    user_ids = np.array([int(preprocessed["user_ids"][i]) for i in indices])

    if max_users and max_users > 0:
        unique_users = sorted(set(user_ids.tolist()))
        selected = set(unique_users[: int(max_users)])
        mask = np.array([u in selected for u in user_ids])
        texts = [t for t, m in zip(texts, mask) if m]
        labels = labels[mask]
        user_ids = user_ids[mask]

    _SPLIT_DATA_CACHE[cache_key] = (texts, labels, user_ids)
    return texts, labels, user_ids


def _build_workers(config: dict) -> List[LLMWorker]:
    defaults = config.get("worker_defaults", {})
    workers = []
    for cfg in config.get("workers", []):
        merged = {**defaults, **(cfg or {})}
        workers.append(LLMWorker(
            model_name=merged.get("name", "deepseek-chat-v3"),
            api_base=merged.get("api_base"),
            temperature=merged.get("temperature", 0.1),
            max_tokens=merged.get("max_tokens", 64),
        ))
    return workers


def _run_evaluation(
    prompt_template: str,
    texts: List[str],
    labels: np.ndarray,
    user_ids: np.ndarray,
    config: dict,
) -> Tuple[np.ndarray, List[List[int]], Dict[str, Any]]:
    """Run ensemble evaluation, return predictions, worker_predictions, metrics."""
    workers = _build_workers(config)
    aggregator = MajorityVoteAggregator()

    predictions = []
    worker_preds = [[] for _ in workers]
    for text in texts:
        sample_preds = []
        for w in workers:
            p = w.predict(text, prompt_template)
            sample_preds.append(p)
        for i, p in enumerate(sample_preds):
            worker_preds[i].append(p)
        predictions.append(aggregator.aggregate(sample_preds))

    pred_arr = np.array(predictions)
    wp_arr = [np.array(wp) for wp in worker_preds]
    metrics = compute_metrics(pred_arr, labels, user_ids, worker_predictions=wp_arr)
    return pred_arr, wp_arr, metrics


def _compute_weighted_fitness(
    acc_hard: float,
    acc_anchor: float,
    kappa_hard: float,
    prompt: str,
) -> float:
    """Score = w1*Acc_Hard + w2*Acc_Anchor + w3*kappa - P_Len"""
    tokens = _estimate_tokens(prompt)
    p_len = 0.0
    if tokens > PROMPT_LEN_LIMIT:
        excess = (tokens - PROMPT_LEN_LIMIT) / 100
        p_len = PENALTY_PER_100_TOKENS * excess
    kappa_clipped = max(0.0, float(kappa_hard))
    return W1_ACC_HARD * acc_hard + W2_ACC_ANCHOR * acc_anchor + W3_KAPPA * kappa_clipped - p_len


def evaluate_fast(
    prompt_template: str,
    active_batch_data: Dict[str, Any],
    pool_texts: List[str],
    pool_labels: List[int],
    pool_user_ids: List[int],
    config: dict,
) -> Dict[str, Any]:
    """
    Evaluate on active batch (Hard + Anchor).
    active_batch_data: { "indices": [...], "hard_indices": [...], "anchor_indices": [...] }
    Pool arrays are indexed by pool index (0..n_pool-1).
    """
    indices = active_batch_data.get("indices", [])
    hard_set = set(active_batch_data.get("hard_indices", []))
    anchor_set = set(active_batch_data.get("anchor_indices", []))

    texts = [pool_texts[i] for i in indices]
    labels = np.array([pool_labels[i] for i in indices])
    user_ids = np.array([pool_user_ids[i] for i in indices])

    pred_arr, wp_arr, metrics = _run_evaluation(
        prompt_template, texts, labels, user_ids, config
    )

    # Split by hard vs anchor
    hard_preds, hard_labels, hard_wp = [], [], []
    anchor_preds, anchor_labels, anchor_wp = [], [], []
    for j, idx in enumerate(indices):
        if idx in hard_set:
            hard_preds.append(pred_arr[j])
            hard_labels.append(labels[j])
            hard_wp.append([wp[j] for wp in wp_arr])
        elif idx in anchor_set:
            anchor_preds.append(pred_arr[j])
            anchor_labels.append(labels[j])
            anchor_wp.append([wp[j] for wp in wp_arr])

    acc_hard = float(np.mean(np.array(hard_preds) == np.array(hard_labels))) if hard_preds else 0.0
    acc_anchor = float(np.mean(np.array(anchor_preds) == np.array(anchor_labels))) if anchor_preds else 1.0

    kappa_hard = 0.0
    if hard_preds and len(wp_arr) > 1:
        try:
            from sklearn.metrics import cohen_kappa_score
        except ImportError:
            pass
        else:
            hard_j = [j for j, idx in enumerate(indices) if idx in hard_set]
            if hard_j:
                hw = [np.array([wp_arr[k][j] for j in hard_j]) for k in range(len(wp_arr))]
                kappas = []
                for i in range(len(hw)):
                    for j in range(i + 1, len(hw)):
                        kappas.append(cohen_kappa_score(hw[i], hw[j], weights="quadratic"))
                kappa_hard = float(np.mean(kappas)) if kappas else 0.0

    combined = _compute_weighted_fitness(acc_hard, acc_anchor, kappa_hard, prompt_template)

    return {
        "combined_score": combined,
        "Acc_Hard": acc_hard,
        "Acc_Anchor": acc_anchor,
        "kappa_Hard": kappa_hard,
        "predictions": pred_arr.tolist(),
        "gold_labels": labels.tolist(),
        "user_ids": user_ids.tolist(),
        "worker_predictions": [wp.tolist() for wp in wp_arr],
        "indices": indices,
        "R_global": metrics.get("R_global", 0),
        "mae": metrics.get("mae", 0),
        "mean_kappa": metrics.get("mean_kappa", 0),
    }


def evaluate_full(
    prompt_template: str,
    config: dict,
    split_name: str = "validation",
) -> Dict[str, Any]:
    """Full evaluation on validation split."""
    texts, labels, user_ids = _load_split_data(config, split_name)
    pred_arr, wp_arr, metrics = _run_evaluation(
        prompt_template, list(texts), labels, user_ids, config
    )
    metrics["combined_score"] = compute_combined_score_unified(metrics, is_ensemble=True)
    metrics["predictions"] = pred_arr.tolist()
    metrics["gold_labels"] = labels.tolist()
    metrics["user_ids"] = user_ids.tolist()
    metrics["worker_predictions"] = [wp.tolist() for wp in wp_arr]
    return metrics


def _format_error_artifacts(
    predictions: list,
    gold_labels: list,
    worker_predictions: list,
    texts: list,
    max_errors: int = 7,
    max_borderline: int = 3,
    max_text_len: int = 250,
) -> str:
    """Format error and borderline examples as artifact text for the LLM mutator."""
    from data_manager import disagreement_score
    from collections import Counter as _Counter

    n_workers = len(worker_predictions)
    errors = []
    borderline = []
    for j in range(len(predictions)):
        pred = predictions[j]
        gold = gold_labels[j]
        wp = [int(worker_predictions[k][j]) for k in range(n_workers)]
        d_score = disagreement_score(wp, rating_min=1, rating_max=5)
        if pred != gold:
            errors.append((texts[j], gold, pred, wp, d_score))
        elif d_score > 0:
            borderline.append((texts[j], gold, pred, wp, d_score))

    errors.sort(key=lambda x: -x[4])
    borderline.sort(key=lambda x: -x[4])

    lines = []

    if errors:
        confusion = _Counter()
        for _, gold, pred, _, _ in errors:
            confusion[f"{gold} -> {pred}"] += 1
        top_conf = confusion.most_common(5)
        lines.append(f"ERROR DISTRIBUTION ({len(errors)} total misclassified):")
        for pair, cnt in top_conf:
            lines.append(f"  gold {pair}: {cnt}")
        lines.append("")
        lines.append("MISCLASSIFIED EXAMPLES (your prompt got these wrong):")
        for i, (text, gold, pred, wp, ds) in enumerate(errors[:max_errors]):
            t = text[:max_text_len] + ("..." if len(text) > max_text_len else "")
            lines.append(f"  {i+1}. Review: \"{t}\"")
            lines.append(f"     Gold: {gold} | Predicted: {pred} | Workers: {wp} | Disagreement: {ds:.2f}")

    if borderline:
        lines.append("")
        lines.append("BORDERLINE EXAMPLES (correct but workers disagreed):")
        for i, (text, gold, pred, wp, ds) in enumerate(borderline[:max_borderline]):
            t = text[:max_text_len] + ("..." if len(text) > max_text_len else "")
            lines.append(f"  {i+1}. Review: \"{t}\"")
            lines.append(f"     Gold: {gold} | Predicted: {pred} | Workers: {wp} | Disagreement: {ds:.2f}")

    if lines:
        total = len(predictions)
        lines.append(f"\nSummary: {len(errors)} errors, {len(borderline)} borderline out of {total} examples.")
    return "\n".join(lines)


def evaluate(prompt_path: Optional[str] = None) -> Any:
    """
    Main entry point for OpenEvolve.
    If active_batch.json exists, uses evaluate_fast. Otherwise evaluate_full.
    Returns EvaluationResult with error artifacts when active batch is present,
    or a plain dict otherwise.
    """
    config = _load_config()
    prompt_file = prompt_path or config.get("prompt_path", "initial_prompt.txt")
    prompt_path_abs = Path(prompt_file)
    if not prompt_path_abs.is_absolute():
        prompt_path_abs = SCRIPT_DIR / prompt_file
    with open(prompt_path_abs, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    active = _load_active_batch()
    if active and "indices" in active:
        pool_texts, pool_labels, pool_user_ids = _load_split_data(config, "train")
        pool_labels_list = pool_labels.tolist()
        pool_user_ids_list = pool_user_ids.tolist()
        result = evaluate_fast(
            prompt_template,
            active,
            pool_texts,
            pool_labels_list,
            pool_user_ids_list,
            config,
        )

        metrics = {
            "combined_score": result["combined_score"],
            "Acc_Hard": result.get("Acc_Hard", 0.0),
            "Acc_Anchor": result.get("Acc_Anchor", 1.0),
            "kappa_Hard": result.get("kappa_Hard", 0.0),
            "R_global": result.get("R_global", 0),
            "mae": result.get("mae", 0),
            "mean_kappa": max(0, result.get("mean_kappa", 0)),
            "prompt_length": min(1.0, _estimate_tokens(prompt_template) / 3000),
        }

        batch_texts = [pool_texts[i] for i in result["indices"]]
        error_text = _format_error_artifacts(
            result["predictions"],
            result["gold_labels"],
            result["worker_predictions"],
            batch_texts,
        )

        if error_text:
            try:
                from openevolve.evaluation_result import EvaluationResult
                return EvaluationResult(metrics=metrics, artifacts={"error_examples": error_text})
            except ImportError:
                pass
        return metrics

    # No active batch: full validation
    result = evaluate_full(prompt_template, config)
    from experiments.feature_dimensions import calculate_all_features
    features = calculate_all_features(prompt_template, metrics={"mean_kappa": result.get("mean_kappa", 0)}, is_ensemble=True)
    result["sentiment_vocabulary_richness"] = features["sentiment_vocabulary_richness"]
    result["mean_kappa"] = max(0, result.get("mean_kappa", 0))
    result["prompt_length"] = features["prompt_length"]
    return result


def evaluate_stage1(prompt_path: Optional[str] = None) -> Dict[str, Any]:
    """Cascade stage 1: fast evaluation on active batch."""
    return evaluate(prompt_path)


def evaluate_stage2(prompt_path: Optional[str] = None) -> Dict[str, Any]:
    """Cascade stage 2: full validation (same as evaluate when no active batch)."""
    return evaluate(prompt_path)

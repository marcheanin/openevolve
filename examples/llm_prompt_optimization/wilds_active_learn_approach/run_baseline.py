"""
Evaluate the current initial_prompt.txt on full train and validation sets.
Establishes baseline metrics before active learning.

Usage:
    python run_baseline.py

Requires OPENROUTER_API_KEY or OPENAI_API_KEY.
"""

import json
import os
import sys
from pathlib import Path

# Load .env before any imports that need OPENAI_API_KEY (e.g. evaluator)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
WILDS_EXPERIMENT = SCRIPT_DIR.parent / "wilds_experiment"
EXPERIMENTS_ROOT = WILDS_EXPERIMENT / "experiments"
if str(WILDS_EXPERIMENT) not in sys.path:
    sys.path.insert(0, str(WILDS_EXPERIMENT))
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

# Use our workers (OpenRouter)
sys.path.insert(0, str(SCRIPT_DIR))
from workers import LLMWorker
from collections import Counter
from experiments.metrics import compute_metrics, compute_combined_score_unified


class MajorityVoteAggregator:
    def aggregate(self, worker_predictions):
        return Counter(worker_predictions).most_common(1)[0][0]


def load_config():
    with open(SCRIPT_DIR / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_split_data(config, split_name, max_samples=None):
    """Load data for a split."""
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
    labels = np.array([preprocessed["labels"][i] for i in indices])
    user_ids = np.array([preprocessed["user_ids"][i] for i in indices])

    # Apply max_users from experiment config
    ds_cfg = config.get("dataset", {})
    if split_name == "train":
        max_users = ds_cfg.get("max_train_users")
    else:
        max_users = ds_cfg.get("max_val_users")  # validation and test
    if max_users and max_users > 0 and len(np.unique(user_ids)) > max_users:
        unique_users = np.unique(user_ids)
        selected_users = set(unique_users[:int(max_users)])
        mask = np.array([u in selected_users for u in user_ids])
        texts = [t for t, m in zip(texts, mask) if m]
        labels = labels[mask]
        user_ids = user_ids[mask]

    if max_samples and len(texts) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(texts), max_samples, replace=False)
        texts = [texts[i] for i in idx]
        labels = labels[idx]
        user_ids = user_ids[idx]

    return texts, labels, user_ids


def evaluate_split(texts, labels, user_ids, prompt_template, config):
    """Run ensemble evaluation on a split."""
    defaults = config.get("worker_defaults", {})
    worker_cfgs = config.get("workers", [])
    workers = []
    for cfg in worker_cfgs:
        merged = {**defaults, **(cfg or {})}
        model_name = merged.get("name", "deepseek-chat-v3")
        workers.append(LLMWorker(
            model_name=model_name,
            api_base=merged.get("api_base"),
            temperature=merged.get("temperature", 0.1),
            max_tokens=merged.get("max_tokens", 64),
        ))
    aggregator = MajorityVoteAggregator()

    predictions = []
    worker_preds = [[] for _ in workers]
    for i, text in enumerate(texts):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(texts)}", flush=True)
        sample_preds = []
        for w_idx, w in enumerate(workers):
            p = w.predict(text, prompt_template)
            worker_preds[w_idx].append(p)
            sample_preds.append(p)
        predictions.append(aggregator.aggregate(sample_preds))

    predictions = np.array(predictions)
    worker_preds = [np.array(wp) for wp in worker_preds]
    metrics = compute_metrics(predictions, labels, user_ids, worker_predictions=worker_preds)
    metrics["combined_score"] = compute_combined_score_unified(metrics, is_ensemble=True)
    return metrics


def main():
    config = load_config()
    prompt_path = SCRIPT_DIR / config.get("prompt_path", "initial_prompt.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    print("Loading train data...")
    train_texts, train_labels, train_users = load_split_data(config, "train")
    print(f"Train: {len(train_texts)} examples")

    print("Loading validation data...")
    val_texts, val_labels, val_users = load_split_data(config, "validation")
    print(f"Validation: {len(val_texts)} examples")

    print("Loading test data...")
    test_texts, test_labels, test_users = load_split_data(config, "test")
    print(f"Test: {len(test_texts)} examples")

    print("Evaluating on train...")
    train_metrics = evaluate_split(train_texts, train_labels, train_users, prompt_template, config)
    print("Evaluating on validation...")
    val_metrics = evaluate_split(val_texts, val_labels, val_users, prompt_template, config)
    print("Evaluating on test...")
    test_metrics = evaluate_split(test_texts, test_labels, test_users, prompt_template, config)

    gap = max(0, train_metrics["R_global"] - val_metrics["R_global"])
    gap_penalty = min(1, 0.5 * max(0, gap - 0.1))
    combined = val_metrics["combined_score"] * (1 - gap_penalty)

    print("\n" + "=" * 50)
    print("BASELINE RESULTS")
    print("=" * 50)
    print(f"Train:       R_global={train_metrics['R_global']:.2%}  R_worst={train_metrics['R_worst']:.2%}  MAE={train_metrics['mae']:.3f}")
    print(f"Validation:  R_global={val_metrics['R_global']:.2%}  R_worst={val_metrics['R_worst']:.2%}  MAE={val_metrics['mae']:.3f}")
    print(f"Test:        R_global={test_metrics['R_global']:.2%}  R_worst={test_metrics['R_worst']:.2%}  MAE={test_metrics['mae']:.3f}")
    print(f"Mean Kappa:  {val_metrics.get('mean_kappa', 0):.3f}")
    print(f"Combined:    {combined:.4f} (gap_penalty={gap_penalty:.3f})")
    print("=" * 50)

    out_dir = SCRIPT_DIR / "results" / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "train": train_metrics,
            "validation": val_metrics,
            "test": test_metrics,
            "combined_score": combined,
        }, f, indent=2)
    print(f"Saved to {out_dir / 'metrics.json'}")

    # Учёт токенов по моделям (для оценки стоимости)
    try:
        from token_usage import get_tracker
        get_tracker().save_json(out_dir / "token_usage.json")
        u = get_tracker().get_usage()
        print(f"Token usage: total={u['total_tokens']} (by model in {out_dir / 'token_usage.json'})")
    except Exception as e:
        print(f"Token usage save skipped: {e}")


if __name__ == "__main__":
    main()

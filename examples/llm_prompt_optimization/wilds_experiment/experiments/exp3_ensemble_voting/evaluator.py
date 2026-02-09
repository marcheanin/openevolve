import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
WILDS_EVALUATOR_PATH = ROOT_DIR / "evaluator.py"

# Ensure parent directory (with the experiments package) is on sys.path,
# but keep the current working directory first so that local evaluator.py
# (this file) is imported when using plain `import evaluator`.
EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]  # .../experiments
PROJECT_ROOT = EXPERIMENTS_ROOT.parent                  # .../wilds_experiment
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

spec = importlib.util.spec_from_file_location("wilds_experiment_evaluator", WILDS_EVALUATOR_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load evaluator from {WILDS_EVALUATOR_PATH}")

wilds_evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wilds_evaluator)

load_wilds_dataset = getattr(wilds_evaluator, "load_wilds_dataset")
preprocess_category_data = getattr(wilds_evaluator, "preprocess_category_data")
create_splits_from_preprocessed = getattr(wilds_evaluator, "create_splits_from_preprocessed")
save_preprocessed_data = getattr(wilds_evaluator, "save_preprocessed_data")
load_preprocessed_data = getattr(wilds_evaluator, "load_preprocessed_data")
from experiments.aggregators import MajorityVoteAggregator  # noqa: E402
from experiments.metrics import compute_metrics, compute_combined_score_unified  # noqa: E402
from experiments.workers import LLMWorker  # noqa: E402
from experiments.feature_dimensions import (  # noqa: E402
    calculate_criteria_explicitness,
    calculate_all_features,
)

# Глобальный кэш для splits (как в старом evaluator)
_DATASET_SPLITS_CACHE = None
_PREPROCESSED_CACHE = None


def load_experiment_config() -> Dict[str, Any]:
    config_path = Path(__file__).resolve().parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    max_train_users = os.getenv("OPENEVOLVE_MAX_TRAIN_USERS")
    if max_train_users:
        config.setdefault("dataset", {})["max_train_users"] = int(max_train_users)

    return config


def _load_dataset_config(path: str) -> Dict[str, Any]:
    config_path = Path(__file__).resolve().parent / path
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _filter_users_by_min_reviews(
    texts: List[str],
    labels: List[int],
    user_ids: List[int],
    min_reviews: int,
    split_name: str,
) -> Tuple[List[str], List[int], List[int]]:
    if min_reviews <= 1:
        return texts, labels, user_ids

    counts: Dict[int, int] = {}
    for uid in user_ids:
        counts[uid] = counts.get(uid, 0) + 1

    allowed_users = {uid for uid, cnt in counts.items() if cnt >= min_reviews}

    if not allowed_users:
        return texts, labels, user_ids

    keep_indices = [i for i, uid in enumerate(user_ids) if uid in allowed_users]
    return (
        [texts[i] for i in keep_indices],
        [labels[i] for i in keep_indices],
        [user_ids[i] for i in keep_indices],
    )


def _limit_users(
    texts: List[str],
    labels: List[int],
    user_ids: List[int],
    max_users: Optional[int],
    split_name: str,
) -> Tuple[List[str], List[int], List[int]]:
    if not max_users or max_users <= 0:
        return texts, labels, user_ids

    unique_users = sorted(set(user_ids))
    selected_users = set(unique_users[:max_users])

    keep_indices = [i for i, uid in enumerate(user_ids) if uid in selected_users]
    return (
        [texts[i] for i in keep_indices],
        [labels[i] for i in keep_indices],
        [user_ids[i] for i in keep_indices],
    )


def _get_cached_splits(dataset_cfg: Dict[str, Any]) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Получает кэшированные splits и preprocessed данные.
    Использует дисковый кэш для работы между процессами (multiprocessing).
    Если кэша нет - создаёт и сохраняет его.
    """
    global _DATASET_SPLITS_CACHE, _PREPROCESSED_CACHE

    if _DATASET_SPLITS_CACHE is not None and _PREPROCESSED_CACHE is not None:
        return _DATASET_SPLITS_CACHE, _PREPROCESSED_CACHE

    _PREPROCESSED_CACHE = load_preprocessed_data(dataset_cfg)

    if _PREPROCESSED_CACHE is not None:
        _DATASET_SPLITS_CACHE = create_splits_from_preprocessed(
            _PREPROCESSED_CACHE,
            train_ratio=dataset_cfg.get("train_ratio", 0.7),
            val_ratio=dataset_cfg.get("validation_ratio", 0.15),
            test_ratio=dataset_cfg.get("test_ratio", 0.15),
            seed=dataset_cfg.get("split_seed", 42),
        )
        return _DATASET_SPLITS_CACHE, _PREPROCESSED_CACHE

    print("[cache] Creating preprocessed cache (first run only, may take a few minutes)...", flush=True)

    dataset, _ = load_wilds_dataset(dataset_cfg)
    category_id = dataset_cfg.get("category_id", 24)
    _PREPROCESSED_CACHE = preprocess_category_data(dataset, category_id)
    save_preprocessed_data(dataset_cfg, _PREPROCESSED_CACHE)

    _DATASET_SPLITS_CACHE = create_splits_from_preprocessed(
        _PREPROCESSED_CACHE,
        train_ratio=dataset_cfg.get("train_ratio", 0.7),
        val_ratio=dataset_cfg.get("validation_ratio", 0.15),
        test_ratio=dataset_cfg.get("test_ratio", 0.15),
        seed=dataset_cfg.get("split_seed", 42),
    )

    print("[cache] Cache created. Future runs will be faster.", flush=True)
    return _DATASET_SPLITS_CACHE, _PREPROCESSED_CACHE


def load_split_data(
    config: Dict[str, Any],
    split_name: str,
    max_samples: Optional[int] = None,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    dataset_cfg = _load_dataset_config(config["dataset"]["config_path"])
    category_id = dataset_cfg.get("category_id", 24)

    splits, preprocessed = _get_cached_splits(dataset_cfg)

    if splits is not None and preprocessed is not None:
        split_map = {"train": "train", "validation": "validation", "test": "test"}
        split_key = split_map.get(split_name, "validation")
        split_indices = splits[split_key]["indices"]

        texts = [preprocessed["texts"][i] for i in split_indices]
        labels = np.array([preprocessed["labels"][i] for i in split_indices], dtype=int)
        user_ids = np.array([preprocessed["user_ids"][i] for i in split_indices], dtype=int)
        texts = list(texts)
    else:
        print(f"[load_split_data] Cache not found, loading full WILDS dataset (slow)...", flush=True)
        dataset, _ = load_wilds_dataset(dataset_cfg)
        print(f"[load_split_data] Dataset loaded, getting subset...", flush=True)
        split_map = {"train": "train", "validation": "val", "test": "test"}
        subset_name = split_map.get(split_name, "val")
        split_data = dataset.get_subset(subset_name)

        metadata_fields = dataset.metadata_fields
        user_idx = metadata_fields.index("user")
        category_idx = metadata_fields.index("category")

        metadata_array = split_data.metadata_array

        category_mask = metadata_array[:, category_idx] == category_id
        indices = np.where(category_mask)[0]

        texts = []
        labels = []
        user_ids = []
        for idx in indices:
            x, y, metadata = split_data[idx]
            texts.append(x)
            labels.append(int(y) + 1)
            user_ids.append(int(metadata_array[idx, user_idx]))

        labels = np.array(labels)
        user_ids = np.array(user_ids)

    if isinstance(labels, np.ndarray):
        labels = labels.tolist()
    if isinstance(user_ids, np.ndarray):
        user_ids = user_ids.tolist()
    if not isinstance(texts, list):
        texts = list(texts)

    min_user_reviews = dataset_cfg.get("min_user_reviews", 1)
    texts, labels, user_ids = _filter_users_by_min_reviews(
        texts, labels, user_ids, min_user_reviews, split_name
    )

    experiment_dataset_cfg = config.get("dataset", {})
    if split_name == "train":
        max_train_users = experiment_dataset_cfg.get("max_train_users")
        if max_train_users is not None:
            try:
                max_train_users = int(max_train_users)
            except (ValueError, TypeError):
                print(
                    f"[load_split_data] WARNING: max_train_users value "
                    f"'{experiment_dataset_cfg.get('max_train_users')}' is not a valid integer, ignoring",
                    flush=True,
                )
                max_train_users = None
        texts, labels, user_ids = _limit_users(
            texts,
            labels,
            user_ids,
            max_train_users,
            split_name,
        )
    elif split_name == "validation":
        max_val_users = experiment_dataset_cfg.get("max_val_users")
        if max_val_users is not None:
            try:
                max_val_users = int(max_val_users)
            except (ValueError, TypeError):
                print(
                    f"[load_split_data] WARNING: max_val_users value "
                    f"'{experiment_dataset_cfg.get('max_val_users')}' is not a valid integer, ignoring",
                    flush=True,
                )
                max_val_users = None
        texts, labels, user_ids = _limit_users(
            texts,
            labels,
            user_ids,
            max_val_users,
            split_name,
        )

    if len(texts) == 0:
        raise ValueError(
            f"No data remaining after filtering for {split_name} split. "
            f"Check min_user_reviews and max_*_users settings."
        )

    texts_arr = np.array(texts)
    labels_arr = np.array(labels)
    user_ids_arr = np.array(user_ids)

    if max_samples and max_samples > 0 and max_samples < len(texts_arr):
        seed = dataset_cfg.get("split_seed", 42)
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(texts_arr), size=max_samples, replace=False)
        texts_arr = texts_arr[indices]
        labels_arr = labels_arr[indices]
        user_ids_arr = user_ids_arr[indices]

    return texts_arr.tolist(), labels_arr, user_ids_arr


def _build_workers(config: Dict[str, Any]) -> Tuple[List[LLMWorker], List[str]]:
    defaults = config.get("worker_defaults", {})
    worker_cfgs = config.get("workers", [])

    if not worker_cfgs:
        model_cfg = config.get("model", {})
        worker_cfgs = [model_cfg]

    workers: List[LLMWorker] = []
    worker_names: List[str] = []
    for idx, cfg in enumerate(worker_cfgs, 1):
        merged = {**defaults, **(cfg or {})}
        model_name = merged.get("name", f"worker_{idx}")
        worker_names.append(model_name)
        workers.append(
            LLMWorker(
                model_name=model_name,
                api_base=merged.get("api_base"),
                temperature=merged.get("temperature", 0.1),
                max_tokens=merged.get("max_tokens", 64),
                timeout=merged.get("timeout", 60),
                max_retries=merged.get("max_retries", 3),
            )
        )

    return workers, worker_names


def _evaluate_split(
    split_name: str,
    prompt_template: str,
    config: Dict[str, Any],
    max_samples: Optional[int],
) -> Tuple[Dict[str, Any], List[int], List[int], List[int], List[List[int]]]:
    texts, gold_labels, user_ids = load_split_data(
        config,
        split_name=split_name,
        max_samples=max_samples,
    )

    workers, worker_names = _build_workers(config)
    aggregator = MajorityVoteAggregator()

    predictions = []
    worker_predictions: List[List[int]] = [[] for _ in workers]
    for idx, text in enumerate(texts, 1):
        sample_preds = []
        for w_idx, worker in enumerate(workers):
            pred = worker.predict(text, prompt_template)
            worker_predictions[w_idx].append(pred)
            sample_preds.append(pred)
        predictions.append(aggregator.aggregate(sample_preds))
        if len(texts) > 50 and idx % 50 == 0:
            print(f"[{split_name}] {idx}/{len(texts)} samples", flush=True)

    predictions_arr = np.array(predictions)
    gold_labels_arr = np.asarray(gold_labels)
    user_ids_arr = np.asarray(user_ids)
    worker_predictions_arr = [np.asarray(wp) for wp in worker_predictions]

    metrics = compute_metrics(
        predictions=predictions_arr,
        gold_labels=gold_labels_arr,
        user_ids=user_ids_arr,
        worker_predictions=worker_predictions_arr,
    )
    metrics["split"] = split_name
    metrics["model_name"] = ", ".join(worker_names)
    metrics["aggregator"] = "majority_vote"
    metrics["worker_names"] = worker_names

    return (
        metrics,
        predictions,
        gold_labels.tolist(),
        user_ids.tolist(),
        [wp.tolist() for wp in worker_predictions_arr],
    )


def evaluate(prompt_path: str | None = None) -> Dict[str, Any]:
    config = load_experiment_config()

    prompt_file = prompt_path or config.get("prompt_path", "initial_prompt.txt")
    prompt_path_abs = Path(__file__).resolve().parent / prompt_file
    with open(prompt_path_abs, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    dataset_cfg = config.get("dataset", {})
    max_samples_train = dataset_cfg.get("max_samples_train", dataset_cfg.get("max_samples"))
    max_samples_val = dataset_cfg.get("max_samples_val", dataset_cfg.get("max_samples"))

    train_metrics, _, _, _, _ = _evaluate_split(
        "train",
        prompt_template,
        config,
        max_samples_train,
    )
    val_metrics, _, _, _, _ = _evaluate_split(
        "validation",
        prompt_template,
        config,
        max_samples_val,
    )

    val_metrics["combined_score"] = compute_combined_score_unified(val_metrics, is_ensemble=True)
    train_metrics["combined_score"] = compute_combined_score_unified(train_metrics, is_ensemble=True)

    train_r_global = train_metrics["R_global"]
    val_r_global = val_metrics["R_global"]
    generalization_gap = max(0.0, float(train_r_global - val_r_global))
    gap_cfg = config.get("generalization", {})
    gap_threshold = float(gap_cfg.get("gap_threshold", 0.1))
    gap_weight = float(gap_cfg.get("gap_penalty", 0.5))
    excess_gap = max(0.0, generalization_gap - gap_threshold)
    gap_penalty = min(1.0, gap_weight * excess_gap)
    combined_score = float(val_metrics["combined_score"] * (1.0 - gap_penalty))

    # Calculate feature dimensions for MAP-Elites (improved metrics for ensemble)
    # For ensemble experiments we use Cohen's Kappa (mean_kappa) as key feature dimension
    # Kappa in [-1, 1]; normalize to [0, 1] for MAP-Elites grid (negative -> 0)
    val_metrics_for_features = {
        'mean_kappa': val_metrics.get('mean_kappa', 0.0),
    }
    features = calculate_all_features(prompt_template, metrics=val_metrics_for_features, is_ensemble=True)
    
    # Primary feature dimensions (used in config.yaml)
    sentiment_vocabulary_richness = features['sentiment_vocabulary_richness']
    # For ensemble: use mean_kappa (Cohen's Kappa), clipped to [0, 1] for grid
    mean_kappa_raw = val_metrics.get('mean_kappa', 0.0)
    mean_kappa_feature = max(0.0, float(mean_kappa_raw))
    
    # Legacy features (for backward compatibility)
    prompt_length_normalized = features['prompt_length']

    metrics = {
        "combined_score": combined_score,
        "R_global": val_metrics["R_global"],
        "R_worst": val_metrics["R_worst"],
        "mae": val_metrics["mae"],
        "mean_kappa": val_metrics.get("mean_kappa", 0.0),
        "train_R_global": train_metrics["R_global"],
        "train_R_worst": train_metrics["R_worst"],
        "train_mae": train_metrics["mae"],
        "train_mean_kappa": train_metrics.get("mean_kappa", 0.0),
        "train_combined_score": train_metrics["combined_score"],
        "val_combined_score": val_metrics["combined_score"],
        "generalization_gap": generalization_gap,
        "generalization_gap_threshold": gap_threshold,
        "generalization_gap_penalty": gap_penalty,
    }

    result = {
        # OpenEvolve expects combined_score at top level
        "combined_score": combined_score,
        "metrics": metrics,
        # Feature dimensions for MAP-Elites (must be at top level)
        "sentiment_vocabulary_richness": sentiment_vocabulary_richness,
        "mean_kappa": mean_kappa_feature,  # Cohen's Kappa, normalized to [0, 1]
        # Legacy features (for backward compatibility)
        "prompt_length": prompt_length_normalized,
        # Additional features for analysis
        "domain_focus": features['domain_focus'],
        "sentiment_vocabulary": features['sentiment_vocabulary'],
        "example_richness": features['example_richness'],
    }

    return result


def save_results(output: Dict[str, Any], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    metrics_path = Path(output_dir) / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(output["metrics"], f, ensure_ascii=False, indent=2)

    predictions_path = Path(output_dir) / "predictions.json"
    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "predictions": output["predictions"],
                "gold_labels": output["gold_labels"],
                "user_ids": output["user_ids"],
                "worker_predictions": output.get("worker_predictions", []),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )



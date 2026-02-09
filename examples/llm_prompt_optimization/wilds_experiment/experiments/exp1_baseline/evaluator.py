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

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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
from experiments.metrics import compute_metrics, compute_combined_score_single  # noqa: E402
from experiments.workers import LLMWorker  # noqa: E402


def load_experiment_config() -> Dict[str, Any]:
    config_path = Path(__file__).resolve().parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
    print(
        f"User filter ({split_name}): min_reviews={min_reviews}, "
        f"users_kept={len(allowed_users)}/{len(counts)}"
    )

    if not allowed_users:
        return texts, labels, user_ids

    keep_indices = [i for i, uid in enumerate(user_ids) if uid in allowed_users]
    return (
        [texts[i] for i in keep_indices],
        [labels[i] for i in keep_indices],
        [user_ids[i] for i in keep_indices],
    )


def load_split_data(
    config: Dict[str, Any],
    max_samples: Optional[int] = None,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    dataset_cfg = _load_dataset_config(config["dataset"]["config_path"])
    dataset, _ = load_wilds_dataset(dataset_cfg)
    split_name = config["dataset"].get("split", "validation")
    split_map = {"train": "train", "validation": "val", "test": "test"}
    subset_name = split_map.get(split_name, "val")
    split_data = dataset.get_subset(subset_name)

    metadata_fields = dataset.metadata_fields
    user_idx = metadata_fields.index("user")
    category_idx = metadata_fields.index("category")

    category_id = dataset_cfg.get("category_id", 14)
    metadata_array = split_data.metadata_array

    category_mask = metadata_array[:, category_idx] == category_id
    indices = np.where(category_mask)[0]

    texts: List[str] = []
    labels: List[int] = []
    user_ids: List[int] = []
    for idx in indices:
        x, y, metadata = split_data[idx]
        texts.append(x)
        labels.append(int(y) + 1)
        user_ids.append(int(metadata_array[idx, user_idx]))

    min_user_reviews = dataset_cfg.get("min_user_reviews", 1)
    texts, labels, user_ids = _filter_users_by_min_reviews(
        texts, labels, user_ids, min_user_reviews, split_name
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


def evaluate(prompt_path: str | None = None) -> Dict[str, Any]:
    config = load_experiment_config()

    prompt_file = prompt_path or config.get("prompt_path", "initial_prompt.txt")
    prompt_path_abs = Path(__file__).resolve().parent / prompt_file
    with open(prompt_path_abs, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    max_samples = config.get("dataset", {}).get("max_samples")
    texts, gold_labels, user_ids = load_split_data(config, max_samples=max_samples)

    model_cfg = config.get("model", {})
    worker = LLMWorker(
        model_name=model_cfg.get("name", "qwen3-235b"),
        api_base=model_cfg.get("api_base"),
        temperature=model_cfg.get("temperature", 0.1),
        max_tokens=model_cfg.get("max_tokens", 64),
        timeout=model_cfg.get("timeout", 60),
        max_retries=model_cfg.get("max_retries", 3),
    )

    predictions = []
    for idx, text in enumerate(texts, 1):
        predictions.append(worker.predict(text, prompt_template))
        if idx % 20 == 0:
            print(f"Processed {idx}/{len(texts)} samples")

    predictions = np.array(predictions)

    metrics = compute_metrics(
        predictions=predictions,
        gold_labels=gold_labels,
        user_ids=user_ids,
    )
    metrics["combined_score"] = compute_combined_score_single(metrics)
    metrics["split"] = config["dataset"].get("split", "validation")
    metrics["model_name"] = model_cfg.get("name", "qwen3-235b")

    return {
        "metrics": metrics,
        "predictions": predictions.tolist(),
        "gold_labels": gold_labels.tolist(),
        "user_ids": user_ids.tolist(),
    }


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
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


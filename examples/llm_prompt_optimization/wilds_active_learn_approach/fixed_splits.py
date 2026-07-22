"""
Fixed split manifest for "plain evolution" ablation (no AL cycles).

Goal:
- Use the *same* underlying capped all-categories payload as the AL pipeline
  (max_train_users/max_val_users/max_reviews_per_user + optional stratify_users),
  but then pick *fixed* train/val/test subsets of target sizes so the run is
  reproducible and comparable.

Manifest format (JSON):
{
  "version": 1,
  "dataset_config_path": "dataset_all_categories.yaml",
  "config_path": "config_all_categories.yaml",
  "split_seed": 42,
  "sizes": {"train": 80, "validation": 225, "test": 225},
  "caps": {... copied from config.dataset ...},
  "indices": {"train": [...], "validation": [...], "test": [...]}
}
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
WILDS_EXPERIMENT = SCRIPT_DIR.parent / "wilds_experiment"


@dataclass(frozen=True)
class FixedSplits:
    texts_by_split: Dict[str, List[str]]
    labels_by_split: Dict[str, np.ndarray]
    user_ids_by_split: Dict[str, np.ndarray]
    indices_by_split: Dict[str, np.ndarray]
    meta: Dict[str, Any]


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_config_path(config_path: str | None) -> Path:
    if config_path:
        p = Path(config_path)
        return p if p.is_absolute() else (SCRIPT_DIR / p)
    # fallback to env used elsewhere
    env = os.environ.get("WILDS_ACTIVE_LEARN_CONFIG")
    if env:
        p = Path(env)
        return p if p.is_absolute() else (SCRIPT_DIR / p)
    return SCRIPT_DIR / "config.yaml"


def _import_base_evaluator():
    import importlib.util

    base_eval_path = WILDS_EXPERIMENT / "evaluator.py"
    spec = importlib.util.spec_from_file_location("wilds_base_evaluator", base_eval_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _rng_for_split(split_seed: int, split_name: str, *, base_offset: int = 1000) -> np.random.RandomState:
    off = {"train": 0, "validation": 1, "val": 1, "test": 2}.get(split_name, 0)
    return np.random.RandomState(int(split_seed) + int(base_offset) + int(off))


def _capped_payload_for_split(config: dict, split_name: str) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Load split payload like data_manager._load_pool_data / AL evaluator:

    - User caps per split (train max_train_users, val max_val_users, test max_test_users
      with fallback to max_val_users for test).
    - Train without user cap: stratified candidate pool (`al_candidate_pool_size`) identical
      to the AL ``DataManager`` path.
    - Optional global max_samples* trim (same RNG rule as data_manager).

    Returns per-review categories for stratified fixed-subset selection in build_fixed_splits.
    """
    from data_manager import _resolve_max_samples

    ds_cfg = config.get("dataset", {}) or {}
    dataset_rel = ds_cfg.get("config_path", "dataset.yaml")
    dataset_cfg_path = SCRIPT_DIR / dataset_rel
    dataset_cfg = _load_yaml(dataset_cfg_path)
    split_seed = int(dataset_cfg.get("split_seed", 42))

    base = _import_base_evaluator()
    preprocessed = base.load_preprocessed_data(dataset_cfg)
    if preprocessed is None:
        dataset, _ = base.load_wilds_dataset(dataset_cfg)
        preprocessed = base.preprocess_category_data(dataset, base.effective_category_id(dataset_cfg))
        base.save_preprocessed_data(dataset_cfg, preprocessed)

    splits = base.create_splits_from_preprocessed(
        preprocessed,
        train_ratio=dataset_cfg.get("train_ratio", 0.7),
        val_ratio=dataset_cfg.get("validation_ratio", 0.15),
        test_ratio=dataset_cfg.get("test_ratio", 0.15),
        seed=split_seed,
    )
    split_key = "validation" if split_name == "val" else split_name
    split = splits[split_key]
    indices = list(split["indices"])

    texts = [preprocessed["texts"][i] for i in indices]
    labels: List[int] = [int(preprocessed["labels"][i]) for i in indices]
    user_ids: List[int] = [int(preprocessed["user_ids"][i]) for i in indices]
    if preprocessed.get("categories") is not None:
        cats_list: List[int] = [int(preprocessed["categories"][i]) for i in indices]
    else:
        cats_list = [0] * len(texts)

    stratify_users = bool(ds_cfg.get("stratify_users", False))
    if split_name == "train":
        max_users = ds_cfg.get("max_train_users")
    elif split_name == "test":
        max_users = ds_cfg.get("max_test_users", ds_cfg.get("max_val_users"))
    else:
        max_users = ds_cfg.get("max_val_users")

    max_reviews_per_user = ds_cfg.get("max_reviews_per_user")
    al_candidate_pool_size = ds_cfg.get("al_candidate_pool_size") or 0
    try:
        al_candidate_pool_size = int(al_candidate_pool_size)
    except (TypeError, ValueError):
        al_candidate_pool_size = 0

    _rng_off = {"train": 0, "validation": 1, "val": 1, "test": 2}.get(split_name, 0)
    rng_split = np.random.RandomState(split_seed + 1000 + _rng_off)

    cats_arr = np.asarray(cats_list, dtype=np.int64)
    uid_arr = np.asarray(user_ids, dtype=np.int64)

    if max_users and int(max_users) > 0:
        if stratify_users and len(cats_list) == len(texts):
            from sample_stratified import select_stratified_users_and_cap

            pick = select_stratified_users_and_cap(
                uid_arr,
                cats_arr,
                int(max_users),
                int(max_reviews_per_user or 0),
                rng_split,
            )
            texts = [texts[i] for i in pick]
            labels = [labels[i] for i in pick]
            user_ids = [user_ids[i] for i in pick]
            cats_list = [cats_list[i] for i in pick]
        else:
            unique_users = sorted(set(user_ids))
            selected_users = set(unique_users[: int(max_users)])
            mask = [u in selected_users for u in user_ids]
            texts = [t for t, m in zip(texts, mask) if m]
            labels = [l for l, m in zip(labels, mask) if m]
            user_ids = [u for u, m in zip(user_ids, mask) if m]
            cats_list = [c for c, m in zip(cats_list, mask) if m]

            uid_arr = np.asarray(user_ids, dtype=np.int64)
            if max_reviews_per_user and int(max_reviews_per_user) > 0:
                final_idx = []
                # Mirror data_manager non-stratify loop (set iteration order).
                for u in selected_users:
                    u_idx = np.where(uid_arr == u)[0]
                    if len(u_idx) > int(max_reviews_per_user):
                        u_idx = rng_split.choice(u_idx, size=int(max_reviews_per_user), replace=False)
                    final_idx.extend(np.asarray(u_idx).ravel().tolist())
                final_idx.sort()
                texts = [texts[i] for i in final_idx]
                labels = [labels[i] for i in final_idx]
                user_ids = [user_ids[i] for i in final_idx]
                cats_list = [cats_list[i] for i in final_idx]

        cats_arr = np.asarray(cats_list, dtype=np.int64)
        uid_arr = np.asarray(user_ids, dtype=np.int64)
    elif (
        split_name == "train"
        and al_candidate_pool_size > 0
        and len(texts) > al_candidate_pool_size
        and len(cats_list) == len(texts)
    ):
        from sample_stratified import (
            select_stratified_users_and_cap,
            stratified_downsample_pick,
        )

        rpu = int(max_reviews_per_user) if max_reviews_per_user else 15
        n_users_for_pool = max(1, (al_candidate_pool_size + rpu - 1) // rpu)
        rng_users = np.random.RandomState(split_seed + 1000)
        uid_arr = np.asarray(user_ids, dtype=np.int64)
        cats_arr = np.asarray(cats_list, dtype=np.int64)
        pick_user_caps = select_stratified_users_and_cap(
            uid_arr,
            cats_arr,
            n_users_for_pool,
            rpu,
            rng_users,
        )
        if len(pick_user_caps) > al_candidate_pool_size:
            cats_after = np.asarray([cats_list[i] for i in pick_user_caps])
            rng_trim = np.random.RandomState(split_seed + 2000)
            sub = stratified_downsample_pick(cats_after, al_candidate_pool_size, rng_trim)
            pick_user_caps = pick_user_caps[sub]
        texts = [texts[i] for i in pick_user_caps]
        labels = [labels[i] for i in pick_user_caps]
        user_ids = [user_ids[i] for i in pick_user_caps]
        cats_list = [cats_list[i] for i in pick_user_caps]

    max_reviews = _resolve_max_samples(ds_cfg, split_name)
    if max_reviews and len(texts) > max_reviews:
        pick = rng_split.choice(len(texts), size=max_reviews, replace=False)
        pick.sort()
        texts = [texts[i] for i in pick]
        labels = [labels[i] for i in pick]
        user_ids = [user_ids[i] for i in pick]
        cats_list = [cats_list[i] for i in pick]

    labels_out = np.asarray(labels, dtype=np.int64)
    user_ids_out = np.asarray(user_ids, dtype=np.int64)
    categories_out = np.asarray(cats_list, dtype=np.int64)
    return texts, labels_out, user_ids_out, categories_out


def build_fixed_splits(
    *,
    config_path: str | None,
    sizes: Dict[str, int],
    out_path: Path,
) -> FixedSplits:
    """
    Create fixed indices for train/validation/test and write manifest JSON to out_path.
    """
    config_file = _resolve_config_path(config_path)
    cfg = _load_yaml(config_file)

    ds_cfg = cfg.get("dataset", {}) or {}
    dataset_rel = ds_cfg.get("config_path", "dataset.yaml")
    dataset_cfg_path = SCRIPT_DIR / dataset_rel
    dataset_cfg = _load_yaml(dataset_cfg_path)
    split_seed = int(dataset_cfg.get("split_seed", 42))

    indices_by_split: Dict[str, np.ndarray] = {}
    texts_by_split: Dict[str, List[str]] = {}
    labels_by_split: Dict[str, np.ndarray] = {}
    user_ids_by_split: Dict[str, np.ndarray] = {}

    from sample_stratified import stratified_downsample_pick

    for split_name in ("train", "validation", "test"):
        target_n = int(sizes[split_name])
        texts, labels, user_ids, cats = _capped_payload_for_split(cfg, split_name)
        rng = _rng_for_split(split_seed, split_name, base_offset=2000)
        pick = stratified_downsample_pick(cats, n_pick=target_n, rng=rng)

        indices_by_split[split_name] = pick.astype(np.int64, copy=False)
        texts_by_split[split_name] = [texts[i] for i in pick]
        labels_by_split[split_name] = labels[pick]
        user_ids_by_split[split_name] = user_ids[pick]

    meta = {
        "version": 1,
        "config_path": str(config_file),
        "dataset_config_path": str(dataset_cfg_path),
        "split_seed": split_seed,
        "sizes": {k: int(v) for k, v in sizes.items()},
        "caps": dict(ds_cfg),
    }
    payload = {
        **meta,
        "indices": {k: indices_by_split[k].tolist() for k in indices_by_split},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return FixedSplits(
        texts_by_split=texts_by_split,
        labels_by_split=labels_by_split,
        user_ids_by_split=user_ids_by_split,
        indices_by_split=indices_by_split,
        meta=meta,
    )


def load_fixed_splits(manifest_path: Path, *, config_path: str | None = None) -> FixedSplits:
    """
    Load fixed indices from manifest and materialize split payloads (texts/labels/user_ids).
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cfg_file = _resolve_config_path(config_path) if config_path else Path(manifest["config_path"])
    cfg = _load_yaml(cfg_file)

    texts_by_split: Dict[str, List[str]] = {}
    labels_by_split: Dict[str, np.ndarray] = {}
    user_ids_by_split: Dict[str, np.ndarray] = {}
    indices_by_split: Dict[str, np.ndarray] = {}

    for split_name in ("train", "validation", "test"):
        pick = np.asarray(manifest["indices"][split_name], dtype=np.int64)
        texts, labels, user_ids, _cats = _capped_payload_for_split(cfg, split_name)
        indices_by_split[split_name] = pick
        texts_by_split[split_name] = [texts[i] for i in pick]
        labels_by_split[split_name] = labels[pick]
        user_ids_by_split[split_name] = user_ids[pick]

    meta = {k: manifest.get(k) for k in ("version", "config_path", "dataset_config_path", "split_seed", "sizes", "caps")}
    return FixedSplits(
        texts_by_split=texts_by_split,
        labels_by_split=labels_by_split,
        user_ids_by_split=user_ids_by_split,
        indices_by_split=indices_by_split,
        meta=meta,
    )


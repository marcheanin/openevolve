"""WILDS Amazon dataset: official OOD train/val/test splits with disk+memory cache."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from prime.config import DatasetCfg
from prime.data.cache import (
    get_cached_splits,
    load_raw_split_pickle,
    raw_split_cache_path,
    resolve_cache_dir,
    save_raw_split_pickle,
    set_cached_splits,
    splits_memory_key,
)


@dataclass(frozen=True)
class ReviewSplit:
    """In-memory split for prompt evolution and evaluation."""

    name: str
    texts: List[str]
    labels: List[int]
    user_ids: List[int]
    example_cluster_ids: Optional[List[int]] = None

    def __len__(self) -> int:
        return len(self.texts)

    def user_disjoint_check(self, other: "ReviewSplit") -> bool:
        a: Set[int] = set(self.user_ids)
        b: Set[int] = set(other.user_ids)
        return len(a & b) == 0


def _load_wilds_dataset(data_root: str):
    from wilds import get_dataset

    return get_dataset(
        dataset="amazon",
        root_dir=data_root,
        download=True,
        unlabeled=False,
    )


def _split_category_filter(
    cfg: DatasetCfg, subset_name: str
) -> Tuple[Optional[int], Optional[List[int]]]:
    """
    Return (include_category_id, exclude_category_ids) for a split.

    Train uses train_category_id (else legacy category_id).
    Val/test use eval_exclude_category_ids when set; else legacy category_id.
    """
    is_train = subset_name in ("train",)
    if is_train:
        include = (
            cfg.train_category_id if cfg.train_category_id is not None else cfg.category_id
        )
        return include, None
    if cfg.eval_exclude_category_ids:
        return None, [int(x) for x in cfg.eval_exclude_category_ids]
    return cfg.category_id, None


def _extract_subset_from_wilds(
    dataset,
    subset_name: str,
    category_id: Optional[int],
    min_user_reviews: int,
    exclude_category_ids: Optional[List[int]] = None,
) -> Tuple[List[str], List[int], List[int]]:
    """
    Extract one official WILDS split (train / val / test).
    Uses metadata masks first; only iterates selected row indices.
    """
    split_map = {"train": "train", "validation": "val", "val": "val", "test": "test"}
    key = split_map.get(subset_name, subset_name)
    split_data = dataset.get_subset(key)
    metadata_fields = dataset.metadata_fields
    user_idx = metadata_fields.index("user")
    category_idx = metadata_fields.index("category")
    metadata_array = split_data.metadata_array

    if exclude_category_ids:
        excl = set(int(x) for x in exclude_category_ids)
        cats = metadata_array[:, category_idx]
        # metadata may be torch tensor
        cats_np = cats.numpy() if hasattr(cats, "numpy") else np.asarray(cats)
        mask = ~np.isin(cats_np.astype(int), list(excl))
        indices = np.where(mask)[0]
    elif category_id is None:
        indices = np.arange(len(split_data), dtype=np.int64)
    else:
        mask = metadata_array[:, category_idx] == category_id
        indices = np.where(mask)[0]

    texts: List[str] = []
    labels: List[int] = []
    user_ids: List[int] = []
    n = len(indices)
    for i, idx in enumerate(indices):
        if i > 0 and i % 5000 == 0:
            print(f"  [{subset_name}] loading {i}/{n}...", flush=True)
        idx = int(idx)
        x, y, _ = split_data[idx]
        texts.append(x)
        labels.append(int(y) + 1)
        user_ids.append(int(metadata_array[idx, user_idx]))

    if min_user_reviews > 1:
        texts, labels, user_ids = _filter_min_reviews(texts, labels, user_ids, min_user_reviews)
    return texts, labels, user_ids


def _load_raw_split(
    dataset,
    cfg: DatasetCfg,
    subset_name: str,
    cache_root: Path,
    use_cache: bool,
) -> Tuple[List[str], List[int], List[int]]:
    include_id, exclude_ids = _split_category_filter(cfg, subset_name)
    cache_path = raw_split_cache_path(
        cache_root,
        subset_name,
        include_id,
        cfg.min_user_reviews,
        exclude_category_ids=exclude_ids,
    )
    if use_cache:
        cached = load_raw_split_pickle(cache_path)
        if cached is not None:
            print(f"[cache] {subset_name}: {len(cached['texts'])} reviews from {cache_path.name}", flush=True)
            return cached["texts"], cached["labels"], cached["user_ids"]

    print(f"[wilds] extracting official {subset_name} split (first run may take 2-3 min)...", flush=True)
    texts, labels, user_ids = _extract_subset_from_wilds(
        dataset,
        subset_name,
        include_id,
        cfg.min_user_reviews,
        exclude_category_ids=exclude_ids,
    )
    if use_cache:
        save_raw_split_pickle(cache_path, texts, labels, user_ids)
        print(f"[cache] saved {subset_name} -> {cache_path.name} ({len(texts)} reviews)", flush=True)
    return texts, labels, user_ids


def _filter_min_reviews(
    texts: List[str],
    labels: List[int],
    user_ids: List[int],
    min_reviews: int,
) -> Tuple[List[str], List[int], List[int]]:
    from collections import Counter

    counts = Counter(user_ids)
    keep_users = {u for u, c in counts.items() if c >= min_reviews}
    out_t, out_l, out_u = [], [], []
    for t, l, u in zip(texts, labels, user_ids):
        if u in keep_users:
            out_t.append(t)
            out_l.append(l)
            out_u.append(u)
    return out_t, out_l, out_u


def _cap_users(
    texts: List[str],
    labels: List[int],
    user_ids: List[int],
    max_users: Optional[int],
    max_reviews_per_user: Optional[int],
    seed: int,
) -> Tuple[List[str], List[int], List[int]]:
    if max_users is None and max_reviews_per_user is None:
        return texts, labels, user_ids
    rng = np.random.RandomState(seed)
    unique = sorted(set(user_ids))
    if max_users is not None and len(unique) > max_users:
        pick = set(rng.choice(unique, size=max_users, replace=False).tolist())
    else:
        pick = set(unique)
    out_t, out_l, out_u = [], [], []
    per_user: Dict[int, List[int]] = {}
    for i, u in enumerate(user_ids):
        if u not in pick:
            continue
        per_user.setdefault(u, []).append(i)
    for u in sorted(per_user.keys()):
        idxs = per_user[u]
        if max_reviews_per_user and len(idxs) > max_reviews_per_user:
            idxs = rng.choice(idxs, size=max_reviews_per_user, replace=False).tolist()
        for i in idxs:
            out_t.append(texts[i])
            out_l.append(labels[i])
            out_u.append(user_ids[i])
    return out_t, out_l, out_u


def load_amazon_splits(cfg: DatasetCfg, seed: int = 42) -> Dict[str, ReviewSplit]:
    """
    Load official WILDS Amazon OOD splits (train / val / test are user-disjoint).

    Caching layers:
    1. Per-split pickle of raw extraction (slow CSV parse) under cache_dir.
    2. In-memory dict keyed by caps + seed (fast within process / repeated experiments).
    """
    use_cache = cfg.use_cache
    mem_key = splits_memory_key(
        cfg.data_root,
        cfg.category_id,
        cfg.min_user_reviews,
        cfg.max_train_users,
        cfg.max_val_users,
        cfg.max_test_users,
        cfg.max_reviews_per_user,
        seed,
        train_category_id=cfg.train_category_id,
        eval_exclude_category_ids=cfg.eval_exclude_category_ids,
    )
    if use_cache:
        hit = get_cached_splits(mem_key)
        if hit is not None:
            return hit

    cache_root = resolve_cache_dir(cfg.data_root, cfg.cache_dir)
    # Lazy: only open the heavy WILDS object if a raw-split pickle misses.
    dataset = None
    splits: Dict[str, ReviewSplit] = {}

    cap_by_split = {
        "train": (cfg.max_train_users, seed),
        "validation": (cfg.max_val_users, seed + 1),
        "test": (cfg.max_test_users, seed + 2),
    }
    for name in ("train", "validation", "test"):
        include_id, exclude_ids = _split_category_filter(cfg, name)
        cache_path = raw_split_cache_path(
            cache_root,
            name,
            include_id,
            cfg.min_user_reviews,
            exclude_category_ids=exclude_ids,
        )
        cached = load_raw_split_pickle(cache_path) if use_cache else None
        if cached is not None:
            print(f"[cache] {name}: {len(cached['texts'])} reviews from {cache_path.name}", flush=True)
            texts, labels, user_ids = cached["texts"], cached["labels"], cached["user_ids"]
        else:
            if dataset is None:
                dataset = _load_wilds_dataset(cfg.data_root)
            texts, labels, user_ids = _load_raw_split(dataset, cfg, name, cache_root, use_cache)
        max_u, cap_seed = cap_by_split[name]
        texts, labels, user_ids = _cap_users(
            texts, labels, user_ids, max_u, cfg.max_reviews_per_user, cap_seed
        )
        splits[name] = ReviewSplit(name=name, texts=texts, labels=labels, user_ids=user_ids)

    if not splits["train"].user_disjoint_check(splits["validation"]):
        raise RuntimeError("Train and validation users overlap — check WILDS loader")
    if not splits["train"].user_disjoint_check(splits["test"]):
        raise RuntimeError("Train and test users overlap — check WILDS loader")
    if not splits["validation"].user_disjoint_check(splits["test"]):
        raise RuntimeError("Validation and test users overlap — check WILDS loader")

    if use_cache:
        set_cached_splits(mem_key, splits)
    return splits


def subsample_split(split: ReviewSplit, max_examples: int, seed: int) -> ReviewSplit:
    if len(split) <= max_examples:
        return split
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(split), size=max_examples, replace=False)
    idx = sorted(int(i) for i in idx)
    return ReviewSplit(
        name=split.name,
        texts=[split.texts[i] for i in idx],
        labels=[split.labels[i] for i in idx],
        user_ids=[split.user_ids[i] for i in idx],
        example_cluster_ids=(
            [split.example_cluster_ids[i] for i in idx]
            if split.example_cluster_ids is not None
            else None
        ),
    )

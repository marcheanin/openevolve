"""Disk + in-memory caches for WILDS split loading and text embeddings."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

CACHE_VERSION = 2  # bump to invalidate stale pickles after format changes

# In-process caches (survive repeated evaluate() / experiment restarts in same interpreter)
_SPLITS_MEMORY_CACHE: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
_EMBEDDINGS_MEMORY_CACHE: Dict[Tuple[str, str, int], np.ndarray] = {}


def resolve_cache_dir(data_root: str, cache_dir: Optional[str] = None) -> Path:
    if cache_dir:
        p = Path(cache_dir)
    else:
        p = Path(data_root) / ".prime_cache"
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def _tag(value: Any) -> str:
    if value is None:
        return "all"
    return str(value).replace("/", "_")


def raw_split_cache_path(
    cache_root: Path,
    split_name: str,
    category_id: Optional[int],
    min_user_reviews: int,
    *,
    exclude_category_ids: Optional[List[int]] = None,
) -> Path:
    if exclude_category_ids:
        excl = "ex" + "-".join(str(int(x)) for x in sorted(exclude_category_ids))
        cat_tag = excl
    else:
        cat_tag = f"cat{_tag(category_id)}"
    fname = f"wilds_amazon_{split_name}_{cat_tag}_minrev{min_user_reviews}_v{CACHE_VERSION}.pkl"
    return cache_root / fname


def splits_memory_key(
    data_root: str,
    category_id: Optional[int],
    min_user_reviews: int,
    max_train_users: Optional[int],
    max_val_users: Optional[int],
    max_test_users: Optional[int],
    max_reviews_per_user: Optional[int],
    seed: int,
    *,
    train_category_id: Optional[int] = None,
    eval_exclude_category_ids: Optional[List[int]] = None,
) -> Tuple[Any, ...]:
    excl = (
        tuple(sorted(int(x) for x in eval_exclude_category_ids))
        if eval_exclude_category_ids
        else ()
    )
    return (
        str(Path(data_root).resolve()),
        category_id,
        train_category_id,
        excl,
        min_user_reviews,
        max_train_users,
        max_val_users,
        max_test_users,
        max_reviews_per_user,
        seed,
    )


def load_raw_split_pickle(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if data.get("version") != CACHE_VERSION:
            return None
        return data
    except Exception:
        return None


def save_raw_split_pickle(path: Path, texts: List[str], labels: List[int], user_ids: List[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CACHE_VERSION,
        "texts": texts,
        "labels": labels,
        "user_ids": user_ids,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def get_cached_splits(memory_key: Tuple[Any, ...]) -> Optional[Dict[str, Any]]:
    return _SPLITS_MEMORY_CACHE.get(memory_key)


def set_cached_splits(memory_key: Tuple[Any, ...], splits: Dict[str, Any]) -> None:
    _SPLITS_MEMORY_CACHE[memory_key] = splits


def embeddings_cache_path(cache_root: Path, model_name: str, tag: str) -> Path:
    model_slug = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:12]
    safe_tag = _tag(tag)
    return cache_root / f"embeddings_{safe_tag}_{model_slug}_v{CACHE_VERSION}.pkl"


def load_embeddings_pickle(path: Path) -> Optional[np.ndarray]:
    if not path.is_file():
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if data.get("version") != CACHE_VERSION:
            return None
        arr = data.get("embeddings")
        if arr is None:
            return None
        return np.asarray(arr, dtype=np.float32)
    except Exception:
        return None


def save_embeddings_pickle(path: Path, embeddings: np.ndarray, texts_len: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CACHE_VERSION,
        "n_texts": texts_len,
        "embeddings": np.asarray(embeddings, dtype=np.float32),
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def encode_texts_cached(
    texts: List[str],
    model_name: str,
    cache_root: Path,
    tag: str,
    use_cache: bool = True,
) -> np.ndarray:
    """
    Encode texts with sentence-transformers; reuse disk/memory cache when len matches.
    """
    mem_key = (model_name, tag, len(texts))
    if use_cache and mem_key in _EMBEDDINGS_MEMORY_CACHE:
        return _EMBEDDINGS_MEMORY_CACHE[mem_key]

    disk_path = embeddings_cache_path(cache_root, model_name, tag)
    if use_cache:
        cached = load_embeddings_pickle(disk_path)
        if cached is not None and len(cached) == len(texts):
            _EMBEDDINGS_MEMORY_CACHE[mem_key] = cached
            return cached

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    emb = model.encode(texts, show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    if use_cache:
        save_embeddings_pickle(disk_path, emb, len(texts))
        _EMBEDDINGS_MEMORY_CACHE[mem_key] = emb
    return emb

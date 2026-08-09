"""CivilComments-WILDS loader with oracle identity groups.

Primary source: HuggingFace ``pietrolesci/civilcomments-wilds`` (CodaLab often 503).
Fallback: local WILDS ``civilcomments_v1.0/all_data_with_identities.csv``.

Official split names: train / validation / test.
Labels: binary toxicity {0, 1}.
Oracle groups: primary identity among the 8 WILDS identity attributes, else
``none`` (group 0). Groups are per-comment; ``user_ids`` are synthetic comment
indices so the ReviewSplit contract stays intact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from prime.config import DatasetCfg
from prime.data.cache import (
    get_cached_splits,
    load_raw_split_pickle,
    resolve_cache_dir,
    set_cached_splits,
)
from prime.data.wilds_loader import ReviewSplit

IDENTITY_VARS: Tuple[str, ...] = (
    "male",
    "female",
    "LGBTQ",
    "christian",
    "muslim",
    "other_religions",
    "black",
    "white",
)

ORACLE_GROUP_NAMES: Tuple[str, ...] = ("none",) + IDENTITY_VARS

CACHE_PREFIX = "wilds_civilcomments"
CACHE_VERSION_CC = 1
HF_REPO = "pietrolesci/civilcomments-wilds"


def oracle_group_name(group_id: int) -> str:
    if 0 <= group_id < len(ORACLE_GROUP_NAMES):
        return ORACLE_GROUP_NAMES[group_id]
    return f"unknown_{group_id}"


def primary_oracle_group(identity_row: Sequence[float]) -> int:
    """First identity ≥ 0.5 wins (WILDS threshold); else 0 (none)."""
    for i, v in enumerate(identity_row):
        if float(v) >= 0.5:
            return i + 1
    return 0


def _cc_raw_cache_path(cache_root: Path, split_name: str) -> Path:
    return cache_root / f"{CACHE_PREFIX}_{split_name}_v{CACHE_VERSION_CC}.pkl"


def _cc_memory_key(cfg: DatasetCfg, seed: int) -> Tuple[object, ...]:
    return (
        "civilcomments",
        str(Path(cfg.data_root).resolve()),
        cfg.max_train_users,
        cfg.max_val_users,
        cfg.max_test_users,
        seed,
        tuple(IDENTITY_VARS),
        CACHE_VERSION_CC,
    )


def _cap_examples(
    texts: List[str],
    labels: List[int],
    user_ids: List[int],
    group_ids: List[int],
    max_examples: Optional[int],
    seed: int,
) -> Tuple[List[str], List[int], List[int], List[int]]:
    n = len(texts)
    if max_examples is None or n <= max_examples:
        return texts, labels, user_ids, group_ids
    rng = np.random.RandomState(seed)
    by_g: Dict[int, List[int]] = {}
    for i, g in enumerate(group_ids):
        by_g.setdefault(int(g), []).append(i)
    groups = sorted(by_g.keys())
    counts = {g: len(by_g[g]) for g in groups}
    total = sum(counts.values())
    alloc = {g: max(1, int(round(max_examples * counts[g] / total))) for g in groups}
    while sum(alloc.values()) > max_examples:
        g_max = max(groups, key=lambda g: alloc[g])
        if alloc[g_max] > 1:
            alloc[g_max] -= 1
        else:
            break
    while sum(alloc.values()) < max_examples:
        remainders = [(g, counts[g] - alloc[g]) for g in groups if counts[g] > alloc[g]]
        if not remainders:
            break
        g_max = max(remainders, key=lambda t: t[1])[0]
        alloc[g_max] += 1
    idx: List[int] = []
    for g in groups:
        pool = by_g[g]
        take = min(alloc[g], len(pool))
        chosen = rng.choice(pool, size=take, replace=False).tolist()
        idx.extend(int(i) for i in chosen)
    idx = sorted(idx)[:max_examples]
    return (
        [texts[i] for i in idx],
        [labels[i] for i in idx],
        [user_ids[i] for i in idx],
        [group_ids[i] for i in idx],
    )


def _rows_from_hf_split(hf_split) -> Tuple[List[str], List[int], List[int], List[int]]:
    """Vectorized materialization (Python row loops are too slow at ~270k)."""
    try:
        df = hf_split.to_pandas()
    except Exception:
        df = None
    if df is not None:
        texts = [str(t) for t in df["comment_text"].tolist()]
        tox = df["toxicity"]
        if tox.dtype == object or str(tox.dtype).startswith("category"):
            labels = [1 if (isinstance(y, float) and y >= 0.5) or int(y) == 1 else 0 for y in tox.tolist()]
        else:
            labels = [1 if float(y) >= 0.5 else 0 for y in tox.tolist()]
        id_mat = df.loc[:, list(IDENTITY_VARS)].to_numpy(dtype=float)
        # First identity ≥ 0.5; else 0 (none). Vectorized via argmax on mask.
        mask = id_mat >= 0.5
        any_id = mask.any(axis=1)
        first = mask.argmax(axis=1) + 1  # 1..8
        group_ids = np.where(any_id, first, 0).astype(int).tolist()
        labels_arr = np.asarray(labels, dtype=int).tolist()
        user_ids = list(range(len(texts)))
        return texts, labels_arr, user_ids, group_ids

    # Fallback: columnar lists
    texts = [str(t) for t in hf_split["comment_text"]]
    tox_col = hf_split["toxicity"]
    labels = [1 if (isinstance(y, float) and y >= 0.5) or int(y) == 1 else 0 for y in tox_col]
    id_cols = [hf_split[name] for name in IDENTITY_VARS]
    group_ids = []
    n = len(texts)
    for i in range(n):
        id_row = [float(col[i]) for col in id_cols]
        group_ids.append(primary_oracle_group(id_row))
    return texts, labels, list(range(n)), group_ids


def _load_from_huggingface() -> Dict[str, Tuple[List[str], List[int], List[int], List[int]]]:
    from datasets import load_dataset

    print(f"[civilcomments] loading from HuggingFace ({HF_REPO})...", flush=True)
    ds = load_dataset(HF_REPO)
    out: Dict[str, Tuple[List[str], List[int], List[int], List[int]]] = {}
    mapping = {"train": "train", "validation": "validation", "test": "test"}
    for our_name, hf_name in mapping.items():
        print(f"[civilcomments] materializing {our_name} ({len(ds[hf_name])} rows)...", flush=True)
        out[our_name] = _rows_from_hf_split(ds[hf_name])
    return out


def _load_from_wilds(data_root: str) -> Dict[str, Tuple[List[str], List[int], List[int], List[int]]]:
    from wilds import get_dataset

    dataset = get_dataset(
        dataset="civilcomments",
        root_dir=data_root,
        download=False,
        unlabeled=False,
    )
    out: Dict[str, Tuple[List[str], List[int], List[int], List[int]]] = {}
    for name, key in (("train", "train"), ("validation", "val"), ("test", "test")):
        split_data = dataset.get_subset(key)
        fields = list(dataset.metadata_fields)
        id_idxs = [fields.index(n) for n in IDENTITY_VARS]
        meta = split_data.metadata_array
        meta_np = meta.numpy() if hasattr(meta, "numpy") else np.asarray(meta)
        texts: List[str] = []
        labels: List[int] = []
        user_ids: List[int] = []
        group_ids: List[int] = []
        n = len(split_data)
        for i in range(n):
            if i > 0 and i % 20000 == 0:
                print(f"  [wilds {name}] {i}/{n}...", flush=True)
            x, y, _ = split_data[i]
            texts.append(str(x))
            labels.append(int(y))
            user_ids.append(i)
            id_row = [float(meta_np[i, j]) for j in id_idxs]
            group_ids.append(primary_oracle_group(id_row))
        out[name] = (texts, labels, user_ids, group_ids)
    return out


def _extract_all_splits(cfg: DatasetCfg) -> Dict[str, Tuple[List[str], List[int], List[int], List[int]]]:
    csv_path = Path(cfg.data_root) / "civilcomments_v1.0" / "all_data_with_identities.csv"
    if csv_path.is_file():
        print(f"[civilcomments] using local WILDS CSV: {csv_path}", flush=True)
        return _load_from_wilds(cfg.data_root)
    try:
        return _load_from_huggingface()
    except Exception as hf_exc:
        print(f"[civilcomments] HuggingFace load failed ({hf_exc}); trying WILDS download...", flush=True)
        from wilds import get_dataset

        get_dataset(dataset="civilcomments", root_dir=cfg.data_root, download=True)
        return _load_from_wilds(cfg.data_root)


def load_civilcomments_splits(cfg: DatasetCfg, seed: int = 42) -> Dict[str, ReviewSplit]:
    """
    Load CivilComments official splits with oracle identity groups on
    ``example_cluster_ids``. Caps reuse ``max_*_users`` as max *examples*.
    """
    use_cache = cfg.use_cache
    mem_key = _cc_memory_key(cfg, seed)
    if use_cache:
        hit = get_cached_splits(mem_key)
        if hit is not None:
            return hit

    cache_root = resolve_cache_dir(cfg.data_root, cfg.cache_dir)
    raw_by_split: Optional[Dict[str, Tuple[List[str], List[int], List[int], List[int]]]] = None
    splits: Dict[str, ReviewSplit] = {}
    cap_by_split = {
        "train": (cfg.max_train_users, seed),
        "validation": (cfg.max_val_users, seed + 1),
        "test": (cfg.max_test_users, seed + 2),
    }
    offset = {"train": 0, "validation": 10_000_000, "test": 20_000_000}

    for name in ("train", "validation", "test"):
        cache_path = _cc_raw_cache_path(cache_root, name)
        cached = load_raw_split_pickle(cache_path) if use_cache else None
        if cached is not None and "group_ids" in cached:
            print(
                f"[cache] {name}: {len(cached['texts'])} comments from {cache_path.name}",
                flush=True,
            )
            texts = cached["texts"]
            labels = cached["labels"]
            user_ids = cached["user_ids"]
            group_ids = list(cached["group_ids"])
        else:
            if raw_by_split is None:
                raw_by_split = _extract_all_splits(cfg)
            texts, labels, user_ids, group_ids = raw_by_split[name]
            if use_cache:
                import pickle

                cache_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "version": 2,
                    "texts": texts,
                    "labels": labels,
                    "user_ids": user_ids,
                    "group_ids": group_ids,
                    "dataset": "civilcomments",
                    "identity_vars": list(IDENTITY_VARS),
                }
                with open(cache_path, "wb") as f:
                    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(
                    f"[cache] saved {name} -> {cache_path.name} ({len(texts)} comments)",
                    flush=True,
                )

        max_n, cap_seed = cap_by_split[name]
        texts, labels, user_ids, group_ids = _cap_examples(
            texts, labels, user_ids, group_ids, max_n, cap_seed
        )
        user_ids = [offset[name] + u for u in user_ids]
        splits[name] = ReviewSplit(
            name=name,
            texts=texts,
            labels=labels,
            user_ids=user_ids,
            example_cluster_ids=group_ids,
        )

    if use_cache:
        set_cached_splits(mem_key, splits)
    return splits


def build_oracle_cluster_artifacts(split: ReviewSplit) -> "ClusterArtifacts":
    """ClusterArtifacts from oracle ``example_cluster_ids`` already on the split."""
    from prime.data.clustering import ClusterArtifacts

    if split.example_cluster_ids is None:
        raise ValueError("oracle artifacts require example_cluster_ids on the split")
    mapping = {int(u): int(g) for u, g in zip(split.user_ids, split.example_cluster_ids)}
    n_clusters = max(len(ORACLE_GROUP_NAMES), max(mapping.values(), default=0) + 1)
    centroids = np.zeros((n_clusters, 1), dtype=np.float32)
    counts: Dict[int, int] = {}
    for g in split.example_cluster_ids:
        counts[int(g)] = counts.get(int(g), 0) + 1
    return ClusterArtifacts(
        n_clusters=n_clusters,
        user_to_cluster=mapping,
        cluster_centroids=centroids,
        cluster_centroids_label_free=centroids.copy(),
        embedding_model="oracle_identity",
        seed=0,
        pipeline=None,
        train_projection_agreement=1.0,
        is_synthetic=False,
        fit_mode="oracle",
        diagnostics={
            "group_names": list(ORACLE_GROUP_NAMES),
            "group_counts": {str(k): v for k, v in sorted(counts.items())},
            "identity_vars": list(IDENTITY_VARS),
        },
    )

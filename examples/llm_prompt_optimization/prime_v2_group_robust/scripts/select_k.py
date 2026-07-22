#!/usr/bin/env python3
"""Offline: recommend k for style clusters via silhouette + multi-seed ARI stability."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))

from prime.config import load_config
from prime.data.cache import encode_texts_cached, resolve_cache_dir
from prime.data.profiles import build_user_profiles, fit_profile_pipeline, profile_feature_matrix
from prime.data.wilds_loader import load_amazon_splits


def _encode(texts, model_name: str, dataset_cfg) -> np.ndarray:
    if dataset_cfg.use_cache:
        cache_root = resolve_cache_dir(dataset_cfg.data_root, dataset_cfg.cache_dir)
        return encode_texts_cached(texts, model_name, cache_root, "select_k_train", use_cache=True)
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name).encode(texts, show_progress_bar=False, convert_to_numpy=True)


def evaluate_k(
    X: np.ndarray,
    k: int,
    seeds: List[int],
) -> Tuple[float, float]:
    """Return (mean_silhouette, mean_pairwise_ARI) across seeds."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    if k < 2 or k >= len(X):
        return float("nan"), float("nan")

    labels_by_seed = []
    sils = []
    for seed in seeds:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labs = km.fit_predict(X)
        labels_by_seed.append(labs)
        if len(set(labs)) > 1:
            sils.append(float(silhouette_score(X, labs)))
    mean_sil = float(np.mean(sils)) if sils else float("nan")

    aris = []
    for i in range(len(labels_by_seed)):
        for j in range(i + 1, len(labels_by_seed)):
            aris.append(float(adjusted_rand_score(labels_by_seed[i], labels_by_seed[j])))
    mean_ari = float(np.mean(aris)) if aris else 1.0
    return mean_sil, mean_ari


def main() -> int:
    parser = argparse.ArgumentParser(description="Select style-cluster k (offline)")
    parser.add_argument("--config", type=Path, default=PKG_ROOT / "configs" / "base.yaml")
    parser.add_argument("--k-min", type=int, default=4)
    parser.add_argument("--k-max", type=int, default=12)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    cfg = load_config(args.config)
    splits = load_amazon_splits(cfg.dataset, seed=cfg.active_learning.seed)
    train = splits["train"]
    emb = _encode(train.texts, cfg.clusters.embedding_model, cfg.dataset)
    profiles = build_user_profiles(train.texts, train.labels, train.user_ids, emb)
    pipeline = fit_profile_pipeline(profiles, n_pca=cfg.clusters.pca_components)
    X, users = profile_feature_matrix(profiles, pipeline=pipeline, mode="full")
    print(f"Users={len(users)}  feature_dim={X.shape[1]}  k in [{args.k_min},{args.k_max}]")
    print(f"{'k':>4}  {'silhouette':>12}  {'ARI_stability':>14}")

    rows = []
    for k in range(args.k_min, args.k_max + 1):
        sil, ari = evaluate_k(X, k, list(args.seeds))
        rows.append((k, sil, ari))
        print(f"{k:4d}  {sil:12.4f}  {ari:14.4f}")

    # Recommend: maximize silhouette among ARI >= 0.5 (else max silhouette)
    eligible = [r for r in rows if r[2] >= 0.5 and not np.isnan(r[1])]
    pool = eligible or [r for r in rows if not np.isnan(r[1])]
    if not pool:
        print("No valid k found.")
        return 1
    best = max(pool, key=lambda r: (r[1], r[2]))
    print(f"\nRecommended k={best[0]}  (silhouette={best[1]:.4f}, ARI={best[2]:.4f})")
    print("Set clusters.n_clusters in your experiment YAML to this value (runtime does not auto-select).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

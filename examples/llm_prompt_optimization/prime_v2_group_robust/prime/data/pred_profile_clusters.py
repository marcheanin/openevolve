"""Prediction-profile clustering: group users by how the ensemble rates them.

Label-free at deploy: uses only model outputs (ensemble preds / optional
worker votes), never gold. Motivated by E0 large-cap finding that worst users
are those with many honest mid/high ratings the model collapses (esp. 4→5),
not unusual writing style (see experiments/OBSERVATIONS.md C4).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from prime.workers.ensemble import disagreement_score


def user_pred_profile_matrix(
    user_ids: np.ndarray,
    predictions: np.ndarray,
    worker_preds: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[int]]:
    """
    Build one feature row per user.

    Features (all from predictions, no gold):
      - fraction of preds in {1,2,3,4,5}  (5 dims)
      - mean pred, std pred
      - entropy of pred histogram
      - optional: mean worker disagreement per review (if worker_preds given)

    worker_preds: shape (n_workers, n_examples) if provided.
    """
    user_ids = np.asarray(user_ids)
    predictions = np.asarray(predictions)
    users = sorted(int(u) for u in np.unique(user_ids))
    rows: List[np.ndarray] = []
    for u in users:
        mask = user_ids == u
        preds_u = predictions[mask].astype(np.int64)
        hist = np.zeros(5, dtype=np.float64)
        for r in range(1, 6):
            hist[r - 1] = float(np.mean(preds_u == r)) if len(preds_u) else 0.0
        mean_p = float(np.mean(preds_u)) if len(preds_u) else 3.0
        std_p = float(np.std(preds_u)) if len(preds_u) else 0.0
        # entropy of rating mix
        p = hist[hist > 0]
        ent = float(-np.sum(p * np.log(p + 1e-12))) if len(p) else 0.0
        feat = [hist[0], hist[1], hist[2], hist[3], hist[4], mean_p / 5.0, std_p / 4.0, ent / np.log(5)]
        if worker_preds is not None:
            wp = np.asarray(worker_preds)
            # mean disagreement across this user's examples
            scores = []
            idxs = np.where(mask)[0]
            for i in idxs:
                votes = [int(wp[w, i]) for w in range(wp.shape[0])]
                scores.append(disagreement_score(votes))
            feat.append(float(np.mean(scores)) if scores else 0.0)
        rows.append(np.asarray(feat, dtype=np.float32))
    return np.vstack(rows), users


def _merge_tiny_clusters(
    labels: np.ndarray,
    Xs: np.ndarray,
    centroids: np.ndarray,
    min_users: int,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, int]]]:
    """
    Iteratively merge clusters with < min_users users into the nearest remaining
    centroid (Euclidean in scaled feature space). Relabel to contiguous 0..K'-1.
    """
    labels = np.asarray(labels, dtype=np.int64).copy()
    cents = np.asarray(centroids, dtype=np.float64).copy()
    merges: List[Dict[str, int]] = []
    if min_users <= 1:
        return labels, cents.astype(np.float32), merges

    while True:
        present = sorted(int(c) for c in np.unique(labels))
        if len(present) <= 1:
            break
        sizes = {c: int(np.sum(labels == c)) for c in present}
        tiny = [c for c in present if sizes[c] < min_users]
        if not tiny:
            break
        # Merge the smallest first (stable under ties via cluster id).
        src = min(tiny, key=lambda c: (sizes[c], c))
        others = [c for c in present if c != src]
        if not others:
            break
        d = [float(np.linalg.norm(cents[src] - cents[c])) for c in others]
        dst = others[int(np.argmin(d))]
        labels[labels == src] = dst
        merges.append({"from": int(src), "to": int(dst), "n_users": sizes[src]})
        # Recompute destination centroid from members still labeled dst.
        members = Xs[labels == dst]
        if len(members):
            cents[dst] = members.mean(axis=0)

    # Compact ids to 0..K'-1 and rebuild centroids.
    present = sorted(int(c) for c in np.unique(labels))
    remap = {old: new for new, old in enumerate(present)}
    labels = np.asarray([remap[int(c)] for c in labels], dtype=np.int64)
    new_cents = np.vstack([Xs[labels == c].mean(axis=0) for c in range(len(present))])
    return labels, new_cents.astype(np.float32), merges


def fit_pred_profile_clusters(
    user_ids: np.ndarray,
    predictions: np.ndarray,
    n_clusters: int,
    seed: int = 42,
    worker_preds: Optional[np.ndarray] = None,
    min_users_per_cluster: int = 0,
) -> Tuple[Dict[int, int], np.ndarray, Dict[str, object]]:
    """
    KMeans on prediction-profile features.
    Returns user→cluster, centroids, diagnostics.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    X, users = user_pred_profile_matrix(user_ids, predictions, worker_preds=worker_preds)
    if len(users) < 2:
        mapping = {users[0]: 0} if users else {}
        return mapping, np.zeros((1, X.shape[1]), dtype=np.float32), {"n_fit_users": len(users)}

    k = max(2, min(int(n_clusters), len(users) // 2))
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(Xs)
    cents = km.cluster_centers_.astype(np.float32)
    merges: List[Dict[str, int]] = []
    if min_users_per_cluster and min_users_per_cluster > 1:
        labels, cents, merges = _merge_tiny_clusters(
            labels, Xs, cents, int(min_users_per_cluster)
        )
    k_final = int(len(np.unique(labels)))
    mapping = {u: int(c) for u, c in zip(users, labels)}

    # Human-readable descriptors from centroid pred mix
    descriptors: Dict[str, str] = {}
    for c in range(k_final):
        members = [u for u, lab in mapping.items() if lab == c]
        if not members:
            continue
        m = np.isin(user_ids, members)
        preds_c = predictions[m]
        hist = {r: float(np.mean(preds_c == r)) for r in range(1, 6)}
        mean_p = float(np.mean(preds_c)) if len(preds_c) else 0.0
        top = max(hist, key=hist.get)
        descriptors[str(c)] = (
            f"pred-type {c}: n_users={len(members)}, mean_pred~={mean_p:.2f}, "
            f"top_mass={top}★({hist[top]:.0%}), "
            f"mix=" + ",".join(f"{r}:{hist[r]:.0%}" for r in range(1, 6))
        )

    sizes = {c: sum(1 for lab in mapping.values() if lab == c) for c in range(k_final)}
    diagnostics = {
        "n_fit_users": len(users),
        "n_clusters": k_final,
        "n_clusters_requested": k,
        "feature_dim": int(X.shape[1]),
        "has_disagreement": worker_preds is not None,
        "descriptors": descriptors,
        "cluster_sizes": sizes,
        "min_users_per_cluster": int(min_users_per_cluster or 0),
        "merges": merges,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    return mapping, cents, diagnostics


def assign_pred_profile_clusters(
    user_ids: np.ndarray,
    predictions: np.ndarray,
    centroids: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    worker_preds: Optional[np.ndarray] = None,
) -> Dict[int, int]:
    """Nearest-centroid assign in the same scaled pred-profile space."""
    X, users = user_pred_profile_matrix(user_ids, predictions, worker_preds=worker_preds)
    scale = np.where(np.asarray(scaler_scale) == 0, 1.0, np.asarray(scaler_scale))
    Xs = (X - np.asarray(scaler_mean)) / scale
    mapping: Dict[int, int] = {}
    for i, u in enumerate(users):
        d = np.linalg.norm(centroids - Xs[i], axis=1)
        mapping[int(u)] = int(np.argmin(d))
    return mapping


def fit_pred_profile_as_cluster_artifacts(
    user_ids: np.ndarray,
    predictions: np.ndarray,
    n_clusters: int,
    seed: int = 42,
    worker_preds: Optional[np.ndarray] = None,
    max_k: Optional[int] = None,
    min_users_per_cluster: int = 0,
):
    """Wrap pred-profile fit into ClusterArtifacts for the controller."""
    from prime.data.clustering import ClusterArtifacts

    k = int(n_clusters)
    if max_k is not None:
        k = max(2, min(k, int(max_k)))
    mapping, centroids, diagnostics = fit_pred_profile_clusters(
        user_ids,
        predictions,
        n_clusters=k,
        seed=seed,
        worker_preds=worker_preds,
        min_users_per_cluster=min_users_per_cluster,
    )
    sm = np.asarray(diagnostics["scaler_mean"], dtype=np.float32)
    ss = np.asarray(diagnostics["scaler_scale"], dtype=np.float32)
    return ClusterArtifacts(
        n_clusters=int(diagnostics["n_clusters"]),
        user_to_cluster=mapping,
        cluster_centroids=centroids,
        cluster_centroids_label_free=centroids,
        embedding_model="pred_profile",
        seed=seed,
        pipeline=None,
        train_projection_agreement=1.0,
        is_synthetic=False,
        fit_mode="pred_profile",
        diagnostics=diagnostics,
        scaler_mean=sm,
        scaler_scale=ss,
    )

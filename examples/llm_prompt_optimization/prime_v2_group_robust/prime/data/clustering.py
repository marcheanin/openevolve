"""Style cluster assignment: train+test users share one cluster space."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from prime.config import ClusterCfg, DatasetCfg
from prime.data.cache import encode_texts_cached, resolve_cache_dir
from prime.data.profiles import (
    ProfilePipeline,
    build_user_profiles,
    build_user_profiles_label_free,
    fit_profile_pipeline,
    projection_agreement,
    transform_emb_only,
    transform_label_free,
)
from prime.data.wilds_loader import ReviewSplit

logger = logging.getLogger(__name__)


@dataclass
class ClusterArtifacts:
    n_clusters: int
    user_to_cluster: Dict[int, int]
    cluster_centroids: np.ndarray
    cluster_centroids_label_free: np.ndarray
    embedding_model: str
    seed: int
    pipeline: Optional[ProfilePipeline] = None
    train_projection_agreement: float = 1.0
    is_synthetic: bool = False
    # label_free | full | emb_only | pred_profile
    fit_mode: str = "full"
    diagnostics: Optional[Dict[str, object]] = None
    # pred_profile scaler (assignment space); None for style geometries
    scaler_mean: Optional[np.ndarray] = None
    scaler_scale: Optional[np.ndarray] = None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_clusters": self.n_clusters,
            "user_to_cluster": {str(k): v for k, v in self.user_to_cluster.items()},
            "cluster_centroids": self.cluster_centroids.tolist(),
            "cluster_centroids_label_free": self.cluster_centroids_label_free.tolist(),
            "embedding_model": self.embedding_model,
            "seed": self.seed,
            "train_projection_agreement": self.train_projection_agreement,
            "is_synthetic": self.is_synthetic,
            "fit_mode": self.fit_mode,
            "diagnostics": self.diagnostics,
        }
        if self.pipeline is not None:
            payload["pipeline"] = self.pipeline.to_dict()
        if self.scaler_mean is not None:
            payload["scaler_mean"] = self.scaler_mean.tolist()
        if self.scaler_scale is not None:
            payload["scaler_scale"] = self.scaler_scale.tolist()
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ClusterArtifacts":
        data = json.loads(path.read_text(encoding="utf-8"))
        pipeline = None
        if "pipeline" in data:
            pipeline = ProfilePipeline.from_dict(data["pipeline"])
        lf = data.get("cluster_centroids_label_free")
        if lf is None:
            lf = data["cluster_centroids"]
            logger.warning(
                "Cluster artifact %s lacks label-free centroids; using full centroids (legacy).",
                path,
            )
        sm = data.get("scaler_mean")
        ss = data.get("scaler_scale")
        return cls(
            n_clusters=int(data["n_clusters"]),
            user_to_cluster={int(k): int(v) for k, v in data["user_to_cluster"].items()},
            cluster_centroids=np.array(data["cluster_centroids"], dtype=np.float32),
            cluster_centroids_label_free=np.array(lf, dtype=np.float32),
            embedding_model=str(data["embedding_model"]),
            seed=int(data["seed"]),
            pipeline=pipeline,
            train_projection_agreement=float(data.get("train_projection_agreement", 1.0)),
            is_synthetic=bool(data.get("is_synthetic", False)),
            fit_mode=str(data.get("fit_mode", "full")),
            diagnostics=data.get("diagnostics"),
            scaler_mean=np.array(sm, dtype=np.float32) if sm is not None else None,
            scaler_scale=np.array(ss, dtype=np.float32) if ss is not None else None,
        )


def _encode_texts(
    texts: List[str],
    model_name: str,
    dataset_cfg: Optional[DatasetCfg] = None,
    tag: str = "generic",
) -> np.ndarray:
    if dataset_cfg is not None and dataset_cfg.use_cache:
        cache_root = resolve_cache_dir(dataset_cfg.data_root, dataset_cfg.cache_dir)
        return encode_texts_cached(texts, model_name, cache_root, tag, use_cache=True)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)


def _compute_label_free_centroids(
    profiles: Dict[int, object],
    user_to_cluster: Dict[int, int],
    pipeline: ProfilePipeline,
    n_clusters: int,
) -> np.ndarray:
    """Mean label-free feature vector per cluster (not a projection of full centroids)."""
    buckets: Dict[int, List[np.ndarray]] = {c: [] for c in range(n_clusters)}
    for u, cid in user_to_cluster.items():
        if u in profiles:
            buckets[cid].append(transform_label_free(profiles[u], pipeline))
    centroids = []
    for c in range(n_clusters):
        if buckets[c]:
            centroids.append(np.mean(buckets[c], axis=0))
        else:
            centroids.append(np.zeros(pipeline.n_pca + len(pipeline.label_free_scaler_mean)))
    return np.vstack(centroids).astype(np.float32)


def _rating_diagnostic(
    profiles: Dict[int, object],
    user_to_cluster: Dict[int, int],
    n_clusters: int,
) -> Dict[str, object]:
    """
    External cluster validation via rating stats (SPEC v3 §4.1 / Р8): rating
    features are NOT in the cluster geometry; here they only report whether
    label-free clusters are behaviorally meaningful (one-way ANOVA F-stat over
    per-cluster mean_rating).
    """
    groups: Dict[int, List[float]] = {c: [] for c in range(n_clusters)}
    for u, cid in user_to_cluster.items():
        p = profiles.get(u)
        if p is not None:
            groups[cid].append(float(p.mean_rating))
    filled = [np.array(v) for v in groups.values() if len(v) > 0]
    per_cluster = {
        str(c): {
            "n_users": len(v),
            "mean_rating": float(np.mean(v)) if v else None,
            "std_rating": float(np.std(v)) if v else None,
        }
        for c, v in groups.items()
    }
    f_stat = None
    if len(filled) >= 2 and sum(len(g) for g in filled) > len(filled):
        grand = np.concatenate(filled).mean()
        ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in filled)
        ss_within = sum(((g - g.mean()) ** 2).sum() for g in filled)
        df_b = len(filled) - 1
        df_w = sum(len(g) for g in filled) - len(filled)
        if df_w > 0 and ss_within > 0:
            f_stat = float((ss_between / df_b) / (ss_within / df_w))
    return {"per_cluster_rating": per_cluster, "anova_f_mean_rating": f_stat}


def _dispersion_diagnostic(
    review_embeddings: np.ndarray,
    user_ids: List[int],
    profiles: Dict[int, object],
) -> Dict[str, float]:
    """
    Intra-user dispersion (SPEC v3 §4.1): mean distance of a user's review
    embeddings to their own profile embedding. High share of "smeared" users is
    the trigger for the soft-membership ablation (Р1).
    """
    by_user: Dict[int, List[int]] = {}
    for i, u in enumerate(user_ids):
        by_user.setdefault(u, []).append(i)
    dispersions: List[float] = []
    for u, idxs in by_user.items():
        p = profiles.get(u)
        if p is None or len(idxs) < 2:
            continue
        center = p.text_embedding
        d = float(np.mean(np.linalg.norm(review_embeddings[idxs] - center, axis=1)))
        dispersions.append(d)
    if not dispersions:
        return {"mean": 0.0, "p90": 0.0, "n_users": 0}
    return {
        "mean": float(np.mean(dispersions)),
        "p90": float(np.percentile(dispersions, 90)),
        "n_users": len(dispersions),
    }


def fit_style_clusters(
    train_split: ReviewSplit,
    cfg: ClusterCfg,
    dataset_cfg: Optional[DatasetCfg] = None,
    fit_mode: str = "label_free",
    min_reviews_for_fit: int = 3,
    max_k: Optional[int] = None,
) -> ClusterArtifacts:
    """
    Fit k-means on user profiles.

    SPEC v3 §4.1 (Р8): default fit is in the **label-free** space (text embedding
    PCA + length/punctuation/caps, no rating stats) — one geometry for
    train/val/test/deploy. Rating stats are used only as external validation
    (see _rating_diagnostic). `fit_mode='emb_only'` is an E0 ablation that
    drops length/punct/caps so semantics can be isolated from length dominance.
    Users with fewer than `min_reviews_for_fit` reviews do not shape the
    geometry; they are assigned to fitted centroids. `max_k` implements the
    power rule K <= |D_select| / n_min (Р9).
    """
    from sklearn.cluster import KMeans

    emb = _encode_texts(
        train_split.texts, cfg.embedding_model, dataset_cfg, tag="train_cluster_fit"
    )
    profiles = build_user_profiles(
        train_split.texts, train_split.labels, train_split.user_ids, emb
    )
    pipeline = fit_profile_pipeline(profiles, n_pca=cfg.pca_components)

    feature_mode = {
        "label_free": "label_free",
        "emb_only": "emb_only",
        "full": "full",
    }.get(fit_mode, "label_free")

    if fit_mode in ("label_free", "emb_only"):
        eligible = {
            u: p for u, p in profiles.items() if p.n_reviews >= min_reviews_for_fit
        }
        if len(eligible) < 2:
            eligible = dict(profiles)
        X, users = _feature_matrix(eligible, pipeline, mode=feature_mode)
    else:
        eligible = dict(profiles)
        X, users = _feature_matrix(eligible, pipeline, mode="full")

    k = min(cfg.n_clusters, max(2, len(users) // 3))
    if max_k is not None:
        k = max(2, min(k, int(max_k)))
    km = KMeans(n_clusters=k, random_state=cfg.seed, n_init=10)
    labels = km.fit_predict(X)
    user_to_cluster = {u: int(l) for u, l in zip(users, labels)}

    if fit_mode in ("label_free", "emb_only"):
        lf_centroids = km.cluster_centers_.astype(np.float32)
        transform = transform_emb_only if fit_mode == "emb_only" else transform_label_free
        # Assign below-threshold users to fitted centroids (they didn't shape geometry).
        for u, p in profiles.items():
            if u not in user_to_cluster:
                vec = transform(p, pipeline)
                user_to_cluster[u] = int(
                    np.argmin(np.linalg.norm(lf_centroids - vec, axis=1))
                )
        # Diagnostic centroids in full space (rating-aware view of same clusters).
        full_centroids = _compute_full_centroids(profiles, user_to_cluster, pipeline, k)
        agreement = _full_vs_label_free_agreement(
            profiles, user_to_cluster, pipeline, full_centroids
        )
    else:
        lf_centroids = _compute_label_free_centroids(profiles, user_to_cluster, pipeline, k)
        full_centroids = km.cluster_centers_.astype(np.float32)
        agreement = projection_agreement(profiles, user_to_cluster, pipeline, lf_centroids)

    diagnostics: Dict[str, object] = _rating_diagnostic(profiles, user_to_cluster, k)
    diagnostics["intra_user_dispersion"] = _dispersion_diagnostic(
        emb, train_split.user_ids, profiles
    )
    diagnostics["n_fit_users"] = len(users)
    diagnostics["n_assigned_below_threshold"] = len(profiles) - len(users)

    # Р13: random-clusters ablation — shuffle assignments after fit.
    if getattr(cfg, "control", "none") == "shuffle":
        rng = np.random.RandomState(cfg.seed + 9973)
        uids = list(user_to_cluster.keys())
        shuffled = rng.permutation([user_to_cluster[u] for u in uids])
        user_to_cluster = {u: int(c) for u, c in zip(uids, shuffled)}
        diagnostics["control"] = "shuffle"
        logger.info("clusters.control=shuffle: remapped %d users (Р13 ablation)", len(uids))
    else:
        diagnostics["control"] = "none"

    # Р4: template type descriptors from nearest neighbors in feature space.
    descriptors: Dict[str, str] = {}
    if getattr(cfg, "descriptors_enabled", True) and len(users) >= 2:
        descriptors = _build_type_descriptors(
            profiles,
            user_to_cluster,
            X,
            users,
            top_n=int(getattr(cfg, "descriptors_top_n", 5)),
        )
        diagnostics["descriptors"] = descriptors

    return ClusterArtifacts(
        n_clusters=k,
        user_to_cluster=user_to_cluster,
        cluster_centroids=full_centroids,
        cluster_centroids_label_free=lf_centroids,
        embedding_model=cfg.embedding_model,
        seed=cfg.seed,
        pipeline=pipeline,
        train_projection_agreement=agreement,
        is_synthetic=False,
        fit_mode=fit_mode,
        diagnostics=diagnostics,
    )


def _build_type_descriptors(
    profiles: Dict[int, object],
    user_to_cluster: Dict[int, int],
    X: np.ndarray,
    users: List[int],
    top_n: int = 5,
) -> Dict[str, str]:
    """Template descriptors from nearest in-cluster neighbors (no LLM editor)."""
    user_to_row = {u: i for i, u in enumerate(users)}
    by_cluster: Dict[int, List[int]] = {}
    for u, cid in user_to_cluster.items():
        if u in user_to_row:
            by_cluster.setdefault(int(cid), []).append(u)

    out: Dict[str, str] = {}
    for cid, members in sorted(by_cluster.items()):
        lengths = [float(profiles[u].mean_length) for u in members if u in profiles]
        mean_len = float(np.mean(lengths)) if lengths else 0.0
        # Nearest to centroid among members.
        rows = [user_to_row[u] for u in members if u in user_to_row]
        if not rows:
            out[str(cid)] = f"type {cid}: (empty)"
            continue
        centroid = X[rows].mean(axis=0)
        dists = [(u, float(np.linalg.norm(X[user_to_row[u]] - centroid))) for u in members if u in user_to_row]
        dists.sort(key=lambda t: t[1])
        neighbors = [u for u, _ in dists[:top_n]]
        punct = float(np.mean([profiles[u].punctuation_density for u in neighbors if u in profiles]))
        caps = float(np.mean([profiles[u].caps_ratio for u in neighbors if u in profiles]))
        style = []
        if mean_len < 80:
            style.append("short reviews")
        elif mean_len > 250:
            style.append("long reviews")
        else:
            style.append("medium-length reviews")
        if punct > 0.05:
            style.append("punctuation-heavy")
        if caps > 0.15:
            style.append("caps-heavy / emphatic")
        out[str(cid)] = (
            f"type {cid}: {', '.join(style)}; "
            f"n_users={len(members)}, mean_len≈{mean_len:.0f}, "
            f"nearest_users={neighbors}"
        )
    return out


def _compute_full_centroids(
    profiles: Dict[int, object],
    user_to_cluster: Dict[int, int],
    pipeline: ProfilePipeline,
    n_clusters: int,
) -> np.ndarray:
    from prime.data.profiles import transform_full

    buckets: Dict[int, List[np.ndarray]] = {c: [] for c in range(n_clusters)}
    for u, cid in user_to_cluster.items():
        if u in profiles:
            buckets[cid].append(transform_full(profiles[u], pipeline))
    dim = pipeline.n_pca + len(pipeline.full_scaler_mean)
    centroids = [
        np.mean(buckets[c], axis=0) if buckets[c] else np.zeros(dim)
        for c in range(n_clusters)
    ]
    return np.vstack(centroids).astype(np.float32)


def _full_vs_label_free_agreement(
    profiles: Dict[int, object],
    user_to_cluster: Dict[int, int],
    pipeline: ProfilePipeline,
    full_centroids: np.ndarray,
) -> float:
    """
    Diagnostic (inverse of legacy check): share of users whose label-free cluster
    matches nearest *full* (rating-aware) centroid — how much rating info would
    change the grouping.
    """
    from prime.data.profiles import transform_full

    if not profiles:
        return 1.0
    agree = 0
    for u, prof in profiles.items():
        vec = transform_full(prof, pipeline)
        full_cluster = int(np.argmin(np.linalg.norm(full_centroids - vec, axis=1)))
        if user_to_cluster.get(u) == full_cluster:
            agree += 1
    return agree / len(profiles)


def _feature_matrix(
    profiles: Dict[int, object],
    pipeline: ProfilePipeline,
    mode: str,
) -> tuple[np.ndarray, List[int]]:
    from prime.data.profiles import profile_feature_matrix

    return profile_feature_matrix(profiles, pipeline=pipeline, mode=mode)


def assign_users_to_clusters(
    split: ReviewSplit,
    artifacts: ClusterArtifacts,
    cfg: ClusterCfg,
    dataset_cfg: Optional[DatasetCfg] = None,
) -> Dict[int, int]:
    """
    Map OOD users to nearest **label-free** train centroid.
    Train users keep their fitted cluster id.
    """
    emb = _encode_texts(
        split.texts,
        artifacts.embedding_model,
        dataset_cfg,
        tag=f"cluster_{split.name}_{len(split.texts)}",
    )
    if artifacts.pipeline is not None:
        profiles = build_user_profiles_label_free(split.texts, split.user_ids, emb)
    else:
        profiles = build_user_profiles(split.texts, split.labels, split.user_ids, emb)

    mapping: Dict[int, int] = {}
    lf_centroids = artifacts.cluster_centroids_label_free

    for u, prof in profiles.items():
        if u in artifacts.user_to_cluster:
            mapping[u] = artifacts.user_to_cluster[u]
            continue
        if artifacts.pipeline is not None:
            if artifacts.fit_mode == "emb_only":
                vec = transform_emb_only(prof, artifacts.pipeline)
            else:
                vec = transform_label_free(prof, artifacts.pipeline)
        else:
            from prime.data.profiles import profile_feature_matrix as pfm

            X, _ = pfm({u: prof})
            vec = X[0]
        dists = np.linalg.norm(lf_centroids - vec, axis=1)
        mapping[u] = int(np.argmin(dists))
    return mapping


def attach_example_clusters(
    split: ReviewSplit,
    user_to_cluster: Dict[int, int],
) -> ReviewSplit:
    cids = [user_to_cluster.get(u, 0) for u in split.user_ids]
    return ReviewSplit(
        name=split.name,
        texts=split.texts,
        labels=split.labels,
        user_ids=split.user_ids,
        example_cluster_ids=cids,
    )

"""User style profiles and feature pipeline for cluster assignment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

_PUNCT_CHARS = set('.,!?;:"\'-()[]')


@dataclass
class UserProfile:
    user_id: int
    mean_rating: float
    rating_std: float
    mean_length: float
    punctuation_density: float
    caps_ratio: float
    n_reviews: int
    text_embedding: np.ndarray  # mean review embedding


@dataclass
class ProfilePipeline:
    """Fitted StandardScaler + PCA pipeline for full and label-free projections."""

    pca_components: np.ndarray
    pca_mean: np.ndarray
    full_scaler_mean: np.ndarray
    full_scaler_scale: np.ndarray
    label_free_scaler_mean: np.ndarray
    label_free_scaler_scale: np.ndarray
    n_pca: int

    def to_dict(self) -> dict:
        return {
            "pca_components": self.pca_components.tolist(),
            "pca_mean": self.pca_mean.tolist(),
            "full_scaler_mean": self.full_scaler_mean.tolist(),
            "full_scaler_scale": self.full_scaler_scale.tolist(),
            "label_free_scaler_mean": self.label_free_scaler_mean.tolist(),
            "label_free_scaler_scale": self.label_free_scaler_scale.tolist(),
            "n_pca": self.n_pca,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProfilePipeline":
        return cls(
            pca_components=np.array(data["pca_components"], dtype=np.float32),
            pca_mean=np.array(data["pca_mean"], dtype=np.float32),
            full_scaler_mean=np.array(data["full_scaler_mean"], dtype=np.float32),
            full_scaler_scale=np.array(data["full_scaler_scale"], dtype=np.float32),
            label_free_scaler_mean=np.array(data["label_free_scaler_mean"], dtype=np.float32),
            label_free_scaler_scale=np.array(data["label_free_scaler_scale"], dtype=np.float32),
            n_pca=int(data["n_pca"]),
        )


def _text_behavior_stats(texts: List[str]) -> Tuple[float, float, float]:
    """Return mean_length, punctuation_density, caps_ratio."""
    if not texts:
        return 0.0, 0.0, 0.0
    lengths = [len(t) for t in texts]
    mean_length = float(np.mean(lengths))
    punct_densities: List[float] = []
    caps_ratios: List[float] = []
    for t in texts:
        if not t:
            punct_densities.append(0.0)
            caps_ratios.append(0.0)
            continue
        punct_densities.append(sum(1 for c in t if c in _PUNCT_CHARS) / len(t))
        letters = [c for c in t if c.isalpha()]
        caps_ratios.append(
            sum(1 for c in letters if c.isupper()) / len(letters) if letters else 0.0
        )
    return mean_length, float(np.mean(punct_densities)), float(np.mean(caps_ratios))


def build_user_profiles(
    texts: List[str],
    labels: List[int],
    user_ids: List[int],
    review_embeddings: np.ndarray,
) -> Dict[int, UserProfile]:
    """Aggregate per-user statistics and mean text embedding (train / full mode)."""
    by_user: Dict[int, List[int]] = {}
    for i, u in enumerate(user_ids):
        by_user.setdefault(u, []).append(i)

    profiles: Dict[int, UserProfile] = {}
    for u, idxs in by_user.items():
        u_labels = [labels[i] for i in idxs]
        u_texts = [texts[i] for i in idxs]
        mean_length, punct_density, caps_ratio = _text_behavior_stats(u_texts)
        emb = review_embeddings[idxs].mean(axis=0)
        profiles[u] = UserProfile(
            user_id=u,
            mean_rating=float(np.mean(u_labels)),
            rating_std=float(np.std(u_labels)) if len(u_labels) > 1 else 0.0,
            mean_length=mean_length,
            punctuation_density=punct_density,
            caps_ratio=caps_ratio,
            n_reviews=len(idxs),
            text_embedding=emb,
        )
    return profiles


def build_user_profiles_label_free(
    texts: List[str],
    user_ids: List[int],
    review_embeddings: np.ndarray,
) -> Dict[int, UserProfile]:
    """Label-free profiles: text embedding + length/style stats only (no rating stats)."""
    by_user: Dict[int, List[int]] = {}
    for i, u in enumerate(user_ids):
        by_user.setdefault(u, []).append(i)

    profiles: Dict[int, UserProfile] = {}
    for u, idxs in by_user.items():
        u_texts = [texts[i] for i in idxs]
        mean_length, punct_density, caps_ratio = _text_behavior_stats(u_texts)
        emb = review_embeddings[idxs].mean(axis=0)
        profiles[u] = UserProfile(
            user_id=u,
            mean_rating=0.0,
            rating_std=0.0,
            mean_length=mean_length,
            punctuation_density=punct_density,
            caps_ratio=caps_ratio,
            n_reviews=len(idxs),
            text_embedding=emb,
        )
    return profiles


def _full_behavioral_row(p: UserProfile) -> np.ndarray:
    return np.array(
        [
            p.mean_rating / 5.0,
            p.rating_std / 4.0,
            np.log1p(p.mean_length) / 10.0,
            np.log1p(p.n_reviews) / 5.0,
            p.punctuation_density,
            p.caps_ratio,
        ],
        dtype=np.float32,
    )


def _label_free_behavioral_row(p: UserProfile) -> np.ndarray:
    return np.array(
        [
            np.log1p(p.mean_length) / 10.0,
            p.punctuation_density,
            p.caps_ratio,
        ],
        dtype=np.float32,
    )


def _pca_transform(emb: np.ndarray, pipeline: ProfilePipeline) -> np.ndarray:
    centered = emb.astype(np.float32) - pipeline.pca_mean
    return centered @ pipeline.pca_components.T


def _scale_row(row: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    safe_scale = np.where(scale == 0, 1.0, scale)
    return (row - mean) / safe_scale


def transform_full(profile: UserProfile, pipeline: ProfilePipeline) -> np.ndarray:
    emb_part = _pca_transform(profile.text_embedding, pipeline)
    beh = _scale_row(
        _full_behavioral_row(profile),
        pipeline.full_scaler_mean,
        pipeline.full_scaler_scale,
    )
    return np.concatenate([emb_part, beh])


def transform_label_free(profile: UserProfile, pipeline: ProfilePipeline) -> np.ndarray:
    emb_part = _pca_transform(profile.text_embedding, pipeline)
    beh = _scale_row(
        _label_free_behavioral_row(profile),
        pipeline.label_free_scaler_mean,
        pipeline.label_free_scaler_scale,
    )
    return np.concatenate([emb_part, beh])


def transform_emb_only(profile: UserProfile, pipeline: ProfilePipeline) -> np.ndarray:
    """PCA(embedding) only — ablation that removes length/punct/caps dominance."""
    return _pca_transform(profile.text_embedding, pipeline)


def fit_profile_pipeline(
    profiles: Dict[int, UserProfile],
    n_pca: int = 32,
) -> ProfilePipeline:
    """Fit PCA on embeddings and StandardScaler on behavioral blocks."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    users = sorted(profiles.keys())
    embeddings = np.vstack([profiles[u].text_embedding.astype(np.float32) for u in users])
    full_beh = np.vstack([_full_behavioral_row(profiles[u]) for u in users])
    lf_beh = np.vstack([_label_free_behavioral_row(profiles[u]) for u in users])

    n_pca = min(n_pca, embeddings.shape[0], embeddings.shape[1])
    n_pca = max(1, n_pca)
    pca = PCA(n_components=n_pca, random_state=0)
    pca.fit(embeddings)

    full_scaler = StandardScaler()
    full_scaler.fit(full_beh)
    lf_scaler = StandardScaler()
    lf_scaler.fit(lf_beh)

    return ProfilePipeline(
        pca_components=pca.components_.astype(np.float32),
        pca_mean=pca.mean_.astype(np.float32),
        full_scaler_mean=full_scaler.mean_.astype(np.float32),
        full_scaler_scale=full_scaler.scale_.astype(np.float32),
        label_free_scaler_mean=lf_scaler.mean_.astype(np.float32),
        label_free_scaler_scale=lf_scaler.scale_.astype(np.float32),
        n_pca=n_pca,
    )


def profile_feature_matrix(
    profiles: Dict[int, UserProfile],
    pipeline: Optional[ProfilePipeline] = None,
    mode: str = "full",
) -> Tuple[np.ndarray, List[int]]:
    """
    Stack profile features for clustering.
    mode='full': PCA(embedding) + scaled behavioral (incl. rating stats) — train k-means.
    mode='label_free': PCA(embedding) + length/style only — OOD assignment.
    mode='emb_only': PCA(embedding) only — E0 ablation vs length-dominated full_T.
    """
    users = sorted(profiles.keys())
    if pipeline is None:
        rows = []
        for u in users:
            p = profiles[u]
            extra = np.array(
                [
                    p.mean_rating / 5.0,
                    p.rating_std / 4.0,
                    np.log1p(p.mean_length) / 10.0,
                    np.log1p(p.n_reviews) / 5.0,
                ],
                dtype=np.float32,
            )
            rows.append(np.concatenate([p.text_embedding.astype(np.float32), extra]))
        return np.vstack(rows), users

    if mode == "full":
        transform = transform_full
    elif mode == "emb_only":
        transform = transform_emb_only
    else:
        transform = transform_label_free
    rows = [transform(profiles[u], pipeline) for u in users]
    return np.vstack(rows), users


def projection_agreement(
    profiles: Dict[int, UserProfile],
    user_to_cluster_full: Dict[int, int],
    pipeline: ProfilePipeline,
    centroids_label_free: np.ndarray,
) -> float:
    """Fraction of train users where full-cluster == nearest label-free centroid."""
    if not profiles:
        return 1.0
    agree = 0
    for u, prof in profiles.items():
        lf = transform_label_free(prof, pipeline)
        lf_cluster = int(np.argmin(np.linalg.norm(centroids_label_free - lf, axis=1)))
        if user_to_cluster_full.get(u) == lf_cluster:
            agree += 1
    return agree / len(profiles)

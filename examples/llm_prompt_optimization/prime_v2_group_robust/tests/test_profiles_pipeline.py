import numpy as np

from prime.data.profiles import (
    build_user_profiles,
    fit_profile_pipeline,
    profile_feature_matrix,
    transform_full,
    transform_label_free,
)
from prime.evolution.qd_features import build_qd_metrics, qd_feature_dimension_names


def test_profile_pipeline_full_vs_label_free_dims():
    texts = ["Great product!", "Terrible.", "Okay item"]
    labels = [5, 1, 3]
    users = [1, 1, 2]
    emb = np.random.RandomState(0).randn(3, 16).astype(np.float32)
    profiles = build_user_profiles(texts, labels, users, emb)
    pipeline = fit_profile_pipeline(profiles, n_pca=4)
    full = transform_full(profiles[1], pipeline)
    lf = transform_label_free(profiles[1], pipeline)
    assert full.shape[0] == pipeline.n_pca + 6
    assert lf.shape[0] == pipeline.n_pca + 3


def test_qd_feature_names_match_n_clusters():
    names = qd_feature_dimension_names(4)
    assert names == ["cluster_acc_0", "cluster_acc_1", "cluster_acc_2", "cluster_acc_3", "prompt_length"]
    metrics = build_qd_metrics({0: 0.8, 2: 0.4}, "short prompt", n_clusters=4)
    assert metrics["cluster_acc_1"] == 0.0
    assert metrics["cluster_acc_2"] == 0.4

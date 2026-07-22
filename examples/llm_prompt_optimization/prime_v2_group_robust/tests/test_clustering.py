import numpy as np

from prime.data.profiles import build_user_profiles, fit_profile_pipeline, profile_feature_matrix


def test_profile_feature_matrix_shape():
    texts = ["good", "bad", "okay"]
    labels = [5, 1, 3]
    users = [10, 10, 20]
    emb = np.random.randn(3, 4).astype(np.float32)
    profiles = build_user_profiles(texts, labels, users, emb)
    X, user_list = profile_feature_matrix(profiles)
    assert X.shape[0] == 2
    assert 10 in user_list and 20 in user_list
    assert X.shape[1] == 4 + 4  # legacy: embedding + 4 stats

    pipeline = fit_profile_pipeline(profiles, n_pca=2)
    X_full, _ = profile_feature_matrix(profiles, pipeline=pipeline, mode="full")
    assert X_full.shape[1] == 2 + 6  # pca + full behavioral

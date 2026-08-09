from prime.experiment.proxy_validation import (
    corr_cvar_vs_rworst_from_predictions,
    leave_one_user_out_cluster_accuracy,
    pearson_correlation,
    proxy_validation_report,
    spearman_correlation,
)
import numpy as np
import pytest


def test_pearson_perfect_correlation():
    assert pearson_correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == 1.0


def test_proxy_report_not_ready_with_one_cycle():
    report = proxy_validation_report([(0.5, 0.4)], min_cycles=2)
    assert report["ready"] is False
    assert report["correlation"] is None


def test_proxy_report_ready_with_two_cycles():
    report = proxy_validation_report([(0.4, 0.3), (0.6, 0.5)], min_cycles=2)
    assert report["ready"] is True
    assert report["correlation"] == pytest.approx(1.0)


def test_loo_excludes_user_from_cluster_acc():
    # Two users in cluster 0: user 1 all-correct, user 2 all-wrong.
    preds = np.array([5, 5, 1, 1])
    gold = np.array([5, 5, 5, 5])
    users = np.array([1, 1, 2, 2])
    clusters = np.array([0, 0, 0, 0])
    # LOO for user 1 → only user 2's wrong reviews remain → low acc
    loo1 = leave_one_user_out_cluster_accuracy(
        preds, gold, users, clusters, user_id=1, cluster_id=0, beta_a=0.0, beta_b=0.0
    )
    assert loo1 == pytest.approx(0.0)
    loo2 = leave_one_user_out_cluster_accuracy(
        preds, gold, users, clusters, user_id=2, cluster_id=0, beta_a=0.0, beta_b=0.0
    )
    assert loo2 == pytest.approx(1.0)


def test_loo_spearman_lower_than_leaky_when_self_dominated():
    # Each user alone in their cluster: leaky corr is perfect, LOO skips all.
    preds = np.array([5, 1, 5, 1])
    gold = np.array([5, 5, 5, 5])
    users = np.array([1, 2, 3, 4])
    clusters = np.array([0, 1, 2, 3])
    out = corr_cvar_vs_rworst_from_predictions(
        preds, gold, users, clusters, beta_a=0.0, beta_b=0.0, n_perm=200
    )
    assert out["n_users_in_corr"] == 0
    assert out["spearman_user_acc_vs_cluster_acc"] is None
    assert out["n_loo_skipped_singleton"] == 4
    # Leaky still has users
    assert out["n_users_in_corr_leaky"] == 4


def test_permutation_kw_detects_separated_groups():
    from prime.experiment.proxy_validation import permutation_kruskal_wallis

    # Cluster 0: high acc users; cluster 1: low acc
    user_acc = {i: 0.9 for i in range(20)}
    user_acc.update({i: 0.2 for i in range(20, 40)})
    user_cluster = {i: 0 if i < 20 else 1 for i in range(40)}
    out = permutation_kruskal_wallis(user_acc, user_cluster, n_perm=1000, seed=0)
    assert out["kruskal_h"] is not None and out["kruskal_h"] > 5
    assert out["p_value"] is not None and out["p_value"] < 0.05


def test_permutation_kw_null_on_random_labels():
    from prime.experiment.proxy_validation import permutation_kruskal_wallis

    rng = np.random.RandomState(0)
    user_acc = {i: float(rng.rand()) for i in range(40)}
    user_cluster = {i: int(i % 4) for i in range(40)}
    # Shuffle acc relative to clusters → should usually not be significant
    accs = list(user_acc.values())
    rng.shuffle(accs)
    user_acc = {i: accs[i] for i in range(40)}
    out = permutation_kruskal_wallis(user_acc, user_cluster, n_perm=1000, seed=1)
    assert out["p_value"] is not None and out["p_value"] > 0.05

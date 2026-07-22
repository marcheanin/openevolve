from prime.evolution.openevolve_config_patch import patch_openevolve_feature_dimensions
from prime.evolution.qd_features import build_qd_metrics, qd_feature_dimension_names


def test_qd_dims_match_actual_k(tmp_path):
    names = qd_feature_dimension_names(3)
    assert names == ["cluster_acc_0", "cluster_acc_1", "cluster_acc_2", "prompt_length"]
    metrics = build_qd_metrics({0: 0.9, 1: 0.2}, "hello world", n_clusters=3)
    assert set(metrics) == set(names)
    assert metrics["cluster_acc_2"] == 0.0

    dest = tmp_path / "oe.yaml"
    patch_openevolve_feature_dimensions(None, dest, n_clusters=3)
    text = dest.read_text(encoding="utf-8")
    assert "cluster_acc_0" in text
    assert "cluster_acc_2" in text
    assert "cluster_acc_7" not in text

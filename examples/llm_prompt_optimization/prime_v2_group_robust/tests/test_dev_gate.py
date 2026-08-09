"""F8: heir better on D_select but worse on D_dev must not promote under reject."""

from prime.experiment.dev_gate import evaluate_dev_gate, generalization_gap, softmin_from_metrics


def test_dev_gate_rejects_ood_drop():
    champ = {"R_soft_min_gba": 0.70, "R_worst_gba": 0.65, "R_gba_mean": 0.71}
    cand = {"R_soft_min_gba": 0.68, "R_worst_gba": 0.63, "R_gba_mean": 0.72}  # drop 0.02 > 0.01
    g = evaluate_dev_gate(
        champion_metrics=champ, candidate_metrics=cand, delta=0.01, mode="reject"
    )
    assert g["accepted"] is False
    assert g["reason"] == "dev_softmin_drop"
    assert g["drop"] == 0.02


def test_dev_gate_accepts_within_delta():
    champ = {"R_soft_min_gba": 0.70}
    cand = {"R_soft_min_gba": 0.695}  # drop 0.005
    g = evaluate_dev_gate(
        champion_metrics=champ, candidate_metrics=cand, delta=0.01, mode="reject"
    )
    assert g["accepted"] is True


def test_dev_gate_monitor_never_blocks():
    champ = {"R_soft_min_gba": 0.70}
    cand = {"R_soft_min_gba": 0.50}
    g = evaluate_dev_gate(
        champion_metrics=champ, candidate_metrics=cand, delta=0.01, mode="monitor"
    )
    assert g["accepted"] is True
    assert g["would_reject"] is True


def test_generalization_gap():
    assert abs(generalization_gap(0.72, 0.70) - 0.02) < 1e-9
    assert generalization_gap(None, 0.7) is None


def test_softmin_from_cluster_gba():
    m = {"cluster_gba": {1: 0.8, 2: 0.6, 3: 0.7}}
    s = softmin_from_metrics(m, tau=0.10)
    assert 0.55 < s < 0.72  # between hard min and mean

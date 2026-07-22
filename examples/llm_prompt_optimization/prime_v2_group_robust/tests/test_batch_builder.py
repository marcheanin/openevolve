import numpy as np

from prime.acquisition.batch_builder import build_active_batch, compute_group_hard_quotas
from prime.config import AcquisitionCfg


def test_group_quotas_favor_weak_clusters():
    quotas = compute_group_hard_quotas({0: 0.9, 1: 0.3, 2: 0.5}, n_hard=10, n_clusters=3)
    assert sum(quotas.values()) == 10
    assert quotas[1] >= quotas[0]


def test_build_active_batch_group_aware_no_binary_prefilter():
    cfg = AcquisitionCfg(policy="group_aware", batch_size=6, hard_ratio=0.6, group_quota_enabled=True)
    seen = list(range(6))
    preds = [1, 1, 1, 1, 1, 1]
    gold = [1, 1, 1, 1, 1, 1]
    disagreements = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    clusters = [0, 0, 1, 1, 2, 2]
    batch = build_active_batch(
        seen,
        preds,
        gold,
        disagreements,
        clusters,
        None,
        cfg,
        seed=0,
        cluster_accuracies={0: 0.9, 1: 0.2, 2: 0.5},
    )
    assert len(batch["indices"]) == 6
    assert len(batch["hard_indices"]) + len(batch["anchor_indices"]) == len(batch["indices"])
    assert batch.get("group_hard_quotas")

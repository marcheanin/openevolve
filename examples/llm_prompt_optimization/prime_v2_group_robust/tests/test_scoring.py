import numpy as np

from prime.acquisition.scoring import score_examples
from prime.config import AcquisitionCfg


def test_lexicographic_orders_err_then_disagreement():
    cfg = AcquisitionCfg(policy="lexicographic")
    scored = score_examples(
        pool_indices=[0, 1, 2, 3],
        predictions=[1, 2, 2, 5],
        gold=[1, 2, 5, 5],
        disagreements=[0.0, 0.5, 0.1, 0.9],
        cluster_ids=[0, 1, 2, 3],
        cfg=cfg,
    )
    # idx2: err=1,d=0.1; idx3: err=0,d=0.9; idx0: err=0; idx1: err=0,d=0.5
    assert scored[0].pool_index == 2
    assert scored[0].err == 1.0
    assert scored[1].pool_index == 3 or scored[1].disagreement >= 0.5


def test_qbc_d_sorts_by_disagreement_only():
    cfg = AcquisitionCfg(policy="qbc_d")
    scored = score_examples(
        pool_indices=[0, 1],
        predictions=[1, 5],
        gold=[5, 5],
        disagreements=[0.2, 0.8],
        cluster_ids=None,
        cfg=cfg,
    )
    assert scored[0].disagreement >= scored[1].disagreement

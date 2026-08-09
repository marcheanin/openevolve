"""Phase3 F5/F6/F9: scorer, uncertainty, balanced cells."""

from __future__ import annotations

import numpy as np

from prime.config import EnsembleCfg, WorkerSpec
from prime.data.balanced_cells import fingerprint_indices, sample_balanced_cells
from prime.workers.scorer import Scorer, uncertainty_from_votes


def test_uncertainty_agreement_zero():
    assert uncertainty_from_votes([0, 0, 0]) == 0.0


def test_uncertainty_split_one_third():
    assert abs(uncertainty_from_votes([0, 0, 1]) - 1.0 / 3.0) < 1e-9


def test_scorer_mock_single():
    cfg = EnsembleCfg(
        mode="single",
        fail_closed=True,
        workers=[WorkerSpec(name="mock/model")],
        aggregation="majority",
    )
    scorer = Scorer(cfg, label_space="binary", use_mock=True)
    assert scorer.is_single
    texts = ["a", "b", "c", "d"]
    labels = [0, 1, 0, 1]
    out = scorer.predict_batch(texts, "Label 0/1. {review}", labels_for_mock=labels)
    assert len(out.preds) == 4
    assert out.n_calls == 4
    unc, n = scorer.self_consistency(texts, "Label 0/1. {review}", k=3, labels_for_mock=labels)
    assert len(unc) == 4
    assert n == 12


def test_balanced_cells_exact():
    labels = []
    groups = []
    for g in range(1, 5):
        for y in (0, 1):
            labels.extend([y] * 20)
            groups.extend([g] * 20)
    idx = sample_balanced_cells(
        labels, groups, groups=[1, 2, 3, 4], per_cell=5, seed=42
    )
    assert len(idx) == 4 * 2 * 5
    # Check cell sizes
    from collections import Counter

    cells = Counter((groups[i], labels[i]) for i in idx)
    assert all(v == 5 for v in cells.values())
    fp = fingerprint_indices(idx)
    idx2 = sample_balanced_cells(
        labels, groups, groups=[1, 2, 3, 4], per_cell=5, seed=42
    )
    assert fingerprint_indices(idx2) == fp

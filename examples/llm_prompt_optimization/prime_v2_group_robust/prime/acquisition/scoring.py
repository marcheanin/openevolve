"""Acquisition scoring policies for active batch selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from prime.config import AcquisitionCfg


@dataclass(frozen=True)
class ScoredExample:
    pool_index: int
    err: float
    disagreement: float
    cluster_id: int = 0

    @property
    def lex_key(self) -> Tuple[float, float]:
        return (self.err, self.disagreement)


def score_examples(
    pool_indices: Sequence[int],
    predictions: Sequence[int],
    gold: Sequence[int],
    disagreements: Sequence[float],
    cluster_ids: Optional[Sequence[int]],
    cfg: AcquisitionCfg,
    seed: int = 0,
) -> List[ScoredExample]:
    """Score pool examples under configured acquisition policy."""
    scored: List[ScoredExample] = []
    for i, idx in enumerate(pool_indices):
        pred = int(predictions[i])
        label = int(gold[i])
        err = 1.0 if pred != label else 0.0
        d = float(disagreements[i])
        cid = int(cluster_ids[i]) if cluster_ids is not None else 0
        scored.append(ScoredExample(pool_index=int(idx), err=err, disagreement=d, cluster_id=cid))

    policy = cfg.policy
    if policy == "random":
        rng = np.random.RandomState(seed)
        rng.shuffle(scored)
        return scored
    if policy == "qbc_d":
        return sorted(scored, key=lambda s: s.disagreement, reverse=True)
    if policy == "hardest":
        return sorted(scored, key=lambda s: s.err, reverse=True)
    if policy == "group_aware":
        # Group weighting is applied in batch_builder via quotas; sort lexicographically within group.
        return sorted(scored, key=lambda s: s.lex_key, reverse=True)
    # lexicographic default: (err, d) descending
    return sorted(scored, key=lambda s: s.lex_key, reverse=True)

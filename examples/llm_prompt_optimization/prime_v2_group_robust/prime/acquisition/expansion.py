"""Disagreement-based pool expansion with style-cluster coverage."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from prime.acquisition.batch_builder import compute_group_hard_quotas
from prime.acquisition.pool import AcquisitionPool
from prime.config import AcquisitionCfg


def select_expansion_by_disagreement(
    unseen_indices: Sequence[int],
    disagreements: Sequence[float],
    cluster_ids: Sequence[int],
    n_add: int,
    cluster_accuracies: Optional[Dict[int, float]] = None,
    seed: int = 0,
) -> List[int]:
    """
    Rank unseen candidates by d_i (descending), allocate slots with group coverage
    quotas ∝ (1 - Acc_k) when accuracies are available, else round-robin over clusters.
    """
    if n_add <= 0 or not unseen_indices:
        return []

    scored = sorted(
        zip(unseen_indices, disagreements, cluster_ids),
        key=lambda t: float(t[1]),
        reverse=True,
    )
    by_cluster: Dict[int, List[int]] = defaultdict(list)
    for idx, _d, cid in scored:
        by_cluster[int(cid)].append(int(idx))

    active = sorted(by_cluster.keys())
    if not active:
        return []

    if cluster_accuracies:
        quotas = compute_group_hard_quotas(cluster_accuracies, n_add, len(active))
        # Keep only clusters that have candidates
        quotas = {c: quotas.get(c, 0) for c in active}
        # Redistribute unused quota from empty clusters
        unused = sum(quotas.get(c, 0) for c in list(quotas) if c not in by_cluster or not by_cluster[c])
        for c in list(quotas):
            if c not in by_cluster or not by_cluster[c]:
                quotas[c] = 0
        donors = sorted(active, key=lambda c: len(by_cluster.get(c, [])), reverse=True)
        i = 0
        while unused > 0 and donors:
            c = donors[i % len(donors)]
            if by_cluster.get(c):
                quotas[c] = quotas.get(c, 0) + 1
                unused -= 1
            i += 1
            if i > n_add * 10:
                break
    else:
        base, rem = divmod(n_add, len(active))
        quotas = {c: base + (1 if i < rem else 0) for i, c in enumerate(active)}

    selected: List[int] = []
    for cid in sorted(quotas.keys()):
        k = quotas[cid]
        if k <= 0:
            continue
        selected.extend(by_cluster[cid][:k])

    if len(selected) < n_add:
        remaining = [idx for idx, _, _ in scored if idx not in selected]
        selected.extend(remaining[: n_add - len(selected)])
    return selected[:n_add]


def expand_pool_disagreement(
    pool: AcquisitionPool,
    disagreements_by_pool_index: Dict[int, float],
    n_add: int,
    cluster_accuracies: Optional[Dict[int, float]] = None,
    seed: int = 0,
) -> List[int]:
    """Select and mark unseen indices as seen using disagreement scores."""
    if n_add <= 0 or not pool.unseen:
        return []
    # Only candidates that were scored in this expansion pass
    candidates = sorted(i for i in disagreements_by_pool_index if i in pool.unseen)
    if not candidates:
        return []
    disagreements = [float(disagreements_by_pool_index[i]) for i in candidates]
    cluster_ids = [int(pool.cluster_ids[i]) for i in candidates]
    picked = select_expansion_by_disagreement(
        candidates,
        disagreements,
        cluster_ids,
        n_add=min(n_add, len(candidates)),
        cluster_accuracies=cluster_accuracies,
        seed=seed,
    )
    for idx in picked:
        pool.mark_seen(idx)
    return picked

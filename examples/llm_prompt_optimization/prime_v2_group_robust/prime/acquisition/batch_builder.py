"""Build active batch: top-k scoring + diversity + group-weighted quotas."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from prime.acquisition.scoring import ScoredExample, score_examples
from prime.config import AcquisitionCfg


def _diversity_pick(
    candidates: List[ScoredExample],
    k: int,
    embeddings: Optional[np.ndarray],
    n_clusters: int,
    seed: int,
) -> List[int]:
    """Pick k indices from top candidates with cluster diversity."""
    if k <= 0 or not candidates:
        return []
    if embeddings is None or len(candidates) <= k:
        return [c.pool_index for c in candidates[:k]]

    rng = np.random.RandomState(seed)
    top_n = min(len(candidates), max(k * 3, k))
    pool = candidates[:top_n]
    idxs = [c.pool_index for c in pool]
    emb = embeddings[idxs]
    from sklearn.cluster import KMeans

    nc = min(n_clusters, len(pool), k)
    if nc <= 1:
        return idxs[:k]
    labels = KMeans(n_clusters=nc, random_state=seed, n_init=10).fit_predict(emb)
    selected: List[int] = []
    per_cluster: Dict[int, List[int]] = {}
    for i, lab in enumerate(labels):
        per_cluster.setdefault(int(lab), []).append(idxs[i])
    clusters = sorted(per_cluster.keys())
    while len(selected) < k and clusters:
        rng.shuffle(clusters)
        for lab in list(clusters):
            if per_cluster[lab]:
                selected.append(per_cluster[lab].pop(0))
                if len(selected) >= k:
                    break
            if not per_cluster[lab]:
                clusters.remove(lab)
    if len(selected) < k:
        for c in pool:
            if c.pool_index not in selected:
                selected.append(c.pool_index)
            if len(selected) >= k:
                break
    return selected[:k]


def compute_group_hard_quotas(
    cluster_accuracies: Dict[int, float],
    n_hard: int,
    n_clusters: int,
) -> Dict[int, int]:
    """
    Hard-slot quotas proportional to group weakness: weight_k = (1 - Acc_k).
    Ensures each active cluster gets at least one slot when n_hard >= n_active.
    """
    if n_hard <= 0:
        return {}
    active = sorted(cluster_accuracies.keys()) if cluster_accuracies else list(range(n_clusters))
    if not active:
        active = list(range(max(1, n_clusters)))

    weights = {cid: max(0.0, 1.0 - float(cluster_accuracies.get(cid, 0.0))) for cid in active}
    total_w = sum(weights.values())
    if total_w <= 0:
        base, rem = divmod(n_hard, len(active))
        return {cid: base + (1 if i < rem else 0) for i, cid in enumerate(active)}

    raw = {cid: n_hard * w / total_w for cid, w in weights.items()}
    quotas = {cid: int(np.floor(v)) for cid, v in raw.items()}
    assigned = sum(quotas.values())
    remainder = n_hard - assigned
    if remainder > 0:
        frac_order = sorted(active, key=lambda c: raw[c] - quotas[c], reverse=True)
        for cid in frac_order[:remainder]:
            quotas[cid] += 1

    zero_slots = [cid for cid in active if quotas[cid] == 0]
    if zero_slots and n_hard >= len(active):
        donors = sorted(active, key=lambda c: quotas[c], reverse=True)
        for cid in zero_slots:
            if not donors:
                break
            donor = donors[0]
            if quotas[donor] > 1:
                quotas[donor] -= 1
                quotas[cid] = 1
                donors = sorted(active, key=lambda c: quotas[c], reverse=True)

    return quotas


def _pick_hard_by_group_quotas(
    scored: List[ScoredExample],
    quotas: Dict[int, int],
    embeddings: Optional[np.ndarray],
    n_diversity_clusters: int,
    seed: int,
) -> List[int]:
    """Within each group: lexicographic rank → oversample ×3 → diversity pick."""
    by_cluster: Dict[int, List[ScoredExample]] = defaultdict(list)
    for s in scored:
        by_cluster[s.cluster_id].append(s)

    selected: List[int] = []
    for cid, k in sorted(quotas.items()):
        if k <= 0:
            continue
        group_scored = by_cluster.get(cid, [])
        if not group_scored:
            continue
        picked = _diversity_pick(group_scored, k, embeddings, n_diversity_clusters, seed + cid)
        selected.extend(picked)

    if len(selected) < sum(quotas.values()):
        need = sum(quotas.values()) - len(selected)
        remaining = [s for s in scored if s.pool_index not in selected]
        selected.extend(_diversity_pick(remaining, need, embeddings, n_diversity_clusters, seed + 999))
    return selected


def _stratified_anchor_pick(
    scored: List[ScoredExample],
    k: int,
    gold_labels: Sequence[int],
    pool_index_to_pos: Dict[int, int],
    embeddings: Optional[np.ndarray],
    n_diversity_clusters: int,
    seed: int,
) -> List[int]:
    """
    Anchor = low (err, d) rank, stratified by (cluster_id, gold_label).
    Replay against forgetting across class × group cells.
    """
    if k <= 0:
        return []
    anchor_candidates = [s for s in scored if s.err == 0.0]
    if not anchor_candidates:
        anchor_candidates = list(reversed(scored))

    cells: Dict[Tuple[int, int], List[ScoredExample]] = defaultdict(list)
    for s in anchor_candidates:
        pos = pool_index_to_pos.get(s.pool_index)
        label = int(gold_labels[pos]) if pos is not None else 0
        cells[(s.cluster_id, label)].append(s)

    rng = np.random.RandomState(seed)
    cell_keys = sorted(cells.keys())
    rng.shuffle(cell_keys)
    selected: List[int] = []
    while len(selected) < k and cell_keys:
        progressed = False
        for key in list(cell_keys):
            bucket = cells[key]
            if bucket:
                selected.append(bucket.pop(0).pool_index)
                progressed = True
                if len(selected) >= k:
                    break
            if not bucket and key in cell_keys:
                cell_keys.remove(key)
        if not progressed:
            break

    if len(selected) < k:
        rest = [s for s in anchor_candidates if s.pool_index not in selected]
        selected.extend(
            _diversity_pick(rest, k - len(selected), embeddings, n_diversity_clusters, seed + 1)
        )
    return selected[:k]


def build_active_batch(
    seen_indices: Sequence[int],
    predictions: Sequence[int],
    gold: Sequence[int],
    disagreements: Sequence[float],
    cluster_ids: Optional[Sequence[int]],
    embeddings: Optional[np.ndarray],
    cfg: AcquisitionCfg,
    seed: int = 0,
    cluster_accuracies: Optional[Dict[int, float]] = None,
) -> Dict[str, List[int]]:
    """
    Build active batch from seen pool.
    Hard slots: group quotas ∝ (1 - Acc_k), within-group lexicographic + diversity.
    Anchor slots: low-score stratified by cluster × label.
    """
    scored = score_examples(
        seen_indices, predictions, gold, disagreements, cluster_ids, cfg, seed=seed
    )

    n_hard = int(round(cfg.batch_size * cfg.hard_ratio))
    n_anchor = cfg.batch_size - n_hard
    n_clusters = max(1, cfg.n_diversity_clusters)

    if cfg.policy == "group_aware" and cfg.group_quota_enabled:
        accs = cluster_accuracies or {}
        quotas = compute_group_hard_quotas(accs, n_hard, n_clusters)
        hard_pick = _pick_hard_by_group_quotas(
            scored, quotas, embeddings, cfg.n_diversity_clusters, seed
        )
    else:
        oversample = max(n_hard, int(n_hard * cfg.oversample_factor))
        hard_pick = _diversity_pick(
            scored[:oversample],
            n_hard,
            embeddings,
            cfg.n_diversity_clusters,
            seed,
        )

    pool_index_to_pos = {int(idx): i for i, idx in enumerate(seen_indices)}
    anchor_pick = _stratified_anchor_pick(
        list(reversed(scored)),
        n_anchor,
        gold,
        pool_index_to_pos,
        embeddings,
        cfg.n_diversity_clusters,
        seed + 1,
    )

    hard_set = set(hard_pick)
    indices = hard_pick + [i for i in anchor_pick if i not in hard_set]
    if len(indices) < cfg.batch_size:
        for s in scored:
            if s.pool_index not in indices:
                indices.append(s.pool_index)
            if len(indices) >= cfg.batch_size:
                break
    indices = indices[: cfg.batch_size]
    hard_in_batch = [i for i in indices if i in hard_set]
    anchor_in_batch = [i for i in indices if i not in hard_set]
    return {
        "indices": indices,
        "hard_indices": hard_in_batch,
        "anchor_indices": anchor_in_batch,
        "group_hard_quotas": (
            compute_group_hard_quotas(cluster_accuracies or {}, n_hard, n_clusters)
            if cfg.policy == "group_aware" and cfg.group_quota_enabled
            else {}
        ),
    }

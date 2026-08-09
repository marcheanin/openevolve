"""Diagnostics for active-batch representativeness (hard/anchor, seen/unseen, roles)."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set


def build_batch_diagnostics(
    *,
    cycle: int,
    batch: Dict[str, List[int]],
    seen_indices: Sequence[int],
    unseen_count: int,
    pool_hard: Sequence[int],
    pool_anchor: Sequence[int],
    predictions: Sequence[int],
    gold: Sequence[int],
    disagreements: Sequence[float],
    cluster_ids: Sequence[int],
    cluster_accuracies: Dict[int, float],
    policy: str,
    d_select: Optional[Sequence[int]] = None,
    d_anchor: Optional[Sequence[int]] = None,
    train_labels_by_pool: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """
    Snapshot whether the mutator batch covers weak groups / rating classes,
    and whether roles stay disjoint from fitness sets.
    """
    seen = [int(i) for i in seen_indices]
    pos = {idx: i for i, idx in enumerate(seen)}
    b_idx = [int(i) for i in batch.get("indices", [])]
    hard = [int(i) for i in batch.get("hard_indices", [])]
    anchor = [int(i) for i in batch.get("anchor_indices", [])]
    hard_set, anchor_set = set(hard), set(anchor)

    def _hist(indices: Sequence[int], key_fn) -> Dict[str, int]:
        c: Counter = Counter()
        for i in indices:
            if i not in pos:
                continue
            c[str(key_fn(pos[i], i))] += 1
        return dict(sorted(c.items(), key=lambda kv: kv[0]))

    def _cluster(p: int, _i: int) -> int:
        return int(cluster_ids[p]) if p < len(cluster_ids) else -1

    def _gold(p: int, _i: int) -> int:
        return int(gold[p]) if p < len(gold) else -1

    def _err(p: int, _i: int) -> int:
        return int(predictions[p] != gold[p]) if p < len(predictions) else -1

    # Seen-pool gold mix vs hard-batch gold mix (representativeness)
    seen_gold = Counter(int(g) for g in gold)
    hard_gold = Counter()
    for i in hard:
        if i in pos:
            hard_gold[int(gold[pos[i]])] += 1

    # Weak clusters (below median Acc) vs hard coverage
    accs = {int(k): float(v) for k, v in cluster_accuracies.items()}
    if accs:
        med = sorted(accs.values())[len(accs) // 2]
        weak = [c for c, a in accs.items() if a <= med]
    else:
        weak = []
    hard_by_c = Counter(_cluster(pos[i], i) for i in hard if i in pos)
    weak_hard_slots = sum(hard_by_c.get(c, 0) for c in weak)
    weak_coverage = {
        str(c): hard_by_c.get(c, 0) for c in weak
    }

    d_select_set: Set[int] = set(int(x) for x in (d_select or []))
    d_anchor_set: Set[int] = set(int(x) for x in (d_anchor or []))
    batch_set = set(b_idx)
    leak_select = sorted(batch_set & d_select_set)
    leak_d_anchor = sorted(batch_set & d_anchor_set)

    # Fraction of seen errors that made it into hard slots
    seen_err_idxs = [
        seen[p]
        for p in range(len(seen))
        if p < len(predictions) and predictions[p] != gold[p]
    ]
    hard_covers_errs = sum(1 for i in seen_err_idxs if i in hard_set)
    n_seen_err = len(seen_err_idxs)

    mean_d_hard = (
        float(sum(disagreements[pos[i]] for i in hard if i in pos) / max(1, len(hard)))
        if hard
        else 0.0
    )
    mean_d_anchor = (
        float(sum(disagreements[pos[i]] for i in anchor if i in pos) / max(1, len(anchor)))
        if anchor
        else 0.0
    )

    return {
        "cycle": cycle,
        "policy": policy,
        "group_hard_quotas": batch.get("group_hard_quotas") or {},
        "pool": {
            "n_seen": len(seen),
            "n_unseen": int(unseen_count),
            "n_pool_hard": len(pool_hard),
            "n_pool_anchor": len(pool_anchor),
            "n_seen_errors": n_seen_err,
            "seen_gold_hist": {str(k): v for k, v in sorted(seen_gold.items())},
            "cluster_accuracies": {str(k): round(v, 4) for k, v in sorted(accs.items())},
        },
        "batch": {
            "size": len(b_idx),
            "n_hard": len(hard),
            "n_anchor": len(anchor),
            "hard_error_rate": round(
                sum(_err(pos[i], i) for i in hard if i in pos) / max(1, len(hard)), 4
            ),
            "anchor_error_rate": round(
                sum(_err(pos[i], i) for i in anchor if i in pos) / max(1, len(anchor)), 4
            ),
            "mean_disagreement_hard": round(mean_d_hard, 4),
            "mean_disagreement_anchor": round(mean_d_anchor, 4),
            "hard_cluster_hist": _hist(hard, _cluster),
            "anchor_cluster_hist": _hist(anchor, _cluster),
            "hard_gold_hist": {str(k): v for k, v in sorted(hard_gold.items())},
            "batch_gold_hist": _hist(b_idx, _gold),
            "hard_covers_seen_errors": hard_covers_errs,
            "hard_covers_seen_errors_frac": round(
                hard_covers_errs / max(1, n_seen_err), 4
            ),
        },
        "representativeness": {
            "weak_clusters": weak,
            "weak_cluster_hard_slots": weak_hard_slots,
            "weak_cluster_hard_coverage": weak_coverage,
            "warn_weak_undercovered": bool(weak)
            and weak_hard_slots < max(1, len(hard) // max(1, len(weak))),
            "warn_quotas_empty": not bool(batch.get("group_hard_quotas")),
            "warn_hard_all_same_cluster": len(_hist(hard, _cluster)) <= 1 and len(hard) > 3,
        },
        "role_integrity": {
            "batch_intersect_d_select": leak_select[:20],
            "batch_intersect_d_anchor_role": leak_d_anchor[:20],
            "n_leak_d_select": len(leak_select),
            "n_leak_d_anchor": len(leak_d_anchor),
            "ok_disjoint_from_d_select": len(leak_select) == 0,
            "note": (
                "Batch is drawn from train seen-pool; D_select/D_anchor roles are "
                "heldout sources. Non-empty intersect is a role bug."
            ),
        },
    }

"""Shared E6 Amazon ordinal metrics (macro-within-cluster + CVaR + op-point)."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


COLLAPSED = {
    1: 0,
    2: 0,
    3: 1,
    4: 2,
    5: 2,
}  # bins: low / mid / high


def collapse_rating(y: int) -> int:
    return COLLAPSED.get(int(y), 1)


def macro_within_group(
    preds: np.ndarray,
    y: np.ndarray,
    cids: np.ndarray,
    group: int,
    classes: Sequence[int] = (1, 2, 3, 4, 5),
) -> float:
    m = cids == group
    if not m.any():
        return 0.5
    accs = []
    for c in classes:
        mc = m & (y == c)
        if not mc.any():
            continue
        accs.append(float((preds[mc] == c).mean()))
    return float(np.mean(accs)) if accs else 0.5


def group_macros(
    preds: np.ndarray,
    y: np.ndarray,
    cids: np.ndarray,
    *,
    exclude: Optional[Iterable[int]] = None,
) -> Dict[int, float]:
    exclude_set = set(exclude or [])
    out = {}
    for g in sorted(set(int(x) for x in cids.tolist())):
        if g in exclude_set:
            continue
        out[int(g)] = macro_within_group(preds, y, cids, int(g))
    return out


def all_metrics(
    preds: np.ndarray,
    y: np.ndarray,
    cids: np.ndarray,
    user_ids: Optional[np.ndarray] = None,
    *,
    shrink_w: float = 50.0,
    tau: float = 0.10,
) -> Dict[str, float]:
    from prime.fitness.metrics import tail_user_accuracy, worst_group_accuracy
    from prime.fitness.objective import soft_min_accuracies

    preds = np.asarray(preds)
    y = np.asarray(y)
    cids = np.asarray(cids)
    g = group_macros(preds, y, cids)
    vals = np.sort(np.asarray(list(g.values()), dtype=float)) if g else np.asarray([0.5])
    k25 = max(1, int(round(0.25 * len(vals))))
    counts = {gi: int((cids == gi).sum()) for gi in g}
    pooled = float(vals.mean())
    if shrink_w > 0 and g:
        shrunk = {
            gi: (counts[gi] * g[gi] + shrink_w * pooled) / (counts[gi] + shrink_w) for gi in g
        }
    else:
        shrunk = dict(g)
    sv = np.sort(np.asarray(list(shrunk.values()))) if shrunk else vals

    per_class = []
    for c in (1, 2, 3, 4, 5):
        m = y == c
        if m.any():
            per_class.append(float((preds[m] == c).mean()))
    worst_class = float(min(per_class)) if per_class else 0.5

    uids = user_ids if user_ids is not None else np.arange(len(y))
    return {
        "hard_min": float(vals[0]),
        "cvar25": float(vals[:k25].mean()),
        "cvar50": float(vals[: max(1, len(vals) // 2)].mean()),
        "mean_macro": float(vals.mean()),
        "softmin": float(soft_min_accuracies({int(k): float(v) for k, v in g.items()}, tau))
        if g
        else float("nan"),
        "softmin_shrunk": float(
            soft_min_accuracies({int(k): float(v) for k, v in shrunk.items()}, tau)
        )
        if shrunk
        else float("nan"),
        "hard_min_shrunk": float(sv[0]),
        "worst_class": worst_class,
        "R_global": float((preds == y).mean()),
        "R_tail": float(tail_user_accuracy(preds, y, uids, quantile=0.2)),
        "R_worst": float(worst_group_accuracy(preds, y, uids, percentile=10.0)),
        "pred_mean": float(preds.mean()),
        "gold_mean": float(y.mean()),
        "op_shift": float(abs(float(preds.mean()) - float(y.mean()))),
        "mae": float(np.mean(np.abs(preds.astype(float) - y.astype(float)))),
    }


def spearman(a: List[float], b: List[float]) -> float:
    ra = np.empty(len(a))
    ra[np.argsort(a)] = np.arange(len(a))
    rb = np.empty(len(b))
    rb[np.argsort(b)] = np.arange(len(b))
    return float(np.corrcoef(ra, rb)[0, 1])

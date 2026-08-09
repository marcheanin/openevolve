"""D_anchor regression gate with CI/δ tolerance (SPEC v3 §4.3, Р15).

Raw point-delta gating is unusable at |D_anchor|=50: one example is worth
1/50 = 0.02 = the entire tolerance, so the gate rejects on a single flipped
prediction. D_anchor is also built from confidently-solved cells, so it sits at
a ceiling and its noise is one-sided downward — a same-quality prompt loses a
couple of anchors by regression to the mean alone.

This module treats δ as a tolerance band and requires a *significant* paired
regression (exact one-sided McNemar over discordant pairs) before rejecting.
Cell-level (cluster × gold) breakdown is reported so a directional trade-off
(e.g. a 4/5 boundary shift) can be told apart from scattered degradation.
"""

from __future__ import annotations

from collections import Counter
from math import comb
from typing import Any, Dict, List, Optional, Sequence


def mcnemar_one_sided_p(n_improved: int, n_worsened: int) -> float:
    """
    P(X >= n_worsened) for X ~ Binom(n_discordant, 0.5): exact one-sided test
    that the candidate lost more anchors than it gained.
    """
    n = int(n_improved) + int(n_worsened)
    if n <= 0:
        return 1.0
    c = int(n_worsened)
    tail = sum(comb(n, k) for k in range(c, n + 1))
    return float(tail) / float(2**n)


def effective_delta(delta: float, n_anchor: int, min_examples: int = 2) -> float:
    """
    Tolerance must not be finer than the sampling grid: at n=50 a δ of 0.02 is
    exactly one example. Floor it at `min_examples` observations.
    """
    if n_anchor <= 0:
        return float(delta)
    return max(float(delta), float(min_examples) / float(n_anchor))


def _cell_breakdown(
    before_correct: Sequence[bool],
    after_correct: Sequence[bool],
    labels: Sequence[int],
    cluster_ids: Optional[Sequence[int]],
) -> Dict[str, Any]:
    """Per (cluster × gold) regression counts + concentration of the damage."""
    cells: Dict[str, Dict[str, int]] = {}
    regressed_per_cell: Counter = Counter()
    for i, (b, a) in enumerate(zip(before_correct, after_correct)):
        cid = int(cluster_ids[i]) if cluster_ids is not None and i < len(cluster_ids) else 0
        gold = int(labels[i]) if i < len(labels) else -1
        key = f"c{cid}_gold{gold}"
        cell = cells.setdefault(key, {"n": 0, "before_correct": 0, "after_correct": 0, "lost": 0, "gained": 0})
        cell["n"] += 1
        cell["before_correct"] += int(bool(b))
        cell["after_correct"] += int(bool(a))
        if b and not a:
            cell["lost"] += 1
            regressed_per_cell[key] += 1
        elif a and not b:
            cell["gained"] += 1

    total_lost = sum(regressed_per_cell.values())
    top_cell, top_n = (regressed_per_cell.most_common(1) or [(None, 0)])[0]
    return {
        "cells": dict(sorted(cells.items())),
        "regressed_cells": dict(regressed_per_cell.most_common()),
        "n_regressed_cells": len(regressed_per_cell),
        "top_regressed_cell": top_cell,
        # 1.0 = all damage in one cluster×class cell (directional trade-off);
        # low values = scattered degradation.
        "regression_concentration": (
            round(top_n / total_lost, 4) if total_lost else 0.0
        ),
    }


def evaluate_anchor_gate(
    *,
    before_correct: Sequence[bool],
    after_correct: Sequence[bool],
    labels: Sequence[int],
    cluster_ids: Optional[Sequence[int]] = None,
    delta: float = 0.02,
    alpha: float = 0.05,
    min_delta_examples: int = 2,
) -> Dict[str, Any]:
    """
    Accept unless the candidate regresses beyond the δ tolerance *and* the
    paired regression is significant at `alpha`.

    Returns the decision plus everything needed to audit it later.
    """
    n = len(before_correct)
    if n == 0:
        return {
            "accepted": True,
            "reason": "empty_anchor",
            "n_anchor": 0,
            "acc_before": 1.0,
            "acc_after": 1.0,
        }

    acc_before = sum(1 for b in before_correct if b) / n
    acc_after = sum(1 for a in after_correct if a) / n
    drop = acc_before - acc_after

    n_improved = sum(1 for b, a in zip(before_correct, after_correct) if a and not b)
    n_worsened = sum(1 for b, a in zip(before_correct, after_correct) if b and not a)
    p_value = mcnemar_one_sided_p(n_improved, n_worsened)
    delta_eff = effective_delta(delta, n, min_delta_examples)

    within_tolerance = drop <= delta_eff
    significant = p_value < alpha
    accepted = within_tolerance or not significant
    if within_tolerance:
        reason = "within_tolerance"
    elif significant:
        reason = "significant_regression"
    else:
        reason = "tolerated_noise"

    out: Dict[str, Any] = {
        "accepted": bool(accepted),
        "reason": reason,
        "n_anchor": n,
        "acc_before": round(acc_before, 4),
        "acc_after": round(acc_after, 4),
        "drop": round(drop, 4),
        "delta_config": float(delta),
        "delta_effective": round(delta_eff, 4),
        "within_tolerance": bool(within_tolerance),
        "n_improved": n_improved,
        "n_worsened": n_worsened,
        "n_discordant": n_improved + n_worsened,
        "mcnemar_p_one_sided": round(p_value, 4),
        "alpha": float(alpha),
        "significant": bool(significant),
        # Smallest net loss that could ever be significant at this n/alpha:
        # useful to know whether the gate has any power at all.
        "min_detectable_worsened": _min_detectable(alpha),
    }
    out.update(_cell_breakdown(before_correct, after_correct, labels, cluster_ids))
    return out


def _min_detectable(alpha: float) -> int:
    """Fewest all-one-way discordant pairs that reach p < alpha (b=0)."""
    c = 1
    while c < 64:
        if mcnemar_one_sided_p(0, c) < alpha:
            return c
        c += 1
    return c


def summarize_gate_history(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Rejection counter + reasons across cycles (Р15 requires the counter)."""
    if not records:
        return {"n_evaluated": 0, "n_rejected": 0, "rejection_rate": 0.0, "reasons": {}}
    n_rej = sum(1 for r in records if not r.get("accepted", True))
    reasons = Counter(str(r.get("reason")) for r in records)
    return {
        "n_evaluated": len(records),
        "n_rejected": n_rej,
        "rejection_rate": round(n_rej / len(records), 4),
        "reasons": dict(reasons),
        "mean_drop": round(
            sum(float(r.get("drop", 0.0)) for r in records) / len(records), 4
        ),
        "cycles_rejected": [r.get("cycle") for r in records if not r.get("accepted", True)],
    }

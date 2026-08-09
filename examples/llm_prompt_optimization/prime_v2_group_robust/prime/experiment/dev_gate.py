"""D_dev generalization gate (ROADMAP_PHASE3 F8).

Promote a cycle heir only if softmin on fixed D_dev does not drop more than
``dev_gate_delta`` relative to the incumbent champion.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def softmin_from_metrics(metrics: Dict[str, Any], *, tau: float = 0.10) -> float:
    """Prefer precomputed R_soft_min_gba; else softmin over GBA / group accs."""
    for key in ("R_soft_min_gba", "R_soft_min_group"):
        if key in metrics and metrics[key] is not None:
            return float(metrics[key])
    from prime.fitness.objective import soft_min_accuracies

    accs = metrics.get("cluster_gba_shrunk") or metrics.get("cluster_gba") or {}
    if not accs:
        accs = metrics.get("cluster_accuracies_balanced_shrunk") or metrics.get(
            "cluster_accuracies_shrunk"
        ) or {}
    if not accs:
        return float(metrics.get("R_gba_mean", metrics.get("R_global", 0.0)))
    return float(soft_min_accuracies({int(k): float(v) for k, v in accs.items()}, tau))


def evaluate_dev_gate(
    *,
    champion_metrics: Dict[str, Any],
    candidate_metrics: Dict[str, Any],
    delta: float = 0.01,
    tau: float = 0.10,
    mode: str = "reject",
) -> Dict[str, Any]:
    """
    Accept candidate unless softmin(D_dev) drops more than ``delta``.

    ``mode``:
      - ``off``: always accept, still report numbers
      - ``monitor``: never block; log would-be rejection
      - ``reject``: block when drop > delta
    """
    soft_champ = softmin_from_metrics(champion_metrics, tau=tau)
    soft_cand = softmin_from_metrics(candidate_metrics, tau=tau)
    drop = soft_champ - soft_cand
    within = drop <= float(delta)
    would_reject = not within
    if mode == "off":
        accepted = True
        reason = "dev_gate_off"
    elif mode == "monitor":
        accepted = True
        reason = "within_tolerance" if within else "would_reject_monitor"
    else:  # reject
        accepted = within
        reason = "within_tolerance" if within else "dev_softmin_drop"

    return {
        "accepted": bool(accepted),
        "reason": reason,
        "mode": mode,
        "softmin_champion": round(soft_champ, 6),
        "softmin_candidate": round(soft_cand, 6),
        "drop": round(drop, 6),
        "delta": float(delta),
        "within_tolerance": bool(within),
        "would_reject": bool(would_reject),
        "R_worst_gba_champion": champion_metrics.get("R_worst_gba"),
        "R_worst_gba_candidate": candidate_metrics.get("R_worst_gba"),
        "R_gba_mean_champion": champion_metrics.get("R_gba_mean"),
        "R_gba_mean_candidate": candidate_metrics.get("R_gba_mean"),
    }


def generalization_gap(
    select_fitness: Optional[float],
    dev_softmin: Optional[float],
) -> Optional[float]:
    """fitness(D_select) − softmin(D_dev); None if either side missing."""
    if select_fitness is None or dev_softmin is None:
        return None
    return float(select_fitness) - float(dev_softmin)

"""MAP-Elites QD feature metrics for OpenEvolve archive."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Optional, Set


def estimate_normalized_prompt_length(prompt: str, scale: float = 2000.0) -> float:
    tokens = max(1, len(re.findall(r"\S+", prompt)))
    return min(1.0, tokens / scale)


def cluster_acc_metric_name(cluster_id: int) -> str:
    return f"cluster_acc_{cluster_id}"


def _normalize_exclude(exclude_clusters: Optional[Iterable[int]]) -> Set[int]:
    return {int(x) for x in (exclude_clusters or [])}


def qd_feature_dimension_names(
    n_clusters: int,
    include_prompt_length: bool = True,
    exclude_clusters: Optional[Iterable[int]] = None,
) -> list[str]:
    """
    QD axes for MAP-Elites.

    When ``gba_exclude_none`` is on, pass ``exclude_clusters={0}`` so the forever-zero
    ``none`` group does not become a false mutator focus (E5 / M37).
    """
    exclude = _normalize_exclude(exclude_clusters)
    dims = [
        cluster_acc_metric_name(c)
        for c in range(n_clusters)
        if c not in exclude
    ]
    if include_prompt_length:
        dims.append("prompt_length")
    return dims


def build_qd_metrics(
    cluster_accuracies: Optional[Dict[int, float]],
    prompt: str,
    n_clusters: int,
    exclude_clusters: Optional[Iterable[int]] = None,
) -> Dict[str, float]:
    """
    Per-group accuracy/GBA profile + prompt length for MAP-Elites descriptors.
    Missing non-excluded clusters default to 0.0.
    """
    metrics: Dict[str, float] = {}
    accs = cluster_accuracies or {}
    exclude = _normalize_exclude(exclude_clusters)
    for c in range(n_clusters):
        if c in exclude:
            continue
        metrics[cluster_acc_metric_name(c)] = float(accs.get(c, 0.0))
    metrics["prompt_length"] = estimate_normalized_prompt_length(prompt)
    return metrics


def merge_qd_into_eval_metrics(
    base_metrics: Dict[str, Any],
    cluster_accuracies: Optional[Dict[int, float]],
    prompt: str,
    n_clusters: int,
    exclude_clusters: Optional[Iterable[int]] = None,
) -> Dict[str, Any]:
    """Attach QD dimensions without polluting fitness (they are listed in feature_dimensions)."""
    out = dict(base_metrics)
    out.update(
        build_qd_metrics(
            cluster_accuracies,
            prompt,
            n_clusters,
            exclude_clusters=exclude_clusters,
        )
    )
    return out

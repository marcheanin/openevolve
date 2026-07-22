"""MAP-Elites QD feature metrics for OpenEvolve archive."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional


def estimate_normalized_prompt_length(prompt: str, scale: float = 2000.0) -> float:
    tokens = max(1, len(re.findall(r"\S+", prompt)))
    return min(1.0, tokens / scale)


def cluster_acc_metric_name(cluster_id: int) -> str:
    return f"cluster_acc_{cluster_id}"


def qd_feature_dimension_names(n_clusters: int, include_prompt_length: bool = True) -> list[str]:
    dims = [cluster_acc_metric_name(c) for c in range(n_clusters)]
    if include_prompt_length:
        dims.append("prompt_length")
    return dims


def build_qd_metrics(
    cluster_accuracies: Optional[Dict[int, float]],
    prompt: str,
    n_clusters: int,
) -> Dict[str, float]:
    """
    Per-group accuracy profile + prompt length for MAP-Elites descriptors.
    Missing clusters default to 0.0 accuracy.
    """
    metrics: Dict[str, float] = {}
    accs = cluster_accuracies or {}
    for c in range(n_clusters):
        metrics[cluster_acc_metric_name(c)] = float(accs.get(c, 0.0))
    metrics["prompt_length"] = estimate_normalized_prompt_length(prompt)
    return metrics


def merge_qd_into_eval_metrics(
    base_metrics: Dict[str, Any],
    cluster_accuracies: Optional[Dict[int, float]],
    prompt: str,
    n_clusters: int,
) -> Dict[str, Any]:
    """Attach QD dimensions without polluting fitness (they are listed in feature_dimensions)."""
    out = dict(base_metrics)
    out.update(build_qd_metrics(cluster_accuracies, prompt, n_clusters))
    return out

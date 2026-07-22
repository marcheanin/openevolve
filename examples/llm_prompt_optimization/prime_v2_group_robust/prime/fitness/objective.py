"""Configurable fitness objective (CVaR + global + kappa - length penalty)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np

from prime.config import FitnessCfg
from prime.fitness.metrics import compute_metrics


def estimate_tokens(text: str) -> int:
    return max(1, len(re.findall(r"\S+", text)))


def length_penalty(prompt: str, cfg: FitnessCfg) -> float:
    tokens = estimate_tokens(prompt)
    if tokens <= cfg.len_penalty_start:
        return 0.0
    excess = (tokens - cfg.len_penalty_start) / 100.0
    return cfg.len_penalty_per_100 * excess


def compute_fitness(
    predictions: np.ndarray,
    gold: np.ndarray,
    user_ids: np.ndarray,
    prompt: str,
    cfg: FitnessCfg,
    worker_predictions: Optional[List[np.ndarray]] = None,
    cluster_ids: Optional[np.ndarray] = None,
    hard_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute fitness from config weights. No hardcoded W1/W2/W3.
    """
    metrics = compute_metrics(
        predictions,
        gold,
        user_ids,
        worker_predictions=worker_predictions,
        cluster_ids=cluster_ids,
        cvar_quantile=cfg.cvar_quantile,
    )

    cvar = float(metrics.get("CVaR_cluster", metrics["R_global"]))
    global_acc = float(metrics["R_global"])
    kappa = float(max(0.0, metrics.get("mean_kappa", 0.0)))

    if cfg.mode == "global":
        base = global_acc
    elif cfg.mode == "v1_weighted" and hard_mask is not None:
        hard_mask = np.asarray(hard_mask, dtype=bool)
        if np.any(hard_mask):
            acc_hard = float(np.mean(predictions[hard_mask] == gold[hard_mask]))
        else:
            acc_hard = 0.0
        anchor_mask = ~hard_mask
        acc_anchor = float(np.mean(predictions[anchor_mask] == gold[anchor_mask])) if np.any(anchor_mask) else 1.0
        base = 0.5 * acc_hard + 0.3 * acc_anchor + 0.2 * kappa
        metrics["Acc_Hard"] = acc_hard
        metrics["Acc_Anchor"] = acc_anchor
    else:
        base = cfg.w_cvar * cvar + cfg.w_global * global_acc + cfg.w_kappa * kappa

    penalty = length_penalty(prompt, cfg)
    fitness = base - penalty

    return {
        "fitness": float(fitness),
        "combined_score": float(fitness),
        "base_score": float(base),
        "length_penalty": float(penalty),
        **metrics,
    }

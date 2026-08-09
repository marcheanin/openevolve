"""Configurable fitness objective (CVaR + global + kappa - length penalty)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np

from prime.config import FitnessCfg
from prime.fitness.metrics import compute_metrics

REJECT_FITNESS = -1e9


def estimate_tokens(text: str) -> int:
    return max(1, len(re.findall(r"\S+", text)))


def length_penalty(prompt: str, cfg: FitnessCfg) -> float:
    tokens = estimate_tokens(prompt)
    if tokens <= cfg.len_penalty_start:
        return 0.0
    excess = (tokens - cfg.len_penalty_start) / 100.0
    return cfg.len_penalty_per_100 * excess


def soft_min_accuracies(accs: Dict[Any, float], tau: float) -> float:
    """Log-sum-exp softmin over group accuracies.

    tau <= 0 → hard min. As tau grows, approaches the mean. Numerically stable.
    """
    vals = np.asarray(list(accs.values()), dtype=float)
    if vals.size == 0:
        return 0.0
    if tau <= 0.0:
        return float(vals.min())
    z = -vals / float(tau)
    z_max = float(z.max())
    return float(-float(tau) * (np.log(np.mean(np.exp(z - z_max))) + z_max))


def _group_accs_for_lex(metrics: Dict[str, Any], cfg: FitnessCfg) -> Dict[Any, float]:
    """Pick Acc_g / GBA_g dict used by min_group_lex and soft_min_lex."""
    group_acc = getattr(cfg, "group_acc", "balanced_global") or "balanced_global"

    if group_acc == "balanced_within":
        if cfg.shrink_prior_weight > 0 and metrics.get("cluster_gba_shrunk"):
            return metrics["cluster_gba_shrunk"]
        return metrics.get("cluster_gba") or {}

    if cfg.shrink_prior_weight > 0:
        if (
            group_acc == "balanced_global"
            and cfg.class_balanced
            and metrics.get("cluster_accuracies_balanced_shrunk")
        ):
            return metrics["cluster_accuracies_balanced_shrunk"]
        if metrics.get("cluster_accuracies_shrunk"):
            return metrics["cluster_accuracies_shrunk"]
        return metrics.get("cluster_accuracies") or {}

    if (
        group_acc == "balanced_global"
        and cfg.class_balanced
        and metrics.get("cluster_accuracies_balanced")
    ):
        return metrics["cluster_accuracies_balanced"]
    return metrics.get("cluster_accuracies") or {}


def _reject(metrics: Dict[str, Any], reason: str) -> Dict[str, Any]:
    return {
        "fitness": REJECT_FITNESS,
        "combined_score": REJECT_FITNESS,
        "base_score": REJECT_FITNESS,
        "length_penalty": 0.0,
        "reject_reason": reason,
        **metrics,
    }


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
        beta_a=cfg.beta_a,
        beta_b=cfg.beta_b,
        tail_quantile=cfg.tail_quantile,
        shrink_prior_weight=cfg.shrink_prior_weight,
        class_balanced=cfg.class_balanced,
        gba_min_pos=getattr(cfg, "gba_min_pos", 10),
        gba_min_neg=getattr(cfg, "gba_min_neg", 10),
        gba_exclude_none=getattr(cfg, "gba_exclude_none", True),
    )

    # Phase3 F1/F2: fail-closed and degenerate guards.
    if getattr(cfg, "fail_closed", False):
        inv = float(metrics.get("invalid_rate", 0.0))
        if inv > float(getattr(cfg, "max_invalid_rate", 0.02)):
            return _reject(metrics, "invalid_rate")
        pos = float(metrics.get("pred_pos_rate", 0.5))
        lo = float(getattr(cfg, "min_pred_pos_rate", 0.02))
        hi = 1.0 - lo
        if pos < lo or pos > hi:
            return _reject(metrics, "degenerate")

    cvar = float(metrics.get("CVaR_cluster", metrics["R_global"]))
    cvar_shrunk = float(metrics.get("CVaR_cluster_shrunk", cvar))
    global_acc = float(metrics["R_global"])
    macro_acc = float(metrics.get("R_macro", global_acc))
    r_tail = float(metrics.get("R_tail", global_acc))
    kappa = float(max(0.0, metrics.get("mean_kappa", 0.0)))
    group_acc_mode = getattr(cfg, "group_acc", "balanced_global") or "balanced_global"

    # When class balancing is on, the tail and the tie-break both switch to the
    # balanced variants so a pure 4/5 threshold shift cannot buy fitness (C11).
    # For balanced_within (GBA), tie-break is mean GBA.
    if group_acc_mode == "balanced_within":
        tie_break_acc = float(
            metrics.get("R_gba_mean_shrunk", metrics.get("R_gba_mean", macro_acc))
        )
    elif cfg.class_balanced:
        cvar_shrunk = float(metrics.get("CVaR_cluster_balanced_shrunk", cvar_shrunk))
        tie_break_acc = macro_acc
    else:
        tie_break_acc = global_acc

    if cfg.mode == "cvar_lex":
        # SPEC v3 §4.4: lexicographic approximation, no magic weight mix, no kappa.
        base = cvar_shrunk + cfg.epsilon_global * tie_break_acc
    elif cfg.mode == "min_group_lex":
        # Phase 2c arm B / Phase3: pure lex worst-group (Acc or GBA).
        accs = _group_accs_for_lex(metrics, cfg)
        if not accs and group_acc_mode == "balanced_within":
            # Never silently fall back to global — that reopens the M31 silence exploit.
            return _reject(metrics, "gba_empty")
        r_min = float(min(accs.values())) if accs else global_acc
        metrics["R_worst_group"] = r_min
        if group_acc_mode == "balanced_within":
            metrics["R_worst_gba"] = r_min
        base = r_min + cfg.epsilon_global * tie_break_acc
    elif cfg.mode == "soft_min_lex":
        # Softmin over Acc_g / GBA_g (Phase3 F4).
        accs = _group_accs_for_lex(metrics, cfg)
        if not accs and group_acc_mode == "balanced_within":
            return _reject(metrics, "gba_empty")
        r_min = float(min(accs.values())) if accs else global_acc
        r_soft = soft_min_accuracies(accs, cfg.soft_min_tau) if accs else global_acc
        metrics["R_worst_group"] = r_min
        metrics["R_soft_min_group"] = float(r_soft)
        if group_acc_mode == "balanced_within":
            metrics["R_worst_gba"] = r_min
            metrics["R_soft_min_gba"] = float(r_soft)
        base = float(r_soft) + cfg.epsilon_global * tie_break_acc
    elif cfg.mode == "global":
        base = global_acc
    elif cfg.mode == "global_tail_mix":
        # Soft mix: mean accuracy + continuous user-tail (not p10 R_worst / not CVaR).
        # Default 0.5/0.5; R_tail = mean acc over worst `tail_quantile` users on the
        # eval set (D_select in the main loop).
        base = cfg.w_global_mix * global_acc + cfg.w_tail * r_tail
    elif cfg.mode == "macro":
        # Class-balanced control twin for `global`: same "no group awareness"
        # position, but immune to the ordinal-threshold exploit.
        base = macro_acc
    elif cfg.mode == "v1_weighted" and hard_mask is not None:
        # v1 ablation arm: the only mode where kappa participates (SPEC v3 §4.2).
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
        # Legacy v2 weighted mix, kept for comparison runs; kappa removed (SPEC v3).
        base = cfg.w_cvar * cvar + cfg.w_global * global_acc

    penalty = length_penalty(prompt, cfg)
    fitness = base - penalty

    return {
        "fitness": float(fitness),
        "combined_score": float(fitness),
        "base_score": float(base),
        "length_penalty": float(penalty),
        "reject_reason": None,
        **metrics,
    }

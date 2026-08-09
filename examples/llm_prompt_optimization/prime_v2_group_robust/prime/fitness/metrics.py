"""Evaluation metrics: accuracy, worst-group, CVaR, kappa, MAE."""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from sklearn.metrics import cohen_kappa_score
except ImportError:  # pragma: no cover
    cohen_kappa_score = None


def accuracy(predictions: np.ndarray, gold: np.ndarray) -> float:
    if len(predictions) == 0:
        return 0.0
    return float(np.mean(predictions == gold))


def mae(predictions: np.ndarray, gold: np.ndarray) -> float:
    if len(predictions) == 0:
        return 0.0
    return float(np.mean(np.abs(predictions - gold)))


def per_user_accuracy(
    predictions: np.ndarray,
    gold: np.ndarray,
    user_ids: np.ndarray,
) -> Dict[int, float]:
    correct = predictions == gold
    out: Dict[int, float] = {}
    for uid in np.unique(user_ids):
        mask = user_ids == uid
        out[int(uid)] = float(np.mean(correct[mask]))
    return out


def worst_group_accuracy(
    predictions: np.ndarray,
    gold: np.ndarray,
    user_ids: np.ndarray,
    percentile: float = 10.0,
) -> float:
    """
    R_worst: percentile (default 10th) of per-user accuracies.
    Matches WILDS Amazon convention for worst-group reporting.
    """
    per_user = per_user_accuracy(predictions, gold, user_ids)
    if not per_user:
        return 0.0
    return float(np.percentile(list(per_user.values()), percentile))


def tail_user_accuracy(
    predictions: np.ndarray,
    gold: np.ndarray,
    user_ids: np.ndarray,
    quantile: float = 0.2,
) -> float:
    """
    R_tail: mean accuracy over the worst `quantile` fraction of users.

    Unlike `worst_group_accuracy` (a single percentile) this averages a whole
    slice, so it moves continuously as individual users improve. With 8 reviews
    per user a percentile is pinned to the 1/8 lattice and stays flat across
    cycles (OBSERVATIONS M9); a mean over ~20% of users does not.
    """
    per_user = per_user_accuracy(predictions, gold, user_ids)
    if not per_user:
        return 0.0
    values = sorted(per_user.values())
    k = max(1, int(np.ceil(len(values) * quantile)))
    return float(np.mean(values[:k]))


def class_balance_weights(gold: np.ndarray) -> np.ndarray:
    """
    Per-example weights that equalise the gold classes, normalised to mean 1.

    Rationale (OBSERVATIONS C11): on this data 5★ is 56% of examples and 4★ 28%,
    and the ensemble is good at 5★ (0.90) and bad at 4★ (0.38). Under unweighted
    accuracy a candidate can score by simply shifting the 4/5 threshold, which is
    exactly what the pair run did (+0.161 on 4★, −0.120 on 5★, macro-neutral). With
    these weights such a shift is scored neutrally, so the search has to find real
    discrimination instead of re-weighting.
    """
    gold = np.asarray(gold)
    weights = np.ones(len(gold), dtype=float)
    for g in np.unique(gold):
        mask = gold == g
        weights[mask] = 1.0 / float(np.sum(mask))
    total = float(np.sum(weights))
    return weights * (len(gold) / total) if total > 0 else weights


def weighted_accuracy(
    predictions: np.ndarray, gold: np.ndarray, weights: Optional[np.ndarray] = None
) -> float:
    if len(predictions) == 0:
        return 0.0
    correct = (predictions == gold).astype(float)
    if weights is None:
        return float(np.mean(correct))
    w = np.asarray(weights, dtype=float)
    denom = float(np.sum(w))
    return float(np.sum(correct * w) / denom) if denom > 0 else 0.0


def macro_class_accuracy(predictions: np.ndarray, gold: np.ndarray) -> float:
    """Unweighted mean of per-class accuracies (a.k.a. balanced accuracy)."""
    if len(predictions) == 0:
        return 0.0
    per_class = [
        float(np.mean(predictions[gold == g] == g)) for g in np.unique(gold)
    ]
    return float(np.mean(per_class)) if per_class else 0.0


def per_class_accuracy(predictions: np.ndarray, gold: np.ndarray) -> Dict[int, float]:
    return {
        int(g): float(np.mean(predictions[gold == g] == g)) for g in np.unique(gold)
    }


def cluster_accuracies(
    predictions: np.ndarray,
    gold: np.ndarray,
    cluster_ids: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Dict[int, float]:
    correct = predictions == gold
    out: Dict[int, float] = {}
    for cid in np.unique(cluster_ids):
        mask = cluster_ids == cid
        if not np.any(mask):
            continue
        if weights is None:
            out[int(cid)] = float(np.mean(correct[mask]))
        else:
            out[int(cid)] = weighted_accuracy(
                predictions[mask], gold[mask], np.asarray(weights)[mask]
            )
    return out


def smoothed_cluster_accuracies(
    predictions: np.ndarray,
    gold: np.ndarray,
    cluster_ids: np.ndarray,
    beta_a: float = 1.0,
    beta_b: float = 1.0,
    prior_weight: float = 0.0,
    prior_mean: Optional[float] = None,
    weights: Optional[np.ndarray] = None,
) -> Dict[int, float]:
    """
    Per-cluster accuracy shrunk toward a prior, to stabilise the CVaR tail.

    Two modes:

    * ``prior_weight > 0`` — shrink toward the **grand mean** with a prior worth
      ``prior_weight`` pseudo-examples: ``(c_k + w·mean) / (n_k + w)``. This is the
      one that works. Measured on the pair run's test split (E-C,
      ``scripts/exp_tail_metric_stability.py``): it cuts the bootstrap SD of the
      worst-cluster statistic from 0.0355 to 0.0217 (−39%) *and* enlarges the
      observed effect, i.e. it improves discriminability where widening the
      quantile made it worse.
    * ``prior_weight == 0`` — legacy Beta(a, b) form ``(c_k + a) / (n_k + a + b)``.
      With the historical ``a = b = 1`` this shrinks toward 0.5 with a prior worth
      two examples, so at ~48 examples per cluster it moves the estimate by ~4%:
      effectively no smoothing at all, which is why `CVaR_cluster_shrunk` tracked
      the raw metric in every run so far.
    """
    correct = predictions == gold
    if prior_weight > 0 and prior_mean is None:
        prior_mean = weighted_accuracy(predictions, gold, weights)
    out: Dict[int, float] = {}
    w_arr = None if weights is None else np.asarray(weights, dtype=float)
    for cid in np.unique(cluster_ids):
        mask = cluster_ids == cid
        n_k = int(np.sum(mask))
        if n_k == 0:
            continue
        if prior_weight > 0:
            if w_arr is None:
                n_eff, c_k = float(n_k), float(np.sum(correct[mask]))
            else:
                n_eff = float(np.sum(w_arr[mask]))
                c_k = float(np.sum(correct[mask] * w_arr[mask]))
            out[int(cid)] = (c_k + prior_weight * float(prior_mean)) / (n_eff + prior_weight)
        else:
            c_k = float(np.sum(correct[mask]))
            out[int(cid)] = (c_k + beta_a) / (n_k + beta_a + beta_b)
    return out


def cvar_from_accuracies(accs: Dict[int, float], quantile: float = 0.33) -> float:
    """Mean accuracy of the worst `quantile` fraction of groups."""
    values = list(accs.values())
    if not values:
        return 0.0
    values_sorted = sorted(values)
    k = max(1, int(np.ceil(len(values_sorted) * quantile)))
    return float(np.mean(values_sorted[:k]))


def cvar_cluster_accuracy(
    predictions: np.ndarray,
    gold: np.ndarray,
    cluster_ids: np.ndarray,
    quantile: float = 0.33,
) -> float:
    """
    CVaR over style clusters: mean accuracy of worst `quantile` fraction of clusters.
    """
    return cvar_from_accuracies(
        cluster_accuracies(predictions, gold, cluster_ids), quantile
    )


def mean_pairwise_kappa(
    worker_predictions: List[np.ndarray],
    weighted: bool = True,
) -> float:
    if cohen_kappa_score is None or len(worker_predictions) < 2:
        return 0.0
    # Infer label space from observed votes (binary toxicity vs Amazon 1–5).
    flat = np.concatenate([np.asarray(w).ravel() for w in worker_predictions])
    if flat.size == 0:
        return 0.0
    # Drop INVALID (-1) pairs for kappa.
    flat_valid = flat[flat >= 0]
    if flat_valid.size == 0:
        return 0.0
    lo, hi = int(flat_valid.min()), int(flat_valid.max())
    if lo >= 0 and hi <= 1:
        labels = [0, 1]
        use_weighted = False
    else:
        labels = [1, 2, 3, 4, 5]
        use_weighted = weighted
    kappas: List[float] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(len(worker_predictions)):
            for j in range(i + 1, len(worker_predictions)):
                a, b = np.asarray(worker_predictions[i]), np.asarray(worker_predictions[j])
                mask = (a >= 0) & (b >= 0)
                if not np.any(mask):
                    kappas.append(0.0)
                    continue
                try:
                    if use_weighted:
                        k = cohen_kappa_score(a[mask], b[mask], weights="quadratic", labels=labels)
                    else:
                        k = cohen_kappa_score(a[mask], b[mask], labels=labels)
                    kappas.append(float(k) if not np.isnan(k) else 0.0)
                except (ValueError, ZeroDivisionError):
                    kappas.append(0.0)
    return float(np.mean(kappas)) if kappas else 0.0


def group_balanced_accuracies(
    predictions: np.ndarray,
    gold: np.ndarray,
    group_ids: np.ndarray,
    *,
    min_pos: int = 10,
    min_neg: int = 10,
    exclude_groups: Optional[set] = None,
) -> Dict[int, float]:
    """
    Within-group balanced accuracy: GBA_g = 0.5 * (TPR_g + TNR_g).

    Groups below min_pos/min_neg cell sizes are omitted (ROADMAP_PHASE3 F3).
    INVALID predictions (-1) count as errors for both classes.
    """
    predictions = np.asarray(predictions)
    gold = np.asarray(gold)
    group_ids = np.asarray(group_ids)
    exclude = exclude_groups or set()
    out: Dict[int, float] = {}
    for gid in np.unique(group_ids):
        g = int(gid)
        if g in exclude:
            continue
        mask = group_ids == g
        pos = mask & (gold == 1)
        neg = mask & (gold == 0)
        n_pos, n_neg = int(np.sum(pos)), int(np.sum(neg))
        if n_pos < min_pos or n_neg < min_neg:
            continue
        tpr = float(np.mean(predictions[pos] == 1)) if n_pos else 0.0
        tnr = float(np.mean(predictions[neg] == 0)) if n_neg else 0.0
        out[g] = 0.5 * (tpr + tnr)
    return out


def smoothed_group_balanced_accuracies(
    predictions: np.ndarray,
    gold: np.ndarray,
    group_ids: np.ndarray,
    *,
    prior_weight: float = 0.0,
    min_pos: int = 10,
    min_neg: int = 10,
    exclude_groups: Optional[set] = None,
) -> Dict[int, float]:
    """Shrink TPR/TNR toward grand means, then average (same prior_weight as Acc shrink)."""
    raw = group_balanced_accuracies(
        predictions,
        gold,
        group_ids,
        min_pos=min_pos,
        min_neg=min_neg,
        exclude_groups=exclude_groups,
    )
    if prior_weight <= 0 or not raw:
        return raw
    predictions = np.asarray(predictions)
    gold = np.asarray(gold)
    # Grand TPR / TNR over all eligible examples.
    pos = gold == 1
    neg = gold == 0
    grand_tpr = float(np.mean(predictions[pos] == 1)) if np.any(pos) else 0.5
    grand_tnr = float(np.mean(predictions[neg] == 0)) if np.any(neg) else 0.5
    group_ids = np.asarray(group_ids)
    exclude = exclude_groups or set()
    out: Dict[int, float] = {}
    for gid in raw:
        if gid in exclude:
            continue
        mask = group_ids == gid
        pos_m = mask & (gold == 1)
        neg_m = mask & (gold == 0)
        n_pos, n_neg = float(np.sum(pos_m)), float(np.sum(neg_m))
        tpr = float(np.sum(predictions[pos_m] == 1))
        tnr = float(np.sum(predictions[neg_m] == 0))
        tpr_s = (tpr + prior_weight * grand_tpr) / (n_pos + prior_weight)
        tnr_s = (tnr + prior_weight * grand_tnr) / (n_neg + prior_weight)
        out[int(gid)] = 0.5 * (tpr_s + tnr_s)
    return out


def compute_metrics(
    predictions: np.ndarray,
    gold_labels: np.ndarray,
    user_ids: np.ndarray,
    worker_predictions: Optional[List[np.ndarray]] = None,
    cluster_ids: Optional[np.ndarray] = None,
    cvar_quantile: float = 0.33,
    beta_a: float = 1.0,
    beta_b: float = 1.0,
    tail_quantile: float = 0.2,
    shrink_prior_weight: float = 0.0,
    class_balanced: bool = False,
    gba_min_pos: int = 10,
    gba_min_neg: int = 10,
    gba_exclude_none: bool = True,
) -> Dict[str, Any]:
    """Full metric bundle for logging and selection."""
    predictions = np.asarray(predictions)
    gold_labels = np.asarray(gold_labels)
    user_ids = np.asarray(user_ids)

    if len(predictions) == 0:
        return {
            "R_global": 0.0,
            "R_worst": 0.0,
            "R_tail": 0.0,
            "mae": 0.0,
            "CVaR_cluster": 0.0,
            "num_users": 0,
            "num_examples": 0,
            "invalid_rate": 0.0,
            "pred_pos_rate": 0.0,
        }

    invalid_mask = predictions < 0
    invalid_rate = float(np.mean(invalid_mask))
    valid_preds = predictions[~invalid_mask]
    pred_pos_rate = float(np.mean(valid_preds == 1)) if len(valid_preds) else 0.0

    r_global = accuracy(predictions, gold_labels)
    r_worst = worst_group_accuracy(predictions, gold_labels, user_ids)
    r_tail = tail_user_accuracy(predictions, gold_labels, user_ids, quantile=tail_quantile)
    mae_val = mae(predictions, gold_labels)

    # Binary diagnostics (toxicity).
    toxic_mask = gold_labels == 1
    nontoxic_mask = gold_labels == 0
    toxic_recall = (
        float(np.mean(predictions[toxic_mask] == 1)) if np.any(toxic_mask) else 0.0
    )
    specificity = (
        float(np.mean(predictions[nontoxic_mask] == 0)) if np.any(nontoxic_mask) else 0.0
    )

    result: Dict[str, Any] = {
        "R_global": r_global,
        "R_worst": r_worst,
        "R_tail": r_tail,
        "R_macro": macro_class_accuracy(predictions, gold_labels),
        "mae": mae_val,
        "accuracy": r_global,
        "num_users": int(len(np.unique(user_ids))),
        "num_examples": int(len(predictions)),
        "accuracy_per_user": per_user_accuracy(predictions, gold_labels, user_ids),
        "accuracy_per_class": per_class_accuracy(predictions, gold_labels),
        "invalid_rate": invalid_rate,
        "pred_pos_rate": pred_pos_rate,
        "toxic_recall": toxic_recall,
        "specificity": specificity,
    }

    if cluster_ids is not None:
        cluster_ids = np.asarray(cluster_ids)
        result["CVaR_cluster"] = cvar_cluster_accuracy(
            predictions, gold_labels, cluster_ids, quantile=cvar_quantile
        )
        result["cluster_accuracies"] = cluster_accuracies(predictions, gold_labels, cluster_ids)
        # Min Acc over groups with ≥1 example. On CivilComments (comment-as-user)
        # R_worst (user p10) collapses to 0; use R_worst_group for headlines (M26).
        ca = result["cluster_accuracies"]
        if ca:
            result["R_worst_group"] = float(min(ca.values()))
        # The selection-facing variant. `shrink_prior_weight > 0` shrinks toward the
        # grand mean, which is what actually stabilises the tail; the legacy
        # Beta(1,1) path is kept so historical runs stay reproducible.
        shrunk = smoothed_cluster_accuracies(
            predictions,
            gold_labels,
            cluster_ids,
            beta_a=beta_a,
            beta_b=beta_b,
            prior_weight=shrink_prior_weight,
        )
        result["cluster_accuracies_shrunk"] = shrunk
        result["CVaR_cluster_shrunk"] = cvar_from_accuracies(shrunk, quantile=cvar_quantile)
        result["shrink_prior_weight"] = float(shrink_prior_weight)

        if class_balanced:
            weights = class_balance_weights(gold_labels)
            bal = cluster_accuracies(predictions, gold_labels, cluster_ids, weights=weights)
            bal_shrunk = smoothed_cluster_accuracies(
                predictions,
                gold_labels,
                cluster_ids,
                beta_a=beta_a,
                beta_b=beta_b,
                prior_weight=shrink_prior_weight,
                weights=weights,
            )
            result["cluster_accuracies_balanced"] = bal
            result["cluster_accuracies_balanced_shrunk"] = bal_shrunk
            result["CVaR_cluster_balanced"] = cvar_from_accuracies(bal, quantile=cvar_quantile)
            result["CVaR_cluster_balanced_shrunk"] = cvar_from_accuracies(
                bal_shrunk, quantile=cvar_quantile
            )

        # Within-group balanced accuracy (Phase3 F3 / M31 fix).
        exclude = {0} if gba_exclude_none else set()
        gba = group_balanced_accuracies(
            predictions,
            gold_labels,
            cluster_ids,
            min_pos=gba_min_pos,
            min_neg=gba_min_neg,
            exclude_groups=exclude,
        )
        gba_shrunk = smoothed_group_balanced_accuracies(
            predictions,
            gold_labels,
            cluster_ids,
            prior_weight=shrink_prior_weight,
            min_pos=gba_min_pos,
            min_neg=gba_min_neg,
            exclude_groups=exclude,
        )
        result["cluster_gba"] = gba
        result["cluster_gba_shrunk"] = gba_shrunk
        result["gba_eligible_groups"] = sorted(gba.keys())
        if gba:
            result["R_worst_gba"] = float(min(gba.values()))
            result["R_gba_mean"] = float(np.mean(list(gba.values())))
            result["worst_gba_group"] = int(min(gba, key=gba.get))
        if gba_shrunk:
            result["R_worst_gba_shrunk"] = float(min(gba_shrunk.values()))
            result["R_gba_mean_shrunk"] = float(np.mean(list(gba_shrunk.values())))
        # none group separately when present.
        gba_all = group_balanced_accuracies(
            predictions,
            gold_labels,
            cluster_ids,
            min_pos=max(1, gba_min_pos // 2),
            min_neg=max(1, gba_min_neg // 2),
            exclude_groups=set(),
        )
        if 0 in gba_all:
            result["R_gba_none"] = float(gba_all[0])

    if worker_predictions and len(worker_predictions) > 1:
        result["mean_kappa"] = mean_pairwise_kappa(worker_predictions, weighted=True)
        disagreements = sum(
            1 for i in range(len(predictions))
            if len({int(wp[i]) for wp in worker_predictions if int(wp[i]) >= 0}) > 1
        )
        result["disagreement_rate"] = disagreements / len(predictions)

    return result

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


def cluster_accuracies(
    predictions: np.ndarray,
    gold: np.ndarray,
    cluster_ids: np.ndarray,
) -> Dict[int, float]:
    correct = predictions == gold
    out: Dict[int, float] = {}
    for cid in np.unique(cluster_ids):
        mask = cluster_ids == cid
        if np.any(mask):
            out[int(cid)] = float(np.mean(correct[mask]))
    return out


def cvar_cluster_accuracy(
    predictions: np.ndarray,
    gold: np.ndarray,
    cluster_ids: np.ndarray,
    quantile: float = 0.33,
) -> float:
    """
    CVaR over style clusters: mean accuracy of worst `quantile` fraction of clusters.
    """
    accs = list(cluster_accuracies(predictions, gold, cluster_ids).values())
    if not accs:
        return 0.0
    accs_sorted = sorted(accs)
    k = max(1, int(np.ceil(len(accs_sorted) * quantile)))
    return float(np.mean(accs_sorted[:k]))


def mean_pairwise_kappa(
    worker_predictions: List[np.ndarray],
    weighted: bool = True,
) -> float:
    if cohen_kappa_score is None or len(worker_predictions) < 2:
        return 0.0
    labels_1_5 = [1, 2, 3, 4, 5]
    kappas: List[float] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(len(worker_predictions)):
            for j in range(i + 1, len(worker_predictions)):
                a, b = worker_predictions[i], worker_predictions[j]
                try:
                    if weighted:
                        k = cohen_kappa_score(a, b, weights="quadratic", labels=labels_1_5)
                    else:
                        k = cohen_kappa_score(a, b, labels=labels_1_5)
                    kappas.append(float(k) if not np.isnan(k) else 0.0)
                except (ValueError, ZeroDivisionError):
                    kappas.append(0.0)
    return float(np.mean(kappas)) if kappas else 0.0


def compute_metrics(
    predictions: np.ndarray,
    gold_labels: np.ndarray,
    user_ids: np.ndarray,
    worker_predictions: Optional[List[np.ndarray]] = None,
    cluster_ids: Optional[np.ndarray] = None,
    cvar_quantile: float = 0.33,
) -> Dict[str, Any]:
    """Full metric bundle for logging and selection."""
    predictions = np.asarray(predictions)
    gold_labels = np.asarray(gold_labels)
    user_ids = np.asarray(user_ids)

    if len(predictions) == 0:
        return {
            "R_global": 0.0,
            "R_worst": 0.0,
            "mae": 0.0,
            "CVaR_cluster": 0.0,
            "num_users": 0,
            "num_examples": 0,
        }

    r_global = accuracy(predictions, gold_labels)
    r_worst = worst_group_accuracy(predictions, gold_labels, user_ids)
    mae_val = mae(predictions, gold_labels)

    result: Dict[str, Any] = {
        "R_global": r_global,
        "R_worst": r_worst,
        "mae": mae_val,
        "accuracy": r_global,
        "num_users": int(len(np.unique(user_ids))),
        "num_examples": int(len(predictions)),
        "accuracy_per_user": per_user_accuracy(predictions, gold_labels, user_ids),
    }

    if cluster_ids is not None:
        cluster_ids = np.asarray(cluster_ids)
        result["CVaR_cluster"] = cvar_cluster_accuracy(
            predictions, gold_labels, cluster_ids, quantile=cvar_quantile
        )
        result["cluster_accuracies"] = cluster_accuracies(predictions, gold_labels, cluster_ids)

    if worker_predictions and len(worker_predictions) > 1:
        result["mean_kappa"] = mean_pairwise_kappa(worker_predictions, weighted=True)
        disagreements = sum(
            1 for i in range(len(predictions))
            if len({int(wp[i]) for wp in worker_predictions}) > 1
        )
        result["disagreement_rate"] = disagreements / len(predictions)

    return result

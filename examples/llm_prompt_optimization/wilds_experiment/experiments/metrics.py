import numpy as np
from typing import Dict, List, Optional, Any

try:
    from sklearn.metrics import cohen_kappa_score
except ImportError:  # pragma: no cover - optional dependency
    cohen_kappa_score = None


def compute_metrics(
    predictions: np.ndarray,
    gold_labels: np.ndarray,
    user_ids: np.ndarray,
    worker_predictions: Optional[List[np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Compute core metrics for WILDS Amazon experiments.

    For ensemble: mean_kappa is quadratic weighted (ordinal 1–5; close errors penalized less).
    mean_kappa_unweighted is also returned for comparison.
    """
    if len(predictions) == 0:
        return {
            "R_global": 0.0,
            "R_worst": 0.0,
            "mae": 0.0,
            "accuracy_per_user": {},
            "num_users": 0,
            "num_examples": 0,
        }

    predictions = np.asarray(predictions)
    gold_labels = np.asarray(gold_labels)
    user_ids = np.asarray(user_ids)

    correct = predictions == gold_labels
    r_global = float(np.mean(correct))
    mae = float(np.mean(np.abs(predictions - gold_labels)))

    accuracy_per_user: Dict[int, float] = {}
    for user_id in np.unique(user_ids):
        mask = user_ids == user_id
        accuracy_per_user[int(user_id)] = float(np.mean(correct[mask]))

    r_worst = (
        float(np.percentile(list(accuracy_per_user.values()), 10))
        if accuracy_per_user
        else 0.0
    )

    result: Dict[str, Any] = {
        "R_global": r_global,
        "R_worst": r_worst,
        "mae": mae,
        "accuracy_per_user": accuracy_per_user,
        "num_users": len(np.unique(user_ids)),
        "num_examples": int(len(predictions)),
    }

    if worker_predictions and len(worker_predictions) > 1:
        kappas_unweighted = []
        kappas_weighted = []
        if cohen_kappa_score is not None:
            for i in range(len(worker_predictions)):
                for j in range(i + 1, len(worker_predictions)):
                    a, b = worker_predictions[i], worker_predictions[j]
                    kappas_unweighted.append(cohen_kappa_score(a, b))
                    # Quadratic weighted kappa: штраф за близкие промахи (1 vs 2) меньше, чем за далёкие (1 vs 5)
                    kappas_weighted.append(cohen_kappa_score(a, b, weights="quadratic"))
        result["mean_kappa_unweighted"] = float(np.mean(kappas_unweighted)) if kappas_unweighted else 0.0
        result["mean_kappa"] = float(np.mean(kappas_weighted)) if kappas_weighted else 0.0

        disagreements = 0
        for idx in range(len(predictions)):
            votes = {int(wp[idx]) for wp in worker_predictions}
            if len(votes) > 1:
                disagreements += 1
        result["disagreement_rate"] = disagreements / len(predictions)

    return result


def compute_combined_score_single(metrics: Dict[str, Any]) -> float:
    """
    Combined score for single-model experiments.
    Formula: R_global * (1 - 0.2 * MAE/4).
    """
    r_global = metrics.get("R_global", 0.0)
    mae = metrics.get("mae", 0.0)
    mae_penalty = mae / 4.0
    return float(r_global * (1 - 0.2 * mae_penalty))


def compute_combined_score_ensemble(metrics: Dict[str, Any]) -> float:
    """
    Combined score for ensemble experiments.
    Formula: 0.4*R_global + 0.4*R_worst - 0.2*(1 - D).
    
    DEPRECATED: Use compute_combined_score_unified() for comparable scores.
    """
    r_global = metrics.get("R_global", 0.0)
    r_worst = metrics.get("R_worst", 0.0)
    disagreement = metrics.get("disagreement_rate", 0.0)
    return float(0.4 * r_global + 0.4 * r_worst - 0.2 * (1 - disagreement))


def compute_combined_score_unified(
    metrics: Dict[str, Any],
    is_ensemble: bool = False,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Унифицированная формула combined score для single model и ensemble.
    
    Позволяет корректно сравнивать результаты single model и ensemble экспериментов.
    
    Базовая формула (одинаковая для всех):
        score = 0.4 * R_global + 0.3 * R_worst + 0.3 * (1 - MAE/4)
    
    Для ensemble добавляется бонус за согласованность (quadratic weighted Kappa, обрезанная до [0,1]):
        score += 0.1 * max(0, mean_kappa)
    mean_kappa в compute_metrics считается как quadratic weighted (для порядковой шкалы 1–5).
    
    Args:
        metrics: Словарь с метриками (R_global, R_worst, mae, mean_kappa для ensemble)
        is_ensemble: True если это ensemble эксперимент
        weights: Опциональные веса компонентов (по умолчанию: global=0.4, worst=0.3, mae=0.3, consistency=0.1)
    
    Returns:
        Combined score в диапазоне [0.0, 1.0]
    
    Example:
        >>> metrics_single = {"R_global": 0.50, "R_worst": 0.30, "mae": 0.72}
        >>> score = compute_combined_score_unified(metrics_single, is_ensemble=False)
        >>> # score ≈ 0.539
        
        >>> metrics_ensemble = {"R_global": 0.65, "R_worst": 0.44, "mae": 0.36, "mean_kappa": 0.50}
        >>> score = compute_combined_score_unified(metrics_ensemble, is_ensemble=True)
        >>> # score ≈ 0.717 (base + 0.1 * 0.50)
    """
    if weights is None:
        weights = {
            "global": 0.4,
            "worst": 0.3,
            "mae": 0.3,
            "consistency": 0.1 if is_ensemble else 0.0,
        }
    
    r_global = metrics.get("R_global", 0.0)
    r_worst = metrics.get("R_worst", 0.0)
    mae = metrics.get("mae", 0.0)
    
    # Нормализуем MAE к диапазону 0-1 (инвертируем: меньше ошибок = лучше)
    mae_normalized = 1.0 - (mae / 4.0)
    
    # Базовая формула (одинаковая для single model и ensemble)
    base_score = (
        weights["global"] * r_global +
        weights["worst"] * r_worst +
        weights["mae"] * mae_normalized
    )
    
    # Бонус за согласованность ансамбля (только для ensemble): Cohen's Kappa
    # Kappa в [-1, 1]; обрезаем до [0, 1], чтобы отрицательная каппа не снижала скор
    if is_ensemble and weights["consistency"] > 0:
        kappa_raw = metrics.get("mean_kappa", 0.0)
        kappa_score = max(0.0, float(kappa_raw))
        consistency_bonus = weights["consistency"] * kappa_score
        score = base_score + consistency_bonus
    else:
        score = base_score
    
    # Ограничиваем диапазон [0.0, 1.0]
    return float(max(0.0, min(1.0, score)))


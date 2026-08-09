"""Proxy validation: CVaR_cluster vs official R_worst correlation (SPEC §4.1 / E0)."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from prime.fitness.metrics import (
    cvar_from_accuracies,
    smoothed_cluster_accuracies,
    worst_group_accuracy,
)


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks with proper tie handling (fractional ranks)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j)  # average of positions i..j
        i = j + 1
    return ranks


def spearman_correlation(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Rank correlation (SPEC E0 primary) with average ranks for ties."""
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def kruskal_wallis_h(groups: Sequence[Sequence[float]]) -> Optional[float]:
    """
    Kruskal–Wallis H with tie correction.
    `groups` = list of per-cluster user-accuracy arrays.
    """
    clean = [np.asarray(g, dtype=np.float64) for g in groups if len(g) > 0]
    if len(clean) < 2:
        return None
    all_vals = np.concatenate(clean)
    n = len(all_vals)
    if n < 3:
        return None
    ranks = _average_ranks(all_vals) + 1.0  # 1-based average ranks
    h = 0.0
    pos = 0
    for g in clean:
        m = len(g)
        r_mean = float(np.mean(ranks[pos : pos + m]))
        h += m * (r_mean - (n + 1) / 2.0) ** 2
        pos += m
    h *= 12.0 / (n * (n + 1))
    _, counts = np.unique(all_vals, return_counts=True)
    tie = 1.0 - float(np.sum(counts**3 - counts)) / (n**3 - n) if n > 1 else 1.0
    if tie <= 0:
        return float(h)
    return float(h / tie)


def permutation_kruskal_wallis(
    user_acc: Dict[int, float],
    user_cluster: Dict[int, int],
    *,
    n_perm: int = 2000,
    seed: int = 0,
) -> Dict[str, object]:
    """
    Unbiased E0 gate (OBSERVATIONS M2/P1): do clusters separate user accuracies?

    Permutes cluster labels across users; reports H_obs and one-sided p-value
    P(H_null >= H_obs). Unlike LOO Spearman, the null is centered correctly.
    """
    users = sorted(user_acc.keys())
    if len(users) < 4:
        return {"kruskal_h": None, "p_value": None, "n_perm": n_perm, "n_users": len(users)}
    clusters = np.array([user_cluster.get(u, -1) for u in users], dtype=np.int64)
    accs = np.array([user_acc[u] for u in users], dtype=np.float64)
    # drop users without cluster
    ok = clusters >= 0
    clusters, accs = clusters[ok], accs[ok]
    uniq = sorted(set(clusters.tolist()))
    if len(uniq) < 2:
        return {"kruskal_h": None, "p_value": None, "n_perm": n_perm, "n_users": int(len(accs))}

    def _groups(lab: np.ndarray) -> List[np.ndarray]:
        return [accs[lab == c] for c in uniq if np.any(lab == c)]

    h_obs = kruskal_wallis_h(_groups(clusters))
    if h_obs is None:
        return {"kruskal_h": None, "p_value": None, "n_perm": n_perm, "n_users": int(len(accs))}

    rng = np.random.RandomState(seed)
    null = np.empty(n_perm, dtype=np.float64)
    for t in range(n_perm):
        perm = rng.permutation(clusters)
        h = kruskal_wallis_h(_groups(perm))
        null[t] = h if h is not None else 0.0
    p_value = float(np.mean(null >= h_obs))
    return {
        "kruskal_h": float(h_obs),
        "p_value": p_value,
        "n_perm": n_perm,
        "n_users": int(len(accs)),
        "n_clusters": len(uniq),
        "null_h_mean": float(null.mean()),
        "null_h_std": float(null.std()),
    }


def proxy_validation_report(
    history: Sequence[Tuple[float, float]],
    min_cycles: int = 2,
) -> dict:
    """
    history: list of (CVaR_cluster, R_worst) per AL cycle on val.
    Returns diagnostic dict for logging.
    """
    n = len(history)
    report = {"n_cycles": n, "min_cycles": min_cycles, "correlation": None, "ready": n >= min_cycles}
    if n >= min_cycles:
        cvars = [h[0] for h in history]
        rworsts = [h[1] for h in history]
        report["correlation"] = pearson_correlation(cvars, rworsts)
        report["spearman"] = spearman_correlation(cvars, rworsts)
        report["cvar_series"] = cvars
        report["r_worst_series"] = rworsts
    return report


def _user_to_cluster_mode(user_ids: np.ndarray, cluster_ids: np.ndarray) -> Dict[int, int]:
    user_cluster: Dict[int, int] = {}
    for u in np.unique(user_ids):
        mask = user_ids == u
        cids = cluster_ids[mask]
        vals, counts = np.unique(cids, return_counts=True)
        user_cluster[int(u)] = int(vals[np.argmax(counts)])
    return user_cluster


def leave_one_user_out_cluster_accuracy(
    predictions: np.ndarray,
    gold: np.ndarray,
    user_ids: np.ndarray,
    cluster_ids: np.ndarray,
    user_id: int,
    cluster_id: int,
    *,
    beta_a: float = 1.0,
    beta_b: float = 1.0,
) -> Optional[float]:
    """
    Beta-smoothed accuracy of `cluster_id` excluding all examples from `user_id`.

    Used by E0 Spearman so a user's own reviews do not inflate the cluster
    accuracy they are correlated against (leakage on small clusters).
    Returns None when the cluster has no remaining examples after exclusion.
    """
    mask = (cluster_ids == int(cluster_id)) & (user_ids != int(user_id))
    n_k = int(np.sum(mask))
    if n_k == 0:
        return None
    c_k = float(np.sum((predictions == gold)[mask]))
    return (c_k + beta_a) / (n_k + beta_a + beta_b)


def corr_cvar_vs_rworst_from_predictions(
    predictions: np.ndarray,
    gold: np.ndarray,
    user_ids: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    cvar_quantile: float = 0.33,
    beta_a: float = 1.0,
    beta_b: float = 1.0,
    n_perm: int = 1000,
    perm_seed: int = 0,
) -> Dict[str, object]:
    """
    Single-pass proxy check used by E0: given fixed predictions, compute
    R_worst (user-level) and CVaR_cluster (group-level) and their relationship
    across users via per-user contribution to both metrics.

    Primary Spearman is **leave-one-user-out**: user accuracy vs accuracy of
    that user's cluster computed without the user's own reviews. The leaky
    (in-sample) Spearman is retained as a diagnostic only.
    """
    predictions = np.asarray(predictions)
    gold = np.asarray(gold)
    user_ids = np.asarray(user_ids)
    cluster_ids = np.asarray(cluster_ids)

    r_worst = worst_group_accuracy(predictions, gold, user_ids)
    shrunk = smoothed_cluster_accuracies(
        predictions, gold, cluster_ids, beta_a=beta_a, beta_b=beta_b
    )
    cvar = cvar_from_accuracies(shrunk, quantile=cvar_quantile)
    r_global = float(np.mean(predictions == gold)) if len(predictions) else 0.0

    from prime.fitness.metrics import per_user_accuracy

    user_acc = per_user_accuracy(predictions, gold, user_ids)
    user_cluster = _user_to_cluster_mode(user_ids, cluster_ids)

    ua_leaky: List[float] = []
    ca_leaky: List[float] = []
    ua_loo: List[float] = []
    ca_loo: List[float] = []
    n_loo_skipped = 0
    for u, acc in user_acc.items():
        cid = user_cluster.get(int(u))
        if cid is None or cid not in shrunk:
            continue
        ua_leaky.append(float(acc))
        ca_leaky.append(float(shrunk[cid]))
        loo = leave_one_user_out_cluster_accuracy(
            predictions,
            gold,
            user_ids,
            cluster_ids,
            int(u),
            int(cid),
            beta_a=beta_a,
            beta_b=beta_b,
        )
        if loo is None:
            n_loo_skipped += 1
            continue
        ua_loo.append(float(acc))
        ca_loo.append(float(loo))

    return {
        "R_global": r_global,
        "R_worst": float(r_worst),
        "CVaR_cluster": float(cvar),
        "cluster_accuracies_shrunk": {str(k): v for k, v in shrunk.items()},
        "n_clusters_present": len(shrunk),
        # Diagnostic correlations (LOO Spearman is biased under null — OBSERVATIONS M2).
        "spearman_user_acc_vs_cluster_acc": spearman_correlation(ua_loo, ca_loo),
        "pearson_user_acc_vs_cluster_acc": pearson_correlation(ua_loo, ca_loo),
        "n_users_in_corr": len(ua_loo),
        "n_loo_skipped_singleton": n_loo_skipped,
        "spearman_leaky": spearman_correlation(ua_leaky, ca_leaky),
        "pearson_leaky": pearson_correlation(ua_leaky, ca_leaky),
        "n_users_in_corr_leaky": len(ua_leaky),
        # Primary E0 gate (P1): permutation Kruskal–Wallis on user accuracies.
        **{
            f"kw_{k}": v
            for k, v in permutation_kruskal_wallis(
                user_acc, user_cluster, n_perm=n_perm, seed=perm_seed
            ).items()
        },
    }

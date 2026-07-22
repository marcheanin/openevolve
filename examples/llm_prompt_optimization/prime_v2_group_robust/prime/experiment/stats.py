"""Bootstrap CI and paired statistical tests."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Return (mean, lower, upper) bootstrap CI."""
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype=float)
    rng = np.random.RandomState(seed)
    means = []
    n = len(arr)
    for _ in range(n_bootstrap):
        sample = arr[rng.randint(0, n, size=n)]
        means.append(float(np.mean(sample)))
    means.sort()
    lo = means[int((alpha / 2) * n_bootstrap)]
    hi = means[int((1 - alpha / 2) * n_bootstrap) - 1]
    return float(np.mean(arr)), lo, hi


def bootstrap_user_metric_ci(
    per_user_correct: Dict[int, List[bool]],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Bootstrap CI over users (resample users, not examples)."""
    user_ids = sorted(per_user_correct.keys())
    user_accs = [float(np.mean(per_user_correct[u])) for u in user_ids]
    return bootstrap_ci(user_accs, n_bootstrap=n_bootstrap, alpha=alpha, seed=seed)


def paired_bootstrap_test(
    a: List[float],
    b: List[float],
    n_bootstrap: int = 5000,
    seed: int = 0,
) -> Dict[str, float]:
    """Paired bootstrap: P(mean(a) > mean(b))."""
    if len(a) != len(b) or not a:
        return {"p_a_better": 0.5, "mean_diff": 0.0}
    rng = np.random.RandomState(seed)
    diffs = []
    n = len(a)
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        diffs.append(float(np.mean(np.array(a)[idx]) - np.mean(np.array(b)[idx])))
    mean_diff = float(np.mean(np.array(a) - np.array(b)))
    p = float(np.mean(np.array(diffs) > 0))
    return {"p_a_better": p, "mean_diff": mean_diff}

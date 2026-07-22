"""Proxy validation: CVaR_cluster vs official R_worst correlation on val."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


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
        report["cvar_series"] = cvars
        report["r_worst_series"] = rworsts
    return report

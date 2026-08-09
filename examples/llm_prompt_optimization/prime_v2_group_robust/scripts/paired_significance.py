"""Paired significance + power analysis for two prompts on the same test slice.

Both eval dirs must come from ``eval_prompt_on_test.py`` / ``_eval_split`` so that
predictions line up example-for-example. Because the two prompts are scored on
identical users, a paired bootstrap removes between-set variance and is the only
honest way to compare runs at our sample sizes.

Usage:
  python scripts/paired_significance.py \
      --a results/<run>/evals/initial_prompt \
      --b results/<run>/evals/final_selected
"""

from __future__ import annotations

import argparse
from math import comb
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

METRICS = ("R_global", "CVaR_cluster", "R_worst", "R_tail")


def _load(d: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.load(d / "ensemble_predictions.npy"),
        np.load(d / "labels.npy"),
        np.load(d / "user_ids.npy", allow_pickle=True),
        np.load(d / "cluster_ids.npy"),
    )


def _user_level(
    correct: np.ndarray, user_ids: Sequence, cluster_ids: Sequence
) -> Tuple[np.ndarray, np.ndarray, List]:
    """Collapse example-level correctness to per-user accuracy (WILDS convention)."""
    acc: Dict = {}
    clu: Dict = {}
    for u, cl, ok in zip(user_ids, cluster_ids, correct.tolist()):
        acc.setdefault(u, []).append(ok)
        clu[u] = int(cl)
    users = sorted(acc)
    return (
        np.array([float(np.mean(acc[u])) for u in users]),
        np.array([clu[u] for u in users]),
        users,
    )


def _metrics(user_acc: np.ndarray, clusters: np.ndarray, tail_q: float) -> Tuple[float, ...]:
    keys = sorted(set(clusters.tolist()))
    cluster_acc = np.array([user_acc[clusters == k].mean() for k in keys])
    k_cl = max(1, int(np.ceil(tail_q * len(cluster_acc))))
    k_us = max(1, int(np.ceil(tail_q * len(user_acc))))
    return (
        float(user_acc.mean()),
        float(np.sort(cluster_acc)[:k_cl].mean()),
        float(cluster_acc.min()),
        float(np.sort(user_acc)[:k_us].mean()),
    )


def _mcnemar_exact(only_a: int, only_b: int) -> float:
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(comb(n, i) for i in range(k + 1))
    return min(1.0, 2 * tail / 2**n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="baseline eval dir")
    ap.add_argument("--b", required=True, help="candidate eval dir")
    ap.add_argument("--tail-quantile", type=float, default=0.2)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pa, ya, ua, ca_id = _load(Path(args.a))
    pb, yb, ub, _ = _load(Path(args.b))
    if not (np.array_equal(ya, yb) and np.array_equal(ua, ub)):
        raise SystemExit("eval dirs are not aligned: labels/user_ids differ")

    ok_a, ok_b = (pa == ya).astype(int), (pb == yb).astype(int)
    print(f"examples={len(ya)} users={len(set(ua.tolist()))}")
    print(
        f"example-level acc: A={ok_a.mean():.4f} B={ok_b.mean():.4f} "
        f"delta={ok_b.mean() - ok_a.mean():+.4f}"
    )

    only_a = int(((ok_a == 1) & (ok_b == 0)).sum())
    only_b = int(((ok_a == 0) & (ok_b == 1)).sum())
    print(
        f"McNemar: A-only-correct={only_a} B-only-correct={only_b} "
        f"p_two_sided={_mcnemar_exact(only_a, only_b):.4f}"
    )

    acc_a, clusters, _ = _user_level(ok_a, ua.tolist(), ca_id.tolist())
    acc_b, _, _ = _user_level(ok_b, ua.tolist(), ca_id.tolist())
    point_a = _metrics(acc_a, clusters, args.tail_quantile)
    point_b = _metrics(acc_b, clusters, args.tail_quantile)

    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, len(acc_a), size=(args.boot, len(acc_a)))
    deltas = np.empty((args.boot, len(METRICS)))
    vals_a = np.empty((args.boot, len(METRICS)))
    for i, sel in enumerate(draws):
        va = _metrics(acc_a[sel], clusters[sel], args.tail_quantile)
        vb = _metrics(acc_b[sel], clusters[sel], args.tail_quantile)
        vals_a[i] = va
        deltas[i] = np.array(vb) - np.array(va)

    print(f"\npaired user-level bootstrap ({args.boot} resamples):")
    for i, name in enumerate(METRICS):
        d = deltas[:, i]
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        print(
            f"  {name:13s} A={point_a[i]:.4f} B={point_b[i]:.4f} "
            f"delta={point_b[i] - point_a[i]:+.4f} CI95=[{lo:+.4f},{hi:+.4f}] p={p:.3f}"
        )

    print("\nnoise floor at this sample size (bootstrap SD of the metric on A):")
    for i, name in enumerate(METRICS):
        sd = float(vals_a[:, i].std())
        print(f"  {name:13s} SD={sd:.4f} -> min detectable effect (80% power) ~ {2.8 * sd:.3f}")

    print("\nper-cluster (users / accuracy):")
    for k in sorted(set(clusters.tolist())):
        m = clusters == k
        print(
            f"  c{k}: users={int(m.sum()):3d} A={acc_a[m].mean():.3f} "
            f"B={acc_b[m].mean():.3f} delta={acc_b[m].mean() - acc_a[m].mean():+.3f}"
        )


if __name__ == "__main__":
    main()

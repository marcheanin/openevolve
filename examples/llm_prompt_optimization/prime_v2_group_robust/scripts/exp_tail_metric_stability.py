#!/usr/bin/env python3
"""E-C: which tail-metric definition can actually discriminate at our sample size?

Zero API cost — works entirely from saved predictions of two prompts scored on the
same slice. For each candidate definition of "robustness" it reports:

  point values on both prompts, the paired delta, the bootstrap SD (noise floor),
  and the discriminability ratio |delta| / SD.

A definition is only usable if its noise floor is small relative to the effects the
optimiser actually produces. Today `CVaR_cluster` at q=0.2 with K=5 collapses to
`R_worst_cluster` (see OBSERVATIONS M11), so this script exists to pick a
replacement on evidence rather than taste.

Usage:
  python scripts/exp_tail_metric_stability.py \
      --a results/<run>/evals/initial_prompt \
      --b results/<run>/evals/final_selected
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np


def _load(d: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.load(d / "ensemble_predictions.npy"),
        np.load(d / "labels.npy"),
        np.load(d / "user_ids.npy", allow_pickle=True),
        np.load(d / "cluster_ids.npy"),
    )


def _to_user_level(
    correct: np.ndarray, uid: np.ndarray, cid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    acc: Dict = defaultdict(list)
    clu: Dict = {}
    for u, c, ok in zip(uid.tolist(), cid.tolist(), correct.tolist()):
        acc[u].append(ok)
        clu[u] = int(c)
    users = sorted(acc)
    return (
        np.array([float(np.mean(acc[u])) for u in users]),
        np.array([clu[u] for u in users]),
    )


def _cluster_acc(ua: np.ndarray, cl: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    keys = sorted(set(cl.tolist()))
    accs = np.array([ua[cl == k].mean() for k in keys])
    sizes = np.array([int((cl == k).sum()) for k in keys], dtype=float)
    return accs, sizes


def _cvar(ua: np.ndarray, cl: np.ndarray, q: float) -> float:
    accs, _ = _cluster_acc(ua, cl)
    k = max(1, int(np.ceil(q * len(accs))))
    return float(np.sort(accs)[:k].mean())


def _shrunk_min(ua: np.ndarray, cl: np.ndarray, prior_weight: float) -> float:
    """Worst cluster after shrinking each cluster toward the global mean.

    Small clusters are pulled hardest, which is exactly what stops the argmin from
    hopping between clusters on noise (OBSERVATIONS C8).
    """
    accs, sizes = _cluster_acc(ua, cl)
    grand = float(ua.mean())
    shrunk = (accs * sizes + grand * prior_weight) / (sizes + prior_weight)
    return float(shrunk.min())


def _shrunk_cvar(ua: np.ndarray, cl: np.ndarray, q: float, prior_weight: float) -> float:
    accs, sizes = _cluster_acc(ua, cl)
    grand = float(ua.mean())
    shrunk = (accs * sizes + grand * prior_weight) / (sizes + prior_weight)
    k = max(1, int(np.ceil(q * len(shrunk))))
    return float(np.sort(shrunk)[:k].mean())


def _user_tail(ua: np.ndarray, q: float) -> float:
    k = max(1, int(np.ceil(q * len(ua))))
    return float(np.sort(ua)[:k].mean())


def _softmin(ua: np.ndarray, cl: np.ndarray, beta: float) -> float:
    """Smooth minimum over clusters: -1/beta * log(mean(exp(-beta*acc)))."""
    accs, _ = _cluster_acc(ua, cl)
    return float(-np.log(np.mean(np.exp(-beta * accs))) / beta)


def _balanced_by_class(pred: np.ndarray, y: np.ndarray) -> float:
    """Macro-averaged per-class accuracy — the class-imbalance-aware view (C11)."""
    return float(np.mean([(pred[y == g] == g).mean() for g in sorted(set(y.tolist()))]))


def build_definitions() -> Dict[str, Callable[[np.ndarray, np.ndarray], float]]:
    return {
        "R_global": lambda ua, cl: float(ua.mean()),
        "R_worst_cluster": lambda ua, cl: float(_cluster_acc(ua, cl)[0].min()),
        "CVaR_cluster q=0.2": lambda ua, cl: _cvar(ua, cl, 0.2),
        "CVaR_cluster q=0.4": lambda ua, cl: _cvar(ua, cl, 0.4),
        "CVaR_cluster q=0.6": lambda ua, cl: _cvar(ua, cl, 0.6),
        "shrunk_min (w=10)": lambda ua, cl: _shrunk_min(ua, cl, 10.0),
        "shrunk_min (w=25)": lambda ua, cl: _shrunk_min(ua, cl, 25.0),
        "shrunk_CVaR q=0.4 (w=10)": lambda ua, cl: _shrunk_cvar(ua, cl, 0.4, 10.0),
        "softmin_cluster (b=10)": lambda ua, cl: _softmin(ua, cl, 10.0),
        "softmin_cluster (b=25)": lambda ua, cl: _softmin(ua, cl, 25.0),
        "R_tail users q=0.2": lambda ua, cl: _user_tail(ua, 0.2),
        "R_tail users q=0.4": lambda ua, cl: _user_tail(ua, 0.4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="optional JSON output path")
    args = ap.parse_args()

    pa, y, uid, cid = _load(Path(args.a))
    pb, yb, ub, _ = _load(Path(args.b))
    if not (np.array_equal(y, yb) and np.array_equal(uid, ub)):
        raise SystemExit("eval dirs are not aligned")

    ua, cl = _to_user_level((pa == y).astype(float), uid, cid)
    ub_acc, _ = _to_user_level((pb == y).astype(float), uid, cid)
    defs = build_definitions()

    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, len(ua), size=(args.boot, len(ua)))

    print(f"users={len(ua)} clusters={len(set(cl.tolist()))} bootstrap={args.boot}\n")
    print(f"{'definition':26s} {'A':>7s} {'B':>7s} {'delta':>8s} {'SD':>7s} {'|d|/SD':>7s}")
    rows: List[Dict[str, object]] = []
    for name, fn in defs.items():
        va, vb = fn(ua, cl), fn(ub_acc, cl)
        boot_a = np.empty(args.boot)
        for i, sel in enumerate(draws):
            boot_a[i] = fn(ua[sel], cl[sel])
        sd = float(boot_a.std())
        ratio = abs(vb - va) / sd if sd > 0 else float("nan")
        print(f"{name:26s} {va:7.4f} {vb:7.4f} {vb - va:+8.4f} {sd:7.4f} {ratio:7.2f}")
        rows.append(
            {"definition": name, "a": va, "b": vb, "delta": vb - va, "boot_sd": sd, "ratio": ratio}
        )

    print("\nclass-balanced view (example level, not user level):")
    print(
        f"  macro per-class accuracy: A={_balanced_by_class(pa, y):.4f} "
        f"B={_balanced_by_class(pb, y):.4f} delta={_balanced_by_class(pb, y) - _balanced_by_class(pa, y):+.4f}"
    )

    accs, sizes = _cluster_acc(ua, cl)
    print(f"\ncluster sizes (users): {[int(s) for s in sizes]}")
    print(f"cluster accuracies A:  {[round(float(a), 3) for a in accs]}")
    print(
        "\nnote: at K=5 any q<=0.2 selects exactly 1 cluster, so R_worst_cluster and "
        "CVaR q=0.2 are the same statistic (M11)."
    )
    print(
        "Lower SD is better; a definition needs |d|/SD >~ 2 to call a real effect at "
        "this sample size."
    )

    if args.out:
        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""E-D: pick the objective by measured signal-to-noise, offline and for free.

Consumes the repeated evaluations saved by ``exp_fitness_noise.py``
(``<run>/exp_A_predictions/``): several independent scorings of the **same prompt**
on the same D_select. Any spread between them is pure measurement noise, so for each
candidate objective we get its noise SD directly — no modelling, no bootstrap.

That SD is the number that matters. In the pair run OpenEvolve claimed +0.055 in
cycle 1 and +0.002 / +0.003 afterwards; an objective whose noise SD is 0.03 cannot
tell those apart, and the search is then ranking noise (OBSERVATIONS M15).

Signal is estimated separately from two genuinely different prompts on the test
split (``--signal-a`` / ``--signal-b``, e.g. initial vs evolved), so the report ends
with noise, signal, and their ratio for every definition.

Usage:
  python scripts/exp_objective_noise_sweep.py --run-dir results/<run> \
      --signal-a results/<run>/evals/initial_prompt \
      --signal-b results/<run>/evals/final_selected
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))

from prime.fitness.metrics import (  # noqa: E402
    class_balance_weights,
    cvar_from_accuracies,
    macro_class_accuracy,
    smoothed_cluster_accuracies,
    weighted_accuracy,
)

Objective = Callable[[np.ndarray, np.ndarray, np.ndarray], float]


def _make_cvar(quantile: float, prior: float, balanced: bool, eps: float = 0.01) -> Objective:
    def fn(pred: np.ndarray, gold: np.ndarray, cid: np.ndarray) -> float:
        weights = class_balance_weights(gold) if balanced else None
        accs = smoothed_cluster_accuracies(
            pred, gold, cid, prior_weight=prior, weights=weights
        )
        tie = macro_class_accuracy(pred, gold) if balanced else weighted_accuracy(pred, gold)
        return cvar_from_accuracies(accs, quantile=quantile) + eps * tie

    return fn


def build_objectives() -> Dict[str, Objective]:
    out: Dict[str, Objective] = {
        "R_global": lambda p, g, c: weighted_accuracy(p, g),
        "R_macro": lambda p, g, c: macro_class_accuracy(p, g),
    }
    for prior, balanced in itertools.product((0.0, 25.0, 50.0, 100.0), (False, True)):
        tag = "bal" if balanced else "raw"
        name = f"cvar_lex q=0.40 w={prior:g} {tag}"
        out[name] = _make_cvar(0.40, prior, balanced)
    out["cvar_lex q=0.20 w=0 raw (legacy argmin)"] = _make_cvar(0.20, 0.0, False)
    out["cvar_lex q=0.60 w=50 bal"] = _make_cvar(0.60, 50.0, True)
    return out


def _load_repeats(run_dir: Path) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    d = run_dir / "exp_A_predictions"
    if not d.is_dir():
        raise SystemExit(
            f"{d} not found — run scripts/exp_fitness_noise.py first (it persists the repeats)"
        )
    reps = sorted(d.glob("ensemble_repeat*.npy"), key=lambda p: p.name)
    if len(reps) < 2:
        raise SystemExit(f"need >= 2 repeats in {d}, found {len(reps)}")
    return (
        [np.load(p) for p in reps],
        np.load(d / "labels.npy"),
        np.load(d / "cluster_ids.npy"),
    )


def _load_eval(d: Optional[Path]):
    if d is None:
        return None
    return (
        np.load(d / "ensemble_predictions.npy"),
        np.load(d / "labels.npy"),
        np.load(d / "cluster_ids.npy"),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--signal-a", type=Path, default=None)
    ap.add_argument("--signal-b", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    repeats, labels, clusters = _load_repeats(args.run_dir)
    sig_a = _load_eval(args.signal_a)
    sig_b = _load_eval(args.signal_b)

    print(
        f"noise from {len(repeats)} identical re-evaluations of one prompt on "
        f"{len(labels)} D_select examples, {len(set(clusters.tolist()))} clusters"
    )
    if sig_a and sig_b:
        print(
            f"signal from initial vs evolved on {len(sig_a[1])} test examples "
            f"(a genuinely different prompt)"
        )
    print()
    header = f"{'objective':36s} {'mean':>7s} {'noise SD':>9s}"
    if sig_a and sig_b:
        header += f" {'signal':>8s} {'sig/noise':>10s}"
    print(header)

    rows: List[Dict[str, object]] = []
    for name, fn in build_objectives().items():
        vals = np.array([fn(r, labels, clusters) for r in repeats])
        sd = float(vals.std(ddof=1))
        line = f"{name:36s} {vals.mean():7.4f} {sd:9.4f}"
        row: Dict[str, object] = {"objective": name, "mean": float(vals.mean()), "noise_sd": sd}
        if sig_a and sig_b:
            signal = abs(fn(*sig_b) - fn(*sig_a))
            ratio = signal / sd if sd > 0 else float("inf")
            line += f" {signal:8.4f} {ratio:10.2f}"
            row["signal"] = signal
            row["signal_over_noise"] = ratio
        print(line)
        rows.append(row)

    flips = [int((repeats[0] != r).sum()) for r in repeats[1:]]
    print(
        f"\nunderlying instability: {flips} of {len(labels)} predictions flip between "
        f"repeats ({[f'{f / len(labels):.1%}' for f in flips]}) at worker temperature 0"
    )
    print(
        "Read the noise SD against the per-cycle gains OpenEvolve reports. An "
        "objective is usable only if its noise is well below the improvements it is "
        "supposed to rank."
    )
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

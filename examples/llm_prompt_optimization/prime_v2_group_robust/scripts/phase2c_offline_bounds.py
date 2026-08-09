#!/usr/bin/env python3
"""E4 offline bounds on CivilComments cached predictions (~0 API).

Portfolio LOO / oracle-over-workers headroom on Phase 2b diagnostic preds
(and optionally later live-run pred_cache). Pattern: phase0_offline_bounds.py.

Usage:
  python scripts/phase2c_offline_bounds.py
  python scripts/phase2c_offline_bounds.py --pred-dir experiments/E4_civilcomments/phase2b_diagnostics/predictions
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))

DEFAULT_PRED = (
    PKG_ROOT / "experiments" / "E4_civilcomments" / "phase2b_diagnostics" / "predictions"
)
OUT_DIR = PKG_ROOT / "experiments" / "E4_civilcomments" / "phase2c_offline_bounds"


def _majority(votes: np.ndarray) -> np.ndarray:
    # votes shape (n_workers, n)
    out = []
    for i in range(votes.shape[1]):
        col = votes[:, i]
        # binary majority; ties → 0
        out.append(1 if int((col == 1).sum()) > int((col == 0).sum()) else 0)
    return np.asarray(out, dtype=np.int16)


def portfolio_loo(
    wp: np.ndarray, y: np.ndarray, g: np.ndarray
) -> Dict[str, Any]:
    """Per-group specialist LOO: pick best worker on other groups, score held-out group."""
    n_w = wp.shape[0]
    groups = sorted(set(g.tolist()))
    # Global best single
    per_w = [float((wp[w] == y).mean()) for w in range(n_w)]
    best_single = float(max(per_w))
    ens = _majority(wp)
    ens_acc = float((ens == y).mean())
    hit_any = np.any(wp == y[None, :], axis=0)
    oracle_acc = float(hit_any.mean())

    # Per-group: assign specialist = argmax worker acc on all other groups
    loo_correct = np.zeros(len(y), dtype=float)
    for gid in groups:
        m = g == gid
        m_other = ~m
        if not m.any() or not m_other.any():
            loo_correct[m] = (ens[m] == y[m]).astype(float)
            continue
        scores = [float((wp[w, m_other] == y[m_other]).mean()) for w in range(n_w)]
        w_star = int(np.argmax(scores))
        loo_correct[m] = (wp[w_star, m] == y[m]).astype(float)
    loo_acc = float(loo_correct.mean())

    # Per-group accuracies under ensemble
    per_g = {}
    for gid in groups:
        m = g == gid
        per_g[int(gid)] = {
            "n": int(m.sum()),
            "ens_acc": float((ens[m] == y[m]).mean()),
            "oracle_acc": float(hit_any[m].mean()),
            "best_single_acc": float(max((wp[w, m] == y[m]).mean() for w in range(n_w))),
        }

    return {
        "n": int(len(y)),
        "per_worker_acc": per_w,
        "best_single": best_single,
        "ensemble": ens_acc,
        "oracle_over_workers": oracle_acc,
        "agg_only_headroom_pp": (oracle_acc - ens_acc) * 100,
        "portfolio_loo": loo_acc,
        "portfolio_loo_vs_best_single_pp": (loo_acc - best_single) * 100,
        "portfolio_loo_vs_ensemble_pp": (loo_acc - ens_acc) * 100,
        "per_group": per_g,
        "go_portfolio": (loo_acc - best_single) >= 0.05,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", type=Path, default=DEFAULT_PRED)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    pred = args.pred_dir
    y = np.load(pred / "labels.npy")
    g = np.load(pred / "group_ids.npy")
    # Prefer repeat0 workers
    wp_path = pred / "workers_repeat0.npy"
    if not wp_path.exists():
        print(f"missing {wp_path}", file=sys.stderr)
        return 1
    wp = np.load(wp_path)
    report = portfolio_loo(wp, y, g)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    decision = "KEEP investigating portfolio/routing" if report["go_portfolio"] else (
        "DEFER portfolio/routing (LOO < +5pp vs best single) — matches Amazon E0a posture"
    )
    lines = [
        "# Phase 2c offline bounds — CivilComments",
        "",
        f"Source: `{pred}`",
        "",
        f"**Portfolio LOO decision:** {decision}",
        "",
        f"- Ensemble acc: {report['ensemble']:.3f}",
        f"- Best single: {report['best_single']:.3f}",
        f"- Oracle-over-workers: {report['oracle_over_workers']:.3f} "
        f"(agg-only +{report['agg_only_headroom_pp']:.1f} pp)",
        f"- Portfolio LOO: {report['portfolio_loo']:.3f} "
        f"({report['portfolio_loo_vs_best_single_pp']:+.1f} pp vs best single)",
        "",
        "Roadmap gate: invest in routing/PCO only if LOO > +5 pp. "
        "Otherwise keep C12 deferred.",
        "",
    ]
    (args.out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {args.out / 'RESULTS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

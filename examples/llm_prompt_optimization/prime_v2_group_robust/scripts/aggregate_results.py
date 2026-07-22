#!/usr/bin/env python3
"""Aggregate experiment results into mean±std tables with bootstrap CI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1]
sys_path_inserted = False


def _collect_summaries(results_root: Path, experiment: str) -> List[Dict[str, Any]]:
    base = results_root / experiment
    if not base.is_dir():
        return []
    out: List[Dict[str, Any]] = []
    for run_dir in sorted(base.iterdir()):
        summary_path = run_dir / "summary.json"
        meta_path = run_dir / "run_metadata.json"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        ft = summary.get("final_test", {})
        out.append({
            "run": run_dir.name,
            "seed": meta.get("seed"),
            "R_global": ft.get("R_global"),
            "R_worst": ft.get("R_worst"),
            "CVaR_cluster": ft.get("CVaR_cluster"),
            "mae": ft.get("mae"),
        })
    return out


def _mean_std(values: List[float]) -> str:
    if not values:
        return "n/a"
    arr = np.asarray(values, dtype=float)
    return f"{arr.mean():.4f}±{arr.std():.4f}"


def main() -> int:
    import sys

    global sys_path_inserted
    if not sys_path_inserted:
        sys.path.insert(0, str(PKG_ROOT))
        sys_path_inserted = True

    from prime.experiment.stats import bootstrap_ci

    parser = argparse.ArgumentParser(description="Aggregate PRIME v2 experiment results")
    parser.add_argument("--results-dir", type=Path, default=PKG_ROOT / "results")
    parser.add_argument("--experiments", nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    lines = ["| experiment | n | R_global | R_worst | CVaR_cluster | mae |",
             "|---|---:|---:|---:|---:|---:|"]
    report: Dict[str, Any] = {}

    for exp in args.experiments:
        rows = _collect_summaries(args.results_dir, exp)
        report[exp] = rows
        for metric in ("R_global", "R_worst", "CVaR_cluster", "mae"):
            vals = [float(r[metric]) for r in rows if r.get(metric) is not None]
            if vals:
                mean, lo, hi = bootstrap_ci(vals, seed=42)
                report[exp + f"_{metric}_ci"] = {"mean": mean, "lo": lo, "hi": hi}
        lines.append(
            f"| {exp} | {len(rows)} | "
            f"{_mean_std([r['R_global'] for r in rows if r.get('R_global') is not None])} | "
            f"{_mean_std([r['R_worst'] for r in rows if r.get('R_worst') is not None])} | "
            f"{_mean_std([r['CVaR_cluster'] for r in rows if r.get('CVaR_cluster') is not None])} | "
            f"{_mean_std([r['mae'] for r in rows if r.get('mae') is not None])} |"
        )

    table = "\n".join(lines)
    print(table)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n\n" + table, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Overlay test-metric curves (report style) for AL vs Plain ablation.

Builds a 1×4 figure (R_global, R_worst, Combined, MAE) with both methods on
each panel. The first point ("Base") uses a shared baseline so ablation starts
from the same initial prompt metric on the capped holdout test set.

Default runs (matched ablation narrative):
  - PRIME + synthetic few-shot: results_all_categories_evolve_subsample
  - Plain evolution (no AL):     results_plain_uncapped_data_match

Usage:
    python plot_ablation_test_report_2x2.py
    python plot_ablation_test_report_2x2.py --baseline-from plain
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent

FONT_TITLE = 16
FONT_LABEL = 15
FONT_TICK = 14
FONT_LEGEND = 15
FONT_ANNOTATION = 12

METRICS = [
    ("R_global", "test_R_global", "Global Accuracy (Test)", r"$R_{\mathrm{global}}$"),
    ("R_worst", "test_R_worst", "Worst-group Accuracy (Test)", r"$R_{\mathrm{worst}}$"),
    ("combined_score", "test_combined_score", "Combined Score (Test)", "Combined score"),
    ("mae", "test_mae", "Mean Absolute Error (Test)", "MAE"),
]

COLOR_AL = "#1f77b4"
COLOR_PLAIN = "#9aa0a6"


def _load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _series(entries: list, log_key: str) -> List[float]:
    return [float(e.get(log_key, 0.0)) for e in entries]


def _shared_baseline(
    run_dirs: Dict[str, Path],
    *,
    source: str,
) -> Dict[str, float]:
    """Pick canonical Base metrics for both curves."""
    keys = [m[0] for m in METRICS]
    if source == "plain":
        base = _load_json(run_dirs["plain"] / "baseline_test_metrics.json")
        if not isinstance(base, dict):
            raise SystemExit("Missing plain baseline_test_metrics.json")
        return {k: float(base[k]) for k in keys}

    if source == "al":
        base = _load_json(run_dirs["al"] / "baseline_test_metrics.json")
        if not isinstance(base, dict):
            raise SystemExit("Missing AL baseline_test_metrics.json")
        return {k: float(base[k]) for k in keys}

    # average: numerically stable compromise when runs differ slightly
    bases = []
    for label in ("al", "plain"):
        b = _load_json(run_dirs[label] / "baseline_test_metrics.json")
        if isinstance(b, dict):
            bases.append(b)
    if not bases:
        raise SystemExit("No baseline_test_metrics.json found.")
    return {k: float(np.mean([float(b[k]) for b in bases])) for k in keys}


def _curve_for_run(
    run_dir: Path,
    shared_baseline: Dict[str, float],
    max_cycles: int,
) -> Tuple[List[str], Dict[str, List[float]]]:
    entries = _load_json(run_dir / "active_loop_log.json")
    final = _load_json(run_dir / "final_test_metrics.json")
    if not isinstance(entries, list) or not entries:
        raise SystemExit(f"No active_loop_log.json in {run_dir}")
    if not isinstance(final, dict):
        raise SystemExit(f"No final_test_metrics.json in {run_dir}")

    entries = sorted(entries, key=lambda e: e.get("al_iter", 0))[:max_cycles]
    cycle_iters = [int(e.get("al_iter", i)) for i, e in enumerate(entries)]

    curves: Dict[str, List[float]] = {}
    for base_key, log_key, _, _ in METRICS:
        per_cycle = _series(entries, log_key)
        curves[base_key] = [
            shared_baseline[base_key],
            *per_cycle,
            float(final.get(base_key, per_cycle[-1] if per_cycle else shared_baseline[base_key])),
        ]

    x_labels = ["Base"] + [str(i) for i in cycle_iters] + ["Final"]
    return x_labels, curves


def _ylim_for_metric(metric_key: str, values: List[float]) -> Tuple[float, float]:
    vmin, vmax = min(values), max(values)
    pad = max(0.02, (vmax - vmin) * 0.12)
    if metric_key == "mae":
        lo = max(0.0, vmin - pad)
        hi = vmax + pad
    else:
        lo = max(0.0, vmin - pad)
        hi = min(1.0, vmax + pad)
        if hi - lo < 0.08:
            mid = 0.5 * (lo + hi)
            lo, hi = max(0.0, mid - 0.04), min(1.0, mid + 0.04)
    return lo, hi


def plot_overlay(
    run_al: Path,
    run_plain: Path,
    *,
    label_al: str,
    label_plain: str,
    baseline_source: str,
    max_cycles: int | None,
    figsize: str,
    suptitle: str | None,
    out_path: Path,
) -> None:
    run_dirs = {"al": run_al.resolve(), "plain": run_plain.resolve()}
    shared_baseline = _shared_baseline(run_dirs, source=baseline_source)

    n_al = len(_load_json(run_al / "active_loop_log.json") or [])
    n_plain = len(_load_json(run_plain / "active_loop_log.json") or [])
    n_common = min(n_al, n_plain)
    if max_cycles is None:
        max_cycles = n_common
    else:
        max_cycles = min(max_cycles, n_common)

    x_al, curves_al = _curve_for_run(run_al, shared_baseline, max_cycles)
    x_plain, curves_plain = _curve_for_run(run_plain, shared_baseline, max_cycles)
    if x_al != x_plain:
        raise SystemExit("Cycle labels mismatch between runs.")

    x_pos = list(range(len(x_al)))

    parts = [float(x.strip()) for x in figsize.split(",")]
    w, h = (parts[0], parts[1]) if len(parts) >= 2 else (parts[0], parts[0] * 0.28)
    fig, axes = plt.subplots(1, 4, figsize=(w, h), sharex=True, sharey=False)
    axes = list(axes)

    for ax, (base_key, _, title, ylab) in zip(axes, METRICS):
        y_al = curves_al[base_key]
        y_plain = curves_plain[base_key]
        all_y = y_al + y_plain

        ax.plot(
            x_pos,
            y_al,
            "o-",
            color=COLOR_AL,
            linewidth=2.0,
            markersize=6,
            label=label_al,
        )
        ax.plot(
            x_pos,
            y_plain,
            "s--",
            color=COLOR_PLAIN,
            linewidth=1.8,
            markersize=6,
            label=label_plain,
        )

        ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold", pad=8)
        ax.set_ylabel(ylab, fontsize=FONT_LABEL, labelpad=6)
        ax.tick_params(axis="both", labelsize=FONT_TICK)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_al)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_ylim(*_ylim_for_metric(base_key, all_y))

        if base_key == "mae":
            ax.text(
                0.02,
                0.04,
                "lower is better",
                transform=ax.transAxes,
                fontsize=FONT_ANNOTATION,
                color="#555555",
                style="italic",
            )

    layout_top = 0.90 if suptitle else 0.92
    if suptitle:
        fig.suptitle(suptitle, fontsize=FONT_TITLE, fontweight="bold", y=0.98)

    fig.subplots_adjust(
        left=0.06,
        right=0.995,
        bottom=0.28,
        top=layout_top,
        wspace=0.28,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=FONT_LEGEND,
        bbox_to_anchor=(0.5, 0.08),
        columnspacing=1.5,
        handletextpad=0.6,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(
        "Shared Base:",
        {k: round(shared_baseline[k], 4) for k in shared_baseline},
        f"(source={baseline_source}, cycles=0..{max_cycles - 1})",
    )


def main() -> None:
    p = argparse.ArgumentParser(description="1x4 AL vs Plain test report overlay.")
    p.add_argument(
        "--run-al",
        type=str,
        default="results_all_categories_evolve_subsample",
        help="PRIME / AL run directory",
    )
    p.add_argument(
        "--run-plain",
        type=str,
        default="results_plain_uncapped_data_match",
        help="Plain evolution run directory",
    )
    p.add_argument(
        "--label-al",
        type=str,
        default="PRIME (Active Learning + synthetic few-shot)",
    )
    p.add_argument("--label-plain", type=str, default="Plain Evolution")
    p.add_argument(
        "--baseline-from",
        choices=("plain", "al", "average"),
        default="plain",
        help="Which run supplies the shared Base point (default: plain control)",
    )
    p.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Align to at most this many cycles (default: min across runs)",
    )
    p.add_argument(
        "--out",
        type=str,
        default="plots/ablation_al_vs_plain_test_report_2x2.png",
    )
    p.add_argument(
        "--figsize",
        type=str,
        default="20,4.5",
        help="Figure size in inches, e.g. 20,4.5 (default) or 24,4.5 (wide)",
    )
    p.add_argument(
        "--suptitle",
        type=str,
        default="",
        help='Figure title (default: none). Use "Metric evolution per cycle" to restore.',
    )
    args = p.parse_args()

    run_al = (SCRIPT_DIR / args.run_al).resolve()
    run_plain = (SCRIPT_DIR / args.run_plain).resolve()
    if not run_al.is_dir() or not run_plain.is_dir():
        raise SystemExit("Run directory not found.")

    plot_overlay(
        run_al,
        run_plain,
        label_al=args.label_al,
        label_plain=args.label_plain,
        baseline_source=args.baseline_from,
        max_cycles=args.max_cycles,
        figsize=args.figsize,
        suptitle=args.suptitle.strip() or None,
        out_path=(SCRIPT_DIR / args.out).resolve(),
    )


if __name__ == "__main__":
    main()

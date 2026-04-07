"""
Lightweight report plots for test metrics only.

This script builds a compact 1x4 figure with the key per-cycle TEST metrics:
  - Global Accuracy (R_global)
  - Worst-group Accuracy (R_worst)
  - Test Combined Score
  - Mean Absolute Error (MAE)

Curves start with baseline value (first point) and end with final test value
(last point). No separate horizontal reference lines.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent


def _load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _field(entries, key):
    return [e.get(key, 0) for e in entries]


def plot_test_report(save_dir: Path | None = None, results_dir: Path | None = None) -> None:
    """
    Build a clean 1x4 test-metrics figure for reports.
    Curves start with baseline and end with final.
    """
    base = (results_dir or SCRIPT_DIR / "results").resolve()
    entries = _load_json(base / "active_loop_log.json")
    if not entries:
        print("No log data (active_loop_log.json). Run active_loop.py first.")
        return

    baseline = _load_json(base / "baseline_test_metrics.json")
    final = _load_json(base / "final_test_metrics.json")

    entries.sort(key=lambda e: e.get("al_iter", 0))
    cycle_iters = [e.get("al_iter", i) for i, e in enumerate(entries)]

    has_test = any("test_R_global" in e for e in entries)
    if not has_test:
        print("No per-cycle test metrics in log; nothing to plot.")
        return

    test_R_global = _field(entries, "test_R_global")
    test_combined = _field(entries, "test_combined_score")
    test_R_worst = _field(entries, "test_R_worst")
    test_mae = _field(entries, "test_mae")

    # Build extended x-axis: ["Base", 0, 1, 2, ..., "Final"]
    x_labels = ["Base"] + [str(i) for i in cycle_iters] + ["Final"]
    x_pos = list(range(len(x_labels)))

    def _extend_curve(data: list, key: str) -> list:
        """Prepend baseline value and append final value to curve data."""
        base_val = baseline.get(key, data[0]) if baseline else data[0]
        final_val = final.get(key, data[-1]) if final else data[-1]
        return [base_val] + data + [final_val]

    out = save_dir or (base / "plots")
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    fig.suptitle(
        "Active Learning Evolution — Test Metrics",
        fontsize=13,
        fontweight="bold",
    )

    def _setup(ax, ylabel: str = "", title: str = ""):
        ax.set_xlabel("AL Cycle")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    # 1) Global Accuracy
    ax = axes[0]
    ax.plot(x_pos, _extend_curve(test_R_global, "R_global"), color="tab:blue", marker="o", ms=5, label="test R_global")
    _setup(ax, ylabel="R_global", title="Global Accuracy (Test)")

    # 2) Worst-group Accuracy
    ax = axes[1]
    ax.plot(
        x_pos,
        _extend_curve(test_R_worst, "R_worst"),
        color="tab:red",
        marker="o",
        ms=5,
        label="test R_worst",
    )
    _setup(ax, ylabel="R_worst", title="Worst-group Accuracy (Test)")

    # 3) Test Combined Score
    ax = axes[2]
    ax.plot(x_pos, _extend_curve(test_combined, "combined_score"), color="tab:purple", marker="o", ms=5, label="test Combined")
    _setup(ax, ylabel="Combined Score", title="Combined Score (Test)")

    # 4) Mean Absolute Error
    ax = axes[3]
    ax.plot(x_pos, _extend_curve(test_mae, "mae"), color="tab:gray", marker="o", ms=5, label="test MAE")
    _setup(ax, ylabel="MAE", title="Mean Absolute Error (Test, ↓ better)")

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    p = out / "active_evolution_test_report.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"Saved: {p}")


def main():
    import argparse

    p = argparse.ArgumentParser(
        description="Report-style plots for per-cycle TEST metrics (no baselines/events)."
    )
    p.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Results directory (default: results)",
    )
    a = p.parse_args()
    rd = Path(a.results_dir).resolve() if a.results_dir else None
    if rd and not rd.is_dir():
        print(f"Not a directory: {rd}")
        sys.exit(1)
    plot_test_report(results_dir=rd)


if __name__ == "__main__":
    main()


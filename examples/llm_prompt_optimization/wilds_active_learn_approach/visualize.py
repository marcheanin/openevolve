"""
Visualization for Active Prompt Evolution results.

Generates plot files from active_loop_log.json, baseline_test_metrics.json,
and final_test_metrics.json:
  - active_evolution_curves.png   (validation metrics)
  - active_evolution_test.png    (test metrics per cycle, when available)
  - active_evolution_secondary.png (diagnostics)
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent


def _load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _field(entries, key):
    return [e.get(key, 0) for e in entries]


def plot_results(save_dir: Path | None = None, results_dir: Path | None = None) -> None:
    base = (results_dir or SCRIPT_DIR / "results").resolve()
    entries = _load_json(base / "active_loop_log.json")
    if not entries:
        print("No log data (active_loop_log.json). Run active_loop.py first.")
        return

    baseline = _load_json(base / "baseline_test_metrics.json")
    final = _load_json(base / "final_test_metrics.json")

    entries.sort(key=lambda e: e.get("al_iter", 0))
    cycle_iters = [e.get("al_iter", i) for i, e in enumerate(entries)]

    # Validation metrics (always)
    val_R_global = _field(entries, "val_R_global")
    val_R_worst = _field(entries, "val_R_worst")
    val_combined = _field(entries, "val_combined_score")
    val_Acc_Hard = _field(entries, "val_Acc_Hard")
    val_Acc_Anchor = _field(entries, "val_Acc_Anchor")
    val_mae = _field(entries, "val_mae")
    val_kappa = _field(entries, "val_mean_kappa")

    # Test metrics (per-cycle, when available)
    has_test = any("test_R_global" in e for e in entries)
    if has_test:
        test_R_global = _field(entries, "test_R_global")
        test_R_worst = _field(entries, "test_R_worst")
        test_combined = _field(entries, "test_combined_score")
        test_Acc_Hard = _field(entries, "test_Acc_Hard")
        test_Acc_Anchor = _field(entries, "test_Acc_Anchor")
        test_mae = _field(entries, "test_mae")
        test_kappa = _field(entries, "test_mean_kappa")

    evo_best = _field(entries, "evo_best_score")
    batch_hard = _field(entries, "batch_n_hard")
    batch_anchor = _field(entries, "batch_n_anchor")
    cycle_time = _field(entries, "cycle_time_s")

    expansion_iters = [e["al_iter"] for e in entries if e.get("expanded")]
    consolidation_iters = [e["al_iter"] for e in entries if e.get("consolidated")]

    # Build extended x-axis: ["Base", 0, 1, 2, ..., "Final"]
    x_labels = ["Base"] + [str(i) for i in cycle_iters] + ["Final"]
    x_pos = list(range(len(x_labels)))  # 0, 1, 2, ..., n+1

    # Shift expansion/consolidation iters by +1 to account for "Base" at position 0
    expansion_x = [i + 1 for i in expansion_iters]
    consolidation_x = [i + 1 for i in consolidation_iters]

    def _extend_curve(data: list, key: str) -> list:
        """Prepend baseline value and append final value to curve data."""
        base_val = baseline.get(key, data[0]) if baseline else data[0]
        final_val = final.get(key, data[-1]) if final else data[-1]
        return [base_val] + data + [final_val]

    def _events(ax):
        for i, xi in enumerate(expansion_x):
            ax.axvline(xi, color="orange", ls="--", alpha=0.7, lw=1.2,
                       label="Expansion" if i == 0 else None)
        for i, xi in enumerate(consolidation_x):
            ax.axvline(xi, color="green", ls=":", alpha=0.5, lw=1.2,
                       label="Consolidation" if i == 0 else None)

    def _setup(ax, ylabel="", title=""):
        ax.set_xlabel("AL Cycle")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    out = save_dir or (base / "plots")
    out.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # Figure 1: Validation metrics (3x2)
    # ================================================================
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle("Active Learning Evolution — Validation Metrics", fontsize=13, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(x_pos, _extend_curve(val_R_global, "R_global"), "b-o", ms=5, label="val R_global")
    _events(ax)
    _setup(ax, ylabel="R_global", title="Global Accuracy (R_global)")

    ax = axes[0, 1]
    ax.plot(x_pos, _extend_curve(val_R_worst, "R_worst"), "c-o", ms=5, label="val R_worst")
    _events(ax)
    _setup(ax, ylabel="R_worst", title="Worst-group Accuracy (R_worst)")

    ax = axes[1, 0]
    ax.plot(x_pos, _extend_curve(val_Acc_Hard, "Acc_Hard"), "r-o", ms=5, label="val Acc_Hard")
    _events(ax)
    _setup(ax, ylabel="Acc_Hard", title="Hard-set Accuracy")

    ax = axes[1, 1]
    curve = _extend_curve(val_Acc_Anchor, "Acc_Anchor")
    ax.plot(x_pos, curve, "g-o", ms=5, label="val Acc_Anchor")
    _events(ax)
    ax.set_ylim(min(min(curve), 0.9) - 0.02, 1.02)
    _setup(ax, ylabel="Acc_Anchor", title="Anchor-set Accuracy")

    ax = axes[2, 0]
    ax.plot(x_pos, _extend_curve(val_combined, "combined_score"), "m-o", ms=5, label="val Combined")
    _events(ax)
    _setup(ax, ylabel="Combined Score", title="Validation Combined Score")

    ax = axes[2, 1]
    ax.plot(x_pos, _extend_curve(val_mae, "mae"), "k-o", ms=5, label="val MAE")
    _events(ax)
    _setup(ax, ylabel="MAE", title="Mean Absolute Error (↓ better)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    p1 = out / "active_evolution_curves.png"
    plt.savefig(p1, dpi=150)
    plt.close()
    print(f"Saved: {p1}")

    # ================================================================
    # Figure 1b: Test metrics (3x2) — when per-cycle test eval available
    # ================================================================
    if has_test:
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        fig.suptitle("Active Learning Evolution — Test Metrics (per cycle)", fontsize=13, fontweight="bold")

        ax = axes[0, 0]
        ax.plot(x_pos, _extend_curve(test_R_global, "R_global"), "b-o", ms=5, label="test R_global")
        _events(ax)
        _setup(ax, ylabel="R_global", title="Global Accuracy (R_global)")

        ax = axes[0, 1]
        ax.plot(x_pos, _extend_curve(test_R_worst, "R_worst"), "c-o", ms=5, label="test R_worst")
        _events(ax)
        _setup(ax, ylabel="R_worst", title="Worst-group Accuracy (R_worst)")

        ax = axes[1, 0]
        ax.plot(x_pos, _extend_curve(test_Acc_Hard, "Acc_Hard"), "r-o", ms=5, label="test Acc_Hard")
        _events(ax)
        _setup(ax, ylabel="Acc_Hard", title="Hard-set Accuracy")

        ax = axes[1, 1]
        curve = _extend_curve(test_Acc_Anchor, "Acc_Anchor")
        ax.plot(x_pos, curve, "g-o", ms=5, label="test Acc_Anchor")
        _events(ax)
        ax.set_ylim(min(min(curve), 0.9) - 0.02, 1.02)
        _setup(ax, ylabel="Acc_Anchor", title="Anchor-set Accuracy")

        ax = axes[2, 0]
        ax.plot(x_pos, _extend_curve(test_combined, "combined_score"), "m-o", ms=5, label="test Combined")
        _events(ax)
        _setup(ax, ylabel="Combined Score", title="Test Combined Score")

        ax = axes[2, 1]
        ax.plot(x_pos, _extend_curve(test_mae, "mae"), "k-o", ms=5, label="test MAE")
        _events(ax)
        _setup(ax, ylabel="MAE", title="Mean Absolute Error (↓ better)")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        p1b = out / "active_evolution_test.png"
        plt.savefig(p1b, dpi=150)
        plt.close()
        print(f"Saved: {p1b}")

    # ================================================================
    # Figure 2: Secondary / diagnostic metrics (2x2)
    # ================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Active Learning Evolution — Diagnostics", fontsize=13, fontweight="bold")

    def _setup_diag(ax, ylabel="", title=""):
        """Setup for diagnostic plots (use cycle_iters, not extended x-axis)."""
        ax.set_xlabel("AL Cycle")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(cycle_iters)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    def _events_diag(ax):
        """Events for diagnostic plots (original iters, not shifted)."""
        for i, xi in enumerate(expansion_iters):
            ax.axvline(xi, color="orange", ls="--", alpha=0.7, lw=1.2,
                       label="Expansion" if i == 0 else None)
        for i, xi in enumerate(consolidation_iters):
            ax.axvline(xi, color="green", ls=":", alpha=0.5, lw=1.2,
                       label="Consolidation" if i == 0 else None)

    # 1. evo_best_score vs val/test combined
    ax = axes[0, 0]
    ax.plot(cycle_iters, evo_best, "r-s", ms=5, label="Evo best (batch fitness)")
    ax.plot(cycle_iters, val_combined, "m-o", ms=5, label="val Combined")
    if has_test:
        ax.plot(cycle_iters, test_combined, "b-o", ms=5, label="test Combined")
    _events_diag(ax)
    _setup_diag(ax, ylabel="Score", title="Evolution Best vs Val/Test Combined")

    # 2. mean_kappa (val + test when available)
    ax = axes[0, 1]
    ax.plot(cycle_iters, val_kappa, "teal", marker="o", ms=5, label="val mean_kappa")
    if has_test:
        ax.plot(cycle_iters, test_kappa, "navy", marker="s", ms=5, label="test mean_kappa")
    _events_diag(ax)
    _setup_diag(ax, ylabel="Kappa", title="Mean Inter-worker Kappa")

    # 3. Hard vs Anchor (batch)
    ax = axes[1, 0]
    ax.bar([i - 0.15 for i in cycle_iters], batch_hard, 0.3, color="salmon", label="Hard")
    ax.bar([i + 0.15 for i in cycle_iters], batch_anchor, 0.3, color="cornflowerblue", label="Anchor")
    _events_diag(ax)
    _setup_diag(ax, ylabel="Count", title="Batch Composition (Hard / Anchor)")

    # 4. Cycle time
    ax = axes[1, 1]
    ax.bar(cycle_iters, cycle_time, 0.5, color="slategray", label="Cycle time")
    for i, t in zip(cycle_iters, cycle_time):
        ax.text(i, t + 20, f"{t:.0f}s", ha="center", fontsize=7)
    _setup_diag(ax, ylabel="Seconds", title="Cycle Wall-clock Time")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    p2 = out / "active_evolution_secondary.png"
    plt.savefig(p2, dpi=150)
    plt.close()
    print(f"Saved: {p2}")


def main():
    import argparse
    p = argparse.ArgumentParser(description="Plot Active Learning evolution curves.")
    p.add_argument("--results-dir", type=str, default=None,
                   help="Results directory (default: results)")
    a = p.parse_args()
    rd = Path(a.results_dir).resolve() if a.results_dir else None
    if rd and not rd.is_dir():
        print(f"Not a directory: {rd}")
        sys.exit(1)
    plot_results(results_dir=rd)


if __name__ == "__main__":
    main()

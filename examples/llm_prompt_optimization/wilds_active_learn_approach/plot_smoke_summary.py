"""
Smoke-run summary plots for an AL experiment.

Reads active_loop_log.json + baseline_test_metrics.json + final_test_metrics.json
from a results dir and produces 4 figures:

    1. metrics_per_cycle.png   - val/test combined_score, R_global, R_worst, mae per cycle
    2. pool_dynamics.png       - n_seen / n_unseen / batch Hard/Anchor and refresh counts
    3. baseline_vs_final.png   - bars: baseline test vs final test for each metric
    4. timing_breakdown.png    - timings per stage (if [smoke] timings are not exposed,
                                 derives cycle_time_s and per-stage estimates from the log)

Run:
    python plot_smoke_summary.py --run results_all_categories_uncapped_train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent


def _load_json(p: Path):
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _series(entries, key):
    return [float(e.get(key, 0.0) or 0.0) for e in entries]


def plot_metrics_per_cycle(entries, baseline, final_, out_path: Path, title: str):
    if not entries:
        return
    cycles = [e["al_iter"] + 1 for e in entries]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=13)

    metric_pairs = [
        ("combined_score", "Combined score"),
        ("R_global", "R_global"),
        ("R_worst", "R_worst (worst-decile user)"),
        ("mae", "MAE (lower is better)"),
    ]

    for ax, (metric, label) in zip(axs.ravel(), metric_pairs):
        v = _series(entries, f"val_{metric}")
        t = _series(entries, f"test_{metric}")

        # Prepend baseline-test as Cycle 0 (only test, baseline_val is not collected per-cycle)
        x_with_base = [0] + cycles
        if baseline and metric in baseline:
            t_full = [float(baseline[metric])] + t
        else:
            t_full = [t[0]] + t
        v_full = [v[0]] + v  # placeholder for visual continuity

        ax.plot(x_with_base, v_full, "o-", color="#1f77b4", label="val")
        ax.plot(x_with_base, t_full, "s--", color="#d62728", label="test")
        if final_ and metric in final_:
            ax.scatter(
                [cycles[-1] + 0.3],
                [float(final_[metric])],
                color="black",
                marker="*",
                s=120,
                zorder=5,
                label="final test (best-by-val)",
            )
        ax.set_title(label)
        ax.set_xlabel("AL cycle (0 = baseline test)")
        ax.set_ylabel(label)
        ax.set_xticks(x_with_base)
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_pool_dynamics(entries, out_path: Path, title: str):
    if not entries:
        return
    cycles = [e["al_iter"] + 1 for e in entries]

    n_seen = _series(entries, "n_seen")
    n_unseen = _series(entries, "n_unseen")
    batch_hard = _series(entries, "batch_n_hard")
    batch_anchor = _series(entries, "batch_n_anchor")
    refresh_added = [int((e.get("refresh_event") or {}).get("added", 0)) for e in entries]
    refresh_hard = [int((e.get("refresh_event") or {}).get("added_hard", 0)) for e in entries]

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(title, fontsize=13)

    ax = axs[0]
    ax.plot(cycles, n_seen, "o-", color="#2ca02c", label="n_seen")
    ax.plot(cycles, n_unseen, "s--", color="#7f7f7f", label="n_unseen")
    ax.set_xticks(cycles)
    ax.set_xlabel("AL cycle")
    ax.set_ylabel("# reviews")
    ax.set_title("Pool: Seen / Unseen")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axs[1]
    width = 0.35
    x = np.arange(len(cycles))
    ax.bar(x - width / 2, batch_hard, width, color="#d62728", label="Hard in batch")
    ax.bar(x + width / 2, batch_anchor, width, color="#1f77b4", label="Anchor in batch")
    ax.set_xticks(x, [str(c) for c in cycles])
    ax.set_xlabel("AL cycle")
    ax.set_ylabel("examples")
    ax.set_title("Active batch composition")
    ax.legend()
    ax.grid(alpha=0.25, axis="y")

    ax = axs[2]
    ax.bar(x - width / 2, refresh_added, width, color="#7f7f7f", label="refresh added")
    ax.bar(x + width / 2, refresh_hard, width, color="#d62728", label="refresh marked Hard")
    ax.set_xticks(x, [str(c) for c in cycles])
    ax.set_xlabel("AL cycle")
    ax.set_ylabel("examples")
    ax.set_title("Pool refresh per cycle")
    ax.legend()
    ax.grid(alpha=0.25, axis="y")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_baseline_vs_final(baseline: dict, final_: dict, entries: list, out_path: Path, title: str):
    if not baseline or not final_:
        return
    metrics_higher = ["combined_score", "R_global", "R_worst", "mean_kappa", "Acc_Hard", "Acc_Anchor"]
    metrics_lower = ["mae"]

    keys = [m for m in metrics_higher + metrics_lower if m in baseline and m in final_]
    base_v = [float(baseline[k]) for k in keys]
    final_v = [float(final_[k]) for k in keys]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(keys))
    w = 0.35
    bars_b = ax.bar(x - w / 2, base_v, w, color="#7f7f7f", label="Baseline (initial prompt)")
    bars_f = ax.bar(x + w / 2, final_v, w, color="#2ca02c", label="Final (best-by-val prompt)")

    for bars, vals in ((bars_b, base_v), (bars_f, final_v)):
        for b, v in zip(bars, vals):
            ax.annotate(
                f"{v:.3f}",
                xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x, keys)
    ax.set_ylabel("metric value")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25, axis="y")

    deltas = [f - b for b, f in zip(base_v, final_v)]
    delta_text = "  |  ".join(
        f"Δ {k}={d:+.3f}" + ("  (lower=better)" if k in metrics_lower else "")
        for k, d in zip(keys, deltas)
    )
    fig.text(0.5, -0.02, delta_text, ha="center", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_timing_breakdown(entries, out_path: Path, title: str):
    if not entries:
        return
    cycles = [e["al_iter"] + 1 for e in entries]
    cycle_t = _series(entries, "cycle_time_s")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(cycles, cycle_t, color="#1f77b4")
    for c, t in zip(cycles, cycle_t):
        ax.annotate(
            f"{t:.0f}s",
            xy=(c, t),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(cycles)
    ax.set_xlabel("AL cycle")
    ax.set_ylabel("Wall time, s")
    ax.set_title(title)
    ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        type=str,
        required=True,
        help="Path to results directory (with active_loop_log.json + baseline/final test metrics).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output dir for plots (default: <run>/_plots/).",
    )
    parser.add_argument("--title", type=str, default="AL smoke run")
    args = parser.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = (SCRIPT_DIR / run_dir).resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Not a directory: {run_dir}")

    entries = _load_json(run_dir / "active_loop_log.json") or []
    baseline = _load_json(run_dir / "baseline_test_metrics.json") or {}
    final_ = _load_json(run_dir / "final_test_metrics.json") or {}

    out_dir = Path(args.out) if args.out else (run_dir / "_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_metrics_per_cycle(entries, baseline, final_, out_dir / "metrics_per_cycle.png", args.title)
    plot_pool_dynamics(entries, out_dir / "pool_dynamics.png", args.title + " — pool dynamics")
    plot_baseline_vs_final(baseline, final_, entries, out_dir / "baseline_vs_final.png", args.title + " — baseline vs final test")
    plot_timing_breakdown(entries, out_dir / "timing_breakdown.png", args.title + " — wall time per cycle")

    print(f"Plots saved to: {out_dir}")
    for f in sorted(out_dir.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()

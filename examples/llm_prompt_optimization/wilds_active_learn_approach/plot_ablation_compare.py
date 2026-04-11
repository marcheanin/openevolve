"""
Overlay two AL runs (e.g. with / without synthetic few-shot) on one figure.

Reads active_loop_log.json (+ optional baseline/final) from each results directory.
"""

from __future__ import annotations

import argparse
import json
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


def _series(entries: list, key: str):
    return [float(e.get(key, 0.0) or 0.0) for e in entries]


def plot_ablation(
    dir_a: Path,
    dir_b: Path,
    label_a: str,
    label_b: str,
    out_path: Path,
    title: str | None = None,
) -> None:
    log_a = _load_json(dir_a / "active_loop_log.json")
    log_b = _load_json(dir_b / "active_loop_log.json")
    if not log_a or not log_b:
        raise SystemExit("Need active_loop_log.json in both directories.")

    log_a = sorted(log_a, key=lambda e: e.get("al_iter", 0))
    log_b = sorted(log_b, key=lambda e: e.get("al_iter", 0))

    base_a = _load_json(dir_a / "baseline_test_metrics.json")
    base_b = _load_json(dir_b / "baseline_test_metrics.json")
    fin_a = _load_json(dir_a / "final_test_metrics.json")
    fin_b = _load_json(dir_b / "final_test_metrics.json")

    x = [int(e.get("al_iter", i)) for i, e in enumerate(log_a)]
    if len(x) != len(log_b):
        raise SystemExit(
            f"Cycle count mismatch: {len(x)} vs {len(log_b)} — align runs before plotting."
        )

    seed_val_a = _series(log_a, "seed_val_score")
    seed_val_b = _series(log_b, "seed_val_score")
    gbest_a = _series(log_a, "global_best_val_score")
    gbest_b = _series(log_b, "global_best_val_score")
    test_c_a = _series(log_a, "test_combined_score")
    test_c_b = _series(log_b, "test_combined_score")
    gap_a = _series(log_a, "val_test_gap")
    gap_b = _series(log_b, "val_test_gap")
    val_r_a = _series(log_a, "val_R_global")
    val_r_b = _series(log_b, "val_R_global")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    st = title or "Ablation: synthetic few-shot (8 AL cycles × 20 evolve iter)"
    fig.suptitle(st, fontsize=12, fontweight="bold")

    xa = x
    color_a, color_b = "#1f77b4", "#ff7f0e"
    ms = 6

    ax = axes[0, 0]
    ax.plot(xa, seed_val_a, "o-", color=color_a, ms=ms, label=f"{label_a} (seed val)")
    ax.plot(xa, seed_val_b, "s-", color=color_b, ms=ms, label=f"{label_b} (seed val)")
    ax.plot(xa, gbest_a, "--", color=color_a, alpha=0.55, label=f"{label_a} best@val")
    ax.plot(xa, gbest_b, "--", color=color_b, alpha=0.55, label=f"{label_b} best@val")
    ax.set_ylabel("score")
    ax.set_title("Validation combined (per cycle + cumulative best)")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=7, loc="best")

    ax = axes[0, 1]
    ax.plot(xa, test_c_a, "o-", color=color_a, ms=ms, label=f"{label_a} test")
    ax.plot(xa, test_c_b, "s-", color=color_b, ms=ms, label=f"{label_b} test")
    if base_a and base_b:
        b_a = float(base_a.get("combined_score", 0))
        b_b = float(base_b.get("combined_score", 0))
        if abs(b_a - b_b) < 1e-9:
            ax.axhline(b_a, color="gray", ls=":", lw=1, alpha=0.8, label="baseline test")
        else:
            ax.axhline(b_a, color="gray", ls=":", lw=1, alpha=0.6, label="baseline A")
            ax.axhline(b_b, color="0.5", ls=":", lw=1, alpha=0.6, label="baseline B")
    if fin_a and fin_b:
        ax.axhline(float(fin_a["combined_score"]), color=color_a, ls=":", lw=1.2, alpha=0.9)
        ax.axhline(float(fin_b["combined_score"]), color=color_b, ls=":", lw=1.2, alpha=0.9)
    ax.set_ylabel("combined")
    ax.set_title("Holdout test combined (per cycle; dotted = final best_val prompt)")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=7, loc="best")

    ax = axes[1, 0]
    ax.plot(xa, val_r_a, "o-", color=color_a, ms=ms, label=label_a)
    ax.plot(xa, val_r_b, "s-", color=color_b, ms=ms, label=label_b)
    ax.set_ylabel("R_global")
    ax.set_title("Validation R_global")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=7, loc="best")

    ax = axes[1, 1]
    ax.plot(xa, gap_a, "o-", color=color_a, ms=ms, label=label_a)
    ax.plot(xa, gap_b, "s-", color=color_b, ms=ms, label=label_b)
    ax.axhline(0.0, color="gray", ls="-", lw=0.8, alpha=0.5)
    ax.set_ylabel("val − test")
    ax.set_title("Generalization gap (seed val − per-cycle test combined)")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=7, loc="best")

    for ax in axes[1, :]:
        ax.set_xlabel("AL cycle (index)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    p = argparse.ArgumentParser(description="Plot ablation comparison on one figure.")
    p.add_argument(
        "--with-synth",
        type=Path,
        default=SCRIPT_DIR / "results_ablation_with_synth_8x20",
        help="Results dir: synthetic few-shot ON",
    )
    p.add_argument(
        "--without-synth",
        type=Path,
        default=SCRIPT_DIR / "results_ablation_without_synth_8x20",
        help="Results dir: synthetic few-shot OFF",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=SCRIPT_DIR / "plots" / "ablation_8x20_comparison.png",
        help="Output PNG path",
    )
    p.add_argument("--title", type=str, default=None)
    args = p.parse_args()

    plot_ablation(
        args.with_synth.resolve(),
        args.without_synth.resolve(),
        "with synth",
        "no synth",
        args.output.resolve(),
        title=args.title,
    )


if __name__ == "__main__":
    main()

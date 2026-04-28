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


def _field(entries: list[dict], key: str):
    return [e.get(key, 0) for e in entries]


def _extend_curve(data: list, baseline: dict | None, final: dict | None, key: str) -> list:
    """Prepend baseline value and append final value to curve data (best-effort)."""
    if not data:
        return []
    base_val = baseline.get(key, data[0]) if baseline else data[0]
    final_val = final.get(key, data[-1]) if final else data[-1]
    return [base_val] + data + [final_val]


def _events_from_entries(entries: list[dict]) -> tuple[list[int], list[int]]:
    """Return x-positions (already shifted by +1 to account for 'Base')."""
    expansion_iters = [e["al_iter"] for e in entries if e.get("expanded")]
    consolidation_iters = [e["al_iter"] for e in entries if e.get("consolidated")]
    expansion_x = [i + 1 for i in expansion_iters]
    consolidation_x = [i + 1 for i in consolidation_iters]
    return expansion_x, consolidation_x


def _plot_events(ax, expansion_x: list[int], consolidation_x: list[int]):
    for i, xi in enumerate(expansion_x):
        ax.axvline(
            xi,
            color="orange",
            ls="--",
            alpha=0.55,
            lw=1.0,
            label="Expansion" if i == 0 else None,
        )
    for i, xi in enumerate(consolidation_x):
        ax.axvline(
            xi,
            color="green",
            ls=":",
            alpha=0.35,
            lw=1.0,
            label="Consolidation" if i == 0 else None,
        )


def plot_overlay_val_test_grids(
    dir_a: Path,
    dir_b: Path,
    label_a: str,
    label_b: str,
    out_dir: Path,
    title_prefix: str | None = None,
    include_events: bool = True,
) -> list[Path]:
    """
    Create 2 figures:
      - overlay_val_metrics.png  (3x2, val_*)
      - overlay_test_metrics.png (3x2, test_* per-cycle)
    """
    log_a = _load_json(dir_a / "active_loop_log.json")
    log_b = _load_json(dir_b / "active_loop_log.json")
    if not log_a or not log_b:
        raise SystemExit("Need active_loop_log.json in both directories.")

    log_a = sorted(log_a, key=lambda e: e.get("al_iter", 0))
    log_b = sorted(log_b, key=lambda e: e.get("al_iter", 0))

    if len(log_a) != len(log_b):
        raise SystemExit(
            f"Cycle count mismatch: {len(log_a)} vs {len(log_b)} — align runs before plotting."
        )

    base_a = _load_json(dir_a / "baseline_test_metrics.json")
    base_b = _load_json(dir_b / "baseline_test_metrics.json")
    fin_a = _load_json(dir_a / "final_test_metrics.json")
    fin_b = _load_json(dir_b / "final_test_metrics.json")

    cycle_iters = [int(e.get("al_iter", i)) for i, e in enumerate(log_a)]
    x_labels = ["Base"] + [str(i) for i in cycle_iters] + ["Final"]
    x_pos = list(range(len(x_labels)))

    exp_x_a, cons_x_a = _events_from_entries(log_a)
    exp_x_b, cons_x_b = _events_from_entries(log_b)
    # union (same x-axis), keep stable ordering
    exp_x = sorted(set(exp_x_a + exp_x_b))
    cons_x = sorted(set(cons_x_a + cons_x_b))

    color_a, color_b = "#1f77b4", "#ff7f0e"
    ms = 4.5

    def _setup(ax, ylabel: str, title: str):
        ax.set_xlabel("AL Cycle")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.grid(True, alpha=0.3)

    saved: list[Path] = []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # Validation metrics (3x2)
    # ---------------------------
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    st = (title_prefix + " — " if title_prefix else "") + "Validation Metrics (overlay)"
    fig.suptitle(st, fontsize=13, fontweight="bold")

    val_specs = [
        ("val_R_global", "R_global", "Global Accuracy (R_global)"),
        ("val_R_worst", "R_worst", "Worst-group Accuracy (R_worst)"),
        ("val_Acc_Hard", "Acc_Hard", "Hard-set Accuracy"),
        ("val_Acc_Anchor", "Acc_Anchor", "Anchor-set Accuracy"),
        ("val_combined_score", "combined_score", "Combined Score"),
        ("val_mae", "mae", "Mean Absolute Error (↓ better)"),
    ]

    for ax, (log_key, base_key, ttl) in zip(axes.ravel(), val_specs):
        a = _field(log_a, log_key)
        b = _field(log_b, log_key)
        aa = _extend_curve(a, base_a, fin_a, base_key)
        bb = _extend_curve(b, base_b, fin_b, base_key)
        ax.plot(x_pos, aa, "o-", color=color_a, ms=ms, label=label_a)
        ax.plot(x_pos, bb, "s-", color=color_b, ms=ms, label=label_b)
        if include_events:
            _plot_events(ax, exp_x, cons_x)
        if log_key == "val_Acc_Anchor":
            ymin = min(min(aa), min(bb), 0.9) - 0.02
            ax.set_ylim(ymin, 1.02)
        _setup(ax, ylabel=log_key.replace("val_", ""), title=ttl)
        ax.legend(fontsize=7, loc="best")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    p_val = out_dir / "overlay_val_metrics.png"
    plt.savefig(p_val, dpi=150)
    plt.close()
    print(f"Saved: {p_val}")
    saved.append(p_val)

    # ---------------------------
    # Test metrics (3x2)
    # ---------------------------
    has_test = any("test_R_global" in e for e in log_a) and any("test_R_global" in e for e in log_b)
    if has_test:
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        st = (title_prefix + " — " if title_prefix else "") + "Test Metrics (per cycle; overlay)"
        fig.suptitle(st, fontsize=13, fontweight="bold")

        test_specs = [
            ("test_R_global", "R_global", "Global Accuracy (R_global)"),
            ("test_R_worst", "R_worst", "Worst-group Accuracy (R_worst)"),
            ("test_Acc_Hard", "Acc_Hard", "Hard-set Accuracy"),
            ("test_Acc_Anchor", "Acc_Anchor", "Anchor-set Accuracy"),
            ("test_combined_score", "combined_score", "Combined Score"),
            ("test_mae", "mae", "Mean Absolute Error (↓ better)"),
        ]

        for ax, (log_key, base_key, ttl) in zip(axes.ravel(), test_specs):
            a = _field(log_a, log_key)
            b = _field(log_b, log_key)
            aa = _extend_curve(a, base_a, fin_a, base_key)
            bb = _extend_curve(b, base_b, fin_b, base_key)
            ax.plot(x_pos, aa, "o-", color=color_a, ms=ms, label=label_a)
            ax.plot(x_pos, bb, "s-", color=color_b, ms=ms, label=label_b)
            if include_events:
                _plot_events(ax, exp_x, cons_x)
            if log_key == "test_Acc_Anchor":
                ymin = min(min(aa), min(bb), 0.9) - 0.02
                ax.set_ylim(ymin, 1.02)
            _setup(ax, ylabel=log_key.replace("test_", ""), title=ttl)
            ax.legend(fontsize=7, loc="best")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        p_test = out_dir / "overlay_test_metrics.png"
        plt.savefig(p_test, dpi=150)
        plt.close()
        print(f"Saved: {p_test}")
        saved.append(p_test)

    return saved


def plot_final_test_bars(
    dir_a: Path,
    dir_b: Path,
    label_a: str,
    label_b: str,
    out_path: Path,
    title: str | None = None,
) -> Path | None:
    fin_a = _load_json(dir_a / "final_test_metrics.json")
    fin_b = _load_json(dir_b / "final_test_metrics.json")
    if not fin_a or not fin_b:
        return None

    metrics = [
        ("combined_score", "combined"),
        ("R_global", "R_global"),
        ("R_worst", "R_worst"),
        ("mean_kappa", "mean_kappa"),
        ("mae", "mae (↓)"),
        ("Acc_Hard", "Acc_Hard"),
        ("Acc_Anchor", "Acc_Anchor"),
    ]
    keys = [k for k, _ in metrics if k in fin_a and k in fin_b]
    if not keys:
        return None

    vals_a = [float(fin_a[k]) for k in keys]
    vals_b = [float(fin_b[k]) for k in keys]
    labels = [dict(metrics).get(k, k) for k in keys]

    fig, ax = plt.subplots(1, 1, figsize=(11, 4))
    st = title or "Final Test Metrics — best@val prompt"
    ax.set_title(st, fontsize=12, fontweight="bold")

    x = list(range(len(keys)))
    w = 0.38
    ax.bar([i - w / 2 for i in x], vals_a, w, label=label_a, color="#1f77b4", alpha=0.9)
    ax.bar([i + w / 2 for i in x], vals_b, w, label=label_b, color="#ff7f0e", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=9, loc="best")

    # annotate values
    for i, (a, b) in enumerate(zip(vals_a, vals_b)):
        ax.text(i - w / 2, a, f"{a:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w / 2, b, f"{b:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


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
    p.add_argument(
        "--out-dir",
        type=Path,
        default=SCRIPT_DIR / "plots",
        help="Directory to save additional overlay figures",
    )
    args = p.parse_args()

    plot_ablation(
        args.with_synth.resolve(),
        args.without_synth.resolve(),
        "with synth",
        "no synth",
        args.output.resolve(),
        title=args.title,
    )

    # Additional overlay plots for key val/test metrics on one figure each
    extra_dir = (args.out_dir / "ablation_overlay_8x20").resolve()
    plot_overlay_val_test_grids(
        args.with_synth.resolve(),
        args.without_synth.resolve(),
        "with synth",
        "no synth",
        extra_dir,
        title_prefix=args.title or "Ablation: synthetic few-shot (8×20)",
        include_events=True,
    )
    plot_final_test_bars(
        args.with_synth.resolve(),
        args.without_synth.resolve(),
        "with synth",
        "no synth",
        extra_dir / "final_test_metrics_bars.png",
        title="Final Test Metrics (best@val prompt)",
    )


if __name__ == "__main__":
    main()

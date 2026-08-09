#!/usr/bin/env python3
"""Plots + metric dump for E1_constraint_global (Phase 1 mechanics check).

Figures written to <run>/analysis/:
  fig1_evolution_trajectory.png   D_select best-so-far per OE iteration × cycle
  fig2_cycle_metrics.png          D_select / val key / gate drops across AL cycles
  fig3_test_comparison.png        initial vs final: global metrics, classes, clusters
  fig4_pred_shifts.png            prediction movement + confusion 4/5
  fig5_vs_lowvar.png              side-by-side with the failed lowvar run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))

from prime.fitness.metrics import (  # noqa: E402
    cluster_accuracies,
    cvar_from_accuracies,
    mae,
    macro_class_accuracy,
    per_class_accuracy,
    worst_group_accuracy,
)


def load_oe_programs(run_dir: Path, cycle: int) -> List[Dict[str, Any]]:
    p = run_dir / f"al_iter_{cycle}" / "openevolve_output_path.txt"
    if not p.is_file():
        return []
    staging = Path(p.read_text(encoding="utf-8").strip())
    cps = sorted(
        (staging / "checkpoints").glob("checkpoint_*"),
        key=lambda d: int(d.name.split("_")[-1]),
    )
    if not cps:
        return []
    out = []
    for f in (cps[-1] / "programs").glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        m = data.get("metrics") or {}
        if "combined_score" not in m:
            continue
        out.append(
            {
                "iteration": int(data.get("iteration_found") or 0),
                "score": float(m["combined_score"]),
            }
        )
    return sorted(out, key=lambda r: r["iteration"])


def best_so_far(programs: List[Dict[str, Any]], n_iters: int = 8) -> List[float]:
    seeds = [p["score"] for p in programs if p["iteration"] == 0]
    best = max(seeds) if seeds else float("nan")
    series = []
    for it in range(1, n_iters + 1):
        for p in programs:
            if p["iteration"] == it:
                best = max(best, p["score"]) if best == best else p["score"]
        series.append(best)
    return series


def load_init_final(run_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Initial = seed ensemble used for cluster_assign_test; final = eval_test."""
    y = np.load(run_dir / "evals" / "eval_test" / "labels.npy")
    final = np.load(run_dir / "evals" / "eval_test" / "ensemble_predictions.npy")
    init = np.load(run_dir / "pred_cache" / "cluster_assign_test_ens.npy")
    cid = np.load(run_dir / "evals" / "eval_test" / "cluster_ids.npy")
    uid = np.load(run_dir / "evals" / "eval_test" / "user_ids.npy")
    return y, init, final, cid, uid


def metric_bundle(pred: np.ndarray, y: np.ndarray, cid: np.ndarray, uid: np.ndarray) -> Dict[str, Any]:
    accs = cluster_accuracies(pred, y, cid)
    return {
        "R_global": float(np.mean(pred == y)),
        "R_macro": macro_class_accuracy(pred, y),
        "R_worst": worst_group_accuracy(pred, y, uid),
        "CVaR_q40": cvar_from_accuracies(accs, 0.40),
        "mae": mae(pred, y),
        "per_class": per_class_accuracy(pred, y),
        "cluster": accs,
    }


def style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
            "axes.grid": True,
            "grid.alpha": 0.35,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )


def fig1_trajectory(run_dir: Path, out: Path) -> None:
    style()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#1f4e79", "#2e7d32", "#c62828"]
    x_off = 0
    xticks, xlabels = [], []
    for c, color in zip((1, 2, 3), colors):
        progs = load_oe_programs(run_dir, c)
        series = best_so_far(progs, 8)
        xs = np.arange(1, len(series) + 1) + x_off
        ax.plot(xs, series, marker="o", color=color, lw=2, label=f"cycle {c}")
        for i, v in enumerate(series):
            xticks.append(xs[i])
            xlabels.append(f"C{c}.{i+1}")
        x_off += len(series) + 1
        # separator
        ax.axvline(x_off + 0.5, color="#bbb", ls=":", lw=1)

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    for c in summary.get("cycles", []):
        # mark cycle best as horizontal guide in that cycle's band — skip for clarity
        pass

    ax.set_ylabel("D_select fitness (= R_global)")
    ax.set_xlabel("OpenEvolve iteration (per cycle)")
    ax.set_title("Phase 1 — evolution trajectory on D_select")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig2_cycle_metrics(run_dir: Path, out: Path) -> None:
    style()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    cycles = summary["cycles"]
    xs = [c["cycle"] for c in cycles]
    evo = [c.get("best_evo_score") for c in cycles]
    fit = [c.get("fitness") for c in cycles]
    val0 = [c["val_selection_key"][0] if c.get("val_selection_key") else np.nan for c in cycles]
    val1 = [c["val_selection_key"][1] if c.get("val_selection_key") else np.nan for c in cycles]
    drops = []
    for c in cycles:
        g = c.get("anchor_gate") or {}
        drops.append(g.get("drop") if g else np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    ax = axes[0]
    ax.plot(xs, fit, "s--", color="#78909c", label="cycle-entry fitness")
    ax.plot(xs, evo, "o-", color="#1f4e79", lw=2, label="best_evo (D_select R_global)")
    ax.set_xticks(xs)
    ax.set_xlabel("AL cycle")
    ax.set_ylabel("score")
    ax.set_title("D_select objective")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(xs, val0, "o-", color="#6a1b9a", label="val key[0] (tail proxy)")
    ax.plot(xs, val1, "s-", color="#00838f", label="val key[1] (R_global)")
    ax.set_xticks(xs)
    ax.set_xlabel("AL cycle")
    ax.set_title("Val selection key")
    ax.legend(fontsize=8)

    ax = axes[2]
    colors = ["#2e7d32" if (d == d and d <= 0.02) else "#c62828" for d in drops]
    ax.bar(xs, [0 if d != d else d for d in drops], color=colors, width=0.55)
    ax.axhline(0.02, color="#c62828", ls="--", lw=1, label="δ = 0.02")
    ax.set_xticks(xs)
    ax.set_xlabel("AL cycle")
    ax.set_ylabel("anchor drop")
    ax.set_title("Anchor gate (reject mode)")
    ax.legend(fontsize=8)
    for i, c in enumerate(cycles):
        g = c.get("anchor_gate")
        if not g:
            ax.text(xs[i], 0.001, "n/a\n(unchanged)", ha="center", va="bottom", fontsize=7)
        else:
            ax.text(
                xs[i],
                (g.get("drop") or 0) + 0.001,
                g.get("reason", "")[:12],
                ha="center",
                va="bottom",
                fontsize=7,
            )

    fig.suptitle("Phase 1 — per-cycle signals", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig3_test_comparison(run_dir: Path, out: Path, init_m: Dict, fin_m: Dict) -> None:
    style()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))

    # global metrics
    ax = axes[0]
    names = ["R_global", "R_macro", "CVaR_q40", "R_worst", "1−MAE/4"]
    init_v = [
        init_m["R_global"],
        init_m["R_macro"],
        init_m["CVaR_q40"],
        init_m["R_worst"],
        1 - init_m["mae"] / 4,
    ]
    fin_v = [
        fin_m["R_global"],
        fin_m["R_macro"],
        fin_m["CVaR_q40"],
        fin_m["R_worst"],
        1 - fin_m["mae"] / 4,
    ]
    x = np.arange(len(names))
    w = 0.36
    ax.bar(x - w / 2, init_v, w, label="initial", color="#90a4ae")
    ax.bar(x + w / 2, fin_v, w, label="final", color="#1f4e79")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0.3, 0.95)
    ax.set_title("Test metrics")
    ax.legend(fontsize=8)

    # per-class
    ax = axes[1]
    classes = sorted(init_m["per_class"])
    xi = np.arange(len(classes))
    ax.bar(xi - w / 2, [init_m["per_class"][c] for c in classes], w, color="#90a4ae", label="initial")
    ax.bar(xi + w / 2, [fin_m["per_class"][c] for c in classes], w, color="#1f4e79", label="final")
    ax.set_xticks(xi)
    ax.set_xticklabels([f"{c}★" for c in classes])
    ax.set_ylim(0, 1)
    ax.set_title("Per-class accuracy")
    ax.legend(fontsize=8)

    # clusters
    ax = axes[2]
    clusters = sorted(init_m["cluster"])
    xi = np.arange(len(clusters))
    ax.bar(xi - w / 2, [init_m["cluster"][c] for c in clusters], w, color="#90a4ae", label="initial")
    ax.bar(xi + w / 2, [fin_m["cluster"][c] for c in clusters], w, color="#1f4e79", label="final")
    ax.set_xticks(xi)
    ax.set_xticklabels([f"c{c}" for c in clusters])
    ax.set_ylim(0.5, 0.9)
    ax.set_title("Per-cluster accuracy")
    ax.legend(fontsize=8)

    fig.suptitle("Phase 1 — initial vs final on official OOD test (240 users)", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig4_pred_shifts(run_dir: Path, out: Path, y: np.ndarray, init: np.ndarray, final: np.ndarray) -> None:
    style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    delta = final.astype(int) - init.astype(int)
    ax = axes[0]
    bins = np.arange(-3, 4) - 0.5
    ax.hist(delta, bins=bins, color="#1f4e79", edgecolor="white")
    ax.axvline(0, color="#c62828", ls="--")
    ax.set_xlabel("final_pred − initial_pred")
    ax.set_ylabel("# examples")
    ax.set_title(f"Prediction shifts (changed={(init != final).sum()} / {len(y)})")
    ax.text(
        0.98,
        0.95,
        f"down={(delta < 0).sum()}   up={(delta > 0).sum()}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax = axes[1]
    pairs = [(4, 5), (5, 4), (4, 3), (3, 4), (5, 3), (3, 5)]
    init_c = [int(np.sum((y == a) & (init == b))) for a, b in pairs]
    fin_c = [int(np.sum((y == a) & (final == b))) for a, b in pairs]
    labels = [f"{a}→{b}" for a, b in pairs]
    x = np.arange(len(pairs))
    w = 0.36
    ax.bar(x - w / 2, init_c, w, color="#90a4ae", label="initial")
    ax.bar(x + w / 2, fin_c, w, color="#1f4e79", label="final")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("count")
    ax.set_title("Key confusions (gold → pred)")
    ax.legend(fontsize=8)

    fig.suptitle("Phase 1 — how predictions moved (promotion, not demotion)", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig5_vs_lowvar(run_dir: Path, lowvar_dir: Optional[Path], out: Path, init_m: Dict, fin_m: Dict) -> None:
    style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # R_global deltas
    ax = axes[0]
    labels = ["Phase 1\nconstraint+global", "lowvar\nCVaR_bal+monitor"]
    p1_delta = fin_m["R_global"] - init_m["R_global"]
    low_delta = np.nan
    if lowvar_dir and (lowvar_dir / "analysis" / "analysis_summary.json").is_file():
        a = json.loads((lowvar_dir / "analysis" / "analysis_summary.json").read_text(encoding="utf-8"))
        tu = a.get("test_uniform") or {}
        if tu.get("initial") and tu.get("final"):
            low_delta = tu["final"]["R_global"] - tu["initial"]["R_global"]
    elif lowvar_dir and (lowvar_dir / "summary.json").is_file():
        # fallback from known FINDINGS numbers
        low_delta = 0.6713541666666667 - 0.7239583333333334

    vals = [p1_delta, low_delta]
    colors = ["#2e7d32" if v >= 0 else "#c62828" for v in vals]
    ax.bar([0, 1], vals, color=colors, width=0.55)
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_ylabel("Δ R_global (test)")
    ax.set_title("Transfer: final − initial")
    for i, v in enumerate(vals):
        if v == v:
            ax.text(i, v + (0.003 if v >= 0 else -0.008), f"{v:+.3f}", ha="center", fontsize=10)

    # shift direction
    ax = axes[1]
    y, init, final, _, _ = load_init_final(run_dir)
    d1 = final.astype(int) - init.astype(int)
    p1_down, p1_up = int((d1 < 0).sum()), int((d1 > 0).sum())
    low_down, low_up = 341, 12  # from FINDINGS_lowvar / raw analysis
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w / 2, [p1_down, low_down], w, label="pred ↓", color="#c62828")
    ax.bar(x + w / 2, [p1_up, low_up], w, label="pred ↑", color="#2e7d32")
    ax.set_xticks(x)
    ax.set_xticklabels(["Phase 1", "lowvar"])
    ax.set_ylabel("# test examples")
    ax.set_title("Direction of prediction changes")
    ax.legend(fontsize=8)

    fig.suptitle("Phase 1 vs failed lowvar objective", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=Path("results/E1_constraint_global/seed42_20260801_002853"),
    )
    ap.add_argument(
        "--lowvar-dir",
        type=Path,
        default=Path("results/E1_pred_profile_cvar_lowvar/seed42_20260730_114314"),
    )
    args = ap.parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else PKG_ROOT / args.run_dir
    lowvar = args.lowvar_dir if args.lowvar_dir.is_absolute() else PKG_ROOT / args.lowvar_dir
    out_dir = run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    y, init, final, cid, uid = load_init_final(run_dir)
    init_m = metric_bundle(init, y, cid, uid)
    fin_m = metric_bundle(final, y, cid, uid)

    fig1_trajectory(run_dir, out_dir / "fig1_evolution_trajectory.png")
    fig2_cycle_metrics(run_dir, out_dir / "fig2_cycle_metrics.png")
    fig3_test_comparison(run_dir, out_dir / "fig3_test_comparison.png", init_m, fin_m)
    fig4_pred_shifts(run_dir, out_dir / "fig4_pred_shifts.png", y, init, final)
    fig5_vs_lowvar(run_dir, lowvar if lowvar.is_dir() else None, out_dir / "fig5_vs_lowvar.png", init_m, fin_m)

    dump = {
        "initial": {k: (v if not isinstance(v, dict) else {str(a): b for a, b in v.items()}) for k, v in init_m.items()},
        "final": {k: (v if not isinstance(v, dict) else {str(a): b for a, b in v.items()}) for k, v in fin_m.items()},
        "delta_R_global": fin_m["R_global"] - init_m["R_global"],
        "figures": [
            "fig1_evolution_trajectory.png",
            "fig2_cycle_metrics.png",
            "fig3_test_comparison.png",
            "fig4_pred_shifts.png",
            "fig5_vs_lowvar.png",
        ],
    }
    (out_dir / "plot_metrics.json").write_text(json.dumps(dump, indent=2), encoding="utf-8")
    print(f"Wrote figures to {out_dir}")
    for f in dump["figures"]:
        print(" ", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

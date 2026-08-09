#!/usr/bin/env python3
"""Post-run analysis + plots for the low-variance E1 run (M17).

Reconstructs the full search trajectory from the surviving OpenEvolve checkpoints,
rescores initial/final test predictions in one uniform metric definition, and draws
the four figures that tell the run's story:

  analysis/fig1_evolution_trajectory.png   best-so-far D_select fitness per iteration,
                                           with the +-2SD noise band and consolidation points
  analysis/fig2_transfer.png               D_select gain vs val selection key vs test outcome
  analysis/fig3_test_comparison.png        initial vs final on test: metrics, clusters, classes
  analysis/fig4_inrun_signals.png          anchor trend, damage report, error triage mix

Usage: python scripts/analyze_lowvar_run.py --run-dir results/E1_pred_profile_cvar_lowvar/<run>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))

from prime.fitness.metrics import compute_metrics  # noqa: E402

NOISE_SD = 0.009  # measured for this objective (exp_objective_noise_sweep, w=50 bal)


# ---------------------------------------------------------------- data loading


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
        origin = (data.get("metadata") or {}).get("prime_origin") or {}
        out.append(
            {
                "id": data.get("id"),
                "iteration": int(data.get("iteration_found") or 0),
                "score": float(m["combined_score"]),
                "origin": origin.get("source"),
                "origin_cycle": origin.get("cycle"),
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
                best = max(best, p["score"])
        series.append(best)
    return series


def uniform_test_metrics(eval_dir: Path) -> Dict[str, Any]:
    pred = np.load(eval_dir / "ensemble_predictions.npy")
    labels = np.load(eval_dir / "labels.npy")
    users = np.load(eval_dir / "user_ids.npy")
    clusters = np.load(eval_dir / "cluster_ids.npy")
    return compute_metrics(
        pred,
        labels,
        users,
        cluster_ids=clusters,
        cvar_quantile=0.4,
        shrink_prior_weight=50.0,
        class_balanced=True,
    )


# ---------------------------------------------------------------------- plots


def fig1_trajectory(run_dir: Path, out: Path, summary: Dict[str, Any]) -> Dict[str, Any]:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x0 = 0
    info: Dict[str, Any] = {}
    colors = ["#1f77b4", "#2ca02c", "#9467bd"]
    for cyc in (1, 2, 3):
        progs = load_oe_programs(run_dir, cyc)
        if not progs:
            continue
        seeds = [p["score"] for p in progs if p["iteration"] == 0]
        series = best_so_far(progs)
        xs = list(range(x0, x0 + len(series) + 1))
        seed_level = max(seeds) if seeds else series[0]
        ys = [seed_level] + series
        ax.step(xs, ys, where="post", color=colors[cyc - 1], lw=2, label=f"cycle {cyc} best-so-far")
        ax.scatter(
            [x0 + p["iteration"] for p in progs if p["iteration"] > 0],
            [p["score"] for p in progs if p["iteration"] > 0],
            s=18,
            color=colors[cyc - 1],
            alpha=0.45,
        )
        ax.fill_between(xs, seed_level - 2 * NOISE_SD, seed_level + 2 * NOISE_SD,
                        color=colors[cyc - 1], alpha=0.10)
        if cyc == 1 and seeds:
            info["initial_d_select"] = min(seeds)
            ax.scatter([x0], [min(seeds)], marker="D", s=70, color="black", zorder=5,
                       label="initial prompt on D_select")
        info[f"c{cyc}_seed_best"] = seed_level
        info[f"c{cyc}_final_best"] = series[-1]
        x0 += len(series) + 1

    cons = [c.get("consolidation", {}) for c in summary.get("cycles", [])]
    cons_x = [9 + 8, 9 + 9 + 8]  # end of cycles 2 and 3
    for cx, c in zip(cons_x, [c for c in cons if c.get("changed")]):
        y = c.get("select_fitness")
        if y:
            ax.scatter([cx], [y], marker="x", s=90, color="red", zorder=5)
    ax.scatter([], [], marker="x", s=90, color="red", label="consolidation candidate")

    ax.set_xlabel("OpenEvolve iteration (3 cycles x 8)")
    ax.set_ylabel("D_select fitness (balanced shrunk cvar_lex)")
    ax.set_title("Search trajectory: one real jump in cycle 1, then nothing above the noise band")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig1_evolution_trajectory.png", dpi=150)
    plt.close(fig)
    return info


def fig2_transfer(out: Path, d_select: Dict[str, float], val_keys: List[List[float]],
                  test_init: Dict[str, Any], test_final: Dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    groups = [
        ("D_select\nfitness", d_select["initial"], d_select["final"]),
        ("val selection key\n(balanced shrunk CVaR)", val_keys[0][0], val_keys[-1][0]),
        ("test CVaR\n(balanced shrunk)", test_init["CVaR_cluster_balanced_shrunk"],
         test_final["CVaR_cluster_balanced_shrunk"]),
        ("test R_global", test_init["R_global"], test_final["R_global"]),
        ("test R_macro", test_init["R_macro"], test_final["R_macro"]),
    ]
    x = np.arange(len(groups))
    w = 0.35
    ax.bar(x - w / 2, [g[1] for g in groups], w, label="initial / cycle-1 state", color="#8ab4d8")
    ax.bar(x + w / 2, [g[2] for g in groups], w, label="final selected", color="#d88a8a")
    for i, (_, a, b) in enumerate(groups):
        d = b - a
        ax.annotate(f"{d:+.3f}", (i, max(a, b) + 0.012), ha="center",
                    fontsize=10, fontweight="bold",
                    color="#1a7a1a" if d > 0 else "#b02020")
    ax.set_xticks(x, [g[0] for g in groups], fontsize=9)
    ax.set_ylim(0.4, 0.85)
    ax.set_title("Where the gain went: D_select up, val flat-ish, test down (overfitting via selection)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig2_transfer.png", dpi=150)
    plt.close(fig)


def fig3_test(out: Path, init: Dict[str, Any], fin: Dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    m_keys = ["R_global", "R_macro", "CVaR_cluster", "CVaR_cluster_balanced_shrunk", "R_tail"]
    x = np.arange(len(m_keys))
    axes[0].bar(x - 0.18, [init[k] for k in m_keys], 0.36, label="initial", color="#8ab4d8")
    axes[0].bar(x + 0.18, [fin[k] for k in m_keys], 0.36, label="final", color="#d88a8a")
    axes[0].set_xticks(x, ["R_global", "R_macro", "CVaR raw", "CVaR bal.shr", "R_tail"],
                       fontsize=8, rotation=15)
    axes[0].set_title("Test metrics (240 users, paired)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    cids = sorted(init["cluster_accuracies"], key=int)
    x = np.arange(len(cids))
    axes[1].bar(x - 0.18, [init["cluster_accuracies"][c] for c in cids], 0.36,
                label="initial", color="#8ab4d8")
    axes[1].bar(x + 0.18, [fin["cluster_accuracies"][c] for c in cids], 0.36,
                label="final", color="#d88a8a")
    for i, c in enumerate(cids):
        d = fin["cluster_accuracies"][c] - init["cluster_accuracies"][c]
        axes[1].annotate(f"{d:+.2f}", (i, max(init["cluster_accuracies"][c],
                                              fin["cluster_accuracies"][c]) + 0.01),
                         ha="center", fontsize=9, color="#b02020")
    axes[1].set_xticks(x, [f"c{c}" for c in cids])
    axes[1].set_title("Per cluster: every cluster worse (uniform damage)")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    classes = sorted(init["accuracy_per_class"], key=int)
    x = np.arange(len(classes))
    axes[2].bar(x - 0.18, [init["accuracy_per_class"][c] for c in classes], 0.36,
                label="initial", color="#8ab4d8")
    axes[2].bar(x + 0.18, [fin["accuracy_per_class"][c] for c in classes], 0.36,
                label="final", color="#d88a8a")
    for i, c in enumerate(classes):
        d = fin["accuracy_per_class"][c] - init["accuracy_per_class"][c]
        axes[2].annotate(f"{d:+.2f}", (i, max(init["accuracy_per_class"][c],
                                              fin["accuracy_per_class"][c]) + 0.01),
                         ha="center", fontsize=9,
                         color="#1a7a1a" if d > 0 else "#b02020")
    axes[2].set_xticks(x, [f"{c}*" for c in classes])
    axes[2].set_title("Per gold class")
    axes[2].legend()
    axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle("Initial vs final prompt on the same 240 test users "
                 "(R_global -0.053 CI[-0.074,-0.032] p<0.001; McNemar broke 216 / fixed 115)")
    fig.tight_layout()
    fig.savefig(out / "fig3_test_comparison.png", dpi=150)
    plt.close(fig)


def fig4_signals(run_dir: Path, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

    # anchor gate
    accs, labels = [], []
    for cyc in (1, 2, 3):
        p = run_dir / f"al_iter_{cyc}" / "anchor_gate.json"
        if not p.is_file():
            continue
        j = json.loads(p.read_text(encoding="utf-8"))
        if not accs and j.get("acc_before") is not None:
            accs.append(float(j["acc_before"]))
            labels.append("entry")
        if j.get("acc_after") is not None:
            accs.append(float(j["acc_after"]))
            labels.append(
                f"after c{cyc}\n({j.get('n_improved', 0)}+/{j.get('n_worsened', 0)}-, "
                f"p={j.get('mcnemar_p_one_sided', float('nan')):.3f})"
            )
    axes[0].plot(range(len(accs)), accs, "o-", color="#b02020", lw=2)
    for i, a in enumerate(accs):
        axes[0].annotate(f"{a:.2f}", (i, a + 0.008), ha="center", fontsize=9)
    axes[0].set_xticks(range(len(accs)), labels, fontsize=8)
    axes[0].set_ylim(min(accs) - 0.04, max(accs) + 0.04)
    axes[0].set_title("Anchor accuracy (monitor mode): C1 winner\nbroke 15/100 (13 of them gold 4-5), 0 improved")
    axes[0].grid(alpha=0.3)

    # damage report
    cycles, fixed, broke = [2, 3], [0, 1], [4, 3]
    x = np.arange(len(cycles))
    axes[1].bar(x - 0.18, fixed, 0.36, label="fixed", color="#79b879")
    axes[1].bar(x + 0.18, broke, 0.36, label="broke", color="#d88a8a")
    axes[1].set_xticks(x, [f"cycle {c}" for c in cycles])
    axes[1].set_title("Damage report on comparable batch rows:\nthe mutator was trading, not improving")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    # error triage mix
    cyc_x = np.arange(3)
    systematic = [5, 6, 2]
    contested = [5, 10, 14]
    axes[2].bar(cyc_x, systematic, 0.5, label="systematic (fixable)", color="#e0a030")
    axes[2].bar(cyc_x, contested, 0.5, bottom=systematic, label="contested (noise)", color="#b0b0b0")
    for i in cyc_x:
        tot = systematic[i] + contested[i]
        axes[2].annotate(f"{systematic[i] / tot:.0%}", (i, tot + 0.3), ha="center", fontsize=9)
    axes[2].set_xticks(cyc_x, [f"cycle {c}" for c in (1, 2, 3)])
    axes[2].set_title("Batch error triage: systematic share collapses\nafter cycle 1 (easy wins consumed)")
    axes[2].legend(fontsize=8)
    axes[2].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / "fig4_inrun_signals.png", dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    run_dir = args.run_dir
    out = run_dir / "analysis"
    out.mkdir(exist_ok=True)

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    val_keys = [c["val_selection_key"] for c in summary["cycles"] if c.get("val_selection_key")]

    test_init = uniform_test_metrics(run_dir / "evals" / "initial_prompt")
    test_final = uniform_test_metrics(run_dir / "evals" / "eval_test")

    info = fig1_trajectory(run_dir, out, summary)
    d_sel = {
        "initial": info.get("initial_d_select", float("nan")),
        "final": info.get("c3_final_best", float("nan")),
    }
    fig2_transfer(out, d_sel, val_keys, test_init, test_final)
    fig3_test(out, test_init, test_final)
    fig4_signals(run_dir, out)

    report = {
        "d_select": {**info, "gain": d_sel["final"] - d_sel["initial"], "noise_sd": NOISE_SD},
        "val_selection_keys": val_keys,
        "test_uniform": {
            "initial": {k: test_init[k] for k in
                        ("R_global", "R_macro", "CVaR_cluster", "CVaR_cluster_balanced_shrunk")},
            "final": {k: test_final[k] for k in
                      ("R_global", "R_macro", "CVaR_cluster", "CVaR_cluster_balanced_shrunk")},
        },
    }
    (out / "analysis_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nfigures in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

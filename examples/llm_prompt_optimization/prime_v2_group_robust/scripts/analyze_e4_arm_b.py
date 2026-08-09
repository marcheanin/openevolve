#!/usr/bin/env python3
"""E4 Arm B (min_group_lex) evolution plots: clusters × sets.

Usage:
  python scripts/analyze_e4_arm_b.py \
    --run-dir results/E4_civilcomments_arm_b_min_group_2x20/seed42_20260807_115056
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

from prime.data.civilcomments_loader import ORACLE_GROUP_NAMES  # noqa: E402
from prime.fitness.metrics import cluster_accuracies, per_class_accuracy  # noqa: E402

GROUP_NAMES = list(ORACLE_GROUP_NAMES)
CYCLE_COLORS = {1: "#1565c0", 2: "#ef6c00", 3: "#2e7d32"}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _group_name(gid: int) -> str:
    return GROUP_NAMES[gid] if 0 <= gid < len(GROUP_NAMES) else f"g{gid}"


def _oe_staging(run: Path, cycle: int) -> Optional[Path]:
    path_file = run / f"al_iter_{cycle}" / "openevolve_output_path.txt"
    if not path_file.is_file():
        return None
    staging = Path(path_file.read_text(encoding="utf-8").strip())
    return staging if staging.is_dir() else None


def _oe_programs(run: Path, cycle: int) -> List[Dict[str, Any]]:
    """All OE programs from the latest checkpoint of a cycle (D_select metrics)."""
    staging = _oe_staging(run, cycle)
    if staging is None:
        return []
    cps = sorted(
        (staging / "checkpoints").glob("checkpoint_*"),
        key=lambda d: int(d.name.split("_")[-1]),
    )
    if not cps:
        return []
    out: List[Dict[str, Any]] = []
    for f in (cps[-1] / "programs").glob("*.json"):
        try:
            data = _load_json(f)
        except (OSError, ValueError):
            continue
        m = data.get("metrics") or {}
        if "combined_score" not in m:
            continue
        cluster = {
            int(k.rsplit("_", 1)[-1]): float(v)
            for k, v in m.items()
            if k.startswith("cluster_acc_")
        }
        out.append(
            {
                "id": str(data.get("id") or f.stem),
                "iteration": int(data.get("iteration_found") or 0),
                "score": float(m["combined_score"]),
                "R_global": float(m.get("R_global", m["combined_score"])),
                "R_worst_group": (
                    float(m["R_worst_group"]) if m.get("R_worst_group") is not None else None
                ),
                "cluster": cluster,
                "metrics": m,
            }
        )
    return sorted(out, key=lambda r: (r["iteration"], -r["score"]))


def _best_so_far_series(
    progs: List[Dict[str, Any]], key: str
) -> Tuple[List[int], List[float]]:
    best: Dict[int, float] = {}
    cur = -1.0
    for p in sorted(progs, key=lambda r: r["iteration"]):
        val = p.get(key)
        if val is None:
            continue
        cur = max(cur, float(val))
        best[p["iteration"]] = cur
    xs = sorted(best)
    return xs, [best[i] for i in xs]


def _champion_cluster_traj(
    progs: List[Dict[str, Any]],
) -> Tuple[List[int], Dict[int, List[float]], List[float]]:
    """At each iteration, cluster_acc of the then-best program (by score)."""
    by_iter: Dict[int, Dict[str, Any]] = {}
    for p in progs:
        it = p["iteration"]
        if it not in by_iter or p["score"] > by_iter[it]["score"]:
            by_iter[it] = p
    # Running champion across iterations
    xs = sorted(by_iter)
    champ_clusters: Dict[int, List[float]] = {}
    scores: List[float] = []
    cur_best: Optional[Dict[str, Any]] = None
    for it in xs:
        cand = by_iter[it]
        if cur_best is None or cand["score"] > cur_best["score"]:
            cur_best = cand
        scores.append(float(cur_best["score"]))
        for gid, acc in (cur_best.get("cluster") or {}).items():
            champ_clusters.setdefault(int(gid), []).append(float(acc))
    # Align missing groups with nan
    n = len(xs)
    for gid in list(champ_clusters):
        if len(champ_clusters[gid]) != n:
            # rebuild properly
            pass
    # Rebuild cleanly
    champ_clusters = {}
    scores = []
    cur_best = None
    for it in xs:
        cand = by_iter[it]
        if cur_best is None or cand["score"] > cur_best["score"]:
            cur_best = cand
        assert cur_best is not None
        scores.append(float(cur_best["score"]))
        for gid, acc in (cur_best.get("cluster") or {}).items():
            champ_clusters.setdefault(int(gid), [])
        for gid in champ_clusters:
            champ_clusters[gid].append(float((cur_best.get("cluster") or {}).get(gid, np.nan)))
    return xs, champ_clusters, scores


def fig_cycle_sets(summary: Dict[str, Any], out: Path) -> None:
    cycles = summary["cycles"]
    xs = [c["cycle"] for c in cycles]
    entry = [c["fitness"] for c in cycles]
    evo = [c.get("best_evo_score") for c in cycles]
    heir = [c.get("pareto", {}).get("heir_select_fitness") for c in cycles]
    val_primary = [c["val_selection_key"][0] for c in cycles]
    val_secondary = [
        c["val_selection_key"][1] if len(c.get("val_selection_key") or []) > 1 else None
        for c in cycles
    ]
    best_key = (summary.get("best_selection_key") or [None])[0]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    ax.plot(xs, entry, "s--", color="#78909c", label="cycle-entry (D_select lex)")
    ax.plot(xs, evo, "o-", color="#c62828", lw=2, label="OE best (D_select lex)")
    ax.plot(xs, heir, "D-", color="#2e7d32", lw=2, label="heir after gate (D_select)")
    ax.set_xlabel("AL cycle")
    ax.set_ylabel("min_group_lex / fitness")
    ax.set_title("D_select trajectory")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(xs)

    ax = axes[1]
    ax.plot(xs, val_primary, "o-", color="#6a1b9a", lw=2, label="val primary key")
    if any(v is not None for v in val_secondary):
        ax.plot(
            xs,
            val_secondary,
            "s--",
            color="#00838f",
            lw=1.5,
            label="val secondary (tie-break)",
        )
    if best_key is not None:
        ax.axhline(
            best_key,
            color="#6a1b9a",
            ls=":",
            alpha=0.7,
            label=f"selected primary={best_key:.3f}",
        )
    ax.set_xlabel("AL cycle")
    ax.set_ylabel("val selection key")
    ax.set_title("Val selection (final = lexicographic best)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(xs)
    fig.tight_layout()
    fig.savefig(out / "fig1_cycle_sets.png", dpi=150)
    plt.close(fig)


def fig_oe_scalar(run: Path, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, key, ylab, title in (
        (axes[0], "score", "best-so-far lex fitness", "OE D_select fitness"),
        (axes[1], "R_worst_group", "best-so-far R_worst_group", "OE D_select worst-group"),
    ):
        for cycle, col in CYCLE_COLORS.items():
            progs = _oe_programs(run, cycle)
            if not progs:
                continue
            xs, ys = _best_so_far_series(progs, key)
            if not xs:
                continue
            ax.plot(xs, ys, "o-", color=col, lw=2, label=f"cycle {cycle}")
        ax.set_xlabel("OE iteration_found")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig2_oe_scalar.png", dpi=150)
    plt.close(fig)


def fig_oe_cluster_lines(run: Path, out: Path) -> None:
    """Champion-on-D_select per-group accuracy over OE iterations."""
    cycles = [c for c in (1, 2, 3) if _oe_programs(run, c)]
    if not cycles:
        return
    fig, axes = plt.subplots(1, len(cycles), figsize=(5.5 * len(cycles), 4.6), sharey=True)
    if len(cycles) == 1:
        axes = [axes]
    cmap = plt.get_cmap("tab10")
    for ax, cycle in zip(axes, cycles):
        progs = _oe_programs(run, cycle)
        xs, clusters, scores = _champion_cluster_traj(progs)
        for gid in sorted(clusters):
            ax.plot(
                xs,
                clusters[gid],
                "-",
                color=cmap(gid % 10),
                lw=1.6,
                label=_group_name(gid),
            )
        ax.set_xlabel("OE iteration")
        ax.set_title(f"C{cycle}: champion cluster_acc (D_select)")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.3, 1.0)
    axes[0].set_ylabel("accuracy")
    axes[-1].legend(fontsize=7, loc="lower right", ncol=2)
    fig.suptitle("Per-group accuracy of current OE champion on D_select", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "fig3_oe_cluster_lines.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_oe_cluster_heatmap(run: Path, out: Path) -> None:
    """Heatmap: groups × OE iterations for champion cluster_acc, one panel per cycle."""
    cycles = [c for c in (1, 2, 3) if _oe_programs(run, c)]
    if not cycles:
        return
    fig, axes = plt.subplots(1, len(cycles), figsize=(5.8 * len(cycles), 4.8))
    if len(cycles) == 1:
        axes = [axes]
    for ax, cycle in zip(axes, cycles):
        progs = _oe_programs(run, cycle)
        xs, clusters, _scores = _champion_cluster_traj(progs)
        gids = sorted(clusters)
        mat = np.array([clusters[g] for g in gids], dtype=float)
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0.45, vmax=0.95)
        ax.set_yticks(range(len(gids)))
        ax.set_yticklabels([_group_name(g) for g in gids], fontsize=8)
        # show a subset of x ticks
        step = max(1, len(xs) // 8)
        tick_idx = list(range(0, len(xs), step))
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([str(xs[i]) for i in tick_idx], fontsize=8)
        ax.set_xlabel("OE iteration")
        ax.set_title(f"C{cycle} champion · D_select")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Heatmap: group accuracy of OE champion over iterations", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "fig4_oe_cluster_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _split_group_accs(run: Path, split: str) -> Tuple[Dict[int, float], Dict[int, int], float]:
    root = run / "evals" / f"eval_{split}"
    y = np.load(root / "labels.npy")
    pred = np.load(root / "ensemble_predictions.npy")
    cid = np.load(root / "cluster_ids.npy")
    accs = {int(k): float(v) for k, v in cluster_accuracies(pred, y, cid).items()}
    ns = {gid: int((cid == gid).sum()) for gid in accs}
    return accs, ns, float((pred == y).mean())


def fig_sets_groups(run: Path, out: Path) -> Dict[str, Any]:
    """Grouped bars: per-group accuracy on D_select champions vs val/test/seed."""
    # D_select: OE champions from C1 / C2 (from program metrics)
    series: Dict[str, Dict[int, float]] = {}
    ns_ref: Dict[int, int] = {}

    for cycle in (1, 2):
        progs = _oe_programs(run, cycle)
        if not progs:
            continue
        champ = max(progs, key=lambda p: p["score"])
        series[f"D_select C{cycle} OE-best"] = {
            int(k): float(v) for k, v in (champ.get("cluster") or {}).items()
        }

    # val / test from saved evals (final selected prompt). Prefer test n= for labels.
    for split, label in (("validation", "val (final prompt)"), ("test", "test (final prompt)")):
        root = run / "evals" / f"eval_{split}"
        if (root / "ensemble_predictions.npy").is_file():
            accs, ns, _g = _split_group_accs(run, split)
            series[label] = accs
            if split == "test" or not ns_ref:
                ns_ref = ns

    # seed from noise report
    noise = run / "evals" / "test_noise" / "noise_report.json"
    if noise.is_file():
        nr = _load_json(noise)
        seed_ca = (nr.get("seed") or {}).get("cluster_accuracies") or {}
        if seed_ca:
            series["test seed"] = {int(k): float(v) for k, v in seed_ca.items()}
        # mean of noise repeats for final
        runs = nr.get("runs") or []
        if runs:
            # optional: first repeat cluster accs if present
            r0 = runs[0] if isinstance(runs[0], dict) else {}
            ca = r0.get("cluster_accuracies")
            if ca:
                series["test final r0"] = {int(k): float(v) for k, v in ca.items()}

    all_gids = sorted({g for d in series.values() for g in d})
    if not all_gids:
        return {}

    names = [_group_name(g) for g in all_gids]
    n_series = len(series)
    x = np.arange(len(all_gids))
    width = 0.8 / max(n_series, 1)
    fig, ax = plt.subplots(figsize=(13, 5.2))
    colors = ["#1565c0", "#ef6c00", "#6a1b9a", "#2e7d32", "#c62828", "#00838f"]
    for i, (label, accs) in enumerate(series.items()):
        vals = [accs.get(g, np.nan) for g in all_gids]
        ax.bar(
            x + i * width - 0.4 + width / 2,
            vals,
            width=width * 0.95,
            label=label,
            color=colors[i % len(colors)],
        )
    xticklabels = []
    for g, name in zip(all_gids, names):
        n = ns_ref.get(g)
        xticklabels.append(f"{name}\nn={n}" if n is not None else name)
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, fontsize=8)
    ax.set_ylim(0.3, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title("Per-group accuracy across sets / prompts")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig5_sets_groups.png", dpi=150)
    plt.close(fig)

    # Compact delta: final test − seed
    if "test (final prompt)" in series and "test seed" in series:
        fig, ax = plt.subplots(figsize=(10, 4.2))
        fin = series["test (final prompt)"]
        seed = series["test seed"]
        deltas = [fin.get(g, np.nan) - seed.get(g, np.nan) for g in all_gids]
        colors_d = ["#2e7d32" if d >= 0 else "#c62828" for d in deltas]
        ax.bar(range(len(all_gids)), deltas, color=colors_d)
        ax.axhline(0, color="black", lw=1)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(xticklabels, fontsize=8)
        ax.set_ylabel("Δ accuracy (final − seed)")
        ax.set_title("Test group deltas vs same-day seed")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "fig6_test_delta_vs_seed.png", dpi=150)
        plt.close(fig)

    # class bars for test final vs seed if available
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    y = np.load(run / "evals" / "eval_test" / "labels.npy")
    pred = np.load(run / "evals" / "eval_test" / "ensemble_predictions.npy")
    pc = per_class_accuracy(pred, y)
    axes[0].bar(
        [0, 1],
        [pc.get(0, 0), pc.get(1, 0)],
        color=["#546e7a", "#bf360c"],
    )
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(
        [f"non-toxic\nn={(y == 0).sum()}", f"toxic\nn={(y == 1).sum()}"]
    )
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title(f"Test final classes (R={float((pred == y).mean()):.3f})")
    axes[0].grid(True, axis="y", alpha=0.3)

    if noise.is_file():
        nr = _load_json(noise)
        seed_pc = (nr.get("seed") or {}).get("accuracy_per_class") or {}
        axes[1].bar(
            [0, 1],
            [float(seed_pc.get("0", seed_pc.get(0, 0))), float(seed_pc.get("1", seed_pc.get(1, 0)))],
            color=["#90a4ae", "#e65100"],
        )
        axes[1].set_xticks([0, 1])
        axes[1].set_xticklabels(["non-toxic", "toxic"])
        axes[1].set_ylim(0, 1.05)
        axes[1].set_title(f"Test seed classes (R={nr['seed'].get('R_global', float('nan')):.3f})")
        axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig7_test_classes.png", dpi=150)
    plt.close(fig)

    return {
        "series": {k: {str(g): v for g, v in d.items()} for k, d in series.items()},
        "ns_ref": {str(k): v for k, v in ns_ref.items()},
    }


def fig_scatter_tradeoff(run: Path, out: Path) -> None:
    """All OE candidates: R_global vs R_worst_group, colored by cycle."""
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for cycle, col in CYCLE_COLORS.items():
        progs = _oe_programs(run, cycle)
        if not progs:
            continue
        xs = [p["R_global"] for p in progs]
        ys = [p["R_worst_group"] if p["R_worst_group"] is not None else np.nan for p in progs]
        ax.scatter(xs, ys, c=col, alpha=0.65, s=36, label=f"C{cycle} OE")
        # mark champion
        champ = max(progs, key=lambda p: p["score"])
        ax.scatter(
            [champ["R_global"]],
            [champ["R_worst_group"]],
            c=col,
            s=140,
            marker="*",
            edgecolors="black",
            linewidths=0.6,
            zorder=5,
        )
    ax.set_xlabel("R_global (D_select)")
    ax.set_ylabel("R_worst_group (D_select)")
    ax.set_title("OE candidates: global vs worst-group (★ = cycle champion)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig8_tradeoff_scatter.png", dpi=150)
    plt.close(fig)


def write_results(
    run: Path,
    out: Path,
    summary: Dict[str, Any],
    groups: Dict[str, Any],
) -> None:
    ft = summary.get("final_test") or {}
    lines = [
        "# E4 Arm B analysis — cluster / set evolution",
        "",
        f"Run: `{run}`",
        f"Fitness: `{summary.get('fitness_mode')}` · geometry: `{summary.get('cluster_geometry')}`",
        "",
        "## Final test",
        "",
        f"- R_global: **{ft.get('R_global')}**",
        f"- R_worst_group: **{ft.get('R_worst_group')}**",
        f"- Selected val key: `{summary.get('best_selection_key')}`",
        "",
        "## Cycles",
        "",
        "| cycle | entry D_select | OE best | heir | val key |",
        "|---:|---:|---:|---:|---|",
    ]
    for c in summary.get("cycles") or []:
        lines.append(
            f"| {c['cycle']} | {c.get('fitness'):.4f} | {c.get('best_evo_score'):.4f} | "
            f"{(c.get('pareto') or {}).get('heir_select_fitness'):.4f} | "
            f"`{c.get('val_selection_key')}` |"
        )
    lines += [
        "",
        "## Figures",
        "",
        "- `fig1_cycle_sets.png` — D_select vs val selection across AL cycles",
        "- `fig2_oe_scalar.png` — OE best-so-far fitness & R_worst_group",
        "- `fig3_oe_cluster_lines.png` — champion per-group accuracy over OE iters",
        "- `fig4_oe_cluster_heatmap.png` — same as heatmap",
        "- `fig5_sets_groups.png` — groups × {D_select C1/C2, val, test, seed}",
        "- `fig6_test_delta_vs_seed.png` — test Δ vs same-day seed",
        "- `fig7_test_classes.png` — toxic/non-toxic final vs seed",
        "- `fig8_tradeoff_scatter.png` — R_global vs R_worst_group candidates",
        "",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "groups_series.json").write_text(
        json.dumps(groups, indent=2), encoding="utf-8"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=PKG_ROOT
        / "results"
        / "E4_civilcomments_arm_b_min_group_2x20"
        / "seed42_20260807_115056",
    )
    args = ap.parse_args()
    run = args.run_dir.resolve()
    out = run / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    summary = _load_json(run / "summary.json")

    fig_cycle_sets(summary, out)
    fig_oe_scalar(run, out)
    fig_oe_cluster_lines(run, out)
    fig_oe_cluster_heatmap(run, out)
    groups = fig_sets_groups(run, out)
    fig_scatter_tradeoff(run, out)
    write_results(run, out, summary, groups)

    print(f"wrote {out}")
    for p in sorted(out.glob("fig*.png")):
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Analyze completed E4 arm-A (global) run: plots + gate audit + test report.

Usage:
  python scripts/analyze_e4_arm_a.py \
    --run-dir results/E4_civilcomments_arm_a_global/seed42_20260805_104411
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


def _load(run: Path) -> Dict[str, Any]:
    return json.loads((run / "summary.json").read_text(encoding="utf-8"))


def _oe_programs(run: Path, cycle: int) -> List[Dict[str, Any]]:
    path_file = run / f"al_iter_{cycle}" / "openevolve_output_path.txt"
    if not path_file.is_file():
        return []
    staging = Path(path_file.read_text(encoding="utf-8").strip())
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
                "id": data.get("id"),
                "iteration": int(data.get("iteration_found") or 0),
                "score": float(m["combined_score"]),
                "R_global": float(m.get("R_global", m["combined_score"])),
                "R_macro": m.get("R_macro"),
                "mean_kappa": m.get("mean_kappa"),
                "R_worst_group": m.get("R_worst_group"),
                "metrics": m,
            }
        )
    return sorted(out, key=lambda r: (r["iteration"], -r["score"]))


def _gate_events(run: Path) -> List[Dict[str, Any]]:
    out = []
    for line in (run / "smoke_trace.jsonl").read_text(encoding="utf-8").splitlines():
        o = json.loads(line)
        if o.get("stage_id") == "10b_anchor_gate":
            d = dict(o.get("details") or {})
            d["cycle"] = d.get("cycle")
            out.append(d)
    return out


def _suspicious(m: Dict[str, Any]) -> List[str]:
    flags = []
    rg = float(m.get("R_global") or m.get("combined_score") or 0)
    rm = m.get("R_macro")
    k = m.get("mean_kappa")
    if rm is not None and rg - float(rm) > 0.25:
        flags.append(f"R_global−R_macro={rg - float(rm):+.2f}")
    if k is not None and float(k) < 0.05 and rg > 0.85:
        flags.append(f"mean_kappa={float(k):.3f} near 0 at high R_global")
    rwb = m.get("R_worst_group")
    if rwb is not None and rg - float(rwb) > 0.25:
        flags.append(f"R_global−R_worst_group={rg - float(rwb):+.2f}")
    return flags


def fig_cycles(run: Path, summary: Dict[str, Any], out: Path) -> None:
    cycles = summary["cycles"]
    xs = [c["cycle"] for c in cycles]
    entry = [c["fitness"] for c in cycles]
    evo = [c.get("best_evo_score") for c in cycles]
    heir = [c.get("pareto", {}).get("heir_select_fitness") for c in cycles]
    val = [c["val_selection_key"][0] for c in cycles]
    best_key = summary.get("best_selection_key", [None])[0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(xs, entry, "s--", color="#78909c", label="cycle-entry fitness (D_select)")
    ax.plot(xs, evo, "o-", color="#c62828", lw=2, label="OE best on D_select")
    ax.plot(xs, heir, "D-", color="#2e7d32", lw=2, label="heir after gate (D_select)")
    ax.set_xlabel("AL cycle")
    ax.set_ylabel("R_global / fitness")
    ax.set_title("Arm A: D_select scores")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(xs)

    ax = axes[1]
    ax.plot(xs, val, "o-", color="#6a1b9a", lw=2, label="val selection key")
    if best_key is not None:
        ax.axhline(best_key, color="#6a1b9a", ls=":", alpha=0.7, label=f"selected key={best_key:.4f}")
    ax.set_xlabel("AL cycle")
    ax.set_ylabel("val key (R_macro)")
    ax.set_title("Cross-cycle selection (final = argmax val)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(xs)
    fig.tight_layout()
    fig.savefig(out / "fig1_cycle_trajectory.png", dpi=140)
    plt.close(fig)


def fig_oe_iters(run: Path, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    colors = ["#1565c0", "#ef6c00", "#2e7d32"]
    for c, col in zip((1, 2, 3), colors):
        progs = _oe_programs(run, c)
        if not progs:
            continue
        # best-so-far by iteration
        best = {}
        cur = -1.0
        for p in sorted(progs, key=lambda r: r["iteration"]):
            cur = max(cur, p["score"])
            best[p["iteration"]] = cur
        xs = sorted(best)
        ax.plot(xs, [best[i] for i in xs], "o-", color=col, lw=2, label=f"cycle {c}")
    ax.set_xlabel("OE iteration_found")
    ax.set_ylabel("best-so-far combined_score")
    ax.set_title("OpenEvolve trajectory (island best-so-far)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig2_oe_iterations.png", dpi=140)
    plt.close(fig)


def _gate_attempts(run: Path) -> List[Dict[str, Any]]:
    """Per-candidate gate attempts from cycle artifacts (includes rejects)."""
    rows: List[Dict[str, Any]] = []
    for cycle in (1, 2, 3):
        path = run / f"al_iter_{cycle}" / "anchor_gate.json"
        if not path.is_file():
            continue
        g = json.loads(path.read_text(encoding="utf-8"))
        for r in g.get("rejected_before_accept") or []:
            rows.append(
                {
                    "cycle": cycle,
                    "accepted": False,
                    "drop": float(r.get("drop") or 0),
                    "reason": str(r.get("reason") or "rejected"),
                    "front_rank": r.get("front_rank"),
                }
            )
        for r in g.get("all_rejected") or []:
            rows.append(
                {
                    "cycle": cycle,
                    "accepted": False,
                    "drop": float(r.get("drop") or 0),
                    "reason": str(r.get("reason") or "rejected"),
                    "front_rank": r.get("front_rank"),
                }
            )
        if g.get("accepted") is True:
            rows.append(
                {
                    "cycle": cycle,
                    "accepted": True,
                    "drop": float(g.get("drop") or 0),
                    "reason": str(g.get("reason") or "accepted"),
                    "front_rank": g.get("front_rank"),
                }
            )
    return rows


def fig_gate(run: Path, out: Path) -> None:
    attempts = _gate_attempts(run)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    drops = [a["drop"] for a in attempts]
    colors = ["#2e7d32" if a["accepted"] else "#c62828" for a in attempts]
    labels = [
        f"C{a['cycle']} r{a.get('front_rank', '?')}\n{a['reason'][:16]}" for a in attempts
    ]
    ax.bar(range(len(drops)), drops, color=colors)
    ax.axhline(0.02, color="black", ls="--", lw=1, label="δ=0.02")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("anchor accuracy drop")
    ax.set_title("Anchor gate attempts (green=accept, red=reject)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig3_anchor_gate.png", dpi=140)
    plt.close(fig)


def fig_test_groups(run: Path, out: Path) -> None:
    y = np.load(run / "evals" / "eval_test" / "labels.npy")
    pred = np.load(run / "evals" / "eval_test" / "ensemble_predictions.npy")
    cid = np.load(run / "evals" / "eval_test" / "cluster_ids.npy")
    accs = cluster_accuracies(pred, y, cid)
    names = []
    vals = []
    ns = []
    for gid in sorted(accs):
        names.append(GROUP_NAMES[gid] if gid < len(GROUP_NAMES) else f"g{gid}")
        vals.append(accs[gid])
        ns.append(int((cid == gid).sum()))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    bars = ax.bar(range(len(vals)), vals, color="#455a64")
    worst = int(np.argmin(vals))
    bars[worst].set_color("#c62828")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"{n}\nn={ns[i]}" for i, n in enumerate(names)], fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title(f"Test per-oracle-group (R_global={float((pred == y).mean()):.3f})")
    ax.axhline(float((pred == y).mean()), color="#1565c0", ls=":", label="global")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    pc = per_class_accuracy(pred, y)
    ax.bar([0, 1], [pc.get(0, 0), pc.get(1, 0)], color=["#546e7a", "#bf360c"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"non-toxic\nn={(y == 0).sum()}", f"toxic\nn={(y == 1).sum()}"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Test per-class accuracy")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig4_test_groups.png", dpi=140)
    plt.close(fig)
    return {
        "R_global": float((pred == y).mean()),
        "R_worst_group": float(min(accs.values())) if accs else None,
        "per_group": {names[i]: {"acc": vals[i], "n": ns[i]} for i in range(len(names))},
        "per_class": {str(k): float(v) for k, v in pc.items()},
    }


def audit_rejected(run: Path) -> List[Dict[str, Any]]:
    """High-fitness OE programs that gate or selection refused."""
    rows = []
    # Same degenerate 0.883 program: rejected in C1 and again as C2 front_rank 0
    pid = "8eb20d2e-4ca6-48eb-abe1-ff1a37f9b650"
    hits = list(run.rglob(f"{pid}.json"))
    if hits:
        m = json.loads(hits[0].read_text(encoding="utf-8")).get("metrics") or {}
        rows.append(
            {
                "program_id": pid,
                "note": (
                    "OE score 0.883; rejected twice (C1 all_rejected + C2 rejected_before_accept "
                    "front_rank=0) with anchor drop 0.15 / McNemar p=0"
                ),
                "combined_score": m.get("combined_score"),
                "R_global": m.get("R_global"),
                "R_macro": m.get("R_macro"),
                "mean_kappa": m.get("mean_kappa"),
                "R_worst_group": m.get("R_worst_group"),
                "suspicious": _suspicious(m),
                "verdict": "CORRECTLY REJECTED (majority-class collapse on D_select)",
            }
        )
    # C3 0.9 accepted by gate but lost on val
    hits = list((run / "seed_c4" / "programs").glob("9b5d0a96*.json"))
    if hits:
        m = json.loads(hits[0].read_text(encoding="utf-8")).get("metrics") or {}
        rows.append(
            {
                "program_id": hits[0].stem,
                "note": "C3 OE best 0.90; gate accepted (tolerated_noise); val key worse → not final",
                "combined_score": m.get("combined_score"),
                "R_global": m.get("R_global"),
                "R_macro": m.get("R_macro"),
                "mean_kappa": m.get("mean_kappa"),
                "R_worst_group": m.get("R_worst_group"),
                "suspicious": _suspicious(m),
                "verdict": "GATE OK; VAL deselected (C2 has higher val key)",
            }
        )
    return rows


def write_results(run: Path, out: Path, test: Dict[str, Any], audit: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    best_key = summary.get("best_selection_key", [None])[0]
    sel_cycle = None
    for c in summary["cycles"]:
        if abs(c["val_selection_key"][0] - best_key) < 1e-9:
            sel_cycle = c["cycle"]
    lines = [
        "# E4 Arm A analysis",
        "",
        f"Run: `{run}`",
        f"Fitness: `{summary.get('fitness_mode')}` · geometry: `{summary.get('cluster_geometry')}`",
        "",
        "## Final test (n=800)",
        "",
        f"- **R_global: {test['R_global']:.4f}**",
        f"- **R_worst_group: {test['R_worst_group']:.4f}** (worst identity)",
        f"- R_macro / class tox recall: see fig4; toxic class is the weak side",
        f"- Selected by val key **{best_key:.4f}** → **cycle {sel_cycle}** heir (not C3's 0.90 D_select)",
        "",
        "### Per-group test",
        "",
        "| group | n | acc |",
        "|---|---:|---:|",
    ]
    for name, row in test["per_group"].items():
        lines.append(f"| {name} | {row['n']} | {row['acc']:.3f} |")
    lines += [
        "",
        "## Gate audit: strong-on-fitness rejects",
        "",
    ]
    for r in audit:
        lines.append(f"### `{r['program_id'][:8]}…`")
        lines.append(f"- {r['note']}")
        lines.append(
            f"- D_select: score={r['combined_score']} R_macro={r['R_macro']} "
            f"κ={r['mean_kappa']} R_worst_group={r['R_worst_group']}"
        )
        if r["suspicious"]:
            lines.append(f"- Red flags: {', '.join(r['suspicious'])}")
        lines.append(f"- **Verdict: {r['verdict']}**")
        lines.append("")
    ag = summary.get("anchor_gate") or {}
    lines += [
        "## Gate summary",
        "",
        f"- Evaluated {ag.get('n_evaluated')}, rejected {ag.get('n_rejected')} "
        f"({ag.get('rejection_rate')})",
        f"- Reasons: {ag.get('reasons')}",
        "",
        "## Figures",
        "",
        "- `fig1_cycle_trajectory.png`",
        "- `fig2_oe_iterations.png`",
        "- `fig3_anchor_gate.png`",
        "- `fig4_test_groups.png`",
        "",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=PKG_ROOT
        / "results"
        / "E4_civilcomments_arm_a_global"
        / "seed42_20260805_104411",
    )
    args = ap.parse_args()
    run = args.run_dir.resolve()
    out = run / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    summary = _load(run)
    fig_cycles(run, summary, out)
    fig_oe_iters(run, out)
    fig_gate(run, out)
    test = fig_test_groups(run, out)
    audit = audit_rejected(run)
    write_results(run, out, test, audit, summary)
    (out / "audit.json").write_text(
        json.dumps({"test": test, "audit": audit, "best_selection_key": summary.get("best_selection_key")}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {out}")
    print(f"TEST R_global={test['R_global']:.4f} R_worst_group={test['R_worst_group']:.4f}")
    for r in audit:
        print(f"  audit {r['program_id'][:8]}: {r['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

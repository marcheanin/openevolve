#!/usr/bin/env python3
"""Quick A/B (+seed) comparison for completed E4 CivilComments runs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
from prime.data.civilcomments_loader import ORACLE_GROUP_NAMES  # noqa: E402

NAMES = list(ORACLE_GROUP_NAMES)


def _ens(path: Path) -> dict:
    m = json.loads(path.read_text(encoding="utf-8"))
    return m.get("ensemble") or m.get("final_test") or m


def _rwg(ens: dict) -> float:
    if ens.get("R_worst_group") is not None:
        return float(ens["R_worst_group"])
    ca = ens.get("cluster_accuracies") or {}
    return float(min(ca.values())) if ca else float("nan")


def _arm(run: Path) -> dict:
    s = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    ft = s.get("final_test") or {}
    cycles = []
    for c in s.get("cycles") or []:
        g = c.get("anchor_gate") or {}
        p = c.get("pareto") or {}
        cycles.append(
            {
                "cycle": c["cycle"],
                "fitness": c.get("fitness"),
                "evo": c.get("best_evo_score"),
                "heir": p.get("heir_source"),
                "heir_fit": p.get("heir_select_fitness"),
                "val_key": (c.get("val_selection_key") or [None])[0],
                "gate_ok": g.get("accepted"),
                "gate_reason": g.get("reason"),
                "drop": g.get("drop"),
            }
        )
    return {
        "run": run,
        "mode": s.get("fitness_mode"),
        "geom": s.get("cluster_geometry"),
        "best_key": s.get("best_selection_key"),
        "ft": ft,
        "cycles": cycles,
        "gate": s.get("anchor_gate") or {},
    }


def main() -> int:
    seed = _ens(PKG / "experiments/E4_civilcomments/baseline_initial_prompt_test/metrics.json")
    a = _arm(PKG / "results/E4_civilcomments_arm_a_global/seed42_20260805_104411")
    b = _arm(PKG / "results/E4_civilcomments_arm_b_min_group/seed42_20260805_135157")
    out = PKG / "experiments/E4_civilcomments/compare_ab_seed"
    out.mkdir(parents=True, exist_ok=True)

    rows = [
        ("seed", seed, None),
        ("A_global", a["ft"], a),
        ("B_min_group_lex", b["ft"], b),
    ]

    lines = [
        "# E4 A/B vs seed (test n=800)",
        "",
        "While Arm C (`style` inferred) is still running.",
        "",
        "## Headline",
        "",
        "| arm | fitness | R_global | R_macro | R_worst_group | Acc tox | CVaR | kappa |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, ens, meta in rows:
        mode = meta["mode"] if meta else "—"
        tox = (ens.get("accuracy_per_class") or {}).get("1") or (ens.get("accuracy_per_class") or {}).get(1)
        lines.append(
            f"| {name} | {mode} | {ens.get('R_global'):.4f} | {ens.get('R_macro'):.4f} | "
            f"{_rwg(ens):.4f} | {float(tox):.3f} | {ens.get('CVaR_cluster'):.3f} | "
            f"{ens.get('mean_kappa'):.3f} |"
        )
    lines += [
        "",
        "### Δ vs seed",
        "",
        "| arm | ΔR_global | ΔR_macro | ΔR_worst_group | ΔAcc tox |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, ens, _ in rows[1:]:
        tox = float((ens.get("accuracy_per_class") or {}).get("1", 0))
        stox = float((seed.get("accuracy_per_class") or {}).get("1", 0))
        lines.append(
            f"| {name} | {ens['R_global']-seed['R_global']:+.4f} | "
            f"{ens['R_macro']-seed['R_macro']:+.4f} | {_rwg(ens)-_rwg(seed):+.4f} | "
            f"{tox-stox:+.3f} |"
        )

    # per-group table
    lines += ["", "## Per-oracle-group (test)", "", "| group | seed | A | B | B−seed | B−A |", "|---|---:|---:|---:|---:|---:|"]
    ca_s = {str(k): float(v) for k, v in (seed.get("cluster_accuracies") or {}).items()}
    ca_a = {str(k): float(v) for k, v in (a["ft"].get("cluster_accuracies") or {}).items()}
    ca_b = {str(k): float(v) for k, v in (b["ft"].get("cluster_accuracies") or {}).items()}
    for gid in sorted(ca_s, key=int):
        g = NAMES[int(gid)] if int(gid) < len(NAMES) else gid
        s, aa, bb = ca_s[gid], ca_a.get(gid, float("nan")), ca_b.get(gid, float("nan"))
        lines.append(f"| {g} | {s:.3f} | {aa:.3f} | {bb:.3f} | {bb-s:+.3f} | {bb-aa:+.3f} |")

    for label, arm in (("A", a), ("B", b)):
        lines += [f"", f"## Arm {label} cycles", ""]
        lines.append("| c | entry fitness | OE best | heir | val key | gate |")
        lines.append("|---:|---:|---:|---|---:|---|")
        for c in arm["cycles"]:
            lines.append(
                f"| {c['cycle']} | {c['fitness']:.4f} | {c['evo']:.4f} | "
                f"{c['heir']}@{c['heir_fit']:.4f} | {c['val_key']:.4f} | "
                f"{'OK' if c['gate_ok'] else 'REJ'}:{c['gate_reason']} drop={c['drop']} |"
            )
        g = arm["gate"]
        lines.append("")
        lines.append(
            f"Gate: evaluated {g.get('n_evaluated')}, rejected {g.get('n_rejected')} "
            f"({g.get('rejection_rate')}); reasons={g.get('reasons')}"
        )

    lines += [
        "",
        "## Verdict (so far)",
        "",
        "- **B (`min_group_lex`) beats seed and A on headline R_worst_group** "
        f"({_rwg(b['ft']):.3f} vs seed {_rwg(seed):.3f} / A {_rwg(a['ft']):.3f}).",
        f"- B also slightly up on R_global ({b['ft']['R_global']:.4f} vs seed {seed['R_global']:.4f}).",
        "- A was ≈seed on global/macro and **worse** on worst-group — global fitness wrong objective here.",
        "- Toxic class still ~0.41–0.42 for all three; gains are mostly group rebalancing, not tox recall.",
        "- Arm C running: same lex, style-inferred groups; judge on **oracle** R_worst_group when done.",
        "",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # bar chart
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    metrics = ["R_global", "R_macro", "R_worst_group"]
    vals = {
        "seed": [seed["R_global"], seed["R_macro"], _rwg(seed)],
        "A": [a["ft"]["R_global"], a["ft"]["R_macro"], _rwg(a["ft"])],
        "B": [b["ft"]["R_global"], b["ft"]["R_macro"], _rwg(b["ft"])],
    }
    x = np.arange(len(metrics))
    w = 0.25
    for i, (lab, color) in enumerate([("seed", "#78909c"), ("A", "#ef6c00"), ("B", "#2e7d32")]):
        axes[0].bar(x + (i - 1) * w, vals[lab], w, label=lab, color=color)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].set_ylim(0.55, 0.95)
    axes[0].legend()
    axes[0].set_title("Test headlines")
    axes[0].grid(True, axis="y", alpha=0.3)

    gids = sorted(ca_s, key=int)
    xs = np.arange(len(gids))
    axes[1].bar(xs - w, [ca_s[g] for g in gids], w, label="seed", color="#78909c")
    axes[1].bar(xs, [ca_a.get(g, 0) for g in gids], w, label="A", color="#ef6c00")
    axes[1].bar(xs + w, [ca_b.get(g, 0) for g in gids], w, label="B", color="#2e7d32")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([NAMES[int(g)][:8] for g in gids], fontsize=7, rotation=30)
    axes[1].set_ylim(0.5, 1.0)
    axes[1].legend(fontsize=8)
    axes[1].set_title("Per-group test acc")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_ab_seed.png", dpi=140)
    plt.close(fig)

    text = (out / "RESULTS.md").read_text(encoding="utf-8")
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

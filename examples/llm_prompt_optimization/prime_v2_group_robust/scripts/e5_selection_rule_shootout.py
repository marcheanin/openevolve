#!/usr/bin/env python
"""Which D_dev selection rule actually predicts test worst-group performance?

Runs entirely on cached predictions from the selection-control harness, so it
costs nothing to add a rule. Includes the rule PRIME actually ships
(`R_soft_min_gba` with shrinkage, from `prime.fitness`), the group-free control,
and a per-group precision budget diagnostic.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def group_gba(preds, y, c) -> dict[int, float]:
    out = {}
    for g in sorted(set(c.tolist())):
        if g == 0:
            continue
        m = c == g
        pos, neg = m & (y == 1), m & (y == 0)
        if not pos.any() or not neg.any():
            continue
        out[int(g)] = 0.5 * (float((preds[pos] == 1).mean()) + float((preds[neg] == 0).mean()))
    return out


def rule_values(preds, y, c, *, shrink_w: float, tau: float) -> dict[str, float]:
    from prime.fitness.objective import soft_min_accuracies

    g = group_gba(preds, y, c)
    v = np.sort(np.asarray(list(g.values())))
    k25 = max(1, int(round(0.25 * len(v))))

    # Empirical-Bayes shrink of each group toward the pooled mean, weight in
    # pseudo-examples, matching prime.fitness.metrics.
    counts = {gi: int((c == gi).sum()) for gi in g}
    pooled = float(v.mean())
    shrunk = {
        gi: (counts[gi] * g[gi] + shrink_w * pooled) / (counts[gi] + shrink_w) for gi in g
    }
    sv = np.sort(np.asarray(list(shrunk.values())))

    recall = float((preds[y == 1] == 1).mean())
    spec = float((preds[y == 0] == 0).mean())
    return {
        "mean_gba": float(v.mean()),
        "worst_class": min(recall, spec),
        "global_acc": float((preds == y).mean()),
        "cvar25": float(v[:k25].mean()),
        "cvar50": float(v[: max(1, len(v) // 2)].mean()),
        "hard_min": float(v[0]),
        "softmin": float(soft_min_accuracies({int(k): float(x) for k, x in g.items()}, tau)),
        "softmin_shrunk": float(
            soft_min_accuracies({int(k): float(x) for k, x in shrunk.items()}, tau)
        ),
        "hard_min_shrunk": float(sv[0]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sel-dir", type=Path, default=ROOT / "results/E5_selection_control/strictness_sweep"
    )
    ap.add_argument(
        "--fixed-dir", type=Path, default=ROOT / "experiments/E5_civilcomments/fixed_sets"
    )
    ap.add_argument("--shrink-weight", type=float, default=40.0)
    ap.add_argument("--tau", type=float, default=0.10)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    dev = json.loads((args.fixed_dir / "d_dev_materialized.json").read_text(encoding="utf-8"))
    big = json.loads(
        (args.fixed_dir / "test_fixed_large_materialized.json").read_text(encoding="utf-8")
    )
    yd, cd = np.asarray(dev["labels"]), np.asarray(dev["cluster_ids"])
    yt, ct = np.asarray(big["labels"]), np.asarray(big["cluster_ids"])

    rep = json.loads(
        (args.sel_dir / "selection_control_report.json").read_text(encoding="utf-8")
    )
    names = [r["name"] for r in sorted(rep["rows"], key=lambda r: r["rank"])]
    dp = {n: np.load(args.sel_dir / f"preds/{n}__dev.npy") for n in names}
    tp = {n: np.load(args.sel_dir / f"preds/{n}__test.npy") for n in names}

    dev_v = {n: rule_values(dp[n], yd, cd, shrink_w=args.shrink_weight, tau=args.tau) for n in names}
    test_cvar = {
        n: rule_values(tp[n], yt, ct, shrink_w=args.shrink_weight, tau=args.tau)["cvar25"]
        for n in names
    }

    rules = [
        "mean_gba", "worst_class", "global_acc", "cvar25", "cvar50",
        "hard_min", "softmin", "softmin_shrunk", "hard_min_shrunk",
    ]

    print(f"pool={args.sel_dir.name}  candidates={len(names)}  "
          f"test cvar25 range [{min(test_cvar.values()):.4f}, {max(test_cvar.values()):.4f}]")
    print()
    print("=== Rule -> pick -> test CVaR@25% (higher is better) ===")
    print(f"{'dev rule':16} {'pick':16} {'test cvar25':>11} {'vs best':>9} {'rank_corr':>10}")
    best_test = max(test_cvar.values())
    order_true = np.argsort([test_cvar[n] for n in names])
    summary = {}
    for rule in rules:
        pick = max(names, key=lambda n: dev_v[n][rule])
        order_dev = np.argsort([dev_v[n][rule] for n in names])
        # Spearman between dev rule and test cvar25 across the pool
        rd = np.empty(len(names)); rd[order_dev] = np.arange(len(names))
        rt = np.empty(len(names)); rt[order_true] = np.arange(len(names))
        rho = float(np.corrcoef(rd, rt)[0, 1])
        summary[rule] = {
            "pick": pick,
            "test_cvar25": test_cvar[pick],
            "regret": best_test - test_cvar[pick],
            "spearman_dev_vs_test_cvar25": rho,
        }
        print(
            f"{rule:16} {pick:16} {test_cvar[pick]:11.4f} "
            f"{best_test - test_cvar[pick]:+9.4f} {rho:10.2f}"
        )

    print()
    print("=== Precision of each dev statistic (bootstrap over D_dev) ===")
    rng = np.random.default_rng(0)
    cells = [
        np.where((cd == g) & (yd == lab))[0]
        for g in sorted(set(cd.tolist()))
        for lab in (0, 1)
    ]
    anchor = "seed_anchor" if "seed_anchor" in names else names[0]
    samples = {r: [] for r in rules}
    per_group = []
    for _ in range(args.n_boot):
        idx = np.concatenate([rng.choice(v, size=len(v), replace=True) for v in cells])
        v = rule_values(dp[anchor][idx], yd[idx], cd[idx], shrink_w=args.shrink_weight, tau=args.tau)
        for r in rules:
            samples[r].append(v[r])
        g = group_gba(dp[anchor][idx], yd[idx], cd[idx])
        per_group.append(g.get(8, np.nan))
    print(f"  (anchor prompt = {anchor})")
    print(f"{'statistic':16} {'sd on D_dev':>12}")
    prec = {}
    for r in rules:
        sd = float(np.std(samples[r], ddof=1))
        prec[r] = sd
        print(f"{r:16} {sd:12.4f}")
    sd_g8 = float(np.nanstd(per_group, ddof=1))
    print(f"{'single group g8':16} {sd_g8:12.4f}   <- the quantity worst-group rules rely on")

    print()
    print("=== Dev value vs test outcome for the two extremes ===")
    for n in (summary["worst_class"]["pick"], summary["softmin_shrunk"]["pick"],
              summary["cvar25"]["pick"]):
        print(
            f"  {n:16} dev cvar25={dev_v[n]['cvar25']:.4f} "
            f"dev softmin_shrunk={dev_v[n]['softmin_shrunk']:.4f} "
            f"dev worst_class={dev_v[n]['worst_class']:.4f} -> test cvar25={test_cvar[n]:.4f}"
        )

    out = args.sel_dir / "selection_rule_shootout.json"
    out.write_text(
        json.dumps(
            {
                "shrink_weight": args.shrink_weight,
                "tau": args.tau,
                "rules": summary,
                "dev_statistic_sd": prec,
                "dev_single_group_sd_g8": sd_g8,
                "dev_values": dev_v,
                "test_cvar25": test_cvar,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

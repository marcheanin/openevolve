#!/usr/bin/env python
"""Where does a worst-group dev statistic stop being anti-correlated?

Grid over the three knobs that define PRIME's selection statistic — CVaR depth k,
soft-min temperature tau, and shrinkage weight — scored on both candidate pools
against two test targets. Runs entirely on cached predictions.

The question is narrow and practical: given a 900-example D_dev with 50 pos and
50 neg per group, which dev statistic best ranks prompts by their *test*
worst-group performance? The answer sets `soft_min_tau` and
`shrink_prior_weight`, and says whether a group-labelled rule is worth using at
all against the group-free control.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

SS = ROOT / "results/E5_s9_matrix/stable_session"
MATRIX = ROOT / "results/E5_s9_matrix"
SEL = ROOT / "results/E5_selection_control"
METHODS = [
    "ape", "ape_k48", "ape_ut", "apo", "gpo", "evoprompt_ga",
    "evoprompt_de", "gepa", "prime", "random_al", "oracle",
]


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


def shrink(g: dict[int, float], counts: dict[int, int], w: float) -> dict[int, float]:
    if w <= 0:
        return dict(g)
    pooled = float(np.mean(list(g.values())))
    return {k: (counts[k] * v + w * pooled) / (counts[k] + w) for k, v in g.items()}


def cvar(vals, k: int) -> float:
    v = np.sort(np.asarray(list(vals)))
    return float(v[: max(1, min(k, len(v)))].mean())


def softmin(vals, tau: float) -> float:
    v = np.asarray(list(vals), dtype=float)
    return float(-tau * np.log(np.mean(np.exp(-v / tau))))


def spearman(a: list[float], b: list[float]) -> float:
    ra = np.empty(len(a)); ra[np.argsort(a)] = np.arange(len(a))
    rb = np.empty(len(b)); rb[np.argsort(b)] = np.arange(len(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def majority(job: Path, kind: str):
    ps = [job / f"{kind}_repeat{r}_preds.npy" for r in range(3)]
    ps = [p for p in ps if p.is_file()]
    return None if not ps else (np.stack([np.load(p) for p in ps]).mean(axis=0) >= 0.5).astype(np.int8)


def load_pools(fixed_dir: Path):
    dev = json.loads((fixed_dir / "d_dev_materialized.json").read_text(encoding="utf-8"))
    yd, cd = np.asarray(dev["labels"]), np.asarray(dev["cluster_ids"])

    pools = {}

    # Pool A: R15 strictness sweep, test = test_fixed_large (n=5251)
    a = SEL / "strictness_sweep"
    if (a / "selection_control_report.json").is_file():
        big = json.loads((fixed_dir / "test_fixed_large_materialized.json").read_text(encoding="utf-8"))
        rep = json.loads((a / "selection_control_report.json").read_text(encoding="utf-8"))
        names = [r["name"] for r in sorted(rep["rows"], key=lambda r: r["rank"])]
        pools["r15_strictness"] = {
            "names": names,
            "dev": {n: np.load(a / f"preds/{n}__dev.npy") for n in names},
            "test": {n: np.load(a / f"preds/{n}__test.npy") for n in names},
            "yt": np.asarray(big["labels"]),
            "ct": np.asarray(big["cluster_ids"]),
        }

    # Pool B: S9 optimizer finals, test = test_fixed (n=1800, majority-of-3)
    b = SEL / "s9_pool"
    if (b / "s9_pool_selection_replication.json").is_file():
        small = json.loads((fixed_dir / "test_fixed_materialized.json").read_text(encoding="utf-8"))
        rep = json.loads((b / "s9_pool_selection_replication.json").read_text(encoding="utf-8"))
        names = list(rep["test_cvar25"].keys())
        test = {}
        for n in names:
            job = SS / ("_shared_seed42" if n == "seed" else f"seed{n.split('_', 1)[0]}_{n.split('_', 1)[1]}")
            test[n] = majority(job, "seed" if n == "seed" else "final")
        pools["s9_optimizers"] = {
            "names": names,
            "dev": {n: np.load(b / f"preds/{n}__dev.npy") for n in names},
            "test": test,
            "yt": np.asarray(small["labels"]),
            "ct": np.asarray(small["cluster_ids"]),
        }

    return yd, cd, pools


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixed-dir", type=Path, default=ROOT / "experiments/E5_civilcomments/fixed_sets")
    ap.add_argument("--out", type=Path, default=SEL / "selection_statistic_ablation.json")
    args = ap.parse_args()

    yd, cd, pools = load_pools(args.fixed_dir)
    if not pools:
        raise SystemExit("no pools found; run the selection-control harness first")
    dev_counts = {int(g): int((cd == g).sum()) for g in sorted(set(cd.tolist())) if g != 0}
    print(f"pools: {', '.join(pools)}   dev per-group n: {sorted(set(dev_counts.values()))}")

    # Candidate dev statistics -------------------------------------------------
    stats: list[tuple[str, str, callable]] = []
    for k in (1, 2, 3, 4, 6, 8):
        stats.append((f"cvar_k{k}", "group", lambda g, k=k: cvar(g.values(), k)))
    for tau in (0.02, 0.05, 0.10, 0.20, 0.50):
        stats.append((f"softmin_t{tau:g}", "group", lambda g, t=tau: softmin(g.values(), t)))

    shrink_ws = (0.0, 10.0, 40.0, 100.0, 400.0)

    targets = {"test_cvar25": 2, "test_hard_min": 1}
    report: dict = {"dev_per_group_n": dev_counts, "pools": {}}

    for pool_name, P in pools.items():
        names = P["names"]
        gdev = {n: group_gba(P["dev"][n], yd, cd) for n in names}
        gtest = {n: group_gba(P["test"][n], P["yt"], P["ct"]) for n in names}
        tvals = {
            t: [cvar(gtest[n].values(), k) for n in names] for t, k in targets.items()
        }
        # group-free controls
        ctrl = {}
        for n in names:
            p, y = P["dev"][n], yd
            rec = float((p[y == 1] == 1).mean())
            spc = float((p[y == 0] == 0).mean())
            ctrl[n] = {"worst_class": min(rec, spc), "global_acc": float((p == y).mean())}

        print()
        print(f"===== POOL {pool_name}  (n={len(names)}) =====")
        for t in targets:
            print(f"  target {t}: range [{min(tvals[t]):.4f}, {max(tvals[t]):.4f}]")

        block: dict = {"n_candidates": len(names), "rules": {}}
        print()
        print(f"{'dev statistic':18} {'w=0':>7} {'w=10':>7} {'w=40':>7} {'w=100':>7} {'w=400':>7}   "
              f"(Spearman vs test_cvar25)")
        for sname, kind, fn in stats:
            cells = []
            for w in shrink_ws:
                dv = [fn(shrink(gdev[n], dev_counts, w)) for n in names]
                rho = spearman(dv, tvals["test_cvar25"])
                pick = names[int(np.argmax(dv))]
                regret = max(tvals["test_cvar25"]) - tvals["test_cvar25"][names.index(pick)]
                block["rules"][f"{sname}__w{w:g}"] = {
                    "spearman_test_cvar25": rho,
                    "spearman_test_hard_min": spearman(dv, tvals["test_hard_min"]),
                    "pick": pick,
                    "regret_test_cvar25": regret,
                }
                cells.append(rho)
            print(f"{sname:18} " + " ".join(f"{c:+7.2f}" for c in cells))

        print()
        for cname in ("worst_class", "global_acc"):
            dv = [ctrl[n][cname] for n in names]
            rho = spearman(dv, tvals["test_cvar25"])
            pick = names[int(np.argmax(dv))]
            regret = max(tvals["test_cvar25"]) - tvals["test_cvar25"][names.index(pick)]
            block["rules"][f"CONTROL_{cname}"] = {
                "spearman_test_cvar25": rho,
                "spearman_test_hard_min": spearman(dv, tvals["test_hard_min"]),
                "pick": pick,
                "regret_test_cvar25": regret,
            }
            print(f"{'CONTROL ' + cname:18} {rho:+7.2f}   pick={pick:16} regret={regret:+.4f}")

        best = min(block["rules"].items(), key=lambda kv: kv[1]["regret_test_cvar25"])
        print(f"\n  lowest regret: {best[0]} -> {best[1]['regret_test_cvar25']:+.4f} "
              f"(pick {best[1]['pick']})")
        report["pools"][pool_name] = block

    # Cross-pool summary: which statistics are positive on BOTH pools? ---------
    if len(report["pools"]) > 1:
        print()
        print("===== statistics with Spearman > 0 on BOTH pools, by mean regret =====")
        common = set.intersection(*(set(b["rules"]) for b in report["pools"].values()))
        rows = []
        for r in common:
            rhos = [report["pools"][p]["rules"][r]["spearman_test_cvar25"] for p in report["pools"]]
            regs = [report["pools"][p]["rules"][r]["regret_test_cvar25"] for p in report["pools"]]
            if min(rhos) > 0:
                rows.append((float(np.mean(regs)), r, rhos, regs))
        for mean_reg, r, rhos, regs in sorted(rows)[:12]:
            print(f"  {r:24} mean regret {mean_reg:.4f}  rho "
                  f"{'/'.join(f'{x:+.2f}' for x in rhos)}  regret "
                  f"{'/'.join(f'{x:.4f}' for x in regs)}")
        report["positive_on_both"] = [
            {"rule": r, "mean_regret": m, "rhos": rhos, "regrets": regs}
            for m, r, rhos, regs in sorted(rows)
        ]

    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""R15 calibration control vs the S9 optimizer matrix, on identical rows.

`test_fixed_large` is a strict superset of `test_fixed`, so the R15 predictions
can be restricted to exactly the 1800 rows the S9 matrix was scored on. That
removes the usual "different eval set" escape hatch: the comparison below is
paired at the example level.

R15 is represented by the candidate its D_dev rule picks — no test peeking. The
sweep's test-oracle is reported separately as an upper bound on what pure
operating-point movement can reach.
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
S9_METHODS = [
    "ape", "ape_k48", "ape_ut", "apo", "gpo", "evoprompt_ga",
    "evoprompt_de", "gepa", "prime", "random_al", "oracle",
]


def cvar25(preds, y, cids, frac=0.25) -> float:
    groups = [g for g in sorted(set(cids.tolist())) if g != 0]
    gba = []
    for g in groups:
        m = cids == g
        pos, neg = m & (y == 1), m & (y == 0)
        tpr = float((preds[pos] == 1).mean()) if pos.any() else 0.5
        tnr = float((preds[neg] == 0).mean()) if neg.any() else 0.5
        gba.append(0.5 * (tpr + tnr))
    gba = np.sort(np.asarray(gba))
    k = max(1, int(round(frac * len(gba))))
    return float(gba[:k].mean())


def paired_boot(a, b, y, cids, n_boot, rng) -> tuple[float, float, float]:
    """CI and two-sided p for cvar25(b) - cvar25(a), resampling within cells."""
    cells = [
        np.where((cids == g) & (y == lab))[0]
        for g in sorted(set(cids.tolist()))
        for lab in (0, 1)
    ]
    cells = [c for c in cells if len(c)]
    d = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(c, size=len(c), replace=True) for c in cells])
        d[i] = cvar25(b[idx], y[idx], cids[idx]) - cvar25(a[idx], y[idx], cids[idx])
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = min(1.0, 2 * min(float((d >= 0).mean()), float((d <= 0).mean())))
    return float(lo), float(hi), p


def holm(pairs: list[tuple[str, float]]) -> dict[str, float]:
    order = sorted(pairs, key=lambda t: t[1])
    m, run, adj = len(order), 0.0, {}
    for i, (name, p) in enumerate(order):
        run = max(run, (m - i) * p)
        adj[name] = min(1.0, run)
    return adj


def majority(job: Path, kind: str):
    ps = [job / f"{kind}_repeat{r}_preds.npy" for r in range(3)]
    ps = [p for p in ps if p.is_file()]
    if not ps:
        return None
    return (np.stack([np.load(p) for p in ps]).mean(axis=0) >= 0.5).astype(np.int8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sel-dir",
        type=Path,
        default=ROOT / "results/E5_selection_control/strictness_sweep",
    )
    ap.add_argument(
        "--fixed-dir", type=Path, default=ROOT / "experiments/E5_civilcomments/fixed_sets"
    )
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--rule", default="cvar25", choices=["cvar25", "mean_gba", "worst_class"])
    args = ap.parse_args()

    rep = json.loads((args.sel_dir / "selection_control_report.json").read_text(encoding="utf-8"))
    rm = json.loads(
        (args.fixed_dir / "test_fixed_large_reuse_map.json").read_text(encoding="utf-8")
    )
    rows_small = np.asarray(rm["rows_in_large_for_each_small_row"], dtype=int)

    small = json.loads(
        (args.fixed_dir / "test_fixed_materialized.json").read_text(encoding="utf-8")
    )
    y = np.asarray(small["labels"], dtype=np.int8)
    cids = np.asarray(small["cluster_ids"], dtype=np.int16)

    large = json.loads(
        (args.fixed_dir / "test_fixed_large_materialized.json").read_text(encoding="utf-8")
    )
    y_l = np.asarray(large["labels"], dtype=np.int8)
    c_l = np.asarray(large["cluster_ids"], dtype=np.int16)
    clean = np.ones(len(y_l), dtype=bool)
    clean[rows_small] = False

    # Sanity: restriction must reproduce the S9 layout exactly.
    if not np.array_equal(y_l[rows_small], y):
        raise SystemExit("row restriction misaligned: labels differ")

    picked = rep["picks"][args.rule]["picked"]
    print(f"R15 pick by dev {args.rule}: {picked}")

    r15 = {}
    for r in rep["rows"]:
        p = np.load(args.sel_dir / f"preds/{r['name']}__test.npy")
        r15[r["name"]] = p
    pick_full = r15[picked]
    pick_1800 = pick_full[rows_small]

    print()
    print("=== R15 sweep, restricted to the S9 1800 rows ===")
    print(f"{'candidate':16} {'cvar25@1800':>12} {'cvar25@5251':>12} {'cvar25@clean3451':>17}")
    for r in sorted(rep["rows"], key=lambda r: r["rank"]):
        p = r15[r["name"]]
        mark = " <- dev pick" if r["name"] == picked else ""
        print(
            f"{r['name']:16} {cvar25(p[rows_small], y, cids):12.4f} "
            f"{cvar25(p, y_l, c_l):12.4f} {cvar25(p[clean], y_l[clean], c_l[clean]):17.4f}{mark}"
        )

    print()
    print("=== S9 methods vs the R15 dev-selected prompt (same 1800 rows, paired) ===")
    print(f"{'method':16} {'cvar25':>8} {'delta_vs_R15':>13} {'CI95':>20} {'p_holm':>8}")
    results = []
    for s in ("42", "43", "44"):
        for m in S9_METHODS + ["__seed__"]:
            job = SS / (f"_shared_seed{s}" if m == "__seed__" else f"seed{s}_{m}")
            kind = "seed" if m == "__seed__" else "final"
            pred = majority(job, kind)
            if pred is None:
                continue
            rng = np.random.default_rng(abs(hash((s, m, "r15"))) % (2**32))
            lo, hi, p = paired_boot(pick_1800, pred, y, cids, args.n_boot, rng)
            results.append(
                {
                    "seed": s,
                    "method": m.strip("_") if m == "__seed__" else m,
                    "cvar25": cvar25(pred, y, cids),
                    "delta_vs_r15": cvar25(pred, y, cids) - cvar25(pick_1800, y, cids),
                    "ci95": [lo, hi],
                    "p_raw": p,
                }
            )
    adj = holm([(f"{r['seed']}_{r['method']}", r["p_raw"]) for r in results])
    for r in sorted(results, key=lambda r: -r["delta_vs_r15"]):
        key = f"{r['seed']}_{r['method']}"
        r["p_holm"] = adj[key]
        flag = "  SIG" if adj[key] < 0.05 else ""
        print(
            f"{key:16} {r['cvar25']:8.4f} {r['delta_vs_r15']:+13.4f} "
            f"[{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}]{adj[key]:9.3f}{flag}"
        )

    beat = [r for r in results if r["p_holm"] < 0.05 and r["delta_vs_r15"] > 0]
    print()
    print(f"R15 dev-pick cvar25 @1800 = {cvar25(pick_1800, y, cids):.4f}")
    print(f"S9 method-seed cells that beat it at Holm<0.05: {len(beat)} / {len(results)}")
    if beat:
        print("  " + ", ".join(f"{b['seed']}_{b['method']}" for b in beat))

    out = args.sel_dir / f"r15_vs_s9__rule_{args.rule}.json"
    out.write_text(
        json.dumps(
            {
                "rule": args.rule,
                "r15_pick": picked,
                "r15_cvar25_1800": cvar25(pick_1800, y, cids),
                "r15_cvar25_5251": cvar25(pick_full, y_l, c_l),
                "r15_cvar25_clean3451": cvar25(pick_full[clean], y_l[clean], c_l[clean]),
                "r15_sweep_1800": {
                    n: cvar25(p[rows_small], y, cids) for n, p in r15.items()
                },
                "s9": results,
                "n_beating_r15": len(beat),
                "n_cells": len(results),
                "caveat": "S9 predictions are majority-of-3 repeats; R15 is a single "
                "repeat. Repeat sd is 0.0008 CVaR@25% (POWER_AUDIT F2), so the "
                "asymmetry is ~5% of the CI half-width.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

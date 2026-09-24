#!/usr/bin/env python
"""E6 power/op-point audit from selection_control cached preds (F1/F3 analogue)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def paired_boot_ci(delta: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.RandomState(seed)
    n = len(delta)
    means = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        means.append(float(delta[idx].mean()))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sel-dir",
        type=Path,
        default=ROOT / "results/E6_amazon_category_controls/selection_control",
    )
    ap.add_argument(
        "--fixed-dir",
        type=Path,
        default=ROOT / "experiments/E6_amazon_category_controls/fixed_sets",
    )
    ap.add_argument("--baseline", default="seed_anchor")
    args = ap.parse_args()

    from e6_metrics import all_metrics
    from e6_score_prompt import load_mat

    mat = load_mat(args.fixed_dir, "test_fixed")
    y = np.asarray(mat["labels"])
    c = np.asarray(mat["cluster_ids"])
    rows = json.loads((args.sel_dir / "rows.json").read_text(encoding="utf-8"))
    preds_dir = args.sel_dir / "preds"

    base_path = preds_dir / f"{args.baseline.replace(':', '__')}__test_fixed__r0.npy"
    if not base_path.is_file():
        # try opt:seed
        alt = preds_dir / "opt__seed__test_fixed__r0.npy"
        base_path = alt if alt.is_file() else base_path
    if not base_path.is_file():
        raise SystemExit(f"missing baseline preds {base_path}")
    base = np.load(base_path)
    base_ok = (base == y).astype(float)

    contrasts = []
    for row in rows:
        name = row["name"]
        if name == args.baseline or name == "opt:seed":
            continue
        p = preds_dir / f"{name.replace(':', '__')}__test_fixed__r0.npy"
        if not p.is_file():
            continue
        pred = np.load(p)
        d = (pred == y).astype(float) - base_ok
        lo, hi = paired_boot_ci(d)
        m = all_metrics(pred, y, c)
        bm = all_metrics(base, y, c)
        resolvable = not (lo <= 0 <= hi)
        contrasts.append(
            {
                "name": name,
                "delta_global": float(m["R_global"] - bm["R_global"]),
                "delta_cvar25": float(m["cvar25"] - bm["cvar25"]),
                "delta_op_shift": float(m["op_shift"] - bm["op_shift"]),
                "paired_global_ci95": [lo, hi],
                "resolvable_vs_seed": resolvable,
            }
        )

    n_res = sum(1 for x in contrasts if x["resolvable_vs_seed"])
    report = {
        "baseline": args.baseline,
        "n_contrasts": len(contrasts),
        "n_resolvable_global": n_res,
        "contrasts": contrasts,
        "seed_metrics": all_metrics(base, y, c),
    }
    out = args.sel_dir / "power_audit.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"resolvable {n_res}/{len(contrasts)} global paired contrasts")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

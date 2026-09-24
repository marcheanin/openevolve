#!/usr/bin/env python
"""Build the two fixed sets for the S11 protocol study.

`truth_large` comes from the official WILDS **test** split and is the ground-truth
ranking of the prompt pool. `dev_universe` comes from the official **validation**
split and is the pool that thousands of simulated dev draws are sampled from —
the piece that no cached artifact can provide, since E5 had exactly one dev draw.

The two never mix, so the official split semantics are preserved. Cells are
capped, not balanced by the binding cell: `other_religions` has only 240 toxic
comments in test and 79 in validation, which is itself a headline number (the
benchmark's own resolution ceiling), so it is reported rather than hidden by
shrinking every other cell to match.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GROUPS = list(range(9))  # 0 = none, 1..8 = WILDS identities
NAMES = ["none", "male", "female", "LGBTQ", "christian", "muslim", "other_religions", "black", "white"]


def fingerprint(idx) -> str:
    return hashlib.sha256(",".join(str(int(i)) for i in sorted(idx)).encode()).hexdigest()[:12]


def draw(labels, gids, cap, seed):
    """Per cell (group x label), take up to `cap` rows. Deterministic given seed.

    The returned order is shuffled, not grouped by cell, so that any prefix of the
    set is itself a stratified random subsample. Scoring writes rows in order, so
    a half-finished prompt still yields an unbiased set of rows to analyse.
    """
    rng = np.random.RandomState(seed)
    chosen, counts = [], {}
    for g in GROUPS:
        for lab in (0, 1):
            pool = np.flatnonzero((gids == g) & (labels == lab))
            k = int(min(cap, len(pool)))
            pick = rng.choice(pool, size=k, replace=False) if k else np.array([], dtype=int)
            chosen.extend(int(i) for i in pick)
            counts[f"g{g}_y{lab}"] = k
    chosen = np.array(sorted(chosen))
    rng.shuffle(chosen)
    return [int(i) for i in chosen], counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "experiments/E5_civilcomments/config_prime_main.yaml")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/S11_protocol_matrix/fixed_sets")
    ap.add_argument("--truth-cap", type=int, default=200, help="rows per cell from the test split")
    ap.add_argument("--dev-cap", type=int, default=150, help="rows per cell from the validation split")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from prime.config import load_config
    from prime.data.civilcomments_loader import load_civilcomments_splits

    cfg = load_config(args.config)
    cfg.dataset.max_val_users = max(int(cfg.dataset.max_val_users or 0), 60_000)
    cfg.dataset.max_test_users = max(int(cfg.dataset.max_test_users or 0), 200_000)
    splits = load_civilcomments_splits(cfg.dataset, seed=args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, split_name, cap in (
        ("truth_large", "test", args.truth_cap),
        ("dev_universe", "validation", args.dev_cap),
    ):
        sp = splits[split_name]
        y, g = np.asarray(sp.labels), np.asarray(sp.example_cluster_ids)
        idx, counts = draw(y, g, cap, args.seed + (0 if name == "truth_large" else 17))
        rec = {
            "name": name,
            "source_split": split_name,
            "cap_per_cell": cap,
            "n": len(idx),
            "fingerprint": fingerprint(idx),
            "indices": [int(i) for i in idx],
            "labels": [int(y[i]) for i in idx],
            "cluster_ids": [int(g[i]) for i in idx],
            "cell_counts": counts,
        }
        (args.out_dir / f"{name}.json").write_text(json.dumps(rec), encoding="utf-8")
        summary[name] = {k: rec[k] for k in ("source_split", "n", "fingerprint", "cap_per_cell")}
        print(f"\n=== {name}: n={rec['n']} from {split_name}, fingerprint {rec['fingerprint']}")
        binding = min((counts[f"g{g_}_y1"], NAMES[g_]) for g_ in range(1, 9))
        for g_ in GROUPS:
            print(f"   {NAMES[g_]:16s} neg {counts[f'g{g_}_y0']:4d}  pos {counts[f'g{g_}_y1']:4d}")
        print(f"   binding identity cell: {binding[1]} with {binding[0]} toxic rows "
              f"-> best achievable GBA sd for that group ~{0.5/np.sqrt(binding[0]):.4f}")

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

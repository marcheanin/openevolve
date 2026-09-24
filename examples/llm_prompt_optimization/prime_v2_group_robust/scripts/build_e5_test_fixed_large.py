#!/usr/bin/env python
"""Build `test_fixed_large` (per_cell=300 -> n=5400) as a superset of `test_fixed`.

Power audit (experiments/E5_civilcomments/POWER_AUDIT.md) showed scorer repeats
contribute ~1/250 of the metric variance while costing 3x. Moving that budget
into examples requires a bigger fixed test set; making it a *superset* of the
existing one means the 1800 already-scored predictions stay valid.

The superset property holds because `RandomState.choice(pool, k, replace=False)`
draws `permutation(len(pool))[:k]`, and the permutation consumes the same RNG
state regardless of k. This script does not assume that — it asserts it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ORIGINAL_FINGERPRINT = "5cfb7ebde3c5"


def _build_capped(test, *, name: str, target_per_cell: int, seed: int, avail: dict):
    """Draw min(target, available) per (group,label) cell.

    Cells are visited in exactly the same order as `sample_balanced_cells`, and
    `RandomState.choice(pool, k, replace=False)` advances the RNG by a full
    `permutation(len(pool))` regardless of k. Keeping that order therefore keeps
    every cell's draw a prefix of the same permutation, which is what makes the
    result a superset of the 100-per-cell set. The caller asserts this.
    """
    import numpy as np

    from prime.data.balanced_cells import cell_counts, fingerprint_indices
    from prime.data.fixed_sets import FixedSet

    labels = test.labels
    gids = test.example_cluster_ids
    by_cell: dict[tuple[int, int], list[int]] = {}
    for i, (lab, gid) in enumerate(zip(labels, gids)):
        by_cell.setdefault((int(gid), int(lab)), []).append(i)

    groups = list(range(9))
    rng = np.random.RandomState(seed)
    chosen: list[int] = []
    per_cell_used: dict[str, int] = {}
    for g in groups:
        for lab in (0, 1):
            pool = by_cell.get((g, lab), [])
            k = min(target_per_cell, len(pool))
            pick = rng.choice(pool, size=k, replace=False)
            chosen.extend(int(i) for i in pick)
            per_cell_used[f"g{g}_y{lab}"] = k
    chosen = sorted(chosen)
    return FixedSet(
        name=name,
        source_split="test",
        indices=chosen,
        fingerprint=fingerprint_indices(chosen),
        design={
            "groups": groups,
            "target_per_cell": target_per_cell,
            "per_cell_used": per_cell_used,
            "include_none": True,
            "seed": seed,
            "balanced": False,
            "note": "min(target, available) per cell; GBA is computed within group "
            "so unequal cell sizes only change per-group precision.",
        },
        cell_counts=cell_counts(labels, gids, chosen),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/E5_civilcomments/config_prime_main.yaml",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "experiments/E5_civilcomments/fixed_sets",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-cell", type=int, default=300)
    ap.add_argument(
        "--allow-non-superset",
        action="store_true",
        help="Do not fail if the larger draw is not a superset (loses pred reuse).",
    )
    args = ap.parse_args()

    import numpy as np

    from prime.config import load_config
    from prime.data.civilcomments_loader import load_civilcomments_splits
    from prime.data.fixed_sets import build_balanced_fixed_set, materialize_split

    cfg = load_config(args.config)
    # Same caps as build_e5_fixed_sets.py, otherwise the candidate pool shifts
    # and the superset property is lost.
    cfg.dataset.max_val_users = max(int(cfg.dataset.max_val_users or 0), 50_000)
    cfg.dataset.max_test_users = max(int(cfg.dataset.max_test_users or 0), 80_000)

    splits = load_civilcomments_splits(cfg.dataset, seed=args.seed)
    test = splits["test"]
    if not test.example_cluster_ids:
        raise RuntimeError("CivilComments test split must carry oracle cluster ids")

    small = build_balanced_fixed_set(
        test,
        name="test_fixed",
        source_split="test",
        per_cell=100,
        seed=args.seed,
        include_none=True,
    )
    print(f"[check] reproduced test_fixed fingerprint = {small.fingerprint}")
    if small.fingerprint != ORIGINAL_FINGERPRINT:
        raise SystemExit(
            f"refusing to proceed: rebuilt test_fixed fingerprint {small.fingerprint} "
            f"!= shipped {ORIGINAL_FINGERPRINT}; the data pipeline drifted"
        )

    # Report per-cell availability before attempting the bigger draw.
    labels = test.labels
    gids = test.example_cluster_ids
    avail: dict[tuple[int, int], int] = {}
    for lab, gid in zip(labels, gids):
        avail[(int(gid), int(lab))] = avail.get((int(gid), int(lab)), 0) + 1
    tight = sorted(avail.items(), key=lambda kv: kv[1])[:5]
    print("[check] scarcest cells (group,label)->n:", tight)
    worst = min(avail[(g, lab)] for g in range(9) for lab in (0, 1))
    print(f"[check] scarcest cell has {worst}; target per-cell {args.per_cell}")

    large = _build_capped(
        test,
        name="test_fixed_large",
        target_per_cell=args.per_cell,
        seed=args.seed,
        avail=avail,
    )

    s_small, s_large = set(small.indices), set(large.indices)
    is_superset = s_small <= s_large
    print(
        f"[check] superset: {is_superset}  "
        f"|small|={len(s_small)} |large|={len(s_large)} "
        f"overlap={len(s_small & s_large)}"
    )
    if not is_superset and not args.allow_non_superset:
        raise SystemExit(
            "larger draw is NOT a superset of test_fixed; existing predictions "
            "could not be reused. Re-run with --allow-non-superset to override."
        )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    large.save(out / "test_fixed_large.json")

    mat = materialize_split(test, large.indices)
    slim = {
        "name": large.name,
        "fingerprint": large.fingerprint,
        "n": large.to_dict()["n"],
        "labels": mat["labels"],
        "user_ids": [str(u) for u in mat["user_ids"]],
        "cluster_ids": mat["cluster_ids"],
        "indices": mat["indices"],
        "source_split": mat["source_split"],
    }
    (out / "test_fixed_large_materialized.json").write_text(json.dumps(slim), encoding="utf-8")

    # Reuse map: for each row of the OLD ordering, its row in the NEW ordering.
    pos_in_large = {int(v): i for i, v in enumerate(large.indices)}
    reuse_rows = [pos_in_large[int(v)] for v in small.indices]
    (out / "test_fixed_large_reuse_map.json").write_text(
        json.dumps(
            {
                "from": {"name": "test_fixed", "fingerprint": small.fingerprint, "n": len(small.indices)},
                "to": {"name": large.name, "fingerprint": large.fingerprint, "n": len(large.indices)},
                "rows_in_large_for_each_small_row": reuse_rows,
            }
        ),
        encoding="utf-8",
    )

    y = np.asarray(mat["labels"])
    cids = np.asarray(mat["cluster_ids"])
    from prime.fitness.metrics import compute_metrics

    m = compute_metrics(
        np.zeros_like(y),
        y,
        np.arange(len(y)),
        cluster_ids=cids,
        class_balanced=True,
        gba_min_pos=10,
        gba_min_neg=10,
    )
    floor_ok = abs(float(m.get("R_worst_gba", -1)) - 0.5) < 1e-9

    print(
        json.dumps(
            {
                "test_fixed_large_fingerprint": large.fingerprint,
                "n": len(large.indices),
                "per_cell": args.per_cell,
                "superset_of_test_fixed": is_superset,
                "ALL_ZERO_R_worst_gba": m.get("R_worst_gba"),
                "ok_floor": floor_ok,
            },
            indent=2,
        )
    )
    return 0 if floor_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

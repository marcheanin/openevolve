#!/usr/bin/env python
"""Build `d_dev_targeted`: the same 900-example budget, concentrated.

R15_CALIBRATION_CONTROL F8 showed every minimum-based dev rule is anti-correlated
with the test worst-group score, and F5 traced the cause to precision: a single
group's GBA on the uniform D_dev has sd 0.0475, while the *identity* of the worst
group is stable. Concentrating the same budget on the known-worst groups is the
one remaining way a group-labelled rule could become usable.

The three target groups are read off the **seed prompt on the uniform D_dev**
(groups 8, 3, 5 at GBA 0.570 / 0.610 / 0.630), which is information available
before any optimization. No test data is consulted.

Per-cell count goes 50 -> 150, so each target group's GBA sd should fall by ~sqrt(3).
The draw keeps the RNG call order of `sample_balanced_cells` (all 9 groups, labels
0 then 1, drawing size 0 for non-target groups), which makes it a strict superset
of the uniform dev on the target cells. Asserted, not assumed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

UNIFORM_FINGERPRINT = "1c10514d553c"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/E5_civilcomments/config_prime_main.yaml",
    )
    ap.add_argument(
        "--out-dir", type=Path, default=ROOT / "experiments/E5_civilcomments/fixed_sets"
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--groups", default="8,3,5")
    ap.add_argument("--per-cell", type=int, default=150)
    args = ap.parse_args()

    import numpy as np

    from prime.config import load_config
    from prime.data.balanced_cells import cell_counts, fingerprint_indices
    from prime.data.civilcomments_loader import load_civilcomments_splits
    from prime.data.fixed_sets import FixedSet, build_d_dev, materialize_split

    targets = {int(x) for x in args.groups.split(",")}
    cfg = load_config(args.config)
    cfg.dataset.max_val_users = max(int(cfg.dataset.max_val_users or 0), 50_000)
    cfg.dataset.max_test_users = max(int(cfg.dataset.max_test_users or 0), 80_000)

    splits = load_civilcomments_splits(cfg.dataset, seed=args.seed)
    val = splits["validation"]

    uniform = build_d_dev(val, seed=args.seed)
    print(f"[check] reproduced d_dev fingerprint = {uniform.fingerprint}")
    if uniform.fingerprint != UNIFORM_FINGERPRINT:
        raise SystemExit(
            f"rebuilt d_dev {uniform.fingerprint} != shipped {UNIFORM_FINGERPRINT}; "
            "pipeline drifted, refusing to build a targeted set against it"
        )

    labels, gids = val.labels, val.example_cluster_ids
    by_cell: dict[tuple[int, int], list[int]] = {}
    for i, (lab, gid) in enumerate(zip(labels, gids)):
        by_cell.setdefault((int(gid), int(lab)), []).append(i)
    for g in sorted(targets):
        for lab in (0, 1):
            have = len(by_cell.get((g, lab), []))
            print(f"[check] cell g{g}_y{lab}: {have} available, need {args.per_cell}")
            if have < args.per_cell:
                raise SystemExit(f"cell (g={g}, y={lab}) has only {have}")

    # Mirror build_d_dev's RNG usage exactly: same seed, same group/label order.
    # `choice(pool, k, replace=False)` draws permutation(len(pool))[:k], and the
    # permutation advances the state independently of k, so drawing 0 for the
    # non-target groups keeps every target cell a prefix of the same permutation.
    rng = np.random.RandomState(args.seed + 17)
    chosen: list[int] = []
    per_cell_used: dict[str, int] = {}
    for g in range(9):
        for lab in (0, 1):
            pool = by_cell.get((g, lab), [])
            k = args.per_cell if g in targets else 0
            pick = rng.choice(pool, size=k, replace=False)
            chosen.extend(int(i) for i in pick)
            per_cell_used[f"g{g}_y{lab}"] = k
    chosen = sorted(chosen)

    fs = FixedSet(
        name="d_dev_targeted",
        source_split="validation",
        indices=chosen,
        fingerprint=fingerprint_indices(chosen),
        design={
            "groups": sorted(targets),
            "per_cell": args.per_cell,
            "seed": args.seed + 17,
            "per_cell_used": per_cell_used,
            "selected_by": "seed prompt per-group GBA on the uniform d_dev "
            "(no test data consulted)",
            "supersedes": "d_dev (uniform, 9 groups x 50)",
        },
        cell_counts=cell_counts(labels, gids, chosen),
    )

    # Superset check against the uniform dev, restricted to the target groups.
    uni_target = {i for i in uniform.indices if int(gids[i]) in targets}
    overlap = uni_target & set(chosen)
    print(
        f"[check] uniform dev rows in target groups: {len(uni_target)}; "
        f"contained in targeted set: {len(overlap)}"
    )
    if overlap != uni_target:
        raise SystemExit(
            "targeted draw is not a superset of the uniform dev on the target "
            "groups; the shared-row consistency check would be impossible"
        )

    out = args.out_dir
    fs.save(out / "d_dev_targeted.json")
    mat = materialize_split(val, fs.indices)
    slim = {
        "name": fs.name,
        "fingerprint": fs.fingerprint,
        "n": fs.to_dict()["n"],
        "labels": mat["labels"],
        "user_ids": [str(u) for u in mat["user_ids"]],
        "cluster_ids": mat["cluster_ids"],
        "indices": mat["indices"],
        "source_split": mat["source_split"],
    }
    (out / "d_dev_targeted_materialized.json").write_text(json.dumps(slim), encoding="utf-8")

    # Map: for each uniform-dev row that survives, its position in the targeted set.
    pos = {int(v): i for i, v in enumerate(fs.indices)}
    shared = [
        {"uniform_row": j, "targeted_row": pos[int(v)]}
        for j, v in enumerate(uniform.indices)
        if int(v) in pos
    ]
    (out / "d_dev_targeted_shared_rows.json").write_text(
        json.dumps(
            {
                "from": {"name": "d_dev", "fingerprint": uniform.fingerprint},
                "to": {"name": fs.name, "fingerprint": fs.fingerprint},
                "n_shared": len(shared),
                "pairs": shared,
            }
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "fingerprint": fs.fingerprint,
                "n": len(fs.indices),
                "groups": sorted(targets),
                "per_cell": args.per_cell,
                "n_shared_with_uniform": len(shared),
                "cell_counts": fs.cell_counts,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

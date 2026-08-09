#!/usr/bin/env python
"""Build and fingerprint E5 D_dev + test_fixed (ROADMAP §4.3 / S6)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/E5_civilcomments/config_prime_main.yaml",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "experiments/E5_civilcomments/fixed_sets",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from prime.config import load_config
    from prime.data.civilcomments_loader import load_civilcomments_splits
    from prime.data.fixed_sets import build_d_dev, build_test_fixed, materialize_split

    cfg = load_config(args.config)
    # Need full enough val/test heads for balanced cells (other_religions).
    cfg.dataset.max_val_users = max(int(cfg.dataset.max_val_users or 0), 50_000)
    cfg.dataset.max_test_users = max(int(cfg.dataset.max_test_users or 0), 80_000)

    splits = load_civilcomments_splits(cfg.dataset, seed=args.seed)
    val, test = splits["validation"], splits["test"]
    if not val.example_cluster_ids or not test.example_cluster_ids:
        raise RuntimeError("CivilComments splits must carry oracle cluster ids")

    d_dev = build_d_dev(val, seed=args.seed)
    test_fixed = build_test_fixed(test, seed=args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d_dev.save(out / "d_dev.json")
    test_fixed.save(out / "test_fixed.json")

    for fs, split in ((d_dev, val), (test_fixed, test)):
        mat = materialize_split(split, fs.indices)
        mat_path = out / f"{fs.name}_materialized.json"
        # Keep texts out of git-sized dumps: write sidecar indices+meta only;
        # full materialization with texts for scoring goes to a separate cache.
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
        mat_path.write_text(json.dumps(slim), encoding="utf-8")

    # Floor check: ALL-ZERO worst-GBA must be 0.5 on test_fixed design.
    import numpy as np
    from prime.fitness.metrics import compute_metrics

    y = np.asarray(materialize_split(test, test_fixed.indices)["labels"])
    cids = np.asarray(materialize_split(test, test_fixed.indices)["cluster_ids"])
    uids = np.arange(len(y))
    zeros = np.zeros_like(y)
    m = compute_metrics(
        zeros, y, uids, cluster_ids=cids, class_balanced=True, gba_min_pos=10, gba_min_neg=10
    )
    floor = {
        "ALL_ZERO_R_worst_gba": m.get("R_worst_gba"),
        "ALL_ZERO_R_gba_mean": m.get("R_gba_mean"),
        "ALL_ZERO_toxic_recall": m.get("toxic_recall"),
        "ok_floor": abs(float(m.get("R_worst_gba", -1)) - 0.5) < 1e-9,
    }

    manifest = {
        "seed": args.seed,
        "d_dev": d_dev.to_dict(),
        "test_fixed": test_fixed.to_dict(),
        "floor_check": floor,
        "config": str(args.config),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"d_dev": d_dev.fingerprint, "test_fixed": test_fixed.fingerprint, **floor}, indent=2))
    if not floor["ok_floor"]:
        print("WARNING: ALL-ZERO worst-GBA is not exactly 0.5", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

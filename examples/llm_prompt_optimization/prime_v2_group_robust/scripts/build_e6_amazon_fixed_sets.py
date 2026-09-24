#!/usr/bin/env python
"""Build fingerprinted E6 Amazon fixed sets (category-shift + pred_profile).

Design: balance cells of (pred_profile cluster × collapsed rating {1-2,3,4-5}).
Clusters are fit on train using seed-prompt predictions (cached), then assigned
to val/test. Cell sizes adapt to availability (min(target, available)).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from e6_metrics import collapse_rating  # noqa: E402


def _sample_cells(
    labels: Sequence,
    cluster_ids: Sequence,
    *,
    clusters: list[int],
    bins: list[int],
    per_cell: int,
    seed: int,
) -> tuple[list[int], dict[str, int]]:
    by: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, (lab, cid) in enumerate(zip(labels, cluster_ids)):
        b = collapse_rating(int(lab))
        if int(cid) in clusters and b in bins:
            by[(int(cid), b)].append(i)

    rng = np.random.RandomState(seed)
    chosen: list[int] = []
    used: dict[str, int] = {}
    for g in clusters:
        for b in bins:
            pool = by.get((g, b), [])
            k = min(per_cell, len(pool))
            if k == 0:
                used[f"g{g}_b{b}"] = 0
                continue
            pick = rng.choice(pool, size=k, replace=False)
            chosen.extend(int(i) for i in pick)
            used[f"g{g}_b{b}"] = k
    return sorted(chosen), used


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/E6_amazon_category_controls/config.yaml",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "experiments/E6_amazon_category_controls/fixed_sets",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-per-cell", type=int, default=100)
    ap.add_argument("--dev-per-cell", type=int, default=50)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    from prime.config import load_config
    from prime.data.balanced_cells import fingerprint_indices
    from prime.data.pred_profile_clusters import (
        assign_pred_profile_clusters,
        fit_pred_profile_clusters,
    )
    from prime.data.wilds_loader import load_amazon_splits
    from prime.workers.ensemble import load_dotenv_if_present
    from prime.workers.scorer import Scorer

    load_dotenv_if_present()
    cfg = load_config(args.config)
    splits = load_amazon_splits(cfg.dataset, seed=args.seed)
    train, val, test = splits["train"], splits["validation"], splits["test"]
    seed_prompt = (ROOT / "prompts/initial_prompt.txt").read_text(encoding="utf-8")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "_train_seed_preds.npy"
    scorer = Scorer(cfg.ensemble, label_space="ordinal5", use_mock=args.mock)

    if cache.is_file() and len(np.load(cache)) == len(train.texts):
        train_preds = np.load(cache)
        print(f"[e6] reuse train preds {cache}", flush=True)
    else:
        print(f"[e6] scoring train n={len(train.texts)} for pred_profile…", flush=True)
        train_preds = np.asarray(
            scorer.predict_batch(train.texts, seed_prompt, labels_for_mock=train.labels).preds,
            dtype=np.int16,
        )
        np.save(cache, train_preds)

    n_clusters = int(cfg.clusters.n_clusters)
    user_to_c, centroids, diag = fit_pred_profile_clusters(
        np.asarray(train.user_ids),
        train_preds,
        n_clusters=n_clusters,
        seed=int(cfg.clusters.seed),
        min_users_per_cluster=int(cfg.clusters.min_users_per_cluster),
    )
    print(f"[e6] fitted clusters K={len(centroids)} diag_keys={list(diag)[:8]}", flush=True)

    def assign_split(split, name: str):
        print(f"[e6] scoring {name} n={len(split.texts)} for assign…", flush=True)
        cpath = out / f"_{name}_seed_preds.npy"
        if cpath.is_file() and len(np.load(cpath)) == len(split.texts):
            preds = np.load(cpath)
        else:
            preds = np.asarray(
                scorer.predict_batch(split.texts, seed_prompt, labels_for_mock=split.labels).preds,
                dtype=np.int16,
            )
            np.save(cpath, preds)
        u2c = assign_pred_profile_clusters(
            np.asarray(split.user_ids),
            preds,
            centroids,
            diag["scaler_mean"],
            diag["scaler_scale"],
        )
        cids = [int(u2c[int(u)]) for u in split.user_ids]
        return preds, cids, u2c

    # Assign val/test (also caches seed preds for later reuse as seed baseline)
    _, val_cids, val_u2c = assign_split(val, "validation")
    _, test_cids, test_u2c = assign_split(test, "test")

    clusters = sorted(set(int(c) for c in test_cids))
    bins = [0, 1, 2]

    def build(name: str, split, cids, per_cell: int, seed: int):
        idxs, used = _sample_cells(
            split.labels,
            cids,
            clusters=clusters,
            bins=bins,
            per_cell=per_cell,
            seed=seed,
        )
        fp = fingerprint_indices(idxs)
        payload = {
            "name": name,
            "source_split": split.name if hasattr(split, "name") else name,
            "indices": idxs,
            "fingerprint": fp,
            "n": len(idxs),
            "design": {
                "clusters": clusters,
                "rating_bins": {"0": "1-2", "1": "3", "2": "4-5"},
                "target_per_cell": per_cell,
                "per_cell_used": used,
                "seed": seed,
                "geometry": "pred_profile",
            },
            "labels": [int(split.labels[i]) for i in idxs],
            "user_ids": [str(split.user_ids[i]) for i in idxs],
            "cluster_ids": [int(cids[i]) for i in idxs],
            "texts": [split.texts[i] for i in idxs],
        }
        return payload

    # Avoid storing huge texts in the index-only json; write slim + materialized.
    def save_pair(payload: dict):
        name = payload["name"]
        slim = {k: v for k, v in payload.items() if k != "texts"}
        (out / f"{name}.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")
        (out / f"{name}_materialized.json").write_text(json.dumps(payload), encoding="utf-8")
        print(
            f"[e6] wrote {name} n={payload['n']} fp={payload['fingerprint']} "
            f"cells_nonzero={sum(1 for v in payload['design']['per_cell_used'].values() if v)}",
            flush=True,
        )

    test_fixed = build("test_fixed", test, test_cids, args.test_per_cell, args.seed)
    d_dev = build("d_dev", val, val_cids, args.dev_per_cell, args.seed + 17)
    save_pair(test_fixed)
    save_pair(d_dev)

    # Cluster artifacts for later targeted builds / PRIME runs
    (out / "pred_profile_fit.json").write_text(
        json.dumps(
            {
                "centroids": np.asarray(centroids).tolist(),
                "scaler_mean": np.asarray(diag["scaler_mean"]).tolist(),
                "scaler_scale": np.asarray(diag["scaler_scale"]).tolist(),
                "n_clusters": len(centroids),
                "train_user_to_cluster": {str(k): int(v) for k, v in user_to_c.items()},
            }
        ),
        encoding="utf-8",
    )
    (out / "cluster_assign_validation.json").write_text(
        json.dumps({str(k): int(v) for k, v in val_u2c.items()}), encoding="utf-8"
    )
    (out / "cluster_assign_test.json").write_text(
        json.dumps({str(k): int(v) for k, v in test_u2c.items()}), encoding="utf-8"
    )

    # Floor: all-3 predictions → report metrics for sanity
    from e6_metrics import all_metrics

    y = np.asarray(test_fixed["labels"])
    c = np.asarray(test_fixed["cluster_ids"])
    u = np.arange(len(y))
    floor = all_metrics(np.full_like(y, 3), y, c, u)
    print(json.dumps({"test_fixed_n": len(y), "floor_all3": floor}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

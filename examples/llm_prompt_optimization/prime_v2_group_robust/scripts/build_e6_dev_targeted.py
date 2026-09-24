#!/usr/bin/env python
"""Build d_dev_targeted: same n≈900, concentrate on k=3 worst clusters (F9).

Worst clusters are chosen from **seed on uniform d_dev** (no test leakage).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/E6_amazon_category_controls/config.yaml",
    )
    ap.add_argument(
        "--fixed-dir",
        type=Path,
        default=ROOT / "experiments/E6_amazon_category_controls/fixed_sets",
    )
    ap.add_argument("--k-worst", type=int, default=3)
    ap.add_argument("--per-cell", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    from e6_metrics import all_metrics, collapse_rating, group_macros
    from prime.config import load_config
    from prime.data.balanced_cells import fingerprint_indices
    from prime.data.wilds_loader import load_amazon_splits
    from prime.workers.ensemble import load_dotenv_if_present
    from prime.workers.scorer import Scorer

    load_dotenv_if_present()
    fixed = Path(args.fixed_dir)
    d_dev = json.loads((fixed / "d_dev_materialized.json").read_text(encoding="utf-8"))
    y = np.asarray(d_dev["labels"])
    c = np.asarray(d_dev["cluster_ids"])
    texts = d_dev["texts"]

    seed_prompt = (ROOT / "prompts/initial_prompt.txt").read_text(encoding="utf-8")
    preds_path = fixed / "_d_dev_seed_preds_for_targeted.npy"
    if preds_path.is_file() and len(np.load(preds_path)) == len(texts):
        preds = np.load(preds_path)
    else:
        cfg = load_config(args.config)
        scorer = Scorer(cfg.ensemble, label_space="ordinal5", use_mock=args.mock)
        print(f"[e6f9] scoring seed on uniform d_dev n={len(texts)}", flush=True)
        preds = np.asarray(
            scorer.predict_batch(texts, seed_prompt, labels_for_mock=list(y)).preds,
            dtype=np.int16,
        )
        np.save(preds_path, preds)

    g = group_macros(preds, y, c)
    worst = sorted(g.items(), key=lambda kv: kv[1])[: args.k_worst]
    targets = [int(g_) for g_, _ in worst]
    print(f"[e6f9] seed macros={ {int(k): round(v,4) for k,v in g.items()} }", flush=True)
    print(f"[e6f9] target clusters={targets} macros={[round(g[t],4) for t in targets]}", flush=True)

    cfg = load_config(args.config)
    splits = load_amazon_splits(cfg.dataset, seed=args.seed)
    val = splits["validation"]
    # Full val cluster assignment from fixed-set build artifacts.
    assign = json.loads((fixed / "cluster_assign_validation.json").read_text(encoding="utf-8"))
    val_cids = [int(assign[str(u)]) for u in val.user_ids]

    by: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, (lab, cid) in enumerate(zip(val.labels, val_cids)):
        b = collapse_rating(int(lab))
        by[(int(cid), b)].append(i)

    rng = np.random.RandomState(args.seed + 17)
    chosen: list[int] = []
    used: dict[str, int] = {}
    # Iterate all clusters seen on test/d_dev design for RNG parity-ish order.
    clusters = sorted(set(int(x) for x in c.tolist()))
    for g_ in clusters:
        for b in (0, 1, 2):
            pool = by.get((g_, b), [])
            k = min(args.per_cell, len(pool)) if g_ in targets else 0
            if k == 0:
                used[f"g{g_}_b{b}"] = 0
                continue
            pick = rng.choice(pool, size=k, replace=False)
            chosen.extend(int(i) for i in pick)
            used[f"g{g_}_b{b}"] = k
    chosen = sorted(chosen)
    payload = {
        "name": "d_dev_targeted",
        "source_split": "validation",
        "indices": chosen,
        "fingerprint": fingerprint_indices(chosen),
        "n": len(chosen),
        "design": {
            "target_clusters": targets,
            "seed_macros_on_uniform_d_dev": {str(k): float(v) for k, v in g.items()},
            "rating_bins": {"0": "1-2", "1": "3", "2": "4-5"},
            "target_per_cell": args.per_cell,
            "per_cell_used": used,
            "seed": args.seed + 17,
            "geometry": "pred_profile",
            "selected_by": "seed macro-within-cluster on uniform d_dev",
        },
        "labels": [int(val.labels[i]) for i in chosen],
        "user_ids": [str(val.user_ids[i]) for i in chosen],
        "cluster_ids": [int(val_cids[i]) for i in chosen],
        "texts": [val.texts[i] for i in chosen],
    }
    slim = {k: v for k, v in payload.items() if k != "texts"}
    (fixed / "d_dev_targeted.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")
    (fixed / "d_dev_targeted_materialized.json").write_text(json.dumps(payload), encoding="utf-8")
    print(
        f"[e6f9] wrote d_dev_targeted n={payload['n']} fp={payload['fingerprint']}",
        flush=True,
    )
    # Quick seed score on targeted for later regret calc.
    m_u = all_metrics(preds, y, c)
    print(f"[e6f9] seed on uniform d_dev cvar25={m_u['cvar25']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

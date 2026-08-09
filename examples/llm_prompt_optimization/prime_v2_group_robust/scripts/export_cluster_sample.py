#!/usr/bin/env python
"""Export a human-readable sample of users/reviews per style cluster (E0 geometry)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1]


def _trunc(s: str, max_chars: int) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= max_chars else s[: max_chars - 1] + "…"


def _build_user_reviews(split, mapping, preds_arr=None):
    by_u: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for i, (t, y, u) in enumerate(zip(split.texts, split.labels, split.user_ids)):
        cid = mapping.get(int(u))
        if cid is None:
            continue
        rec: Dict[str, Any] = {"text": t, "label": int(y), "idx": i}
        if preds_arr is not None:
            rec["pred"] = int(preds_arr[i])
            rec["correct"] = bool(preds_arr[i] == y)
        by_u[int(u)].append(rec)
    return by_u


def main() -> int:
    from prime.config import load_config
    from prime.data.clustering import (
        assign_users_to_clusters,
        fit_style_clusters,
    )
    from prime.data.wilds_loader import load_amazon_splits, subsample_split

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=PKG_ROOT / "experiments/E0_proxy_diag/config.yaml",
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=PKG_ROOT / "results/E0_proxy_diag/seed42_20260727_111020",
    )
    p.add_argument("--k", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--users-per-cluster", type=int, default=3)
    p.add_argument("--reviews-per-user", type=int, default=2)
    p.add_argument("--max-chars", type=int, default=400)
    args = p.parse_args()

    cfg = load_config(args.config)
    seed = args.seed
    k = args.k
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    splits = load_amazon_splits(cfg.dataset, seed=seed)
    train, val = splits["train"], splits["validation"]
    cap = cfg.experiment.smoke_max_examples or 600
    if len(val) > cap:
        val = subsample_split(val, cap, seed)
    if len(train) > max(cap * 2, 400):
        train = subsample_split(train, max(cap * 2, 400), seed)

    preds_path = run_dir / "val_predictions.npy"
    preds = np.load(preds_path) if preds_path.is_file() else None

    cluster_cfg = replace(cfg.clusters, n_clusters=k, control="none", seed=seed)
    art = fit_style_clusters(
        train,
        cluster_cfg,
        dataset_cfg=cfg.dataset,
        fit_mode="label_free",
        min_reviews_for_fit=cfg.data_roles.min_reviews_for_fit,
        max_k=k,
    )
    fit_map = dict(art.user_to_cluster)
    val_map = assign_users_to_clusters(val, art, cluster_cfg, dataset_cfg=cfg.dataset)
    descriptors = (art.diagnostics or {}).get("descriptors", {}) or {}

    train_by_u = _build_user_reviews(train, fit_map)
    val_by_u = _build_user_reviews(val, val_map, preds)
    rng = np.random.RandomState(seed)

    lines: List[str] = [
        f"# Cluster sample — E0, K={k}, full_T, seed={seed}",
        "",
        f"- run_dir: `{run_dir}`",
        f"- train fit users: {len(fit_map)} | val assigned users: {len(val_map)}",
        f"- train reviews: {len(train)} | val reviews: {len(val)}",
        f"- sample: ≤{args.users_per_cluster} users/cluster, "
        f"{args.reviews_per_user} reviews/user (truncated {args.max_chars} chars)",
        "- Fit = train users that shaped k-means; Val = OOD users assigned to centroids",
        "",
    ]
    out: Dict[str, Any] = {"K": k, "seed": seed, "clusters": {}}

    def sample_users(users: List[int], by_u: Dict[int, List[Dict[str, Any]]]) -> List[int]:
        if not users:
            return []
        users_sorted = sorted(users, key=lambda u: -len(by_u.get(u, [])))
        n = args.users_per_cluster
        if len(users_sorted) <= n:
            return users_sorted
        pick = users_sorted[: n - 1]
        rest = users_sorted[n - 1 :]
        pick.append(int(rng.choice(rest)))
        return pick

    def append_block(
        title: str,
        users: List[int],
        by_u: Dict[int, List[Dict[str, Any]]],
        is_val: bool,
    ) -> List[Dict[str, Any]]:
        lines.append(f"### {title}")
        lines.append("")
        if not users:
            lines.append("_empty_")
            lines.append("")
            return []
        samples: List[Dict[str, Any]] = []
        for u in sample_users(users, by_u):
            revs = by_u.get(u, [])
            user_sample: Dict[str, Any] = {"user_id": u, "n_reviews": len(revs), "reviews": []}
            if is_val and revs and "correct" in revs[0]:
                acc = float(np.mean([r["correct"] for r in revs]))
                user_sample["ensemble_acc"] = acc
                lines.append(f"#### User `{u}` — {len(revs)} reviews, ensemble acc={acc:.2f}")
            else:
                lines.append(f"#### User `{u}` — {len(revs)} reviews")
            lines.append("")
            order = sorted(range(len(revs)), key=lambda i: len(revs[i]["text"]))
            chosen: List[int] = []
            if order:
                chosen.append(order[0])
            if len(order) > 1:
                chosen.append(order[-1])
            chosen = chosen[: args.reviews_per_user]
            for j in chosen:
                r = revs[j]
                meta = f"label={r['label']}"
                if "pred" in r:
                    flag = "OK" if r["correct"] else "WRONG"
                    meta += f", pred={r['pred']}, {flag}"
                lines.append(f"- ({meta}) {_trunc(r['text'], args.max_chars)}")
                user_sample["reviews"].append({kk: r[kk] for kk in r if kk != "idx"})
            lines.append("")
            samples.append(user_sample)
        return samples

    for cid in range(int(art.n_clusters)):
        desc = descriptors.get(str(cid), descriptors.get(cid, "")) or ""
        fit_users = [u for u, c in fit_map.items() if int(c) == cid]
        val_users = [u for u, c in val_map.items() if int(c) == cid]
        lines.append(f"## Cluster {cid}")
        lines.append("")
        lines.append(f"**Descriptor:** {desc if desc else '(none)'}")
        lines.append(f"- fit users: {len(fit_users)} | val users: {len(val_users)}")
        lines.append("")
        payload = {
            "descriptor": desc,
            "fit_users_n": len(fit_users),
            "val_users_n": len(val_users),
            "fit_sample": append_block("Fit (train)", fit_users, train_by_u, False),
            "val_sample": append_block("Val (OOD assigned)", val_users, val_by_u, True),
        }
        out["clusters"][str(cid)] = payload

    md_path = run_dir / f"cluster_sample_K{k}.md"
    json_path = run_dir / f"cluster_sample_K{k}.json"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""
Deep offline analysis of a finished E0 run (0 new API calls).

Answers:
  1. Is the LOO Spearman statistically meaningful at this K (permutation test)?
  2. Does cluster membership predict user accuracy at all (Kruskal-Wallis,
     user-level permutation of cluster labels)?
  3. Where do errors actually live (per-gold-label, per-cluster label mix)?
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))

from scripts.run_e0_proxy_diag import _feature_variant_assign  # noqa: E402


def kruskal_h(groups: List[np.ndarray]) -> float:
    """Kruskal-Wallis H on user accuracies grouped by cluster (ties-corrected)."""
    all_vals = np.concatenate(groups)
    n = len(all_vals)
    # average ranks
    order = np.argsort(all_vals, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    sx = all_vals[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    h = 0.0
    pos = 0
    for g in groups:
        m = len(g)
        r_mean = float(np.mean(ranks[pos : pos + m]))
        h += m * (r_mean - (n + 1) / 2.0) ** 2
        pos += m
    h *= 12.0 / (n * (n + 1))
    # tie correction
    _, counts = np.unique(all_vals, return_counts=True)
    tie = 1.0 - float(np.sum(counts**3 - counts)) / (n**3 - n)
    return h / tie if tie > 0 else h


def main() -> int:
    from prime.config import load_config
    from prime.data.wilds_loader import load_amazon_splits, subsample_split
    from prime.experiment.proxy_validation import (
        corr_cvar_vs_rworst_from_predictions,
        spearman_correlation,
    )
    from prime.fitness.metrics import per_user_accuracy
    import logging

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=PKG_ROOT / "experiments/E0_proxy_diag/config_large.yaml")
    p.add_argument("--run-dir", type=Path, default=PKG_ROOT / "results/E0_proxy_diag_large/seed42_20260727_115955")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--n-perm", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING)
    log = logging.getLogger("analyze")

    cfg = load_config(args.config)
    run_dir = args.run_dir.resolve()
    preds = np.load(run_dir / "val_predictions.npy")

    splits = load_amazon_splits(cfg.dataset, seed=args.seed)
    train, val = splits["train"], splits["validation"]
    cap = cfg.experiment.smoke_max_examples or 600
    if len(val) > cap:
        val = subsample_split(val, cap, args.seed)
    if len(train) > max(cap * 2, 400):
        train = subsample_split(train, max(cap * 2, 400), args.seed)
    assert len(preds) == len(val)

    gold = np.asarray(val.labels)
    uids = np.asarray(val.user_ids)

    out: Dict[str, object] = {"K": args.k, "n_perm": args.n_perm}
    print(f"=== Deep E0 analysis | K={args.k} full_T | val={len(val)} ex / {len(set(val.user_ids))} users ===")

    # ---------- 1. Global error structure (cluster-independent) ----------
    print("\n--- 1. Error structure by gold label ---")
    lab_stats = {}
    for lab in range(1, 6):
        m = gold == lab
        if not np.any(m):
            continue
        acc = float(np.mean(preds[m] == lab))
        # most common confusion
        wrong = preds[m][preds[m] != lab]
        conf = None
        if len(wrong):
            v, c = np.unique(wrong, return_counts=True)
            conf = int(v[np.argmax(c)])
        lab_stats[lab] = {"n": int(np.sum(m)), "acc": acc, "top_confusion": conf}
        print(f"  gold={lab}: n={np.sum(m):4d}  acc={acc:.3f}  top_confusion->{conf}")
    out["label_stats"] = lab_stats

    ua = per_user_accuracy(preds, gold, uids)
    accs = np.array(list(ua.values()))
    print(f"\n  user acc: mean={accs.mean():.3f} std={accs.std():.3f} min={accs.min():.3f} p10={np.percentile(accs,10):.3f}")

    # Label mix of worst users vs rest
    worst_users = [u for u, a in ua.items() if a <= np.percentile(accs, 10)]
    rest_users = [u for u, a in ua.items() if a > np.percentile(accs, 10)]
    def label_mix(users):
        m = np.isin(uids, users)
        v, c = np.unique(gold[m], return_counts=True)
        tot = c.sum()
        return {int(k): round(float(x) / tot, 3) for k, x in zip(v, c)}
    print(f"  label mix worst-10% users: {label_mix(worst_users)}")
    print(f"  label mix rest:            {label_mix(rest_users)}")
    out["worst_label_mix"] = label_mix(worst_users)
    out["rest_label_mix"] = label_mix(rest_users)

    # ---------- 2. Cluster fit and per-cluster stats ----------
    cfg_seed = replace(cfg, clusters=replace(cfg.clusters, seed=args.seed))
    art, val_c, mapping = _feature_variant_assign(train, val, cfg_seed, args.k, "full_T", cfg.dataset, log)
    cids = np.asarray(val_c.example_cluster_ids)

    print(f"\n--- 2. Per-cluster stats (K={art.n_clusters}) ---")
    user_cluster: Dict[int, int] = {}
    for u in np.unique(uids):
        vals, counts = np.unique(cids[uids == u], return_counts=True)
        user_cluster[int(u)] = int(vals[np.argmax(counts)])
    cluster_stats = {}
    groups = []
    for c in sorted(set(user_cluster.values())):
        users_c = [u for u, cc in user_cluster.items() if cc == c]
        a = np.array([ua[u] for u in users_c])
        groups.append(a)
        m = np.isin(uids, users_c)
        mix = label_mix(users_c)
        desc = (art.diagnostics or {}).get("descriptors", {}).get(str(c), "")
        cluster_stats[c] = {
            "n_users": len(users_c),
            "user_acc_mean": float(a.mean()),
            "user_acc_std": float(a.std()),
            "label_mix": mix,
        }
        print(f"  cluster {c}: users={len(users_c):3d}  acc={a.mean():.3f}±{a.std():.3f}  labels={mix}")
        if desc:
            print(f"    {desc[:100]}")
    out["cluster_stats"] = {str(k): v for k, v in cluster_stats.items()}

    # ---------- 3. Honest significance tests ----------
    print(f"\n--- 3. Significance (user-level permutation, {args.n_perm} perms) ---")
    metrics = corr_cvar_vs_rworst_from_predictions(
        preds, gold, uids, cids,
        cvar_quantile=cfg.fitness.cvar_quantile,
        beta_a=cfg.fitness.beta_a, beta_b=cfg.fitness.beta_b,
    )
    obs_loo = metrics["spearman_user_acc_vs_cluster_acc"]
    print(f"  observed LOO Spearman (avg-rank fix): {obs_loo}")

    h_obs = kruskal_h(groups)
    print(f"  observed Kruskal-Wallis H: {h_obs:.3f} (df={len(groups)-1})")

    # Permutation null: shuffle user->cluster assignments
    rng = np.random.RandomState(args.seed)
    users_arr = np.array(sorted(user_cluster.keys()))
    clusters_arr = np.array([user_cluster[u] for u in users_arr])
    acc_arr = np.array([ua[int(u)] for u in users_arr])

    null_h = np.empty(args.n_perm)
    null_rho = np.empty(args.n_perm)
    for t in range(args.n_perm):
        perm = rng.permutation(clusters_arr)
        g = [acc_arr[perm == c] for c in sorted(set(perm.tolist())) if np.any(perm == c)]
        null_h[t] = kruskal_h(g)
        # LOO rho under permuted assignment (user-level approximation:
        # cluster acc from user accs excluding self)
        rho_vals_u, rho_vals_c = [], []
        for c in set(perm.tolist()):
            m = perm == c
            if m.sum() < 2:
                continue
            s = acc_arr[m].sum()
            n_c = m.sum()
            for a in acc_arr[m]:
                rho_vals_u.append(a)
                rho_vals_c.append((s - a) / (n_c - 1))
        r = spearman_correlation(rho_vals_u, rho_vals_c)
        null_rho[t] = r if r is not None else 0.0

    p_h = float(np.mean(null_h >= h_obs))
    print(f"  P(H_null >= H_obs) = {p_h:.4f}")
    # Same user-level LOO rho for observed clusters (comparable to null)
    rho_u, rho_c = [], []
    for c in set(clusters_arr.tolist()):
        m = clusters_arr == c
        if m.sum() < 2:
            continue
        s = acc_arr[m].sum()
        n_c = m.sum()
        for a in acc_arr[m]:
            rho_u.append(a)
            rho_c.append((s - a) / (n_c - 1))
    obs_rho_ul = spearman_correlation(rho_u, rho_c) or 0.0
    p_rho = float(np.mean(null_rho >= obs_rho_ul))
    print(f"  observed user-level LOO rho: {obs_rho_ul:.4f}")
    print(f"  null LOO rho: mean={null_rho.mean():.4f} std={null_rho.std():.4f} (NOT centered at 0 if biased)")
    print(f"  P(rho_null >= rho_obs) = {p_rho:.4f}")
    out["tests"] = {
        "kruskal_h_obs": h_obs,
        "p_kruskal": p_h,
        "loo_rho_obs": obs_rho_ul,
        "null_rho_mean": float(null_rho.mean()),
        "null_rho_std": float(null_rho.std()),
        "p_rho": p_rho,
    }

    # ---------- 4. Length as difficulty driver ----------
    print("\n--- 4. Is difficulty driven by length / label mix? ---")
    user_meanlen = {}
    for u in np.unique(uids):
        m = uids == u
        user_meanlen[int(u)] = float(np.mean([len(t) for t, mm in zip(val.texts, m) if mm]))
    ml = np.array([user_meanlen[int(u)] for u in users_arr])
    r_len = spearman_correlation(ml, acc_arr)
    frac5 = np.array([
        float(np.mean(gold[uids == u] == 5)) for u in users_arr
    ])
    r_5 = spearman_correlation(frac5, acc_arr)
    frac_mid = np.array([
        float(np.mean((gold[uids == u] >= 2) & (gold[uids == u] <= 4))) for u in users_arr
    ])
    r_mid = spearman_correlation(frac_mid, acc_arr)
    print(f"  Spearman(user mean length, acc) = {r_len:.4f}")
    print(f"  Spearman(share of 5-star gold, acc) = {r_5:.4f}")
    print(f"  Spearman(share of 2-4 gold, acc) = {r_mid:.4f}")
    out["difficulty_drivers"] = {"rho_len": r_len, "rho_frac5": r_5, "rho_frac_mid": r_mid}

    out_path = run_dir / f"deep_analysis_K{args.k}.json"
    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

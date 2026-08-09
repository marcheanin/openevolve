#!/usr/bin/env python
"""
Recompute E0 on a finished run with:
  - permutation Kruskal–Wallis as primary gate (OBSERVATIONS P1)
  - variants: full_T, emb_only, shuffle, pred_profile

pred_profile uses ensemble predictions only (worker votes optional if saved).
For this offline pass, pred_profile is fit on val users (no train preds cached);
features never use gold — KW permutation remains a valid test.

0 new API calls.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))

from scripts.run_e0_proxy_diag import _feature_variant_assign  # noqa: E402


def _decide(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    go: best structural variant has kw_p_value < 0.05 AND
        (no shuffle row OR structural H > shuffle H mean + margin).
    weak_go: best p < 0.10
    else no_go_proxy
    """
    structural = {"full_T", "emb_only", "pred_profile"}
    struct_rows = [r for r in rows if r["variant"] in structural and r.get("kw_p_mean") is not None]
    shuffle_rows = [r for r in rows if r["variant"] == "shuffle" and r.get("kw_h_mean") is not None]
    decision = "no_go_proxy"
    recommended_k = None
    recommended_variant = None
    best_p = None
    best_h = None
    shuffle_h = None
    if not struct_rows:
        return {
            "decision": decision,
            "recommended_K": None,
            "recommended_variant": None,
            "best_kw_p": None,
            "best_kw_h": None,
            "shuffle_kw_h": None,
        }
    # Prefer low p, then high H
    best = min(
        struct_rows,
        key=lambda r: (
            r["kw_p_mean"] if r["kw_p_mean"] is not None else 1.0,
            -(r["kw_h_mean"] if r["kw_h_mean"] is not None else 0.0),
        ),
    )
    recommended_k = int(best["K"])
    recommended_variant = best["variant"]
    best_p = best["kw_p_mean"]
    best_h = best["kw_h_mean"]
    if shuffle_rows:
        # compare against shuffle at same K if present, else best shuffle H
        same_k = [r for r in shuffle_rows if r["K"] == recommended_k]
        sh = same_k[0] if same_k else max(shuffle_rows, key=lambda r: r["kw_h_mean"] or 0.0)
        shuffle_h = sh["kw_h_mean"]
    margin = 0.5  # H units
    beats_shuffle = shuffle_h is None or (best_h is not None and best_h > (shuffle_h or 0.0) + margin)
    if best_p is not None and best_p < 0.05 and beats_shuffle:
        decision = "go"
    elif best_p is not None and best_p < 0.10:
        decision = "weak_go"
    return {
        "decision": decision,
        "recommended_K": recommended_k,
        "recommended_variant": recommended_variant,
        "best_kw_p": best_p,
        "best_kw_h": best_h,
        "shuffle_kw_h": shuffle_h,
    }


def main() -> int:
    from prime.config import load_config
    from prime.data.pred_profile_clusters import fit_pred_profile_clusters
    from prime.data.wilds_loader import load_amazon_splits, subsample_split
    from prime.experiment.proxy_validation import corr_cvar_vs_rworst_from_predictions

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        type=Path,
        default=PKG_ROOT / "results/E0_proxy_diag_large/seed42_20260727_115955",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=PKG_ROOT / "experiments/E0_proxy_diag/config_large.yaml",
    )
    p.add_argument("--n-perm", type=int, default=2000)
    args = p.parse_args()

    run_dir = args.run_dir.resolve()
    out_dir = run_dir / "recompute_kw_pred"
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(out_dir / "recompute.log", encoding="utf-8"),
        ],
    )
    log = logging.getLogger("recompute_kw")

    cfg = load_config(args.config)
    seed = 42
    preds = np.load(run_dir / "val_predictions.npy")
    wp_path = run_dir / "val_worker_predictions.npy"
    worker_preds = np.load(wp_path) if wp_path.is_file() else None
    if worker_preds is None:
        log.warning("No val_worker_predictions.npy — pred_profile without disagreement dim")

    splits = load_amazon_splits(cfg.dataset, seed=seed)
    train, val = splits["train"], splits["validation"]
    cap = cfg.experiment.smoke_max_examples or 600
    if len(val) > cap:
        val = subsample_split(val, cap, seed)
    if len(train) > max(cap * 2, 400):
        train = subsample_split(train, max(cap * 2, 400), seed)
    assert len(preds) == len(val)

    import yaml

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    e0 = raw.get("e0", {})
    k_grid = list(e0.get("k_grid", [3, 4, 6, 8]))
    # Force include pred_profile; keep style + shuffle
    variants = ["full_T", "emb_only", "pred_profile", "shuffle"]
    cluster_seeds = list(e0.get("cluster_seeds", [42, 43, 44]))

    log.info(
        "Recompute KW+pred | val=%d/%d users | K=%s | variants=%s | n_perm=%d",
        len(val),
        len(set(val.user_ids)),
        k_grid,
        variants,
        args.n_perm,
    )

    events_path = out_dir / "events.jsonl"
    if events_path.exists():
        events_path.unlink()

    rows: List[Dict[str, Any]] = []

    def _metrics_for(cluster_ids: np.ndarray, k: int, variant: str, cseed: int, descriptors=None):
        from prime.experiment.proxy_validation import permutation_kruskal_wallis
        from prime.fitness.metrics import per_user_accuracy

        m = corr_cvar_vs_rworst_from_predictions(
            preds,
            np.asarray(val.labels),
            np.asarray(val.user_ids),
            cluster_ids,
            cvar_quantile=cfg.fitness.cvar_quantile,
            beta_a=cfg.fitness.beta_a,
            beta_b=cfg.fitness.beta_b,
        )
        # Recompute KW with requested n_perm / seed (override default inside corr)
        ua = per_user_accuracy(preds, np.asarray(val.labels), np.asarray(val.user_ids))
        user_cluster: Dict[int, int] = {}
        uids = np.asarray(val.user_ids)
        for u in np.unique(uids):
            cids = cluster_ids[uids == u]
            vals, counts = np.unique(cids, return_counts=True)
            user_cluster[int(u)] = int(vals[np.argmax(counts)])
        kw = permutation_kruskal_wallis(ua, user_cluster, n_perm=args.n_perm, seed=int(cseed))
        m.update({f"kw_{k}": v for k, v in kw.items()})
        m.update(
            {
                "K_requested": int(k),
                "K_effective": int(len(set(user_cluster.values()))),
                "variant": variant,
                "cluster_seed": int(cseed),
                "descriptors": descriptors,
            }
        )
        return m

    for variant in variants:
        for k in k_grid:
            seed_metrics: List[Dict[str, Any]] = []
            for cseed in cluster_seeds:
                cfg_seed = replace(cfg, clusters=replace(cfg.clusters, seed=int(cseed)))
                descriptors = None
                if variant == "pred_profile":
                    mapping, _cent, diag = fit_pred_profile_clusters(
                        np.asarray(val.user_ids),
                        preds,
                        n_clusters=int(k),
                        seed=int(cseed),
                        worker_preds=worker_preds,
                    )
                    descriptors = diag.get("descriptors")
                    cluster_ids = np.array(
                        [mapping[int(u)] for u in val.user_ids], dtype=np.int64
                    )
                elif variant == "shuffle":
                    # Style geometry + shuffled val labels (falsification)
                    _art, val_c, mapping = _feature_variant_assign(
                        train, val, cfg_seed, int(k), "shuffle", cfg.dataset, log
                    )
                    cluster_ids = np.asarray(val_c.example_cluster_ids)
                    descriptors = (_art.diagnostics or {}).get("descriptors")
                else:
                    _art, val_c, mapping = _feature_variant_assign(
                        train, val, cfg_seed, int(k), variant, cfg.dataset, log
                    )
                    cluster_ids = np.asarray(val_c.example_cluster_ids)
                    descriptors = (_art.diagnostics or {}).get("descriptors")

                metrics = _metrics_for(cluster_ids, k, variant, cseed, descriptors)
                seed_metrics.append(metrics)
                rec = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "event": "cell",
                    "k": k,
                    "variant": variant,
                    "seed": cseed,
                    "kw_h": metrics.get("kw_kruskal_h"),
                    "kw_p": metrics.get("kw_p_value"),
                    "spearman_loo": metrics.get("spearman_user_acc_vs_cluster_acc"),
                    "CVaR_cluster": metrics["CVaR_cluster"],
                    "R_worst": metrics["R_worst"],
                }
                with open(events_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                log.info(
                    "K=%s %s seed=%s | H=%.3f p=%s LOO=%s",
                    k,
                    variant,
                    cseed,
                    metrics.get("kw_kruskal_h") if metrics.get("kw_kruskal_h") is not None else float("nan"),
                    f"{metrics['kw_p_value']:.4f}" if metrics.get("kw_p_value") is not None else "n/a",
                    metrics.get("spearman_user_acc_vs_cluster_acc"),
                )

            hs = [m["kw_kruskal_h"] for m in seed_metrics if m.get("kw_kruskal_h") is not None]
            ps = [m["kw_p_value"] for m in seed_metrics if m.get("kw_p_value") is not None]
            spears = [
                m["spearman_user_acc_vs_cluster_acc"]
                for m in seed_metrics
                if m.get("spearman_user_acc_vs_cluster_acc") is not None
            ]
            rows.append(
                {
                    "K": int(k),
                    "variant": variant,
                    "R_worst": float(seed_metrics[0]["R_worst"]),
                    "CVaR_cluster_mean": float(np.mean([m["CVaR_cluster"] for m in seed_metrics])),
                    "kw_h_mean": float(np.mean(hs)) if hs else None,
                    "kw_h_std": float(np.std(hs)) if hs else None,
                    "kw_p_mean": float(np.mean(ps)) if ps else None,
                    "kw_p_std": float(np.std(ps)) if ps else None,
                    "spearman_loo_mean": float(np.mean(spears)) if spears else None,
                    "n_seeds": len(seed_metrics),
                    "per_seed": seed_metrics,
                }
            )

    gate = _decide(rows)
    report = {
        "experiment": "E0_recompute_kw_pred",
        "source_run": str(run_dir),
        "primary_metric": "permutation_kruskal_wallis",
        "n_perm": args.n_perm,
        **gate,
        "criteria": {
            "go": "min kw_p_mean < 0.05 AND H beats shuffle by ≥0.5",
            "weak_go": "min kw_p_mean < 0.10",
            "no_go_proxy": "otherwise",
            "note": "LOO Spearman retained as diagnostic only (biased under null; OBSERVATIONS M2)",
        },
        "rows": rows,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "e0_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )

    md_lines = [
        f"# E0 recompute (KW + pred_profile) — `{gate['decision']}`",
        "",
        f"- Source: `{run_dir.name}` (0 new API calls)",
        f"- Primary gate: **permutation Kruskal–Wallis** (n_perm={args.n_perm})",
        f"- Recommended: **K={gate['recommended_K']}** / `{gate['recommended_variant']}`",
        f"- Best kw_p={gate['best_kw_p']}  H={gate['best_kw_h']}  shuffle_H={gate['shuffle_kw_h']}",
        "",
        "| K | variant | H±std | p±std | LOO Spearman | CVaR |",
        "|---|---------|-------|-------|--------------|------|",
    ]
    for r in sorted(rows, key=lambda x: (x["variant"], x["K"])):
        h = f"{r['kw_h_mean']:.3f}±{r['kw_h_std']:.3f}" if r["kw_h_mean"] is not None else "n/a"
        pv = f"{r['kw_p_mean']:.4f}±{r['kw_p_std']:.4f}" if r["kw_p_mean"] is not None else "n/a"
        lo = f"{r['spearman_loo_mean']:.4f}" if r["spearman_loo_mean"] is not None else "n/a"
        md_lines.append(
            f"| {r['K']} | {r['variant']} | {h} | {pv} | {lo} | {r['CVaR_cluster_mean']:.4f} |"
        )
    md_lines.extend(
        [
            "",
            "See `experiments/OBSERVATIONS.md` and source `FINDINGS.md`.",
            "",
        ]
    )
    (out_dir / "e0_report.md").write_text("\n".join(md_lines), encoding="utf-8")

    log.info(
        "Decision=%s K=%s variant=%s p=%s H=%s",
        gate["decision"],
        gate["recommended_K"],
        gate["recommended_variant"],
        gate["best_kw_p"],
        gate["best_kw_h"],
    )
    print(json.dumps(gate, indent=2))
    return 0 if gate["decision"] in ("go", "weak_go") else 2


if __name__ == "__main__":
    raise SystemExit(main())

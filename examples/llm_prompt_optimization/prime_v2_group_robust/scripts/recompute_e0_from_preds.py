#!/usr/bin/env python
"""
Recompute E0 cluster grid from a finished live run's val_predictions.npy.

No new LLM calls — only cluster fit + LOO Spearman / emb_only / shuffle.
Writes into run_dir/recompute_loo/.
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

# Reuse E0 helpers
from scripts.run_e0_proxy_diag import _feature_variant_assign  # noqa: E402


def main() -> int:
    from prime.config import load_config
    from prime.data.wilds_loader import load_amazon_splits, subsample_split
    from prime.experiment.proxy_validation import corr_cvar_vs_rworst_from_predictions

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        type=Path,
        default=PKG_ROOT / "results/E0_proxy_diag/seed42_20260727_111020",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=PKG_ROOT / "experiments/E0_proxy_diag/config.yaml",
    )
    args = p.parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = run_dir / "recompute_loo"
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
    log = logging.getLogger("recompute_e0")

    cfg = load_config(args.config)
    seed = 42
    preds = np.load(run_dir / "val_predictions.npy")

    splits = load_amazon_splits(cfg.dataset, seed=seed)
    train, val = splits["train"], splits["validation"]
    cap = cfg.experiment.smoke_max_examples or 600
    if len(val) > cap:
        val = subsample_split(val, cap, seed)
    if len(train) > max(cap * 2, 400):
        train = subsample_split(train, max(cap * 2, 400), seed)

    assert len(preds) == len(val), f"preds {len(preds)} != val {len(val)}"

    import yaml

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    e0 = raw.get("e0", {})
    k_grid = list(e0.get("k_grid", [3, 4, 6, 8]))
    variants = list(e0.get("variants", ["full_T", "emb_only", "shuffle"]))
    cluster_seeds = list(e0.get("cluster_seeds", [42, 43, 44]))

    log.info(
        "Recompute LOO E0 | val=%d users=%d | K=%s variants=%s",
        len(val),
        len(set(val.user_ids)),
        k_grid,
        variants,
    )
    log.info(
        "Why val: official WILDS OOD split — proxy must predict unseen-user "
        "tail, not train ID. Cap is budget (3 workers × N reviews), not a "
        "scientific minimum."
    )

    events_path = out_dir / "events.jsonl"
    if events_path.exists():
        events_path.unlink()

    rows: List[Dict[str, Any]] = []
    for variant in variants:
        for k in k_grid:
            seed_metrics: List[Dict[str, Any]] = []
            for cseed in cluster_seeds:
                cfg_seed = replace(cfg, clusters=replace(cfg.clusters, seed=int(cseed)))
                art, val_c, _mapping = _feature_variant_assign(
                    train, val, cfg_seed, int(k), variant, cfg.dataset, log
                )
                metrics = corr_cvar_vs_rworst_from_predictions(
                    preds,
                    np.asarray(val.labels),
                    np.asarray(val.user_ids),
                    np.asarray(val_c.example_cluster_ids),
                    cvar_quantile=cfg.fitness.cvar_quantile,
                    beta_a=cfg.fitness.beta_a,
                    beta_b=cfg.fitness.beta_b,
                )
                metrics.update(
                    {
                        "K_requested": int(k),
                        "K_effective": int(art.n_clusters),
                        "variant": variant,
                        "cluster_seed": int(cseed),
                        "anova_f": (art.diagnostics or {}).get("anova_f_mean_rating"),
                        "n_fit_users": (art.diagnostics or {}).get("n_fit_users"),
                        "descriptors": (art.diagnostics or {}).get("descriptors"),
                    }
                )
                seed_metrics.append(metrics)
                rec = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "event": "cell",
                    "k": k,
                    "variant": variant,
                    "seed": cseed,
                    "spearman_loo": metrics["spearman_user_acc_vs_cluster_acc"],
                    "spearman_leaky": metrics.get("spearman_leaky"),
                    "CVaR_cluster": metrics["CVaR_cluster"],
                    "R_worst": metrics["R_worst"],
                    "n_users_in_corr": metrics["n_users_in_corr"],
                    "n_loo_skipped": metrics.get("n_loo_skipped_singleton"),
                }
                with open(events_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                log.info(
                    "K=%s %s seed=%s LOO=%.4f leaky=%s n=%s skip=%s",
                    k,
                    variant,
                    cseed,
                    metrics["spearman_user_acc_vs_cluster_acc"] or float("nan"),
                    f"{metrics.get('spearman_leaky'):.4f}"
                    if metrics.get("spearman_leaky") is not None
                    else "n/a",
                    metrics["n_users_in_corr"],
                    metrics.get("n_loo_skipped_singleton"),
                )

            spears = [
                m["spearman_user_acc_vs_cluster_acc"]
                for m in seed_metrics
                if m["spearman_user_acc_vs_cluster_acc"] is not None
            ]
            spears_leaky = [
                m["spearman_leaky"] for m in seed_metrics if m.get("spearman_leaky") is not None
            ]
            rows.append(
                {
                    "K": int(k),
                    "variant": variant,
                    "R_worst": float(seed_metrics[0]["R_worst"]),
                    "CVaR_cluster_mean": float(np.mean([m["CVaR_cluster"] for m in seed_metrics])),
                    "CVaR_cluster_std": float(np.std([m["CVaR_cluster"] for m in seed_metrics])),
                    "spearman_mean": float(np.mean(spears)) if spears else None,
                    "spearman_std": float(np.std(spears)) if spears else None,
                    "spearman_leaky_mean": float(np.mean(spears_leaky)) if spears_leaky else None,
                    "anova_f_mean": float(
                        np.mean(
                            [m["anova_f"] for m in seed_metrics if m["anova_f"] is not None]
                        )
                    )
                    if any(m["anova_f"] is not None for m in seed_metrics)
                    else None,
                    "n_seeds": len(seed_metrics),
                    "per_seed": seed_metrics,
                }
            )

    structural = {"full_T", "emb_only"}
    full_rows = [r for r in rows if r["variant"] in structural and r["spearman_mean"] is not None]
    shuffle_rows = [r for r in rows if r["variant"] == "shuffle" and r["spearman_mean"] is not None]
    decision = "no_go_proxy"
    recommended_k: Optional[int] = None
    recommended_variant: Optional[str] = None
    spear = 0.0
    shuffle_spear = None
    if full_rows:
        best = max(full_rows, key=lambda r: r["spearman_mean"] if r["spearman_mean"] is not None else -1.0)
        recommended_k = int(best["K"])
        recommended_variant = best["variant"]
        spear = best["spearman_mean"] or 0.0
        if shuffle_rows:
            shuffle_spear = max(
                shuffle_rows,
                key=lambda r: r["spearman_mean"] if r["spearman_mean"] is not None else -1.0,
            )["spearman_mean"]
        if spear > 0.05 and (shuffle_spear is None or spear > (shuffle_spear or 0.0) + 0.02):
            decision = "go"
        elif spear > 0.0:
            decision = "weak_go"

    report = {
        "experiment": "E0_recompute_loo",
        "source_run": str(run_dir),
        "decision": decision,
        "recommended_K": recommended_k,
        "recommended_variant": recommended_variant,
        "best_spearman_loo": spear,
        "best_shuffle_spearman_loo": shuffle_spear,
        "criteria": {
            "go": "spearman_LOO(full_T|emb_only) > 0.05 AND beats shuffle by ≥0.02",
            "weak_go": "spearman_LOO > 0",
            "no_go_proxy": "LOO Spearman ≤ 0",
            "metric": "leave-one-user-out",
        },
        "why_val": (
            "Official WILDS Amazon validation is the OOD user split used as the "
            "selection/proxy target in the paper protocol. Measuring proxy quality "
            "on train would overfit style geometry to ID users and not test whether "
            "types transfer to unseen sources — the claim we need for E1."
        ),
        "why_val_small": (
            f"This run uses max_val_users={cfg.dataset.max_val_users} × "
            f"max_reviews_per_user={cfg.dataset.max_reviews_per_user} "
            f"(→ {len(val)} reviews) as an API-budget cap for the single ensemble "
            "pass (3 workers × N). It is not the full WILDS val (~100k). After a "
            "honest LOO go, a larger-cap confirmation pass is the optional Step 4."
        ),
        "rows": rows,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "e0_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )

    md = [
        f"# E0 recompute (LOO) — `{decision}`",
        "",
        f"- Source: `{run_dir.name}` (reused `val_predictions.npy`, 0 new API calls)",
        f"- Recommended: **K={recommended_k}** / `{recommended_variant}`",
        f"- Best LOO Spearman: **{spear:.4f}** | best shuffle LOO: "
        f"**{shuffle_spear if shuffle_spear is not None else 'n/a'}**",
        "",
        "## Why we measure on val (and why it is small)",
        "",
        report["why_val"],
        "",
        report["why_val_small"],
        "",
        "## Grid (LOO primary)",
        "",
        "| K | variant | CVaR±std | Spearman_LOO | leaky | ANOVA-F |",
        "|---|---------|----------|--------------|-------|---------|",
    ]
    for r in sorted(rows, key=lambda x: (x["variant"], x["K"])):
        md.append(
            f"| {r['K']} | {r['variant']} | "
            f"{r['CVaR_cluster_mean']:.4f}±{r['CVaR_cluster_std']:.4f} | "
            f"{r['spearman_mean'] if r['spearman_mean'] is not None else 'n/a'} | "
            f"{r.get('spearman_leaky_mean') if r.get('spearman_leaky_mean') is not None else 'n/a'} | "
            f"{r['anova_f_mean'] if r['anova_f_mean'] is not None else 'n/a'} |"
        )
    md.extend(
        [
            "",
            "## Interpretation notes",
            "",
            "- **Spearman_LOO** = user accuracy vs cluster accuracy **without** that user's reviews.",
            "- **leaky** = old definition (diagnostic); often inflated when clusters are small.",
            "- **shuffle** must fall near 0 under LOO if the metric is honest; if shuffle ≈ structural, types are not predictive.",
            "- **emb_only** drops length/punct/caps to test whether semantics alone recover the signal.",
            "",
        ]
    )
    (out_dir / "e0_report.md").write_text("\n".join(md), encoding="utf-8")
    log.info("Decision=%s K=%s variant=%s LOO=%.4f shuffle=%s", decision, recommended_k, recommended_variant, spear, shuffle_spear)
    log.info("Wrote %s", out_dir / "e0_report.md")
    print(json.dumps({"decision": decision, "recommended_K": recommended_k, "recommended_variant": recommended_variant, "best_spearman_loo": spear, "shuffle": shuffle_spear}, indent=2))
    return 0 if decision in ("go", "weak_go") else 2


if __name__ == "__main__":
    raise SystemExit(main())

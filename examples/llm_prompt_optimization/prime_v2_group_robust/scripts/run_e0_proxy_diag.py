#!/usr/bin/env python3
"""E0 proxy diagnostics (SPEC §7 / Р12).

One ensemble pass of the *starting* prompt on official WILDS val (capped).
Predictions are reused while cluster assignments are recomputed for a grid of K
and feature variants. Reports Spearman(CVaR_cluster surrogate structure vs
user-level R_worst diagnostics) and picks a recommended K for E1.

Usage:
  cd prime_v2_group_robust
  python scripts/run_e0_proxy_diag.py --config experiments/E0_proxy_diag/config.yaml
  python scripts/run_e0_proxy_diag.py --config experiments/E0_proxy_diag/config_smoke.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))


def _setup_logging(run_dir: Path, verbose: bool) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("e0")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(run_dir / "e0.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def _load_prompt(project_root: Path, prompt_path: str) -> str:
    p = Path(prompt_path)
    if not p.is_file():
        p = project_root / prompt_path
    return p.read_text(encoding="utf-8")


def _make_run_dir(results_dir: Path, name: str, seed: int) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_dir / name / f"seed{seed}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _synthetic_splits(n_users: int = 40, reviews_per: int = 5, seed: int = 42):
    """Fallback when WILDS is unavailable (smoke / offline)."""
    from prime.data.wilds_loader import ReviewSplit

    rng = np.random.RandomState(seed)
    texts, labels, user_ids = [], [], []
    for u in range(n_users):
        for _ in range(reviews_per):
            length = int(rng.randint(20, 200))
            texts.append("word " * (length // 5))
            labels.append(int(rng.randint(1, 6)))
            user_ids.append(u)
    # Train / val user-disjoint
    train_users = set(range(0, n_users * 2 // 3))
    val_users = set(range(n_users * 2 // 3, n_users))

    def _subset(name, users):
        idxs = [i for i, u in enumerate(user_ids) if u in users]
        return ReviewSplit(
            name=name,
            texts=[texts[i] for i in idxs],
            labels=[labels[i] for i in idxs],
            user_ids=[user_ids[i] for i in idxs],
        )

    return {"train": _subset("train", train_users), "validation": _subset("validation", val_users)}


def _run_inference(
    texts: List[str],
    labels: List[int],
    prompt: str,
    cfg,
    use_mock: bool,
    seed: int,
    log: logging.Logger,
    budget,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    from prime.workers.ensemble import LLMWorker, mock_predict, parallel_predict

    n_workers = max(1, len(cfg.ensemble.workers))
    if use_mock:
        log.info("Inference: MOCK ensemble (n_workers=%d, n=%d)", n_workers, len(texts))
        ens, wp = mock_predict(
            texts, labels, n_workers=n_workers, seed=seed, aggregation=cfg.ensemble.aggregation
        )
        budget.charge("e0_infer", n_calls=0, note="mock_no_api")
        return np.asarray(ens), [np.asarray(w) for w in wp]

    workers = [
        LLMWorker(
            w.name,
            api_base=cfg.ensemble.api_base,
            temperature=w.temperature,
            max_tokens=w.max_tokens,
            timeout=cfg.ensemble.timeout,
            max_retries=cfg.ensemble.max_retries,
            reasoning_effort=getattr(w, "reasoning_effort", "none"),
        )
        for w in cfg.ensemble.workers
    ]
    log.info(
        "Inference: LIVE OpenRouter (%d workers × %d texts = %d calls)",
        len(workers),
        len(texts),
        len(workers) * len(texts),
    )
    t0 = time.time()
    ens, wp = parallel_predict(
        workers,
        texts,
        prompt,
        max_parallel=cfg.ensemble.max_parallel,
        tie_break=cfg.ensemble.tie_break,
        aggregation=cfg.ensemble.aggregation,
    )
    elapsed = time.time() - t0
    n_calls = len(workers) * len(texts)
    budget.charge("e0_infer", n_calls=n_calls, note=f"val_infer_{len(texts)}")
    log.info("Inference done in %.1fs (%d calls charged)", elapsed, n_calls)
    return np.asarray(ens), [np.asarray(w) for w in wp]


def _fit_and_assign(
    train,
    val,
    cfg,
    k: int,
    control: str,
    dataset_cfg,
    log: logging.Logger,
    fit_mode: str = "label_free",
):
    from dataclasses import replace

    from prime.data.clustering import (
        assign_users_to_clusters,
        attach_example_clusters,
        fit_style_clusters,
    )

    # Fit without shuffle; for Р13 ablation we permute *val* assignments so the
    # falsification hits the evaluation geometry (centroids alone would leave
    # OOD assignment unchanged).
    cluster_cfg = replace(cfg.clusters, n_clusters=k, control="none", seed=cfg.clusters.seed)
    log.debug(
        "Fitting clusters K=%d control=%s seed=%d fit_mode=%s",
        k,
        control,
        cluster_cfg.seed,
        fit_mode,
    )
    art = fit_style_clusters(
        train,
        cluster_cfg,
        dataset_cfg=dataset_cfg,
        fit_mode=fit_mode,
        min_reviews_for_fit=cfg.data_roles.min_reviews_for_fit,
        max_k=k,
    )
    mapping = assign_users_to_clusters(val, art, cluster_cfg, dataset_cfg=dataset_cfg)
    if control == "shuffle":
        rng = np.random.RandomState(cfg.clusters.seed + 9973 + int(k))
        uids = list(mapping.keys())
        shuffled = rng.permutation([mapping[u] for u in uids])
        mapping = {u: int(c) for u, c in zip(uids, shuffled)}
        log.debug("Shuffled val assignments for %d users (R13 ablation)", len(uids))
    val_c = attach_example_clusters(val, mapping)
    return art, val_c, mapping


def _feature_variant_assign(
    train,
    val,
    cfg,
    k: int,
    variant: str,
    dataset_cfg,
    log: logging.Logger,
):
    """
    variant:
      - full_T: label-free (emb PCA + length/punct/caps)
      - emb_only: PCA(embedding) only (length-dominance ablation)
      - shuffle: full_T geometry + permuted val assignments (Р13)
    """
    if variant == "shuffle":
        return _fit_and_assign(
            train, val, cfg, k, "shuffle", dataset_cfg, log, fit_mode="label_free"
        )
    if variant == "emb_only":
        return _fit_and_assign(
            train, val, cfg, k, "none", dataset_cfg, log, fit_mode="emb_only"
        )
    return _fit_and_assign(
        train, val, cfg, k, "none", dataset_cfg, log, fit_mode="label_free"
    )


def run_e0(config_path: Path, smoke: bool = False, verbose: bool = True) -> Dict[str, Any]:
    from prime.config import load_config
    from prime.data.wilds_loader import load_amazon_splits, subsample_split
    from prime.experiment.budget import TokenTracker
    from prime.experiment.proxy_validation import corr_cvar_vs_rworst_from_predictions
    from prime.workers.ensemble import load_dotenv_if_present

    overrides: Dict[str, Any] = {}
    if smoke:
        overrides["experiment"] = {"smoke": True}
    if verbose:
        overrides.setdefault("experiment", {})["verbose"] = True

    cfg = load_config(config_path, overrides=overrides if overrides else None)
    seed = cfg.active_learning.seed
    results_root = Path(cfg.experiment.results_dir or "results")
    if not results_root.is_absolute():
        results_root = PKG_ROOT / results_root
    run_dir = _make_run_dir(results_root, cfg.experiment.name or "E0_proxy_diag", seed)
    log = _setup_logging(run_dir, verbose=verbose or cfg.experiment.verbose)

    log.info("=" * 60)
    log.info("E0 PROXY DIAGNOSTICS — start")
    log.info("config=%s  run_dir=%s", config_path, run_dir)
    log.info("=" * 60)

    load_dotenv_if_present()
    has_api = bool(os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"))
    use_mock = cfg.experiment.force_mock or not has_api or cfg.experiment.smoke
    log.info(
        "Backend: %s (force_mock=%s smoke=%s has_api=%s)",
        "MOCK" if use_mock else "LIVE",
        cfg.experiment.force_mock,
        cfg.experiment.smoke,
        has_api,
    )

    budget = TokenTracker.from_cfg(cfg.budget)
    prompt = _load_prompt(PKG_ROOT, cfg.prompt_path)
    log.info("Prompt loaded (%d chars) from %s", len(prompt), cfg.prompt_path)

    # --- data ---
    used_synthetic = False
    try:
        log.info("Loading official WILDS Amazon splits...")
        splits = load_amazon_splits(cfg.dataset, seed=seed)
        train, val = splits["train"], splits["validation"]
        log.info(
            "Loaded train=%d/%d users  val=%d/%d users",
            len(train),
            len(set(train.user_ids)),
            len(val),
            len(set(val.user_ids)),
        )
    except Exception as exc:
        log.warning("WILDS load failed (%s) — using synthetic splits", exc)
        splits = _synthetic_splits(seed=seed)
        train, val = splits["train"], splits["validation"]
        used_synthetic = True

    if cfg.experiment.smoke or cfg.experiment.smoke_max_examples:
        cap = cfg.experiment.smoke_max_examples or 200
        # Cap val for the single inference pass (E0 cost control).
        if len(val) > cap:
            log.info("Capping val %d → %d (smoke_max_examples)", len(val), cap)
            val = subsample_split(val, cap, seed)
        if len(train) > max(cap * 2, 400):
            train = subsample_split(train, max(cap * 2, 400), seed)
            log.info("Capped train → %d for cluster fit", len(train))

    # Persist run metadata
    meta = {
        "experiment": "E0_proxy_diag",
        "config": str(config_path),
        "seed": seed,
        "use_mock": use_mock,
        "used_synthetic": used_synthetic,
        "train_n": len(train),
        "val_n": len(val),
        "train_users": len(set(train.user_ids)),
        "val_users": len(set(val.user_ids)),
        "k_grid": getattr(cfg.experiment, "k_grid", None),
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # --- ONE inference pass ---
    log.info("--- Stage: single ensemble inference on val ---")
    preds, worker_preds = _run_inference(
        val.texts, val.labels, prompt, cfg, use_mock, seed, log, budget
    )
    np.save(run_dir / "val_predictions.npy", preds)
    # Persist worker votes for offline disagreement / pred_profile features (OBSERVATIONS O3).
    try:
        wp_arr = np.asarray(worker_preds, dtype=np.int16)
        np.save(run_dir / "val_worker_predictions.npy", wp_arr)
        log.info("Saved worker predictions shape=%s", tuple(wp_arr.shape))
    except Exception as exc:
        log.warning("Could not save worker predictions: %s", exc)
    log.info(
        "Predictions saved. R_global(raw)=%.4f",
        float(np.mean(preds == np.asarray(val.labels))),
    )

    # --- K / variant grid (reuse predictions) ---
    k_grid = list(getattr(cfg, "_e0_k_grid", None) or [4, 6, 8, 10, 12])
    # Allow override from raw yaml via experiment section stored on cfg
    e0_k = None
    try:
        raw = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    except Exception:
        raw = {}
    # Read k_grid from config file again for clarity
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        raw_yaml = yaml.safe_load(f) or {}
    # merge includes roughly — k_grid lives on experiment
    k_grid = list(raw_yaml.get("e0", {}).get("k_grid", k_grid))
    variants = list(raw_yaml.get("e0", {}).get("variants", ["full_T", "shuffle"]))
    cluster_seeds = list(raw_yaml.get("e0", {}).get("cluster_seeds", [cfg.clusters.seed]))

    log.info("K grid=%s  variants=%s  cluster_seeds=%s", k_grid, variants, cluster_seeds)
    log.info("--- Stage: refit clusters / recompute proxy metrics (0 extra LLM calls) ---")

    rows: List[Dict[str, Any]] = []
    events_path = run_dir / "events.jsonl"

    def _log_event(event: str, **fields: Any) -> None:
        rec = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
        with open(events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        log.debug("event=%s %s", event, {k: fields[k] for k in list(fields)[:6]})

    from dataclasses import replace

    for variant in variants:
        for k in k_grid:
            seed_metrics: List[Dict[str, Any]] = []
            for cseed in cluster_seeds:
                cfg_seed = replace(cfg, clusters=replace(cfg.clusters, seed=int(cseed)))
                try:
                    art, val_c, mapping = _feature_variant_assign(
                        train, val, cfg_seed, int(k), variant, cfg.dataset, log
                    )
                except Exception as exc:
                    log.exception("Fit failed for K=%s variant=%s seed=%s: %s", k, variant, cseed, exc)
                    _log_event("fit_failed", k=k, variant=variant, seed=cseed, error=str(exc))
                    continue

                cluster_ids = np.asarray(val_c.example_cluster_ids)
                metrics = corr_cvar_vs_rworst_from_predictions(
                    preds,
                    np.asarray(val.labels),
                    np.asarray(val.user_ids),
                    cluster_ids,
                    cvar_quantile=cfg.fitness.cvar_quantile,
                    beta_a=cfg.fitness.beta_a,
                    beta_b=cfg.fitness.beta_b,
                    n_perm=1000,
                    perm_seed=int(cseed),
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
                _log_event(
                    "cell",
                    k=k,
                    variant=variant,
                    seed=cseed,
                    R_worst=metrics["R_worst"],
                    CVaR_cluster=metrics["CVaR_cluster"],
                    spearman=metrics["spearman_user_acc_vs_cluster_acc"],
                    spearman_leaky=metrics.get("spearman_leaky"),
                    n_loo_skipped=metrics.get("n_loo_skipped_singleton"),
                    K_effective=metrics["K_effective"],
                )
                log.info(
                    "K=%d/%d  %s  seed=%s | R_worst=%.4f CVaR=%.4f Spearman_LOO=%s leaky=%s n_users=%d",
                    k,
                    art.n_clusters,
                    variant,
                    cseed,
                    metrics["R_worst"],
                    metrics["CVaR_cluster"],
                    metrics["spearman_user_acc_vs_cluster_acc"],
                    metrics.get("spearman_leaky"),
                    metrics["n_users_in_corr"],
                )

            if not seed_metrics:
                continue
            # Aggregate across cluster seeds
            spears = [
                m["spearman_user_acc_vs_cluster_acc"]
                for m in seed_metrics
                if m["spearman_user_acc_vs_cluster_acc"] is not None
            ]
            spears_leaky = [
                m["spearman_leaky"] for m in seed_metrics if m.get("spearman_leaky") is not None
            ]
            kw_hs = [m["kw_kruskal_h"] for m in seed_metrics if m.get("kw_kruskal_h") is not None]
            kw_ps = [m["kw_p_value"] for m in seed_metrics if m.get("kw_p_value") is not None]
            rows.append(
                {
                    "K": int(k),
                    "variant": variant,
                    "K_effective_mean": float(np.mean([m["K_effective"] for m in seed_metrics])),
                    "R_worst": float(seed_metrics[0]["R_worst"]),  # preds fixed → same R_worst
                    "CVaR_cluster_mean": float(np.mean([m["CVaR_cluster"] for m in seed_metrics])),
                    "CVaR_cluster_std": float(np.std([m["CVaR_cluster"] for m in seed_metrics])),
                    "spearman_mean": float(np.mean(spears)) if spears else None,
                    "spearman_std": float(np.std(spears)) if spears else None,
                    "spearman_leaky_mean": float(np.mean(spears_leaky)) if spears_leaky else None,
                    "kw_h_mean": float(np.mean(kw_hs)) if kw_hs else None,
                    "kw_h_std": float(np.std(kw_hs)) if kw_hs else None,
                    "kw_p_mean": float(np.mean(kw_ps)) if kw_ps else None,
                    "kw_p_std": float(np.std(kw_ps)) if kw_ps else None,
                    "anova_f_mean": float(
                        np.mean([m["anova_f"] for m in seed_metrics if m["anova_f"] is not None])
                    )
                    if any(m["anova_f"] is not None for m in seed_metrics)
                    else None,
                    "n_seeds": len(seed_metrics),
                    "per_seed": seed_metrics,
                }
            )

    # --- decide: permutation KW primary (OBSERVATIONS P1); LOO Spearman diagnostic ---
    structural = {"full_T", "emb_only", "pred_profile"}
    full_rows = [r for r in rows if r["variant"] in structural and r.get("kw_p_mean") is not None]
    shuffle_rows = [r for r in rows if r["variant"] == "shuffle" and r.get("kw_h_mean") is not None]
    recommended_k = None
    recommended_variant = None
    decision = "no_go_proxy"
    if full_rows:
        best = min(
            full_rows,
            key=lambda r: (
                r["kw_p_mean"] if r["kw_p_mean"] is not None else 1.0,
                -(r["kw_h_mean"] if r["kw_h_mean"] is not None else 0.0),
            ),
        )
        recommended_k = int(best["K"])
        recommended_variant = best["variant"]
        best_p = best["kw_p_mean"]
        best_h = best["kw_h_mean"]
        shuffle_h = None
        if shuffle_rows:
            same = [r for r in shuffle_rows if r["K"] == recommended_k]
            sh = same[0] if same else max(shuffle_rows, key=lambda r: r["kw_h_mean"] or 0.0)
            shuffle_h = sh["kw_h_mean"]
        beats_shuffle = shuffle_h is None or (best_h is not None and best_h > (shuffle_h or 0.0) + 0.5)
        if best_p is not None and best_p < 0.05 and beats_shuffle:
            decision = "go"
        elif best_p is not None and best_p < 0.10:
            decision = "weak_go"
        else:
            decision = "no_go_proxy"
        log.info(
            "Decision=%s  recommended_K=%s variant=%s  kw_p=%.4f H=%.3f shuffle_H=%s",
            decision,
            recommended_k,
            recommended_variant,
            best_p or -1.0,
            best_h or -1.0,
            shuffle_h,
        )
    else:
        log.warning("No successful structural cells — cannot recommend K")

    report = {
        "experiment": "E0_proxy_diag",
        "decision": decision,
        "recommended_K": recommended_k,
        "recommended_variant": recommended_variant,
        "criteria": {
            "go": "kw_p_mean < 0.05 AND H beats shuffle by ≥0.5",
            "weak_go": "kw_p_mean < 0.10",
            "no_go_proxy": "otherwise — fix features/K before E1",
            "metric": "permutation Kruskal–Wallis on user accuracies (OBSERVATIONS P1)",
            "diagnostic": "LOO Spearman (biased under null — do not gate)",
        },
        "rows": rows,
        "budget": budget.snapshot(),
        "meta": meta,
        "note": (
            "R_worst is identical across K for a fixed prediction vector; "
            "primary gate is permutation KW. LOO Spearman is diagnostic only."
        ),
    }
    (run_dir / "e0_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (run_dir / "budget_report.json").write_text(json.dumps(budget.snapshot(), indent=2), encoding="utf-8")

    # Markdown table
    md_lines = [
        f"# E0 Proxy Diagnostics — `{decision}`",
        "",
        f"- Recommended K: **{recommended_k}** ({recommended_variant})",
        f"- Primary metric: **permutation Kruskal–Wallis** (LOO Spearman = diagnostic)",
        f"- Backend: `{'mock' if use_mock else 'live'}`",
        f"- Val examples: {len(val)} / users: {len(set(val.user_ids))}",
        f"- Budget calls: {budget.total_calls}",
        "",
        "| K | variant | CVaR±std | KW H | KW p | LOO Spearman |",
        "|---|---------|----------|------|------|--------------|",
    ]
    for r in sorted(rows, key=lambda x: (x["variant"], x["K"])):
        md_lines.append(
            f"| {r['K']} | {r['variant']} | "
            f"{r['CVaR_cluster_mean']:.4f}±{r['CVaR_cluster_std']:.4f} | "
            f"{r.get('kw_h_mean') if r.get('kw_h_mean') is not None else 'n/a'} | "
            f"{r.get('kw_p_mean') if r.get('kw_p_mean') is not None else 'n/a'} | "
            f"{r['spearman_mean'] if r['spearman_mean'] is not None else 'n/a'} |"
        )
    md_lines.append("")
    md_lines.append("See `e0_report.json` and `e0.log` for full details.")
    (run_dir / "e0_report.md").write_text("\n".join(md_lines), encoding="utf-8")

    log.info("Wrote %s", run_dir / "e0_report.json")
    log.info("Wrote %s", run_dir / "e0_report.md")
    log.info(
        "E0 DONE — decision=%s recommended_K=%s variant=%s",
        decision,
        recommended_k,
        recommended_variant,
    )
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="E0 proxy diagnostics for GRAPE/PRIME v3")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    report = run_e0(args.config, smoke=args.smoke, verbose=not args.quiet)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "recommended_K": report["recommended_K"],
                "recommended_variant": report.get("recommended_variant"),
            },
            indent=2,
        )
    )
    return 0 if report["decision"] in ("go", "weak_go") else 2


if __name__ == "__main__":
    raise SystemExit(main())

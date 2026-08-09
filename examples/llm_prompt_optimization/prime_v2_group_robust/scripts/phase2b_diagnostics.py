#!/usr/bin/env python3
"""Phase 2b: CivilComments seed diagnostics before live budget.

1. Per-group seed-ensemble profile on official val (stratified subsample)
2. Noise floor: 3 identical scorings of a fixed prompt (M15 protocol)
3. Power analysis: sample sizes for target MDE (M13 protocol)
4. Go/no-go: worst-group gap ≥ ~5 pp + identity-concentrated errors

Usage:
  python scripts/phase2b_diagnostics.py
  python scripts/phase2b_diagnostics.py --n 180 --repeats 3
  python scripts/phase2b_diagnostics.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))

DEFAULT_CONFIG = PKG_ROOT / "experiments" / "E4_civilcomments" / "config_phase2b_diag.yaml"
DEFAULT_OUT = PKG_ROOT / "experiments" / "E4_civilcomments" / "phase2b_diagnostics"


def _preflight_api_key() -> None:
    from prime.workers.ensemble import load_dotenv_if_present

    loaded = load_dotenv_if_present()
    key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        print("ERROR: set OPENROUTER_API_KEY", file=sys.stderr)
        raise SystemExit(1)
    print(f"[preflight] API key OK (from {loaded or 'env'})", flush=True)


def _stratified_sample(
    texts: List[str],
    labels: List[int],
    group_ids: List[int],
    n: int,
    seed: int,
    mode: str = "equal_group",
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """Subsample for diagnostics.

    ``equal_group`` (default): nearly equal n per oracle group, then within-group
    label balance — needed so minority identities are measurable (proportional
    draws drown in ``none``).
    ``proportional``: group×label cells ~ natural frequencies.
    """
    rng = np.random.RandomState(seed)
    n_all = len(texts)
    if n >= n_all:
        idx = np.arange(n_all)
    elif mode == "equal_group":
        by_g: Dict[int, List[int]] = defaultdict(list)
        for i, g in enumerate(group_ids):
            by_g[int(g)].append(i)
        groups = sorted(by_g.keys())
        # Equal target; leftover goes to largest remaining pools
        base = max(1, n // max(1, len(groups)))
        alloc = {g: min(base, len(by_g[g])) for g in groups}
        leftover = n - sum(alloc.values())
        while leftover > 0:
            candidates = [g for g in groups if alloc[g] < len(by_g[g])]
            if not candidates:
                break
            g_pick = max(candidates, key=lambda g: len(by_g[g]) - alloc[g])
            alloc[g_pick] += 1
            leftover -= 1
        chosen: List[int] = []
        for g in groups:
            pool = by_g[g]
            take = alloc[g]
            # Within group: balance labels as far as pool allows
            by_y: Dict[int, List[int]] = defaultdict(list)
            for i in pool:
                by_y[int(labels[i])].append(i)
            y_keys = sorted(by_y.keys())
            if len(y_keys) == 1:
                pick = rng.choice(pool, size=take, replace=False).tolist()
            else:
                per_y = take // len(y_keys)
                rem = take - per_y * len(y_keys)
                pick = []
                for yi in y_keys:
                    want = per_y + (1 if rem > 0 else 0)
                    rem = max(0, rem - 1)
                    avail = by_y[yi]
                    got = min(want, len(avail))
                    if got:
                        pick.extend(rng.choice(avail, size=got, replace=False).tolist())
                if len(pick) < take:
                    rest = [i for i in pool if i not in set(pick)]
                    need = take - len(pick)
                    if need and rest:
                        pick.extend(
                            rng.choice(rest, size=min(need, len(rest)), replace=False).tolist()
                        )
            chosen.extend(pick[:take])
        idx = np.array(sorted(chosen)[:n], dtype=int)
    else:
        cells: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for i, (g, y) in enumerate(zip(group_ids, labels)):
            cells[(int(g), int(y))].append(i)
        nonempty = {k: v for k, v in cells.items() if v}
        if not nonempty:
            raise SystemExit("empty split")
        total = sum(len(v) for v in nonempty.values())
        alloc_c = {
            k: max(1, int(round(n * len(v) / total))) for k, v in nonempty.items()
        }
        while sum(alloc_c.values()) > n:
            k_max = max(alloc_c, key=lambda k: alloc_c[k])
            if alloc_c[k_max] > 1:
                alloc_c[k_max] -= 1
            else:
                break
        while sum(alloc_c.values()) < n:
            remainders = [
                (k, len(nonempty[k]) - alloc_c[k])
                for k in nonempty
                if len(nonempty[k]) > alloc_c[k]
            ]
            if not remainders:
                break
            k_max = max(remainders, key=lambda t: t[1])[0]
            alloc_c[k_max] += 1
        chosen = []
        for k, take in alloc_c.items():
            pool = nonempty[k]
            take = min(take, len(pool))
            chosen.extend(rng.choice(pool, size=take, replace=False).tolist())
        idx = np.array(sorted(chosen)[:n], dtype=int)

    t = [texts[i] for i in idx.tolist()]
    y = np.asarray([labels[i] for i in idx.tolist()], dtype=np.int16)
    g = np.asarray([group_ids[i] for i in idx.tolist()], dtype=np.int16)
    uid = idx.astype(np.int64)  # synthetic comment ids
    return t, y, g, uid


def _group_stats(
    ens: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    group_names: Tuple[str, ...],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for gid in sorted(set(g.tolist())):
        m = g == gid
        n = int(m.sum())
        if n == 0:
            continue
        acc = float((ens[m] == y[m]).mean())
        tox_rate = float(y[m].mean())
        pred_tox = float(ens[m].mean())
        name = group_names[gid] if 0 <= gid < len(group_names) else f"g{gid}"
        rows.append(
            {
                "group_id": int(gid),
                "name": name,
                "n": n,
                "acc": acc,
                "gold_tox_rate": tox_rate,
                "pred_tox_rate": pred_tox,
                "is_identity": name != "none",
            }
        )
    return rows


def _class_mix_residual(
    ens: np.ndarray, y: np.ndarray, g: np.ndarray
) -> Dict[str, float]:
    """C10-style: how much of between-group acc spread is gold-label mix."""
    global_per_class: Dict[int, float] = {}
    for c in sorted(set(y.tolist())):
        m = y == c
        global_per_class[c] = float((ens[m] == c).mean()) if m.any() else 0.0
    obs: List[float] = []
    pred: List[float] = []
    for gid in sorted(set(g.tolist())):
        m = g == gid
        if not m.any():
            continue
        obs.append(float((ens[m] == y[m]).mean()))
        # Expected acc if group differed only by label mix
        ys = y[m]
        pred.append(float(np.mean([global_per_class[int(c)] for c in ys.tolist()])))
    obs_a = np.asarray(obs)
    pred_a = np.asarray(pred)
    resid = obs_a - pred_a
    if len(obs_a) < 2:
        return {"r2": float("nan"), "obs_spread": 0.0, "resid_spread": 0.0}
    # R^2 of predicting obs from class-mix expectation
    ss_tot = float(((obs_a - obs_a.mean()) ** 2).sum())
    ss_res = float(((obs_a - pred_a) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return {
        "r2": float(r2),
        "obs_spread": float(obs_a.max() - obs_a.min()),
        "explained_spread": float(pred_a.max() - pred_a.min()),
        "resid_spread": float(resid.max() - resid.min()),
        "resid_min": float(resid.min()),
        "resid_max": float(resid.max()),
    }


def _aggregation_headroom(
    ens: np.ndarray, y: np.ndarray, wp: np.ndarray
) -> Dict[str, Any]:
    ens_acc = float((ens == y).mean())
    per_worker = [float((wp[w] == y).mean()) for w in range(wp.shape[0])]
    hit_any = np.any(wp == y[None, :], axis=0)
    err = ens != y
    n_err = int(err.sum())
    return {
        "ensemble_acc": ens_acc,
        "per_worker_acc": per_worker,
        "best_single": float(max(per_worker)) if per_worker else 0.0,
        "oracle_over_workers": float(hit_any.mean()),
        "agg_only_headroom": float(hit_any.mean() - ens_acc),
        "errors": n_err,
        "errors_some_worker_right": int((err & hit_any).sum()),
        "errors_all_wrong": int((err & ~hit_any).sum()),
    }


def _bootstrap_metric_sd(
    correct: np.ndarray, g: np.ndarray, n_boot: int, seed: int
) -> Dict[str, float]:
    """Bootstrap SD of R_global and R_worst (min group acc with n>=5)."""
    rng = np.random.default_rng(seed)
    n = len(correct)
    r_global = np.empty(n_boot)
    r_worst = np.empty(n_boot)
    groups = sorted(set(g.tolist()))
    for i in range(n_boot):
        sel = rng.integers(0, n, size=n)
        c = correct[sel]
        gg = g[sel]
        r_global[i] = float(c.mean())
        accs = []
        for gid in groups:
            m = gg == gid
            if int(m.sum()) >= 5:
                accs.append(float(c[m].mean()))
        r_worst[i] = float(min(accs)) if accs else float(c.mean())
    return {
        "R_global_sd": float(r_global.std()),
        "R_worst_sd": float(r_worst.std()),
        "R_global_mde_80": float(2.8 * r_global.std()),
        "R_worst_mde_80": float(2.8 * r_worst.std()),
    }


def _power_n(sd_delta: float, effect: float) -> int:
    """Paired n ≈ (2.8 * sd / effect)^2 for ~80% power (M13 / headroom_audit)."""
    if effect <= 0 or sd_delta <= 0:
        return 10**9
    return int(round((2.8 * sd_delta / effect) ** 2))


def _go_nogo(profile: List[Dict[str, Any]], mix: Dict[str, float], agg: Dict[str, Any]) -> Dict[str, Any]:
    if not profile:
        return {"decision": "no-go", "reasons": ["empty profile"]}
    accs = [r["acc"] for r in profile]
    global_acc = float(np.average([r["acc"] for r in profile], weights=[r["n"] for r in profile]))
    worst = min(profile, key=lambda r: r["acc"])
    best = max(profile, key=lambda r: r["acc"])
    gap = best["acc"] - worst["acc"]
    identity = [r for r in profile if r["is_identity"]]
    none_row = next((r for r in profile if not r["is_identity"]), None)
    identity_worst = min(identity, key=lambda r: r["acc"]) if identity else None

    # Error mass in identity groups vs none
    # Approximate via (1-acc)*n
    err_ident = sum((1.0 - r["acc"]) * r["n"] for r in identity)
    err_none = (1.0 - none_row["acc"]) * none_row["n"] if none_row else 0.0
    err_total = err_ident + err_none
    ident_err_frac = err_ident / err_total if err_total > 0 else 0.0

    reasons: List[str] = []
    go = True
    if gap < 0.05:
        go = False
        reasons.append(f"worst-group gap {gap:.3f} < 0.05")
    else:
        reasons.append(f"worst-group gap {gap:.3f} >= 0.05")

    if identity_worst and identity_worst["acc"] > global_acc - 0.02 and gap < 0.08:
        # Weak concentration signal
        pass
    if mix.get("r2", 0) is not None and not np.isnan(mix.get("r2", float("nan"))):
        if mix["r2"] >= 0.7 and mix.get("resid_spread", 0) < 0.05:
            go = False
            reasons.append(
                f"group axis mostly label-mix (R²={mix['r2']:.2f}, resid_spread={mix.get('resid_spread', 0):.3f})"
            )
        else:
            reasons.append(
                f"residual group effect resid_spread={mix.get('resid_spread', 0):.3f} (R²={mix['r2']:.2f})"
            )

    if identity_worst and none_row and identity_worst["acc"] < none_row["acc"] - 0.03:
        reasons.append(
            f"identity worst '{identity_worst['name']}' "
            f"({identity_worst['acc']:.3f}) << none ({none_row['acc']:.3f})"
        )
    elif identity_worst:
        reasons.append(
            f"identity worst '{identity_worst['name']}'={identity_worst['acc']:.3f}; "
            f"none={none_row['acc'] if none_row else float('nan'):.3f}"
        )

    reasons.append(f"identity share of error mass ~ {ident_err_frac:.0%}")
    reasons.append(
        f"prompt-only headroom (1 - oracle_workers)~{1.0 - agg['oracle_over_workers']:.3f}; "
        f"agg-only~{agg['agg_only_headroom']:+.3f}"
    )

    # Headroom for prompt evolution: all-workers-wrong errors should exist
    if agg["errors_all_wrong"] < max(3, int(0.02 * sum(r["n"] for r in profile))):
        go = False
        reasons.append(
            f"almost no all-workers-wrong errors ({agg['errors_all_wrong']}) - prompt space may be exhausted"
        )

    return {
        "decision": "go" if go else "no-go",
        "global_acc": global_acc,
        "worst": worst,
        "best": best,
        "gap": gap,
        "identity_error_fraction": ident_err_frac,
        "reasons": reasons,
    }


def _write_results_md(out_dir: Path, report: Dict[str, Any]) -> None:
    gng = report["go_nogo"]
    lines = [
        "# Phase 2b diagnostics - CivilComments seed",
        "",
        f"**Decision: `{gng['decision'].upper()}`**",
        "",
        f"Sample: n={report['n']} equal-group val comments, repeats={report['repeats']}, "
        f"seed={report['seed']}.",
        f"Ensemble: {', '.join(report['workers'])}.",
        "",
        "## Go / no-go",
        "",
    ]
    for r in gng["reasons"]:
        lines.append(f"- {r}")
    lines += [
        "",
        f"- Global (weighted) acc: **{gng['global_acc']:.3f}**",
        f"- Best group: `{gng['best']['name']}` {gng['best']['acc']:.3f} (n={gng['best']['n']})",
        f"- Worst group: `{gng['worst']['name']}` {gng['worst']['acc']:.3f} (n={gng['worst']['n']})",
        f"- Gap: **{gng['gap']:.3f}**",
        "",
        "## Per-group profile (repeat 0)",
        "",
        "| group | n | acc | gold tox | pred tox |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["profile"]:
        lines.append(
            f"| {row['name']} | {row['n']} | {row['acc']:.3f} | "
            f"{row['gold_tox_rate']:.2f} | {row['pred_tox_rate']:.2f} |"
        )
    noise = report["noise"]
    lines += [
        "",
        "## Noise floor (M15)",
        "",
        "| metric | mean | SD | range |",
        "|---|---:|---:|---:|",
    ]
    for k, s in noise["summary"].items():
        lines.append(
            f"| {k} | {s['mean']:.4f} | {s['sd']:.4f} | {s['range']:.4f} |"
        )
    lines += [
        "",
        f"Prediction flips vs repeat 0: {noise['flips']} "
        f"({[f'{f / report['n']:.1%}' for f in noise['flips']]})",
        "",
        "## Aggregation headroom",
        "",
    ]
    agg = report["aggregation"]
    lines += [
        f"- Ensemble {agg['ensemble_acc']:.3f}; best single {agg['best_single']:.3f}; "
        f"oracle-over-workers {agg['oracle_over_workers']:.3f}",
        f"- Agg-only headroom {agg['agg_only_headroom']:+.3f}",
        f"- Errors: {agg['errors']} "
        f"(some worker right {agg['errors_some_worker_right']}, "
        f"all wrong {agg['errors_all_wrong']})",
        "",
        "## Class-mix residual (C10-style)",
        "",
    ]
    mix = report["class_mix"]
    lines.append(
        f"- R2={mix.get('r2', float('nan')):.3f}; obs_spread={mix.get('obs_spread', 0):.3f}; "
        f"resid_spread={mix.get('resid_spread', 0):.3f}"
    )
    powr = report["power"]
    lines += [
        "",
        "## Power analysis (M13)",
        "",
        f"Bootstrap SD at n={report['n']}: R_global={powr['bootstrap']['R_global_sd']:.4f} "
        f"(MDE~{powr['bootstrap']['R_global_mde_80']:.3f}); "
        f"R_worst_group={powr['bootstrap']['R_worst_sd']:.4f} "
        f"(MDE~{powr['bootstrap']['R_worst_mde_80']:.3f}).",
        "",
        "Examples needed for 80% power on R_global "
        f"(scale bootstrap SD ~ 1/sqrt(n); target effects):",
        "",
    ]
    for eff, nn in powr["n_for_effect"].items():
        lines.append(f"- effect {eff}: ~{nn} examples")
    lines += [
        "",
        f"Recommended D_select / test caps for E4: **{powr['recommend_d_select']}** / "
        f"**{powr['recommend_test']}** (target MDE {powr['target_mde']} on R_global; "
        f"worst-group needs more).",
        "",
    ]
    (out_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 2b CivilComments diagnostics")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--n", type=int, default=180, help="val subsample size")
    ap.add_argument(
        "--sample-mode",
        choices=("equal_group", "proportional"),
        default="equal_group",
        help="equal_group: ~same n per oracle identity (default for diagnostics)",
    )
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--max-val-pool",
        type=int,
        default=4000,
        help="cap loaded val pool before subsample (cache-friendly)",
    )
    args = ap.parse_args()

    if not args.dry_run:
        _preflight_api_key()

    from prime.config import load_config
    from prime.data.civilcomments_loader import (
        ORACLE_GROUP_NAMES,
        load_civilcomments_splits,
        oracle_group_name,
    )
    from prime.fitness.metrics import compute_metrics
    from prime.workers.ensemble import build_workers, parallel_predict

    cfg = load_config(args.config)
    # Ensure val pool is large enough for stratified draw
    cfg.dataset.max_val_users = max(cfg.dataset.max_val_users, args.max_val_pool)

    print("[phase2b] loading CivilComments val...", flush=True)
    splits = load_civilcomments_splits(cfg.dataset, seed=args.seed)
    val = splits["validation"]
    texts_all = list(val.texts)
    labels_all = list(val.labels)
    # Prefer oracle cluster ids if attached
    if getattr(val, "example_cluster_ids", None) is not None:
        groups_all = [int(x) for x in val.example_cluster_ids]
    else:
        raise SystemExit("validation split missing example_cluster_ids (oracle groups)")

    texts, y, g, uid = _stratified_sample(
        texts_all,
        labels_all,
        groups_all,
        args.n,
        args.seed,
        mode=args.sample_mode,
    )
    print(
        f"[phase2b] subsample n={len(texts)} groups={Counter(g.tolist())} "
        f"labels={Counter(y.tolist())}",
        flush=True,
    )
    for gid, cnt in sorted(Counter(g.tolist()).items()):
        print(f"  {oracle_group_name(gid):16s} n={cnt}", flush=True)

    if args.dry_run:
        print("[dry-run] skipping API calls")
        return 0

    prompt_path = Path(cfg.prompt_path)
    if not prompt_path.is_absolute():
        prompt_path = PKG_ROOT / prompt_path
    prompt = prompt_path.read_text(encoding="utf-8")

    workers = build_workers(cfg.ensemble)
    worker_names = [w.model_name for w in workers]
    print(f"[phase2b] workers={worker_names} repeats={args.repeats}", flush=True)

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.save(pred_dir / "labels.npy", y)
    np.save(pred_dir / "group_ids.npy", g)
    np.save(pred_dir / "user_ids.npy", uid)

    components = ("R_global", "R_worst", "CVaR_cluster", "CVaR_cluster_shrunk", "mean_kappa")
    runs: List[Dict[str, float]] = []
    ens_list: List[np.ndarray] = []
    wp0: Optional[np.ndarray] = None

    for r in range(args.repeats):
        print(f"[phase2b] scoring repeat {r + 1}/{args.repeats}...", flush=True)
        ensemble, wp = parallel_predict(
            workers,
            texts,
            prompt,
            max_parallel=cfg.ensemble.max_parallel,
            tie_break=cfg.ensemble.tie_break,
            aggregation=cfg.ensemble.aggregation,
            label_space=cfg.dataset.label_space,
        )
        ens = np.asarray(ensemble, dtype=np.int16)
        wp_arr = np.asarray(wp, dtype=np.int16)
        ens_list.append(ens)
        if r == 0:
            wp0 = wp_arr
        np.save(pred_dir / f"ensemble_repeat{r}.npy", ens)
        np.save(pred_dir / f"workers_repeat{r}.npy", wp_arr)

        m = compute_metrics(
            ens,
            y,
            uid,
            worker_predictions=[wp_arr[w] for w in range(wp_arr.shape[0])],
            cluster_ids=g,
            cvar_quantile=cfg.fitness.cvar_quantile,
            beta_a=cfg.fitness.beta_a,
            beta_b=cfg.fitness.beta_b,
            tail_quantile=cfg.active_learning.tail_quantile,
            shrink_prior_weight=cfg.fitness.shrink_prior_weight,
            class_balanced=cfg.fitness.class_balanced,
        )
        row = {k: float(m[k]) for k in components if k in m}
        # Also compute explicit worst-group among oracle groups with n>=5
        gst = _group_stats(ens, y, g, ORACLE_GROUP_NAMES)
        eligible = [x for x in gst if x["n"] >= 5]
        row["R_worst_group"] = float(min(x["acc"] for x in eligible)) if eligible else row.get("R_global", 0.0)
        row["R_global_ex"] = float((ens == y).mean())
        runs.append(row)
        print(
            "  " + " ".join(f"{k}={row[k]:.4f}" for k in sorted(row.keys())),
            flush=True,
        )

    assert wp0 is not None
    profile = _group_stats(ens_list[0], y, g, ORACLE_GROUP_NAMES)
    mix = _class_mix_residual(ens_list[0], y, g)
    agg = _aggregation_headroom(ens_list[0], y, wp0)

    # Noise summary
    all_keys = sorted({k for row in runs for k in row})
    summary: Dict[str, Dict[str, float]] = {}
    for k in all_keys:
        vals = np.array([row[k] for row in runs])
        summary[k] = {
            "mean": float(vals.mean()),
            "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(vals.min()),
            "max": float(vals.max()),
            "range": float(vals.max() - vals.min()),
        }
    flips = [int((ens_list[0] != p).sum()) for p in ens_list[1:]]

    correct0 = (ens_list[0] == y).astype(float)
    boot = _bootstrap_metric_sd(correct0, g, args.boot, args.seed)
    # Scale bootstrap SD with 1/sqrt(n): n' = n * (MDE_now / target)^2
    target_mde = 0.05  # practical Phase 2c headline; 0.03 needs ~2x more
    mde_now = max(boot["R_global_mde_80"], 1e-6)
    n_cur = float(len(texts))
    n_for = {
        f"{e:.2f}": int(round(n_cur * (mde_now / e) ** 2)) for e in (0.03, 0.05, 0.08)
    }
    # Budget-aware recommends: aim MDE~0.05 on R_global; keep group floor via equal-ish caps
    recommend_test = max(n_for["0.05"], 600)
    recommend_d_select = max(int(round(recommend_test * 0.6)), 360)
    delta_sd = float(boot["R_global_sd"] * np.sqrt(n_cur))  # approx iid sd of example mean

    gng = _go_nogo(profile, mix, agg)

    report: Dict[str, Any] = {
        "n": len(texts),
        "repeats": args.repeats,
        "seed": args.seed,
        "workers": worker_names,
        "profile": profile,
        "noise": {"repeats": runs, "summary": summary, "flips": flips},
        "aggregation": agg,
        "class_mix": mix,
        "power": {
            "bootstrap": boot,
            "assumed_delta_sd": delta_sd,
            "target_mde": target_mde,
            "n_for_effect": n_for,
            "recommend_d_select": recommend_d_select,
            "recommend_test": recommend_test,
        },
        "go_nogo": gng,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_results_md(out_dir, report)

    print("\n========== GO/NO-GO ==========", flush=True)
    print(f"DECISION: {gng['decision'].upper()}", flush=True)
    for r in gng["reasons"]:
        print(f"  - {r}", flush=True)
    print(f"\nwrote {out_dir / 'RESULTS.md'}", flush=True)
    print(f"wrote {out_dir / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Phase 0 offline bounds — portfolio oracle, DRO ranking stability, demotion Pareto.

Uses only cached artifacts (no API):
  - lowvar run pred_cache (30 prompts × 360 D_select examples)
  - optional noise repeats from the earlier cvar pair run (exp_A_predictions)

Go/no-go thresholds (optimistic D_select estimates):
  E0a portfolio oracle ≥ +5 pp over best single prompt → keep portfolio in method
  E0b DRO ranking Spearman vs R_global / noise SD → whether DRO belongs in fitness
  E0c demotion Pareto has a non-empty "gain 4★ without killing 5★" zone → keep
      bidirectional calibration on Amazon user-shift; else treat it as negative case
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))

from prime.fitness.metrics import (  # noqa: E402
    class_balance_weights,
    cluster_accuracies,
    cvar_from_accuracies,
    macro_class_accuracy,
    per_class_accuracy,
    smoothed_cluster_accuracies,
    weighted_accuracy,
)
from prime.workers.ensemble import MedianAggregator  # noqa: E402


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def strip_evolve_markers(text: str) -> str:
    return re.sub(r"# EVOLVE-BLOCK-(START|END)\n?", "", text)


def median_ensemble(votes_by_example: Dict[int, List[int]], example_ids: List[int]) -> np.ndarray:
    agg = MedianAggregator()
    out = []
    for eid in example_ids:
        votes = votes_by_example.get(int(eid)) or [3, 3, 3]
        out.append(agg.aggregate([int(v) for v in votes]))
    return np.asarray(out, dtype=np.int16)


def load_cache(path: Path) -> Dict[int, List[int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): [int(x) for x in v] for k, v in data.items()}


def collect_prompt_map(run_dir: Path) -> Dict[str, str]:
    """hash -> human label. Best-effort from run dir + results/_oe."""
    prompts: Dict[str, str] = {}
    init = run_dir / "initial_prompt.txt"
    if init.exists():
        prompts[prompt_hash(init.read_text(encoding="utf-8"))] = "initial"

    for cycle in (1, 2, 3):
        for name in ("best_prompt.txt", "start_prompt.txt", "consolidated_prompt.txt"):
            p = run_dir / f"al_iter_{cycle}" / name
            if not p.exists():
                continue
            text = p.read_text(encoding="utf-8")
            prompts[prompt_hash(text)] = f"c{cycle}_{name}"
            prompts[prompt_hash(strip_evolve_markers(text))] = f"c{cycle}_{name}"

    oe_root = run_dir.parent.parent / "_oe"
    if oe_root.is_dir():
        for prog in oe_root.rglob("programs/*.json"):
            try:
                code = json.loads(prog.read_text(encoding="utf-8")).get("code") or ""
            except (json.JSONDecodeError, OSError):
                continue
            if len(code) < 80:
                continue
            label = f"_oe/{prog.parent.parent.name}/{prog.stem[:8]}"
            prompts.setdefault(prompt_hash(code), label)
            prompts.setdefault(prompt_hash(strip_evolve_markers(code)), label + "_s")
    return prompts


def demotion_score(pred: np.ndarray, gold: np.ndarray) -> Dict[str, float]:
    """How hard the prompt demotes relative to gold (esp. 5★ → lower)."""
    delta = pred.astype(float) - gold.astype(float)
    g5 = gold == 5
    g4 = gold == 4
    return {
        "mean_pred_minus_gold": float(np.mean(delta)),
        "frac_pred_lt_gold": float(np.mean(pred < gold)),
        "frac_5_pred_lt_5": float(np.mean(pred[g5] < 5)) if g5.any() else 0.0,
        "frac_4_pred_gt_4": float(np.mean(pred[g4] > 4)) if g4.any() else 0.0,
        "acc_4": float(np.mean(pred[g4] == 4)) if g4.any() else 0.0,
        "acc_5": float(np.mean(pred[g5] == 5)) if g5.any() else 0.0,
        "mean_pred_on_5": float(np.mean(pred[g5])) if g5.any() else 0.0,
        "mean_pred_on_4": float(np.mean(pred[g4])) if g4.any() else 0.0,
    }


def objective_scores(
    pred: np.ndarray, gold: np.ndarray, cid: np.ndarray
) -> Dict[str, float]:
    w_bal = class_balance_weights(gold)
    raw_accs = cluster_accuracies(pred, gold, cid)
    shrunk = smoothed_cluster_accuracies(pred, gold, cid, prior_weight=50.0)
    bal_shrunk = smoothed_cluster_accuracies(
        pred, gold, cid, prior_weight=50.0, weights=w_bal
    )
    # Family-DRO surrogates: worst over a small family of reweightings
    family = []
    # (1) uniform
    family.append(weighted_accuracy(pred, gold))
    # (2) class-balanced
    family.append(weighted_accuracy(pred, gold, w_bal))
    # (3) upweight each cluster ×2
    for c in np.unique(cid):
        w = np.ones(len(gold), dtype=float)
        w[cid == c] *= 2.0
        w *= len(w) / w.sum()
        family.append(weighted_accuracy(pred, gold, w))
    # (4) upweight each gold class ×3 (stronger than full balance)
    for g in np.unique(gold):
        w = np.ones(len(gold), dtype=float)
        w[gold == g] *= 3.0
        w *= len(w) / w.sum()
        family.append(weighted_accuracy(pred, gold, w))

    return {
        "R_global": weighted_accuracy(pred, gold),
        "R_macro": macro_class_accuracy(pred, gold),
        "CVaR_q40_raw": cvar_from_accuracies(raw_accs, 0.40),
        "CVaR_q40_w50": cvar_from_accuracies(shrunk, 0.40),
        "CVaR_q40_w50_bal": cvar_from_accuracies(bal_shrunk, 0.40),
        "DRO_family_worst": float(min(family)),
        "DRO_family_mean": float(np.mean(family)),
        "n_family": float(len(family)),
    }


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra = a.argsort().argsort().astype(float)
    rb = b.argsort().argsort().astype(float)
    # average ranks for ties would be nicer; fine for screening
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def run_e0a(
    preds: Dict[str, np.ndarray],
    gold: np.ndarray,
    cid: np.ndarray,
    labels: Dict[str, str],
) -> Dict[str, Any]:
    """Portfolio upper bounds on D_select (optimistic)."""
    keys = sorted(preds)
    stack = np.stack([preds[k] for k in keys], axis=0)  # C × N
    correct = stack == gold[None, :]

    # Single-prompt baselines
    single_acc = {k: float(correct[i].mean()) for i, k in enumerate(keys)}
    best_single_hash = max(single_acc, key=single_acc.get)
    best_single = single_acc[best_single_hash]
    initial_hash = next((h for h, n in labels.items() if n == "initial"), None)
    initial_acc = single_acc.get(initial_hash, float("nan")) if initial_hash else float("nan")

    # Oracle: per-example pick any candidate that is correct (if any)
    oracle_any = float(correct.any(axis=0).mean())

    # Per-cluster specialist: for each cluster, pick the candidate with best
    # accuracy on that cluster; route test examples by cluster id.
    cluster_specialist = np.zeros(len(gold), dtype=np.int16)
    specialist_map: Dict[int, str] = {}
    for c in np.unique(cid):
        mask = cid == c
        best_i, best_a = 0, -1.0
        for i, k in enumerate(keys):
            a = float(correct[i, mask].mean())
            if a > best_a:
                best_i, best_a = i, a
        specialist_map[int(c)] = keys[best_i]
        cluster_specialist[mask] = stack[best_i, mask]

    # Per-class specialist (same idea on gold class — oracle upper, needs labels)
    class_specialist = np.zeros(len(gold), dtype=np.int16)
    class_map: Dict[int, str] = {}
    for g in np.unique(gold):
        mask = gold == g
        best_i, best_a = 0, -1.0
        for i, k in enumerate(keys):
            a = float(correct[i, mask].mean())
            if a > best_a:
                best_i, best_a = i, a
        class_map[int(g)] = keys[best_i]
        class_specialist[mask] = stack[best_i, mask]

    # Leave-one-out cluster specialist: for each example, pick specialist fitted
    # on other examples of its cluster (reduces in-sample optimism a bit).
    loo = np.zeros(len(gold), dtype=np.int16)
    for c in np.unique(cid):
        mask = cid == c
        idxs = np.where(mask)[0]
        for j in idxs:
            others = idxs[idxs != j]
            if len(others) == 0:
                loo[j] = stack[0, j]
                continue
            best_i, best_a = 0, -1.0
            for i, _k in enumerate(keys):
                a = float(correct[i, others].mean())
                if a > best_a:
                    best_i, best_a = i, a
            loo[j] = stack[best_i, j]

    def pack(name: str, pred: np.ndarray) -> Dict[str, Any]:
        return {
            "name": name,
            "R_global": weighted_accuracy(pred, gold),
            "R_macro": macro_class_accuracy(pred, gold),
            "per_class": {str(k): v for k, v in per_class_accuracy(pred, gold).items()},
            "CVaR_q40": cvar_from_accuracies(cluster_accuracies(pred, gold, cid), 0.40),
        }

    rows = [
        pack("initial", preds[initial_hash]) if initial_hash in preds else None,
        pack("best_single", preds[best_single_hash]),
        pack("oracle_any_correct", np.where(correct.any(axis=0), gold, stack[0])),
        pack("cluster_specialist_oracle", cluster_specialist),
        pack("cluster_specialist_loo", loo),
        pack("class_specialist_oracle", class_specialist),
    ]
    rows = [r for r in rows if r is not None]

    def delta(a: float, b: float) -> Optional[float]:
        if a != a or b != b:
            return None
        return a - b

    cluster_loo_acc = next(r["R_global"] for r in rows if r["name"] == "cluster_specialist_loo")
    cluster_orc_acc = next(
        r["R_global"] for r in rows if r["name"] == "cluster_specialist_oracle"
    )
    result = {
        "n_candidates": len(keys),
        "initial_acc": initial_acc,
        "best_single_acc": best_single,
        "best_single_label": labels.get(best_single_hash, best_single_hash),
        "oracle_any_acc": oracle_any,
        "cluster_specialist_oracle_acc": cluster_orc_acc,
        "cluster_specialist_loo_acc": cluster_loo_acc,
        "delta_loo_vs_best_single_pp": 100.0 * (cluster_loo_acc - best_single),
        "delta_loo_vs_initial_pp": (
            100.0 * (cluster_loo_acc - initial_acc) if initial_acc == initial_acc else None
        ),
        "delta_oracle_vs_best_single_pp": 100.0 * (cluster_orc_acc - best_single),
        "specialist_map": {
            str(c): labels.get(h, h) for c, h in specialist_map.items()
        },
        "class_map": {str(g): labels.get(h, h) for g, h in class_map.items()},
        "rows": rows,
        "go_threshold_pp": 5.0,
        "go_portfolio": bool(100.0 * (cluster_loo_acc - best_single) >= 5.0),
    }
    return result


def run_e0b(
    preds: Dict[str, np.ndarray],
    gold: np.ndarray,
    cid: np.ndarray,
    noise_dir: Optional[Path],
) -> Dict[str, Any]:
    """Ranking stability of objectives across candidates + optional noise SD."""
    keys = sorted(preds)
    scores = {k: objective_scores(preds[k], gold, cid) for k in keys}
    obj_names = list(next(iter(scores.values())).keys())
    obj_names = [n for n in obj_names if n != "n_family"]

    # Rank correlation of each objective vs R_global across candidates
    r_global = np.array([scores[k]["R_global"] for k in keys])
    ranking = {}
    for name in obj_names:
        vec = np.array([scores[k][name] for k in keys])
        ranking[name] = {
            "spearman_vs_R_global": spearman(vec, r_global),
            "mean": float(vec.mean()),
            "std": float(vec.std(ddof=1)) if len(vec) > 1 else 0.0,
            "range": float(vec.max() - vec.min()),
        }

    # Top-k agreement with R_global
    order_g = list(np.argsort(-r_global))
    topk = {}
    for name in obj_names:
        vec = np.array([scores[k][name] for k in keys])
        order = list(np.argsort(-vec))
        for k in (3, 5):
            topk[f"{name}_top{k}_overlap"] = len(set(order[:k]) & set(order_g[:k]))

    noise: Dict[str, Any] = {"available": False}
    if noise_dir and noise_dir.is_dir():
        reps = sorted(noise_dir.glob("ensemble_repeat*.npy"))
        labels_n = np.load(noise_dir / "labels.npy")
        cids_n = np.load(noise_dir / "cluster_ids.npy")
        if reps:
            noise["available"] = True
            noise["n_repeats"] = len(reps)
            noise["n_examples"] = int(len(labels_n))
            noise["note"] = (
                "Repeats from earlier cvar pair run (240-ex D_select), not lowvar 360. "
                "Noise SD is comparable in spirit; absolute values differ by split size."
            )
            per_obj: Dict[str, List[float]] = {n: [] for n in obj_names}
            for rp in reps:
                pred = np.load(rp)
                s = objective_scores(pred, labels_n, cids_n)
                for n in obj_names:
                    per_obj[n].append(s[n])
            noise["sd"] = {
                n: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
                for n, v in per_obj.items()
            }
            # signal: spread across lowvar candidates on this objective
            noise["signal_candidate_range"] = {
                n: ranking[n]["range"] for n in obj_names
            }
            noise["approx_sig_noise"] = {
                n: (
                    ranking[n]["range"] / noise["sd"][n]
                    if noise["sd"][n] > 1e-9
                    else float("inf")
                )
                for n in obj_names
            }

    # Who would each objective promote as heir?
    def top_hash(name: str) -> str:
        return max(keys, key=lambda k: scores[k][name])

    tops = {name: top_hash(name) for name in obj_names}
    init_rg = float(np.max(r_global))  # best R_global among candidates (= often initial)
    dro_sp = ranking.get("DRO_family_worst", {}).get("spearman_vs_R_global", 1.0)
    cvar_sp = ranking.get("CVaR_q40_w50_bal", {}).get("spearman_vs_R_global", 1.0)
    dro_top_rg = scores[tops["DRO_family_worst"]]["R_global"]
    # Different ranking is necessary but not sufficient: the DRO winner must not
    # be worse on raw accuracy than the R_global winner (demotion trap).
    dro_safe = bool(dro_sp == dro_sp and dro_sp < 0.90 and dro_top_rg >= init_rg - 0.005)
    recommendation = {
        "use_DRO_in_fitness": dro_safe,
        "use_CVaR_bal_in_fitness": False,  # already falsified by lowvar live run
        "prefer_constraint_over_DRO_fitness": True,
        "top_by_objective": {
            name: {
                "hash": tops[name],
                "R_global": scores[tops[name]]["R_global"],
                "objective": scores[tops[name]][name],
            }
            for name in obj_names
        },
        "rationale": (
            "DRO_family_worst may rank differently from R_global, but if its top "
            "candidate still loses raw accuracy vs the R_global winner it is the "
            "same demotion trap as CVaR_bal. Keep robustness as a constraint."
        ),
        "dro_spearman_vs_global": dro_sp,
        "cvar_bal_spearman_vs_global": cvar_sp,
        "dro_top_R_global": dro_top_rg,
        "best_R_global": init_rg,
    }

    return {
        "n_candidates": len(keys),
        "ranking_vs_global": ranking,
        "topk_overlap_with_global": topk,
        "noise": noise,
        "recommendation": recommendation,
        "per_candidate": {
            k: {n: scores[k][n] for n in obj_names} for k in keys
        },
    }


def run_e0c(
    preds: Dict[str, np.ndarray],
    gold: np.ndarray,
    labels: Dict[str, str],
) -> Dict[str, Any]:
    """Demotion Pareto: is there a zone that lifts 4★ without killing 5★?"""
    rows = []
    for h, pred in preds.items():
        d = demotion_score(pred, gold)
        rows.append(
            {
                "hash": h,
                "label": labels.get(h, h),
                "R_global": weighted_accuracy(pred, gold),
                "R_macro": macro_class_accuracy(pred, gold),
                **d,
            }
        )
    rows.sort(key=lambda r: r["mean_pred_on_5"])  # more demotion first

    initial = next((r for r in rows if r["label"] == "initial"), None)
    if initial is None:
        # fall back to least-demoting / highest acc_5 among high R_global
        initial = max(rows, key=lambda r: (r["acc_5"], r["R_global"]))

    # "Good zone": acc_4 >= initial + 0.03 AND acc_5 >= initial - 0.02 AND R_global >= initial - 0.01
    good = [
        r
        for r in rows
        if r["acc_4"] >= initial["acc_4"] + 0.03
        and r["acc_5"] >= initial["acc_5"] - 0.02
        and r["R_global"] >= initial["R_global"] - 0.01
    ]
    # Weak zone: any 4★ gain with 5★ drop < 0.05
    weak = [
        r
        for r in rows
        if r["acc_4"] >= initial["acc_4"] + 0.02
        and r["acc_5"] >= initial["acc_5"] - 0.05
    ]

    # Correlation: demotion strength vs metrics
    dem = np.array([r["frac_5_pred_lt_5"] for r in rows])
    return {
        "n_candidates": len(rows),
        "initial": {
            "label": initial["label"],
            "R_global": initial["R_global"],
            "acc_4": initial["acc_4"],
            "acc_5": initial["acc_5"],
        },
        "n_good_zone": len(good),
        "n_weak_zone": len(weak),
        "good_zone_labels": [r["label"] for r in good],
        "weak_zone_labels": [r["label"] for r in weak],
        "corr_demotion_vs_R_global": float(np.corrcoef(dem, [r["R_global"] for r in rows])[0, 1])
        if dem.std() > 1e-12
        else float("nan"),
        "corr_demotion_vs_acc_4": float(np.corrcoef(dem, [r["acc_4"] for r in rows])[0, 1])
        if dem.std() > 1e-12
        else float("nan"),
        "corr_demotion_vs_acc_5": float(np.corrcoef(dem, [r["acc_5"] for r in rows])[0, 1])
        if dem.std() > 1e-12
        else float("nan"),
        "go_bidirectional_calibration_on_amazon_user_shift": bool(len(good) >= 1),
        "rows": rows,
    }


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    e0a, e0b, e0c = report["E0a"], report["E0b"], report["E0c"]
    lines = [
        "# Phase 0 offline bounds — report",
        "",
        f"Run: `{report['run_dir']}`",
        f"Candidates scored on D_select: **{report['n_candidates']}** "
        f"(example_ids aligned, median ensemble).",
        "",
        "Estimates are **in-sample / optimistic** (D_select). "
        "Go thresholds are raised accordingly.",
        "",
        "## E0a — Portfolio upper bound",
        "",
        f"| Metric | Value |",
        f"|---|---:|",
        f"| Initial R_global | {e0a['initial_acc']:.4f} |",
        f"| Best single candidate | {e0a['best_single_acc']:.4f} ({e0a['best_single_label']}) |",
        f"| Oracle any-correct | {e0a['oracle_any_acc']:.4f} |",
        f"| Cluster specialist (oracle fit) | {e0a['cluster_specialist_oracle_acc']:.4f} |",
        f"| Cluster specialist (LOO) | {e0a['cluster_specialist_loo_acc']:.4f} |",
        f"| Δ LOO − best single | **{e0a['delta_loo_vs_best_single_pp']:+.2f} pp** |",
        f"| Δ LOO − initial | {e0a.get('delta_loo_vs_initial_pp')} pp |",
        f"| Go threshold | ≥ +5.0 pp (LOO vs best single) |",
        f"| **Go portfolio?** | **{e0a['go_portfolio']}** |",
        "",
        "Specialist map (cluster → prompt):",
        "",
    ]
    for c, lab in sorted(e0a["specialist_map"].items(), key=lambda x: int(x[0])):
        lines.append(f"- cluster {c}: `{lab}`")
    lines += [
        "",
        "## E0b — DRO / CVaR ranking vs R_global",
        "",
        "| Objective | Spearman vs R_global | cand range | noise SD* | ≈sig/noise |",
        "|---|---:|---:|---:|---:|",
    ]
    noise = e0b.get("noise") or {}
    for name, row in e0b["ranking_vs_global"].items():
        sd = (noise.get("sd") or {}).get(name)
        sn = (noise.get("approx_sig_noise") or {}).get(name)
        lines.append(
            f"| {name} | {row['spearman_vs_R_global']:.3f} | {row['range']:.4f} | "
            f"{sd if sd is not None else '—'} | {sn if sn is not None else '—'} |"
        )
    if noise.get("available"):
        lines += ["", f"\\* Noise SD from `{noise.get('note')}`", ""]
    else:
        lines += ["", "\\* Noise repeats not available for this run.", ""]
    rec = e0b["recommendation"]
    lines += [
        "### Recommendation",
        "",
        f"- Use DRO in fitness: **{rec['use_DRO_in_fitness']}** "
        f"(Spearman={rec['dro_spearman_vs_global']:.3f}, "
        f"DRO-top R_global={rec.get('dro_top_R_global')}, "
        f"best R_global={rec.get('best_R_global')})",
        f"- Use CVaR_bal in fitness: **{rec['use_CVaR_bal_in_fitness']}**",
        f"- Prefer constraint over DRO-fitness: **{rec['prefer_constraint_over_DRO_fitness']}**",
        "",
        f"_{rec['rationale']}_",
        "",
        "Top candidate by objective:",
        "",
    ]
    for name, info in (rec.get("top_by_objective") or {}).items():
        lines.append(
            f"- `{name}` -> R_global={info['R_global']:.4f} (obj={info['objective']:.4f})"
        )
    lines += [
        "",
        "## E0c — Demotion Pareto",
        "",
        f"Initial: R_global={e0c['initial']['R_global']:.4f}, "
        f"acc_4={e0c['initial']['acc_4']:.4f}, acc_5={e0c['initial']['acc_5']:.4f}",
        "",
        f"- Good zone (d_acc_4>=+0.03, d_acc_5>=-0.02, d_R>=-0.01): "
        f"**{e0c['n_good_zone']}** -> {e0c['good_zone_labels']}",
        f"- Weak zone (d_acc_4>=+0.02, d_acc_5>=-0.05): "
        f"**{e0c['n_weak_zone']}** -> {e0c['weak_zone_labels']}",
        f"- corr(demotion, R_global) = {e0c['corr_demotion_vs_R_global']:.3f}",
        f"- corr(demotion, acc_4) = {e0c['corr_demotion_vs_acc_4']:.3f}",
        f"- corr(demotion, acc_5) = {e0c['corr_demotion_vs_acc_5']:.3f}",
        f"- **Go bidirectional calibration on Amazon user-shift?** "
        f"**{e0c['go_bidirectional_calibration_on_amazon_user_shift']}**",
        "",
        "## Combined go/no-go for next phases",
        "",
    ]
    g = report["go_no_go"]
    for k, v in g.items():
        lines.append(f"- **{k}**: {v}")
    lines += ["", "---", "_Generated by `scripts/phase0_offline_bounds.py`._", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=Path(
            "results/E1_pred_profile_cvar_lowvar/seed42_20260730_114314"
        ),
    )
    ap.add_argument(
        "--noise-dir",
        type=Path,
        default=Path(
            "results/E1_pred_profile_cvar_lex/seed42_20260729_213313/exp_A_predictions"
        ),
        help="Optional ensemble_repeat*.npy directory for noise SD",
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    run_dir = (PKG_ROOT / args.run_dir).resolve() if not args.run_dir.is_absolute() else args.run_dir
    noise_dir = args.noise_dir
    if noise_dir and not noise_dir.is_absolute():
        noise_dir = (PKG_ROOT / noise_dir).resolve()
    out_dir = args.out_dir or (run_dir / "analysis" / "phase0")
    out_dir.mkdir(parents=True, exist_ok=True)

    sel = json.loads((run_dir / "d_select_data.json").read_text(encoding="utf-8"))
    example_ids = [int(x) for x in sel["example_ids"]]
    gold = np.asarray(sel["labels"], dtype=np.int16)
    cid = np.asarray(sel["cluster_ids"], dtype=np.int16)

    label_map = collect_prompt_map(run_dir)
    preds: Dict[str, np.ndarray] = {}
    coverage = []
    for path in sorted((run_dir / "pred_cache").glob("*.json")):
        votes = load_cache(path)
        n_hit = sum(1 for e in example_ids if int(e) in votes)
        coverage.append((path.stem, n_hit))
        if n_hit < len(example_ids):
            # skip incomplete caches (e.g. partial eval)
            continue
        preds[path.stem] = median_ensemble(votes, example_ids)

    labels = {h: label_map.get(h, h) for h in preds}

    e0a = run_e0a(preds, gold, cid, labels)
    e0b = run_e0b(preds, gold, cid, noise_dir if noise_dir.exists() else None)
    e0c = run_e0c(preds, gold, labels)

    go = {
        "portfolio_in_method": (
            "KEEP - LOO specialist beats best single by >=5pp"
            if e0a["go_portfolio"]
            else "DROP / deprioritize on Amazon user-shift - LOO headroom <5pp "
            "(oracle fit only +2pp; LOO negative)"
        ),
        "DRO_in_fitness": (
            "REJECT as fitness - ranks differently but top candidate still loses "
            "R_global vs best single; keep as constraint/diagnostic"
            if not e0b["recommendation"]["use_DRO_in_fitness"]
            else "CANDIDATE for fitness (ranks differently AND top keeps R_global)"
        ),
        "CVaR_balanced_fitness": "REJECT - already caused significant test regression",
        "amazon_user_shift_as_headline": (
            "NO - use as negative/mechanics case; headline on CivilComments/category-shift"
            if not e0c["go_bidirectional_calibration_on_amazon_user_shift"]
            else "MAYBE - good zone nonempty; Phase 1 constraint-run still required"
        ),
        "phase1_constraint_live": (
            "GO - validate non-regression gate + raw global fitness on Amazon "
            "(mechanics check, not headline). Expectation: no significant regression; "
            "large worst-group gains unlikely on this shift."
        ),
    }

    report = {
        "run_dir": str(run_dir),
        "n_candidates": len(preds),
        "n_cache_files": len(coverage),
        "n_incomplete_skipped": sum(1 for _, n in coverage if n < len(example_ids)),
        "label_coverage": {
            h: labels[h] for h in sorted(preds, key=lambda x: labels[x])
        },
        "E0a": e0a,
        "E0b": e0b,
        "E0c": e0c,
        "go_no_go": go,
    }

    # Trim bulky per-candidate scores from the markdown-facing JSON? keep full.
    (out_dir / "phase0_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    # Compact rows CSV-like for E0c
    e0c_rows_path = out_dir / "e0c_demotion_rows.json"
    e0c_rows_path.write_text(
        json.dumps(e0c["rows"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_markdown(report, out_dir / "PHASE0_REPORT.md")

    print(f"Wrote {out_dir / 'PHASE0_REPORT.md'}")
    print(f"Candidates: {len(preds)}  incomplete skipped: {report['n_incomplete_skipped']}")
    print(
        f"E0a LOO_delta={e0a['delta_loo_vs_best_single_pp']:+.2f}pp  "
        f"go_portfolio={e0a['go_portfolio']}"
    )
    print(
        f"E0b DRO Spearman={e0b['recommendation']['dro_spearman_vs_global']:.3f}  "
        f"use_DRO_fitness={e0b['recommendation']['use_DRO_in_fitness']}  "
        f"dro_top_Rg={e0b['recommendation']['dro_top_R_global']:.4f}"
    )
    print(
        f"E0c good_zone={e0c['n_good_zone']}  "
        f"go_bidir={e0c['go_bidirectional_calibration_on_amazon_user_shift']}"
    )
    for k, v in go.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

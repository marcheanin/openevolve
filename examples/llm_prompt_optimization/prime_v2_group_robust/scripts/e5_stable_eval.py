#!/usr/bin/env python
"""Stable single-scorer evaluation on E5 test_fixed (repeat mean ± SD).

E5 headlines were single-shot; this script re-scores seed + final (or any
candidate) ``--repeats`` times on the fixed test set with gemma-3 only, then
reports mean/SD/range for raw worst-GBA, softmin, recall/spec, plus McNemar /
bootstrap on the first paired repeat (and on majority-vote aggregates).

Usage:
  python scripts/e5_stable_eval.py \\
    --run-dir results/E5_civilcomments_prime_main_v2/seed42_20260809_080918 \\
    --repeats 3

  python scripts/e5_stable_eval.py \\
    --seed-prompt prompts/initial_prompt_civilcomments.txt \\
    --final-prompt path/to/best_prompt.txt \\
    --out-dir results/.../evals/stable_test_fixed \\
    --repeats 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

METRIC_KEYS = [
    "R_worst_gba_raw",
    "R_soft_min_gba",
    "R_gba_mean",
    "toxic_recall",
    "specificity",
    "R_global",
    "R_macro",
    "pred_pos_rate",
    "invalid_rate",
]


def _load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _resolve_prompts(run_dir: Optional[Path], seed_path: Optional[Path], final_path: Optional[Path]) -> Tuple[str, str, str, str]:
    """Return (seed_text, final_text, seed_note, final_note)."""
    if run_dir is not None:
        run_dir = run_dir.resolve()
        seed_p = seed_path or (run_dir / "initial_prompt.txt")
        if final_path is None:
            # Prefer last cycle best_prompt (selection heir), else summary cycle.
            summary = {}
            sm = run_dir / "summary.json"
            if sm.is_file():
                summary = json.loads(sm.read_text(encoding="utf-8"))
            cycles = summary.get("cycles") or []
            sel_cycle = None
            best_key = summary.get("best_selection_key")
            if best_key is not None:
                for c in cycles:
                    if c.get("val_selection_key") == best_key:
                        sel_cycle = int(c["cycle"])
                        break
            if sel_cycle is None and cycles:
                sel_cycle = int(max(cycles, key=lambda c: tuple(c.get("val_selection_key") or [float("-inf")]))["cycle"])
            candidates = []
            if sel_cycle is not None:
                candidates.append(run_dir / f"al_iter_{sel_cycle}" / "best_prompt.txt")
            for c in range(8, 0, -1):
                candidates.append(run_dir / f"al_iter_{c}" / "best_prompt.txt")
            candidates.append(run_dir / "final_prompt.txt")
            final_p = next((p for p in candidates if p.is_file()), None)
            if final_p is None:
                raise FileNotFoundError(f"no final prompt under {run_dir}")
        else:
            final_p = final_path
        return (
            _load_prompt(seed_p),
            _load_prompt(final_p),
            str(seed_p),
            str(final_p),
        )
    if seed_path is None or final_path is None:
        raise ValueError("Provide --run-dir or both --seed-prompt and --final-prompt")
    return _load_prompt(seed_path), _load_prompt(final_path), str(seed_path), str(final_path)


def _materialize_test_fixed(cfg, fixed_dir: Path) -> Dict[str, Any]:
    from prime.data.civilcomments_loader import load_civilcomments_splits
    from prime.data.fixed_sets import FixedSet, build_test_fixed, materialize_split

    path = fixed_dir / "test_fixed.json"
    splits = load_civilcomments_splits(cfg.dataset, seed=cfg.active_learning.seed)
    split = splits["test"]
    fs = FixedSet.load(path) if path.is_file() else build_test_fixed(split, seed=cfg.active_learning.seed)
    mat = materialize_split(split, fs.indices)
    mat["fingerprint"] = fs.fingerprint
    return mat


def _score_once(
    texts: List[str],
    labels: List[int],
    cluster_ids: List[int],
    user_ids: Sequence[Any],
    prompt: str,
    cfg,
    *,
    use_mock: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    from prime.fitness.metrics import compute_metrics, group_balanced_accuracies
    from prime.fitness.objective import soft_min_accuracies
    from prime.workers.ensemble import build_workers, mock_predict, parallel_predict

    if use_mock:
        ens, _wp = mock_predict(
            texts, labels, 1, seed=0, aggregation="majority", label_space="binary"
        )
    else:
        workers = build_workers(cfg.ensemble)
        ens, _wp = parallel_predict(
            workers,
            texts,
            prompt,
            max_parallel=cfg.ensemble.max_parallel,
            tie_break=cfg.ensemble.tie_break,
            aggregation=cfg.ensemble.aggregation,
            label_space="binary",
            fail_closed=bool(getattr(cfg.ensemble, "fail_closed", True)),
        )
    preds = np.asarray(ens, dtype=np.int16)
    y = np.asarray(labels, dtype=np.int16)
    cids = np.asarray(cluster_ids, dtype=np.int16)
    uids = np.asarray(user_ids)
    m = compute_metrics(
        preds,
        y,
        uids,
        cluster_ids=cids,
        shrink_prior_weight=float(getattr(cfg.fitness, "shrink_prior_weight", 40.0)),
        class_balanced=True,
        gba_min_pos=int(getattr(cfg.fitness, "gba_min_pos", 10)),
        gba_min_neg=int(getattr(cfg.fitness, "gba_min_neg", 10)),
        gba_exclude_none=bool(getattr(cfg.fitness, "gba_exclude_none", True)),
    )
    raw = group_balanced_accuracies(
        preds,
        y,
        cids,
        min_pos=int(getattr(cfg.fitness, "gba_min_pos", 10)),
        min_neg=int(getattr(cfg.fitness, "gba_min_neg", 10)),
        exclude_groups={0} if getattr(cfg.fitness, "gba_exclude_none", True) else set(),
    )
    m["cluster_gba_raw"] = {int(k): float(v) for k, v in raw.items()}
    m["R_worst_gba_raw"] = float(min(raw.values())) if raw else float("nan")
    gba_fit = m.get("cluster_gba_shrunk") or m.get("cluster_gba") or raw
    tau = float(getattr(cfg.fitness, "soft_min_tau", 0.10))
    m["R_soft_min_gba"] = (
        float(soft_min_accuracies({int(k): float(v) for k, v in gba_fit.items()}, tau))
        if gba_fit
        else float("nan")
    )
    return preds, m


def _summarize(runs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for key in METRIC_KEYS:
        vals = [float(r[key]) for r in runs if r.get(key) is not None and np.isfinite(float(r[key]))]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        out[key] = {
            "mean": float(arr.mean()),
            "sd": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "range": float(arr.max() - arr.min()),
            "values": vals,
        }
    return out


def _majority_vote(pred_stack: np.ndarray) -> np.ndarray:
    # pred_stack: R x N
    return (pred_stack.mean(axis=0) >= 0.5).astype(np.int16)


def _mcnemar(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    from prime.experiment.anchor_gate import mcnemar_one_sided_p

    a_ok = a == y
    b_ok = b == y
    b_only = int(np.sum(~a_ok & b_ok))
    a_only = int(np.sum(a_ok & ~b_ok))
    return {
        "seed_only_ok": a_only,
        "final_only_ok": b_only,
        "net_final_minus_seed": b_only - a_only,
        "agreement": float(np.mean(a == b)),
        "mcnemar_p_final_worse": round(mcnemar_one_sided_p(b_only, a_only), 6),
        "mcnemar_p_seed_worse": round(mcnemar_one_sided_p(a_only, b_only), 6),
    }


def _bootstrap_delta_worst(
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    cids: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    min_pos: int,
    min_neg: int,
) -> Dict[str, Any]:
    from prime.fitness.metrics import group_balanced_accuracies

    rng = np.random.RandomState(seed)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        ga = group_balanced_accuracies(
            a[idx], y[idx], cids[idx], min_pos=min_pos, min_neg=min_neg, exclude_groups={0}
        )
        gb = group_balanced_accuracies(
            b[idx], y[idx], cids[idx], min_pos=min_pos, min_neg=min_neg, exclude_groups={0}
        )
        keys = set(ga) & set(gb)
        if not keys:
            continue
        deltas.append(min(gb[k] for k in keys) - min(ga[k] for k in keys))
    if not deltas:
        return {"n_boot": 0}
    arr = np.asarray(deltas, dtype=float)
    return {
        "n_boot": int(len(arr)),
        "delta_mean": float(arr.mean()),
        "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
        "p_positive": float(np.mean(arr > 0)),
    }


def run_stable_eval(
    *,
    cfg,
    seed_prompt: Path,
    final_prompt: Path,
    fixed_dir: Path,
    out_dir: Path,
    repeats: int = 3,
    n_boot: int = 2000,
    use_mock: bool = False,
    force: bool = False,
    seed_note: Optional[str] = None,
    final_note: Optional[str] = None,
    seed_only: bool = False,
) -> Dict[str, Any]:
    """Core stable eval used by CLI and controller post-run hook."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "stable_report.json"
    if report_path.is_file() and not force and not seed_only:
        return json.loads(report_path.read_text(encoding="utf-8"))
    if seed_only and (out / "seed_baseline.json").is_file() and not force:
        return json.loads((out / "seed_baseline.json").read_text(encoding="utf-8"))

    seed_text = _load_prompt(Path(seed_prompt))
    final_text = _load_prompt(Path(final_prompt))
    seed_note = seed_note or str(seed_prompt)
    final_note = final_note or str(final_prompt)
    (out / "seed_prompt.txt").write_text(seed_text, encoding="utf-8")
    (out / "final_prompt.txt").write_text(final_text, encoding="utf-8")
    (out / "prompt_sources.json").write_text(
        json.dumps({"seed": seed_note, "final": final_note}, indent=2), encoding="utf-8"
    )

    cfg.dataset.max_test_users = max(int(cfg.dataset.max_test_users or 0), 80_000)
    print("[stable] materialize test_fixed...", flush=True)
    mat = _materialize_test_fixed(cfg, Path(fixed_dir))
    texts, labels, cids, uids = (
        mat["texts"],
        mat["labels"],
        mat["cluster_ids"],
        mat["user_ids"],
    )
    y = np.asarray(labels, dtype=np.int16)
    cids_a = np.asarray(cids, dtype=np.int16)
    print(
        f"[stable] n={len(texts)} fp={mat['fingerprint']} repeats={repeats} "
        f"worker_mode={cfg.ensemble.mode} workers={[w.name for w in cfg.ensemble.workers]}",
        flush=True,
    )

    def _metrics_from_preds(preds: np.ndarray) -> Dict[str, Any]:
        from prime.fitness.metrics import compute_metrics, group_balanced_accuracies
        from prime.fitness.objective import soft_min_accuracies

        m = compute_metrics(
            preds,
            y,
            np.asarray(uids),
            cluster_ids=cids_a,
            shrink_prior_weight=float(cfg.fitness.shrink_prior_weight),
            class_balanced=True,
            gba_min_pos=int(cfg.fitness.gba_min_pos),
            gba_min_neg=int(cfg.fitness.gba_min_neg),
            gba_exclude_none=True,
        )
        raw = group_balanced_accuracies(
            preds,
            y,
            cids_a,
            min_pos=int(cfg.fitness.gba_min_pos),
            min_neg=int(cfg.fitness.gba_min_neg),
            exclude_groups={0},
        )
        m["cluster_gba_raw"] = {int(k): float(v) for k, v in raw.items()}
        m["R_worst_gba_raw"] = float(min(raw.values())) if raw else float("nan")
        gba = m.get("cluster_gba_shrunk") or raw
        m["R_soft_min_gba"] = (
            float(
                soft_min_accuracies(
                    {int(k): float(v) for k, v in gba.items()},
                    float(cfg.fitness.soft_min_tau),
                )
            )
            if gba
            else float("nan")
        )
        return m

    def _repeat_block(name: str, prompt: str) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        runs: List[Dict[str, Any]] = []
        preds: List[np.ndarray] = []
        for r in range(repeats):
            pred_path = out / f"{name}_repeat{r}_preds.npy"
            print(f"[stable] {name} repeat {r + 1}/{repeats}...", flush=True)
            if pred_path.is_file() and not force:
                ens = np.load(pred_path)
                if len(ens) == len(texts):
                    print(f"  resume cache hit {pred_path.name}", flush=True)
                    met = _metrics_from_preds(ens)
                else:
                    ens, met = _score_once(
                        texts, labels, cids, uids, prompt, cfg, use_mock=use_mock
                    )
                    np.save(pred_path, ens)
            else:
                ens, met = _score_once(
                    texts, labels, cids, uids, prompt, cfg, use_mock=use_mock
                )
                np.save(pred_path, ens)
            preds.append(ens)
            slim = {k: met.get(k) for k in METRIC_KEYS}
            slim["worst_gba_group"] = met.get("worst_gba_group")
            slim["cluster_gba_raw"] = met.get("cluster_gba_raw")
            runs.append(slim)
            (out / f"{name}_repeat{r}_metrics.json").write_text(
                json.dumps(slim, indent=2, default=str), encoding="utf-8"
            )
            print(
                f"  raw_worst_gba={slim['R_worst_gba_raw']:.4f} "
                f"softmin={slim['R_soft_min_gba']:.4f} "
                f"recall={slim['toxic_recall']:.4f} spec={slim['specificity']:.4f}",
                flush=True,
            )
        return runs, np.stack(preds, axis=0)

    seed_runs, seed_stack = _repeat_block("seed", seed_text)
    if seed_only:
        seed_sum = _summarize(seed_runs)
        seed_maj = _majority_vote(seed_stack)
        np.save(out / "seed_majority_preds.npy", seed_maj)
        report = {
            "fingerprint": mat["fingerprint"],
            "n": int(len(texts)),
            "repeats": repeats,
            "scorer": [w.name for w in cfg.ensemble.workers],
            "mode": cfg.ensemble.mode,
            "seed_source": seed_note,
            "seed_only": True,
            "seed_repeats": seed_runs,
            "seed_summary": seed_sum,
        }
        (out / "seed_baseline.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        print(json.dumps({"seed_only": True, "seed_summary": seed_sum}, indent=2), flush=True)
        print(f"[stable] wrote seed baseline {out}", flush=True)
        return report

    final_runs, final_stack = _repeat_block("final", final_text)

    seed_sum = _summarize(seed_runs)
    final_sum = _summarize(final_runs)
    delta_mean = {
        k: final_sum[k]["mean"] - seed_sum[k]["mean"]
        for k in seed_sum
        if k in final_sum
    }

    seed_maj = _majority_vote(seed_stack)
    final_maj = _majority_vote(final_stack)
    np.save(out / "seed_majority_preds.npy", seed_maj)
    np.save(out / "final_majority_preds.npy", final_maj)
    from prime.fitness.metrics import compute_metrics, group_balanced_accuracies
    from prime.fitness.objective import soft_min_accuracies

    def _from_preds(preds: np.ndarray) -> Dict[str, Any]:
        m = compute_metrics(
            preds,
            y,
            np.asarray(uids),
            cluster_ids=cids_a,
            shrink_prior_weight=float(cfg.fitness.shrink_prior_weight),
            class_balanced=True,
            gba_min_pos=int(cfg.fitness.gba_min_pos),
            gba_min_neg=int(cfg.fitness.gba_min_neg),
            gba_exclude_none=True,
        )
        raw = group_balanced_accuracies(
            preds,
            y,
            cids_a,
            min_pos=int(cfg.fitness.gba_min_pos),
            min_neg=int(cfg.fitness.gba_min_neg),
            exclude_groups={0},
        )
        m["R_worst_gba_raw"] = float(min(raw.values())) if raw else float("nan")
        gba = m.get("cluster_gba_shrunk") or raw
        m["R_soft_min_gba"] = (
            float(
                soft_min_accuracies(
                    {int(k): float(v) for k, v in gba.items()},
                    float(cfg.fitness.soft_min_tau),
                )
            )
            if gba
            else float("nan")
        )
        return {k: m.get(k) for k in METRIC_KEYS}

    seed_maj_met = _from_preds(seed_maj)
    final_maj_met = _from_preds(final_maj)
    paired_r0 = _mcnemar(y, seed_stack[0], final_stack[0])
    paired_maj = _mcnemar(y, seed_maj, final_maj)
    boot_r0 = _bootstrap_delta_worst(
        y,
        seed_stack[0],
        final_stack[0],
        cids_a,
        n_boot=n_boot,
        seed=0,
        min_pos=max(5, int(cfg.fitness.gba_min_pos) // 2),
        min_neg=max(5, int(cfg.fitness.gba_min_neg) // 2),
    )
    boot_maj = _bootstrap_delta_worst(
        y,
        seed_maj,
        final_maj,
        cids_a,
        n_boot=n_boot,
        seed=1,
        min_pos=max(5, int(cfg.fitness.gba_min_pos) // 2),
        min_neg=max(5, int(cfg.fitness.gba_min_neg) // 2),
    )

    seed_flips = [int((seed_stack[r] != seed_stack[0]).sum()) for r in range(1, len(seed_stack))]
    final_flips = [int((final_stack[r] != final_stack[0]).sum()) for r in range(1, len(final_stack))]

    report: Dict[str, Any] = {
        "fingerprint": mat["fingerprint"],
        "n": int(len(texts)),
        "repeats": repeats,
        "scorer": [w.name for w in cfg.ensemble.workers],
        "mode": cfg.ensemble.mode,
        "seed_source": seed_note,
        "final_source": final_note,
        "seed_repeats": seed_runs,
        "final_repeats": final_runs,
        "seed_summary": seed_sum,
        "final_summary": final_sum,
        "delta_mean_final_minus_seed": delta_mean,
        "seed_majority": seed_maj_met,
        "final_majority": final_maj_met,
        "delta_majority": {
            k: float(final_maj_met.get(k) or 0) - float(seed_maj_met.get(k) or 0)
            for k in METRIC_KEYS
            if seed_maj_met.get(k) is not None
        },
        "paired_repeat0": paired_r0,
        "paired_majority": paired_maj,
        "bootstrap_delta_worst_gba_repeat0": boot_r0,
        "bootstrap_delta_worst_gba_majority": boot_maj,
        "seed_flips_vs_r0": seed_flips,
        "final_flips_vs_r0": final_flips,
        "stable": bool(
            seed_sum.get("R_worst_gba_raw", {}).get("sd", 1) < 0.02
            and final_sum.get("R_worst_gba_raw", {}).get("sd", 1) < 0.02
        ),
    }
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    lines = [
        "# E5 stable eval — test_fixed (single scorer)",
        "",
        f"- n={len(texts)}, fingerprint=`{mat['fingerprint']}`, repeats={repeats}",
        f"- scorer: `{report['scorer']}` mode=`{report['mode']}`",
        f"- seed: `{seed_note}`",
        f"- final: `{final_note}`",
        "",
        "## Repeat summary (mean ± SD)",
        "",
        "| metric | seed mean | seed SD | final mean | final SD | Δ mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for k in METRIC_KEYS:
        if k not in seed_sum or k not in final_sum:
            continue
        lines.append(
            f"| {k} | {seed_sum[k]['mean']:.4f} | {seed_sum[k]['sd']:.4f} | "
            f"{final_sum[k]['mean']:.4f} | {final_sum[k]['sd']:.4f} | "
            f"{delta_mean[k]:+.4f} |"
        )
    lines += [
        "",
        "## Majority-vote aggregate (across repeats)",
        "",
        f"| R_worst_gba_raw | {seed_maj_met['R_worst_gba_raw']:.4f} → "
        f"{final_maj_met['R_worst_gba_raw']:.4f} "
        f"(Δ {report['delta_majority'].get('R_worst_gba_raw', float('nan')):+.4f}) |",
        "",
        "## Paired tests",
        "",
        f"- repeat0 McNemar net_final_minus_seed={paired_r0['net_final_minus_seed']} "
        f"(final_only={paired_r0['final_only_ok']}, seed_only={paired_r0['seed_only_ok']})",
        f"- majority McNemar net={paired_maj['net_final_minus_seed']}",
        f"- bootstrap Δworst-GBA (r0): mean={boot_r0.get('delta_mean')} "
        f"CI95={boot_r0.get('ci95')} P(Δ>0)={boot_r0.get('p_positive')}",
        f"- bootstrap Δworst-GBA (maj): mean={boot_maj.get('delta_mean')} "
        f"CI95={boot_maj.get('ci95')} P(Δ>0)={boot_maj.get('p_positive')}",
        "",
        f"Prediction flips vs r0: seed={seed_flips}, final={final_flips}",
        "",
        f"Noise floor OK (SD worst-GBA < 0.02 both): **{report['stable']}**",
        "",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {"delta_mean": delta_mean, "stable": report["stable"], "boot_maj": boot_maj},
            indent=2,
        ),
        flush=True,
    )
    print(f"[stable] wrote {out}", flush=True)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--seed-prompt", type=Path, default=None)
    ap.add_argument("--final-prompt", type=Path, default=None)
    ap.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/E5_civilcomments/config_prime_main.yaml",
    )
    ap.add_argument(
        "--fixed-dir",
        type=Path,
        default=ROOT / "experiments/E5_civilcomments/fixed_sets",
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--seed-only",
        action="store_true",
        help="Score seed prompt only (shared baseline for batch).",
    )
    args = ap.parse_args()

    from prime.config import load_config
    from prime.workers.ensemble import load_dotenv_if_present

    load_dotenv_if_present()
    cfg_path = args.config
    if args.run_dir is not None:
        for cand in (
            args.run_dir / "config_resolved.yaml",
            args.run_dir / "config_used.yaml",
        ):
            if cand.is_file():
                cfg_path = cand
                break
    cfg = load_config(cfg_path)
    if getattr(cfg.ensemble, "mode", "single") != "single":
        print(
            f"[stable] WARNING: ensemble.mode={cfg.ensemble.mode}; "
            "E5 stable eval expects single scorer",
            flush=True,
        )

    if args.seed_only:
        if args.seed_prompt is not None:
            seed_p = Path(args.seed_prompt)
            seed_text = _load_prompt(seed_p)
            seed_note = str(seed_p)
        elif args.run_dir is not None:
            seed_p = args.run_dir / "initial_prompt.txt"
            if not seed_p.is_file():
                raise FileNotFoundError(f"missing {seed_p}")
            seed_text = _load_prompt(seed_p)
            seed_note = str(seed_p)
        else:
            raise ValueError("--seed-only needs --seed-prompt or --run-dir")
        final_text, final_note = seed_text, seed_note
        final_p = seed_p
    else:
        seed_text, final_text, seed_note, final_note = _resolve_prompts(
            args.run_dir, args.seed_prompt, args.final_prompt
        )
        seed_p = Path(seed_note) if Path(seed_note).is_file() else None
        final_p = Path(final_note) if Path(final_note).is_file() else None

    out = args.out_dir
    if out is None:
        if args.run_dir is None:
            raise ValueError("--out-dir required without --run-dir")
        out = args.run_dir.resolve() / "evals" / "stable_test_fixed"
    out.mkdir(parents=True, exist_ok=True)
    if seed_p is None or not seed_p.is_file():
        seed_p = out / "_seed_prompt.txt"
        seed_p.write_text(seed_text, encoding="utf-8")
    if not args.seed_only and (final_p is None or not final_p.is_file()):
        final_p = out / "_final_prompt.txt"
        final_p.write_text(final_text, encoding="utf-8")

    run_stable_eval(
        cfg=cfg,
        seed_prompt=seed_p,
        final_prompt=final_p if not args.seed_only else seed_p,
        fixed_dir=args.fixed_dir,
        out_dir=out,
        repeats=args.repeats,
        n_boot=args.n_boot,
        use_mock=args.mock,
        force=args.force,
        seed_note=seed_note,
        final_note=final_note,
        seed_only=bool(args.seed_only),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Post-run M15-style noise floor on the official test slice for an E4 run.

Re-scores the selected final prompt --repeats times (default 3) on the same
test examples as evals/eval_test. Also scores seed (initial_prompt) once if
baseline preds are missing, for paired Δ.

Usage:
  python scripts/e4_test_noise.py --run-dir results/.../seed42_...
  python scripts/e4_test_noise.py --run-dir ... --repeats 3 --force
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))


def _selected_prompt(run_dir: Path) -> Tuple[str, str]:
    """Return (prompt_text, source_note) for the val-selected heir."""
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    best_key = summary.get("best_selection_key")
    cycles = summary.get("cycles") or []
    sel_cycle = None
    if best_key is not None:
        for c in cycles:
            if c.get("val_selection_key") == best_key:
                sel_cycle = int(c["cycle"])
                break
    if sel_cycle is None and cycles:
        # lex max on first component
        def key_t(c):
            k = c.get("val_selection_key") or [float("-inf")]
            return tuple(k)

        sel_cycle = int(max(cycles, key=key_t)["cycle"])
    for name in (
        f"al_iter_{sel_cycle}/best_prompt.txt",
        f"al_iter_{sel_cycle}/prompt_with_injected_fewshot.txt",
        "initial_prompt.txt",
    ):
        p = run_dir / name
        if p.is_file():
            return p.read_text(encoding="utf-8"), name
    raise FileNotFoundError(f"no prompt found under {run_dir} for cycle {sel_cycle}")


def _load_test(run_dir: Path, cfg):
    from scripts.eval_prompt_on_test import _load_test_slice

    return _load_test_slice(run_dir, cfg)


def _score_once(workers, texts, prompt, cfg) -> np.ndarray:
    from prime.workers.ensemble import parallel_predict

    ens, _wp = parallel_predict(
        workers,
        texts,
        prompt,
        max_parallel=cfg.ensemble.max_parallel,
        tie_break=cfg.ensemble.tie_break,
        aggregation=cfg.ensemble.aggregation,
        label_space=cfg.dataset.label_space,
    )
    return np.asarray(ens, dtype=np.int16)


def _metrics(ens, labels, user_ids, cluster_ids, cfg) -> Dict[str, Any]:
    from prime.fitness.metrics import compute_metrics

    m = compute_metrics(
        ens,
        labels,
        user_ids,
        cluster_ids=cluster_ids,
        cvar_quantile=cfg.fitness.cvar_quantile,
        beta_a=cfg.fitness.beta_a,
        beta_b=cfg.fitness.beta_b,
        tail_quantile=cfg.active_learning.tail_quantile,
        shrink_prior_weight=cfg.fitness.shrink_prior_weight,
        class_balanced=cfg.fitness.class_balanced,
    )
    keys = [
        "R_global",
        "R_macro",
        "R_worst_group",
        "CVaR_cluster",
        "mean_kappa",
        "accuracy_per_class",
        "cluster_accuracies",
    ]
    return {k: m.get(k) for k in keys}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--also-seed",
        action="store_true",
        help="Also re-score initial_prompt.txt for paired noise",
    )
    args = ap.parse_args()

    from prime.config import load_config
    from prime.workers.ensemble import build_workers, load_dotenv_if_present

    load_dotenv_if_present()
    run_dir = args.run_dir.resolve()
    out = run_dir / "evals" / "test_noise"
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "noise_report.json"
    if report_path.is_file() and not args.force:
        print(f"[noise] cache hit {report_path}", flush=True)
        print(report_path.read_text(encoding="utf-8"), flush=True)
        return 0

    cfg_path = run_dir / "config_resolved.yaml"
    if not cfg_path.is_file():
        cfg_path = run_dir / "config_used.yaml"
    cfg = load_config(cfg_path)
    prompt, src = _selected_prompt(run_dir)
    (out / "final_prompt.txt").write_text(prompt, encoding="utf-8")
    (out / "final_prompt_source.txt").write_text(src + "\n", encoding="utf-8")

    slice_ = _load_test(run_dir, cfg)
    texts, labels, user_ids, cluster_ids = (
        slice_["texts"],
        slice_["labels"],
        slice_["user_ids"],
        slice_["cluster_ids"],
    )
    workers = build_workers(cfg.ensemble)
    print(
        f"[noise] n={len(texts)} repeats={args.repeats} prompt_src={src} "
        f"workers={[w.model_name for w in workers]}",
        flush=True,
    )

    runs: List[Dict[str, Any]] = []
    preds: List[np.ndarray] = []
    for r in range(args.repeats):
        print(f"[noise] final prompt repeat {r + 1}/{args.repeats}...", flush=True)
        ens = _score_once(workers, texts, prompt, cfg)
        preds.append(ens)
        np.save(out / f"ensemble_repeat{r}.npy", ens)
        met = _metrics(ens, labels, user_ids, cluster_ids, cfg)
        runs.append(met)
        print(
            f"  R_global={met['R_global']:.4f} R_worst_group={met.get('R_worst_group')} "
            f"R_macro={met['R_macro']:.4f}",
            flush=True,
        )

    flips = [int((preds[r] != preds[0]).sum()) for r in range(1, len(preds))]

    def _summ(key: str) -> Dict[str, float]:
        vals = [float(m[key]) for m in runs if m.get(key) is not None]
        arr = np.asarray(vals, dtype=float)
        return {
            "mean": float(arr.mean()),
            "sd": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "range": float(arr.max() - arr.min()),
            "values": vals,
        }

    summary = {
        k: _summ(k)
        for k in ("R_global", "R_macro", "R_worst_group", "CVaR_cluster", "mean_kappa")
        if runs[0].get(k) is not None
    }

    seed_block: Optional[Dict[str, Any]] = None
    if args.also_seed:
        seed_p = (run_dir / "initial_prompt.txt").read_text(encoding="utf-8")
        print("[noise] scoring seed prompt once...", flush=True)
        ens_s = _score_once(workers, texts, seed_p, cfg)
        np.save(out / "ensemble_seed.npy", ens_s)
        seed_block = _metrics(ens_s, labels, user_ids, cluster_ids, cfg)
        # paired vs mean of finals is messy; vs repeat0
        c0, cs = preds[0] == labels, ens_s == labels
        b01 = int((cs & ~c0).sum())
        b10 = int((~cs & c0).sum())
        seed_block["paired_vs_final_r0"] = {
            "seed_ok_final_wrong": b01,
            "seed_wrong_final_ok": b10,
            "net_final_minus_seed": b10 - b01,
        }

    # Official single-shot final_test from the run (for reference)
    official = None
    sm = run_dir / "summary.json"
    if sm.is_file():
        official = (json.loads(sm.read_text(encoding="utf-8")).get("final_test") or {})
        official = {
            k: official.get(k)
            for k in ("R_global", "R_macro", "R_worst_group", "CVaR_cluster")
        }

    report = {
        "run_dir": str(run_dir),
        "prompt_source": src,
        "n_examples": int(len(texts)),
        "repeats": args.repeats,
        "official_final_test": official,
        "runs": runs,
        "summary": summary,
        "flips_vs_repeat0": flips,
        "seed": seed_block,
    }
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Test noise floor (M15 on official test)",
        "",
        f"Run: `{run_dir}`",
        f"Prompt: `{src}`",
        f"n={len(texts)}, repeats={args.repeats}",
        "",
        "## Summary",
        "",
        "| metric | mean | SD | range |",
        "|---|---:|---:|---:|",
    ]
    for k, s in summary.items():
        lines.append(f"| {k} | {s['mean']:.4f} | {s['sd']:.4f} | {s['range']:.4f} |")
    lines += [
        "",
        f"Prediction flips vs repeat 0: {flips}",
        "",
        f"Official final_test (single shot): {official}",
        "",
    ]
    if seed_block:
        lines += [
            "## Seed (one shot, same test)",
            "",
            f"- R_global: {seed_block.get('R_global')}",
            f"- R_worst_group: {seed_block.get('R_worst_group')}",
            f"- paired vs final r0: {seed_block.get('paired_vs_final_r0')}",
            "",
        ]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[noise] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

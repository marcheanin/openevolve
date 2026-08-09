#!/usr/bin/env python3
"""E-A: how much of the D_select fitness is measurement noise?

Workers run at temperature 0, but OpenRouter providers are not bit-deterministic
(OBSERVATIONS O14: two runs at the same seed produced different cluster fits). This
script scores **one** prompt on D_select several times with the prediction cache
bypassed and reports the SD of every fitness component.

Read it against the improvements OpenEvolve claims per cycle (~0.005-0.05). If the
SD is the same order, the search is partly ranking noise and no amount of extra
mutation budget helps.

Usage:
  python scripts/exp_fitness_noise.py --run-dir results/<run> --repeats 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))

COMPONENTS = (
    "R_global",
    "R_worst",
    "R_tail",
    "CVaR_cluster",
    "CVaR_cluster_shrunk",
    "fitness_scalar",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--prompt", type=Path, default=None, help="default: <run>/initial_prompt.txt")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from prime.config import load_config
    from prime.fitness.metrics import compute_metrics
    from prime.workers.ensemble import build_workers, load_dotenv_if_present, parallel_predict

    load_dotenv_if_present()
    run_dir = args.run_dir.resolve()
    cfg = load_config(run_dir / "config_used.yaml")
    sel = json.loads((run_dir / "d_select_data.json").read_text(encoding="utf-8"))
    texts = sel["texts"]
    labels = np.asarray(sel["labels"], dtype=np.int16)
    user_ids = np.asarray(sel["user_ids"])
    cluster_ids = np.asarray(sel["cluster_ids"], dtype=np.int16)

    prompt = (args.prompt or (run_dir / "initial_prompt.txt")).read_text(encoding="utf-8")
    workers = build_workers(cfg.ensemble)
    print(
        f"[E-A] D_select n={len(texts)} users={len(set(map(str, user_ids.tolist())))} "
        f"clusters={len(set(cluster_ids.tolist()))} workers={[w.model_name for w in workers]} "
        f"repeats={args.repeats} (cache bypassed)",
        flush=True,
    )

    # Raw votes of every repeat are persisted so that *any* candidate objective can
    # be re-scored offline for free. Re-running the ensemble to compare two metric
    # definitions would be pure waste (E-D reuses exactly these arrays).
    pred_dir = run_dir / "exp_A_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.save(pred_dir / "labels.npy", labels)
    np.save(pred_dir / "user_ids.npy", user_ids)
    np.save(pred_dir / "cluster_ids.npy", cluster_ids)

    runs: List[Dict[str, float]] = []
    preds: List[np.ndarray] = []
    for r in range(args.repeats):
        ensemble, wp = parallel_predict(
            workers,
            texts,
            prompt,
            max_parallel=cfg.ensemble.max_parallel,
            tie_break=cfg.ensemble.tie_break,
            aggregation=cfg.ensemble.aggregation,
        )
        ens = np.asarray(ensemble, dtype=np.int16)
        wp_arr = np.asarray(wp, dtype=np.int16)
        preds.append(ens)
        np.save(pred_dir / f"ensemble_repeat{r}.npy", ens)
        np.save(pred_dir / f"workers_repeat{r}.npy", wp_arr)
        m = compute_metrics(
            ens,
            labels,
            user_ids,
            worker_predictions=[wp_arr[w] for w in range(wp_arr.shape[0])],
            cluster_ids=cluster_ids,
            cvar_quantile=cfg.fitness.cvar_quantile,
            beta_a=cfg.fitness.beta_a,
            beta_b=cfg.fitness.beta_b,
            tail_quantile=cfg.active_learning.tail_quantile,
        )
        scalar = (
            float(m["CVaR_cluster_shrunk"]) + cfg.fitness.epsilon_global * float(m["R_global"])
            if cfg.fitness.mode == "cvar_lex"
            else float(m["R_global"])
        )
        row = {k: float(m.get(k, 0.0)) for k in COMPONENTS if k != "fitness_scalar"}
        row["fitness_scalar"] = scalar
        runs.append(row)
        print(
            f"  repeat {r + 1}: " + " ".join(f"{k}={row[k]:.4f}" for k in COMPONENTS),
            flush=True,
        )

    print(f"\n[E-A] variability over {args.repeats} identical evaluations")
    summary: Dict[str, Dict[str, float]] = {}
    for k in COMPONENTS:
        vals = np.array([r[k] for r in runs])
        summary[k] = {
            "mean": float(vals.mean()),
            "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(vals.min()),
            "max": float(vals.max()),
        }
        print(
            f"  {k:22s} mean={summary[k]['mean']:.4f} sd={summary[k]['sd']:.4f} "
            f"range={summary[k]['max'] - summary[k]['min']:.4f}"
        )

    flips = [int((preds[0] != p).sum()) for p in preds[1:]]
    print(
        f"\n  prediction flips vs repeat 1: {flips} of {len(labels)} examples "
        f"({[f'{f / len(labels):.1%}' for f in flips]})"
    )
    print(
        "\n  Compare fitness_scalar SD against the per-cycle OpenEvolve improvements "
        "(cvar run: +0.055 in C1, then +0.002 and +0.003)."
    )

    if args.out:
        args.out.write_text(
            json.dumps({"repeats": runs, "summary": summary, "flips": flips}, indent=2),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

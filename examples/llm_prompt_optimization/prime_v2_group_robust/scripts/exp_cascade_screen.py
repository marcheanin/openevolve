#!/usr/bin/env python3
"""E-B: can one cheap worker screen candidates as well as the full ensemble?

Every candidate evaluation today costs |workers| x |D_select| calls, which is why
D_select is only 240 examples — and 240 examples is small enough for OpenEvolve to
memorise in ~10 mutations (OBSERVATIONS O20). A cascade fixes that: screen with one
worker on a much larger D_select, then re-score the top-k with the full ensemble.

That is only safe if the single-worker ranking agrees with the ensemble ranking.
This script measures the agreement on candidates we already paid for: it reads the
OpenEvolve checkpoint programs of a finished run (each carries its ensemble
D_select metrics), re-scores them with one worker on the same D_select, and reports
Spearman plus top-k recall.

Usage:
  python scripts/exp_cascade_screen.py --run-dir results/<run> --max-programs 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))


def _rank(values: List[float]) -> np.ndarray:
    """Average (fractional) ranks — rating scores tie constantly (OBSERVATIONS M4)."""
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return ranks


def _spearman(a: List[float], b: List[float]) -> float:
    ra, rb = _rank(a), _rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt((ra**2).sum() * (rb**2).sum()))
    return float((ra * rb).sum() / denom) if denom else float("nan")


def _collect_programs(run_dir: Path, limit: int) -> List[Tuple[str, str, Dict[str, float]]]:
    """(program_id, prompt_code, ensemble_metrics) deduped by code hash."""
    seen: Dict[str, bool] = {}
    out: List[Tuple[str, str, Dict[str, float]]] = []
    for cycle in sorted(run_dir.glob("al_iter_*")):
        ptr = cycle / "openevolve_output_path.txt"
        if not ptr.exists():
            continue
        oe = Path(ptr.read_text(encoding="utf-8").strip())
        cps = sorted(
            oe.glob("checkpoints/checkpoint_*"), key=lambda p: int(p.name.split("_")[-1])
        )
        if not cps:
            continue
        for pj in sorted((cps[-1] / "programs").glob("*.json")):
            d = json.loads(pj.read_text(encoding="utf-8"))
            code = d.get("code") or ""
            if not code:
                continue
            h = hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]
            if h in seen:
                continue
            seen[h] = True
            out.append((d.get("id", h), code, d.get("metrics") or {}))
    # Spread across the fitness range rather than taking an arbitrary prefix.
    out.sort(key=lambda t: float(t[2].get("combined_score", 0.0)))
    if len(out) > limit:
        pick = np.linspace(0, len(out) - 1, limit).round().astype(int)
        out = [out[i] for i in sorted(set(pick.tolist()))]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--max-programs", type=int, default=20)
    ap.add_argument(
        "--screen-worker",
        default=None,
        help="model name for the cheap screen (default: first configured worker)",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from prime.config import WorkerSpec, load_config
    from prime.evolution.prompt_blocks import strip_evolve_markers
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

    programs = _collect_programs(run_dir, args.max_programs)
    if len(programs) < 5:
        raise SystemExit(f"only {len(programs)} usable programs found; need >=5")

    screen_name = args.screen_worker or cfg.ensemble.workers[0].name
    ens_cfg = cfg.ensemble
    original = ens_cfg.workers
    ens_cfg.workers = [WorkerSpec(screen_name)]
    screen_workers = build_workers(ens_cfg)
    ens_cfg.workers = original

    print(
        f"[E-B] programs={len(programs)} D_select n={len(texts)} "
        f"screen_worker={screen_name} full_ensemble={[w.name for w in original]}",
        flush=True,
    )
    print(f"[E-B] cost: {len(programs)} x {len(texts)} = {len(programs) * len(texts)} calls", flush=True)

    rows: List[Dict[str, object]] = []
    for i, (pid, code, metrics) in enumerate(programs, start=1):
        prompt = strip_evolve_markers(code)
        ensemble, wp = parallel_predict(
            screen_workers,
            texts,
            prompt,
            max_parallel=cfg.ensemble.max_parallel,
            tie_break=cfg.ensemble.tie_break,
            aggregation=cfg.ensemble.aggregation,
        )
        m = compute_metrics(
            np.asarray(ensemble, dtype=np.int16),
            labels,
            user_ids,
            cluster_ids=cluster_ids,
            cvar_quantile=cfg.fitness.cvar_quantile,
            beta_a=cfg.fitness.beta_a,
            beta_b=cfg.fitness.beta_b,
            tail_quantile=cfg.active_learning.tail_quantile,
        )
        screen_scalar = (
            float(m["CVaR_cluster_shrunk"]) + cfg.fitness.epsilon_global * float(m["R_global"])
            if cfg.fitness.mode == "cvar_lex"
            else float(m["R_global"])
        )
        row = {
            "program_id": pid,
            "ensemble_combined": float(metrics.get("combined_score", 0.0)),
            "ensemble_R_global": float(metrics.get("R_global", 0.0)),
            "screen_scalar": screen_scalar,
            "screen_R_global": float(m["R_global"]),
        }
        rows.append(row)
        print(
            f"  [{i}/{len(programs)}] ens={row['ensemble_combined']:.4f} "
            f"screen={screen_scalar:.4f} ens_R={row['ensemble_R_global']:.4f} "
            f"screen_R={row['screen_R_global']:.4f}",
            flush=True,
        )

    ens_scores = [float(r["ensemble_combined"]) for r in rows]
    scr_scores = [float(r["screen_scalar"]) for r in rows]
    ens_r = [float(r["ensemble_R_global"]) for r in rows]
    scr_r = [float(r["screen_R_global"]) for r in rows]

    rho_scalar = _spearman(ens_scores, scr_scores)
    rho_r = _spearman(ens_r, scr_r)
    print(f"\n[E-B] Spearman(ensemble fitness, screen fitness) = {rho_scalar:+.3f}")
    print(f"[E-B] Spearman(ensemble R_global, screen R_global) = {rho_r:+.3f}")

    n = len(rows)
    for k in (3, 5):
        if k >= n:
            continue
        top_ens = set(np.argsort(ens_scores)[-k:].tolist())
        for m_ in (k, 2 * k):
            if m_ > n:
                continue
            top_scr = set(np.argsort(scr_scores)[-m_:].tolist())
            print(
                f"[E-B] recall of ensemble top-{k} inside screen top-{m_}: "
                f"{len(top_ens & top_scr)}/{k}"
            )

    verdict = (
        "PASS - cascade is safe" if rho_scalar >= 0.8
        else "MARGINAL - cascade usable only with a wide top-k" if rho_scalar >= 0.5
        else "FAIL - single-worker screen does not preserve the ranking"
    )
    print(f"\n[E-B] verdict: {verdict}")

    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "screen_worker": screen_name,
                    "spearman_fitness": rho_scalar,
                    "spearman_R_global": rho_r,
                    "verdict": verdict,
                    "rows": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

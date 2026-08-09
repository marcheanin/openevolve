#!/usr/bin/env python3
"""Launch E1 pred_profile pair (cvar_lex then global) with full artifact packaging."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))

EXP = PKG_ROOT / "experiments" / "E1_pred_profile_cvar_vs_global"
CVAR_CFG = EXP / "config_cvar_live.yaml"
GLOBAL_CFG = EXP / "config_global_live.yaml"
# Reuse expensive seed-prompt ensemble + any D_select evals from earlier runs.
# The completed cvar_lex run holds the largest cache (seed prompt over 140 fit
# users plus every D_select candidate it evaluated).
SEED_PRED_CACHE = (
    PKG_ROOT
    / "results"
    / "E1_pred_profile_cvar_lex"
    / "seed42_20260728_125748"
    / "pred_cache"
)


def _write_findings(run_dir: Path, summary: Dict[str, Any], cfg_path: Path) -> None:
    ft = summary.get("final_test") or {}
    lines = [
        f"# FINDINGS — `{run_dir.name}`",
        "",
        f"- Finished (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"- Config: `{cfg_path}`",
        f"- Geometry: `{summary.get('cluster_geometry')}`",
        f"- Fitness: `{summary.get('fitness_mode')}`",
        f"- Actual K: `{summary.get('actual_n_clusters')}`",
        f"- Backend: `{summary.get('inference_backend')}`",
        "",
        "## Final test (official OOD)",
        "",
        f"- R_global: {ft.get('R_global')}",
        f"- R_worst: {ft.get('R_worst')}",
        f"- R_tail: {ft.get('R_tail')}",
        f"- CVaR_cluster: {ft.get('CVaR_cluster')}",
        f"- MAE: {ft.get('mae')}",
        "",
        "## Cycles",
        "",
    ]
    for c in summary.get("cycles") or []:
        par = c.get("pareto") or {}
        cons = c.get("consolidation") or {}
        lines.append(
            f"- cycle {c.get('cycle')}: fitness={c.get('fitness')} "
            f"evo_best={c.get('best_evo_score')} "
            f"front={par.get('front_size')} heir={par.get('heir_source')} "
            f"cons_delta={cons.get('delta_vs_evolved')} "
            f"val_key={c.get('val_selection_key')} "
            f"proxy_corr={c.get('proxy_cvar_tail_corr')}"
        )
    pareto = summary.get("pareto") or {}
    lines.extend(
        [
            "",
            "## Pareto front",
            "",
            f"- heir per cycle: {pareto.get('heir_sources')}",
            f"- consolidation delta on D_select: {pareto.get('consolidation_deltas')}",
            f"- OE seeded from front: {pareto.get('seed_openevolve_from_front')}",
            "",
            "```json",
            json.dumps(pareto.get("final_front") or {}, indent=2),
            "```",
            "",
            "## Budget / tokens",
            "",
            "```json",
            json.dumps(summary.get("token_usage") or {}, indent=2),
            "```",
            "",
            "## Manual notes",
            "",
            "- [ ] Compare twin arm CVaR_cluster / R_tail (cvar vs global)",
            "- [ ] Confirm both arms used the SAME clusters.json (pinned artifact)",
            "- [ ] Check whether the consolidated prompt ever won the front",
            "- [ ] Check no surviving prompt names a cluster/group",
            "",
            "See `PROTOCOL.md` and `OBSERVATIONS.md` in this directory.",
            "",
        ]
    )
    (run_dir / "FINDINGS.md").write_text("\n".join(lines), encoding="utf-8")


def _package_run(run_dir: Path, cfg_path: Path, summary: Dict[str, Any]) -> None:
    shutil.copy2(EXP / "PROTOCOL.md", run_dir / "PROTOCOL.md")
    shutil.copy2(EXP / "README.md", run_dir / "README_experiment.md")
    _write_findings(run_dir, summary, cfg_path)


def _run_one(
    cfg_path: Path,
    dry_run: bool,
    cluster_artifact: Optional[Path] = None,
    eval_sets_artifact: Optional[Path] = None,
    seed_pred_cache: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    from scripts.run_e1_wilds_live import (
        _preflight_api_key,
        _preflight_openevolve,
        _preflight_workers,
    )
    from prime.config import load_config
    from prime.controller import PrimeController

    _preflight_api_key()
    # Both arms MUST share one group definition. Refitting per arm produced
    # {0:23,1:38,2:30,3:16,4:33} for cvar and {0:24,1:30,2:22,3:44,4:20} for
    # global at the same seed — the seed-prompt ensemble is not bit-identical, and
    # a CVaR comparison across different partitions means nothing.
    overrides: Dict[str, Any] = {}
    if cluster_artifact is not None:
        overrides["dataset"] = {"cluster_artifact": str(cluster_artifact)}
        print(f"[pair] pinning cluster_artifact: {cluster_artifact}", flush=True)
    # D_anchor construction runs a live ensemble pass over heldout candidates, so
    # the arms can diverge on D_select/D_anchor even with identical clusters.
    if eval_sets_artifact is not None:
        overrides["data_roles"] = {"eval_sets_artifact": str(eval_sets_artifact)}
        print(f"[pair] pinning eval_sets_artifact: {eval_sets_artifact}", flush=True)
    cfg = load_config(cfg_path, overrides=overrides or None)
    _preflight_openevolve()
    print("=" * 60, flush=True)
    print(f"E1 pred_profile | fitness={cfg.fitness.mode} | geometry={cfg.clusters.geometry}", flush=True)
    print(f"config: {cfg_path}", flush=True)
    print("=" * 60, flush=True)
    if dry_run:
        print("Dry-run OK", flush=True)
        return None
    _preflight_workers(cfg)
    cache = seed_pred_cache or SEED_PRED_CACHE
    if cache.is_dir():
        os.environ["PRIME_SEED_PRED_CACHE"] = str(cache.resolve())
        print(f"[reuse] PRIME_SEED_PRED_CACHE={cache}", flush=True)
    ctrl = PrimeController(cfg, PKG_ROOT, cfg_path)
    # Snapshot already copies OBSERVATIONS; add protocol early
    shutil.copy2(EXP / "PROTOCOL.md", ctrl.ctx.run_dir / "PROTOCOL.md")
    shutil.copy2(EXP / "README.md", ctrl.ctx.run_dir / "README_experiment.md")
    print(f"run_dir: {ctrl.ctx.run_dir}", flush=True)
    summary = ctrl.run()
    _package_run(ctrl.ctx.run_dir, cfg_path, summary)
    print(json.dumps({"run_dir": str(ctrl.ctx.run_dir), "final_test": summary.get("final_test")}, indent=2), flush=True)
    return {"run_dir": str(ctrl.ctx.run_dir), "summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["cvar", "global", "both"], default="both")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--cluster-artifact",
        type=Path,
        default=None,
        help="Pin an existing clusters.json for every arm (required for a fair pair; "
        "when running --arm both, the first arm's fit is reused automatically).",
    )
    parser.add_argument(
        "--eval-sets-artifact",
        type=Path,
        default=None,
        help="Pin an existing eval_sets.json (D_select/D_anchor/D_audit) for every arm. "
        "With --arm both the first arm's split is reused automatically.",
    )
    parser.add_argument(
        "--seed-pred-cache",
        type=Path,
        default=None,
        help="Override the pred_cache to seed from (default: the completed cvar_lex run).",
    )
    args = parser.parse_args()

    arms: List[Path] = []
    if args.arm in ("cvar", "both"):
        arms.append(CVAR_CFG)
    if args.arm in ("global", "both"):
        arms.append(GLOBAL_CFG)

    results = []
    pinned_clusters: Optional[Path] = args.cluster_artifact
    pinned_eval_sets: Optional[Path] = args.eval_sets_artifact
    pinned_cache: Optional[Path] = args.seed_pred_cache
    for cfg in arms:
        if not cfg.is_file():
            print(f"Missing config: {cfg}", file=sys.stderr)
            return 1
        out = _run_one(
            cfg,
            dry_run=args.dry_run,
            cluster_artifact=pinned_clusters,
            eval_sets_artifact=pinned_eval_sets,
            seed_pred_cache=pinned_cache,
        )
        if not out:
            continue
        results.append(out)
        # Everything the next arm must inherit so the comparison is on identical
        # groups and identical eval splits, and so it starts with a warm cache.
        run_dir = Path(out["run_dir"])
        for label, attr, name in (
            ("clusters", "clusters", "clusters.json"),
            ("eval_sets", "eval_sets", "eval_sets.json"),
            ("pred_cache", "pred_cache", "pred_cache"),
        ):
            current = {"clusters": pinned_clusters, "eval_sets": pinned_eval_sets, "pred_cache": pinned_cache}[attr]
            if current is not None:
                continue
            candidate = run_dir / name
            if not candidate.exists():
                continue
            resolved = candidate.resolve()
            if attr == "clusters":
                pinned_clusters = resolved
            elif attr == "eval_sets":
                pinned_eval_sets = resolved
            else:
                pinned_cache = resolved
            print(f"[pair] next arm will reuse {label}: {resolved}", flush=True)

    if len(results) == 2 and not args.dry_run:
        # Pair comparison stub next to experiment folder
        pair = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "arms": [
                {
                    "fitness": r["summary"].get("fitness_mode"),
                    "run_dir": r["run_dir"],
                    "final_test": r["summary"].get("final_test"),
                }
                for r in results
            ],
        }
        pair_path = EXP / f"pair_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        pair_path.write_text(json.dumps(pair, indent=2, default=str), encoding="utf-8")
        print(f"Wrote pair compare: {pair_path}", flush=True)
        # Quick R_worst delta
        try:
            rw = [r["summary"]["final_test"]["R_worst"] for r in results]
            modes = [r["summary"].get("fitness_mode") for r in results]
            print(f"R_worst by arm: {dict(zip(modes, rw))}", flush=True)
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

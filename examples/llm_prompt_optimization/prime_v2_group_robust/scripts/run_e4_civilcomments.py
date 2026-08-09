#!/usr/bin/env python3
"""Launch E4 CivilComments Phase 2c arms (A → B → optional C) with shared eval_sets.

Arm A: global + reject (control)
Arm B: min_group_lex on oracle groups
Arm C: min_group_lex on style (inferred) groups — after A/B

Usage:
  python scripts/run_e4_civilcomments.py --arm a --dry-run
  python scripts/run_e4_civilcomments.py --arm ab
  python scripts/run_e4_civilcomments.py --arm all
"""

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

EXP = PKG_ROOT / "experiments" / "E4_civilcomments"
ARM_CFGS = {
    "a": EXP / "config_arm_a_global.yaml",
    "b": EXP / "config_arm_b_min_group.yaml",
    "c": EXP / "config_arm_c_style.yaml",
}


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
        "",
        "## Final test",
        "",
        f"- R_global: {ft.get('R_global')}",
        f"- R_worst_group: {ft.get('R_worst_group')}",
        f"- R_worst (user p10, often degenerate on CC): {ft.get('R_worst')}",
        f"- CVaR_cluster: {ft.get('CVaR_cluster')}",
        f"- CVaR_cluster_shrunk: {ft.get('CVaR_cluster_shrunk')}",
        "",
        "## Cycles",
        "",
    ]
    for c in summary.get("cycles") or []:
        lines.append(
            f"- cycle {c.get('cycle')}: fitness={c.get('fitness')} "
            f"evo_best={c.get('best_evo_score')} "
            f"val_key={c.get('val_selection_key')}"
        )
    lines.extend(
        [
            "",
            "## Budget",
            "",
            "```json",
            json.dumps(summary.get("token_usage") or {}, indent=2),
            "```",
            "",
            "Headline on CivilComments: prefer R_worst_group / per-identity table over R_worst.",
            "",
        ]
    )
    (run_dir / "FINDINGS.md").write_text("\n".join(lines), encoding="utf-8")


def _run_one(
    cfg_path: Path,
    dry_run: bool,
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
    overrides: Dict[str, Any] = {}
    if eval_sets_artifact is not None:
        overrides["data_roles"] = {"eval_sets_artifact": str(eval_sets_artifact)}
        print(f"[e4] pinning eval_sets_artifact: {eval_sets_artifact}", flush=True)
    cfg = load_config(cfg_path, overrides=overrides or None)
    _preflight_openevolve()
    print("=" * 60, flush=True)
    print(
        f"E4 CivilComments | fitness={cfg.fitness.mode} | geometry={cfg.clusters.geometry}",
        flush=True,
    )
    print(f"config: {cfg_path}", flush=True)
    print("=" * 60, flush=True)
    if dry_run:
        print("Dry-run OK", flush=True)
        return None
    _preflight_workers(cfg)
    if seed_pred_cache is not None and seed_pred_cache.is_dir():
        os.environ["PRIME_SEED_PRED_CACHE"] = str(seed_pred_cache.resolve())
        print(f"[reuse] PRIME_SEED_PRED_CACHE={seed_pred_cache}", flush=True)
    ctrl = PrimeController(cfg, PKG_ROOT, cfg_path)
    print(f"run_dir: {ctrl.ctx.run_dir}", flush=True)
    summary = ctrl.run()
    _write_findings(ctrl.ctx.run_dir, summary, cfg_path)
    # Snapshot RESULTS pointer under experiments/E4
    pointer = {
        "run_dir": str(ctrl.ctx.run_dir),
        "fitness_mode": summary.get("fitness_mode"),
        "cluster_geometry": summary.get("cluster_geometry"),
        "final_test": summary.get("final_test"),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    arm_tag = cfg.experiment.name
    (EXP / f"last_{arm_tag}.json").write_text(json.dumps(pointer, indent=2), encoding="utf-8")
    print(json.dumps(pointer, indent=2), flush=True)
    return {"run_dir": str(ctrl.ctx.run_dir), "summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        choices=["a", "b", "c", "ab", "all"],
        default="ab",
        help="a=global control, b=oracle min_group, c=style inferred, ab=A then B, all=A+B+C",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--eval-sets-artifact", type=Path, default=None)
    parser.add_argument("--seed-pred-cache", type=Path, default=None)
    args = parser.parse_args()

    order: List[str]
    if args.arm == "ab":
        order = ["a", "b"]
    elif args.arm == "all":
        order = ["a", "b", "c"]
    else:
        order = [args.arm]

    pinned_eval: Optional[Path] = args.eval_sets_artifact
    pinned_cache: Optional[Path] = args.seed_pred_cache
    results = []
    for key in order:
        cfg = ARM_CFGS[key]
        if not cfg.is_file():
            print(f"Missing config: {cfg}", file=sys.stderr)
            return 1
        # Share D_select/D_anchor across oracle arms A/B only (same geometry).
        use_pin = pinned_eval if key in ("a", "b") else None
        if key == "c":
            use_pin = None  # style groups → different stratification
        out = _run_one(cfg, args.dry_run, eval_sets_artifact=use_pin, seed_pred_cache=pinned_cache)
        if out is None:
            continue
        results.append({"arm": key, **{k: out[k] for k in ("run_dir",)}})
        run_dir = Path(out["run_dir"])
        if key == "a" and pinned_eval is None:
            es = run_dir / "eval_sets.json"
            if es.is_file():
                pinned_eval = es
                print(f"[e4] will pin eval_sets for arm B: {pinned_eval}", flush=True)
            cache = run_dir / "pred_cache"
            if cache.is_dir() and pinned_cache is None:
                pinned_cache = cache

    if len(results) >= 2:
        compare_path = EXP / f"pair_compare_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        compare_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {compare_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

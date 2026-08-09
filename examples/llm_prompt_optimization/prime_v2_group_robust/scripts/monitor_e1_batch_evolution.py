#!/usr/bin/env python3
"""Tail E1 run: batch diagnostics + evolution stall signals.

Usage:
  python scripts/monitor_e1_batch_evolution.py
  python scripts/monitor_e1_batch_evolution.py --run-dir results/E1_.../seed42_...
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PKG = Path(__file__).resolve().parents[1]


def _latest_run(*names: str) -> Optional[Path]:
    cands: List[Path] = []
    for name in names:
        root = PKG / "results" / name
        if root.is_dir():
            cands.extend([p for p in root.iterdir() if p.is_dir() and p.name.startswith("seed")])
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _summarize_cycle(cycle_dir: Path) -> Dict[str, Any]:
    diag = _load(cycle_dir / "batch_diagnostics.json") or {}
    info = _load(cycle_dir / "best_program_info.json") or {}
    oe_path = cycle_dir / "openevolve_output_path.txt"
    oe_dir = None
    if oe_path.is_file():
        oe_dir = Path(oe_path.read_text(encoding="utf-8").strip())
    early = None
    if oe_dir and oe_dir.is_dir():
        # OpenEvolve logs often mention early stopping
        for log in sorted(oe_dir.glob("**/*.log"))[-2:]:
            text = log.read_text(encoding="utf-8", errors="replace")
            if "early" in text.lower() or "stopping" in text.lower():
                early = "see_log"
                break
    rep = diag.get("representativeness") or {}
    batch = diag.get("batch") or {}
    pool = diag.get("pool") or {}
    roles = diag.get("role_integrity") or {}
    return {
        "cycle": cycle_dir.name,
        "seen": pool.get("n_seen"),
        "unseen": pool.get("n_unseen"),
        "hard": batch.get("n_hard"),
        "anchor": batch.get("n_anchor"),
        "hard_err_rate": batch.get("hard_error_rate"),
        "anchor_err_rate": batch.get("anchor_error_rate"),
        "weak_slots": rep.get("weak_cluster_hard_slots"),
        "weak_cov": rep.get("weak_cluster_hard_coverage"),
        "quotas": diag.get("group_hard_quotas"),
        "warns": [k for k, v in rep.items() if str(k).startswith("warn_") and v],
        "leak_Dsel": roles.get("n_leak_d_select"),
        "cluster_acc": pool.get("cluster_accuracies"),
        "oe_best": info.get("best_score"),
        "oe_early": early,
        "has_artifacts": (cycle_dir / "error_artifacts.txt").is_file(),
    }


def report(run_dir: Path) -> None:
    print(f"=== monitor {run_dir} ===", flush=True)
    meta = _load(run_dir / "run_metadata.json")
    if meta:
        print(
            f"fitness={meta.get('fitness_mode')} geometry={meta.get('cluster_geometry')} "
            f"K={meta.get('actual_n_clusters')}",
            flush=True,
        )
    clusters = _load(run_dir / "clusters.json")
    if clusters:
        diag = clusters.get("diagnostics") or clusters
        sizes = diag.get("cluster_sizes") if isinstance(diag, dict) else None
        merges = diag.get("merges") if isinstance(diag, dict) else None
        if sizes:
            print(f"cluster_sizes={sizes} merges={len(merges or [])}", flush=True)

    cycles = sorted(
        [p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("al_iter_")],
        key=lambda p: p.name,
    )
    if not cycles:
        print("(no al_iter_* yet)", flush=True)
        return
    for cdir in cycles:
        s = _summarize_cycle(cdir)
        print(
            f"{s['cycle']}: seen={s['seen']} unseen={s['unseen']} "
            f"H/A={s['hard']}/{s['anchor']} hard_err={s['hard_err_rate']} "
            f"weak_slots={s['weak_slots']} quotas={s['quotas']} "
            f"warns={s['warns'] or 'none'} leak={s['leak_Dsel']} "
            f"oe_best={s['oe_best']}",
            flush=True,
        )
        if s.get("weak_cov"):
            print(f"  weak_cov={s['weak_cov']} cluster_acc={s['cluster_acc']}", flush=True)

    events = run_dir / "events.jsonl"
    if events.is_file():
        lines = events.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        print(f"events: {len(lines)} lines; last=", flush=True)
        for line in lines[-3:]:
            try:
                e = json.loads(line)
                print(
                    f"  {e.get('stage_id') or e.get('stage')}: {e.get('message') or e.get('msg')}",
                    flush=True,
                )
            except Exception:
                print(f"  {line[:120]}", flush=True)

    if (run_dir / "summary.json").is_file():
        ft = (_load(run_dir / "summary.json") or {}).get("final_test") or {}
        print(f"DONE final_test R_worst={ft.get('R_worst')} R_global={ft.get('R_global')}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--watch", type=float, default=0.0, help="Re-report every N seconds")
    args = ap.parse_args()
    run_dir = args.run_dir
    if run_dir is None:
        run_dir = _latest_run("E1_pred_profile_cvar_lex", "E1_pred_profile_global")
    if run_dir is None or not run_dir.is_dir():
        print("No run dir found", flush=True)
        return 1
    while True:
        report(run_dir.resolve())
        if args.watch <= 0:
            break
        time.sleep(args.watch)
        # Pick newest if still launching
        latest = _latest_run("E1_pred_profile_cvar_lex", "E1_pred_profile_global")
        if latest is not None:
            run_dir = latest
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

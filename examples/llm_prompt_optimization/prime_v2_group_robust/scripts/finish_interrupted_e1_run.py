#!/usr/bin/env python3
"""Finish an interrupted E1 run that completed all AL cycles but died before final test.

Usage:
  python scripts/finish_interrupted_e1_run.py \\
    --run-dir results/E1_pred_profile_cvar_lex/seed42_20260729_213313
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _smoke_stages(run_dir: Path) -> List[Dict[str, Any]]:
    path = run_dir / "smoke_trace.jsonl"
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _pick_final_prompt(run_dir: Path) -> tuple[str, List[float], int]:
    """Replicate controller selection: keep the prompt with the best val key."""
    best_key = (-1.0, -1.0)
    best_cycle = 1
    for stage in _smoke_stages(run_dir):
        if stage.get("stage_id") != "12_selection_val":
            continue
        d = stage.get("details") or {}
        key = tuple(d.get("selection_key") or (-1.0, -1.0))
        cycle = int(d.get("cycle") or 0)
        if key >= best_key:
            best_key = key  # type: ignore[assignment]
            best_cycle = cycle
    prompt_path = run_dir / f"al_iter_{best_cycle}" / "best_prompt.txt"
    if not prompt_path.is_file():
        raise FileNotFoundError(f"missing selected prompt: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8"), list(best_key), best_cycle


def _rebuild_cycle_summaries(run_dir: Path) -> List[Dict[str, Any]]:
    stages = _smoke_stages(run_dir)
    by_cycle: Dict[int, Dict[str, Any]] = {}
    for stage in stages:
        d = stage.get("details") or {}
        cycle = d.get("cycle")
        if cycle is None:
            continue
        cycle = int(cycle)
        slot = by_cycle.setdefault(cycle, {"cycle": cycle})
        sid = stage.get("stage_id")
        if sid == "08_fitness":
            slot["fitness"] = d.get("fitness")
        elif sid == "10_evolution":
            slot["best_evo_score"] = d.get("best_score")
            slot["oe_seed_mode"] = d.get("seed_mode")
        elif sid == "12_selection_val":
            slot["val_selection_key"] = list(d.get("selection_key") or [])
            slot["proxy_cvar_tail_corr"] = d.get("proxy_cvar_tail_corr")
        elif sid == "11_consolidation":
            slot.setdefault("pareto", {}).update(
                {
                    "n_candidates": d.get("n_candidates"),
                    "front_size": d.get("front_size"),
                    "front_sources": d.get("front_sources"),
                    "heir_source": d.get("heir_source"),
                    "heir_select_fitness": d.get("heir_select_fitness"),
                }
            )
        elif sid == "11b_base_consolidation":
            slot["consolidation"] = {
                "enabled": d.get("enabled"),
                "changed": d.get("changed"),
                "select_fitness": d.get("select_fitness"),
                "evolved_select_fitness": d.get("evolved_select_fitness"),
                "delta_vs_evolved": d.get("delta_vs_evolved"),
                "won_front": d.get("won_front"),
            }
        elif sid == "10b_anchor_gate":
            slot["anchor_gate"] = {
                "accepted": True,
                "reason": None,
                "drop": None,
                "mcnemar_p_one_sided": d.get("mcnemar_p_one_sided"),
                "candidate_source": d.get("candidate_source"),
                "front_rank": d.get("front_rank"),
            }

    out: List[Dict[str, Any]] = []
    for cycle in sorted(by_cycle):
        slot = by_cycle[cycle]
        # Prefer on-disk artifacts when present (more complete).
        cons_path = run_dir / f"al_iter_{cycle}" / "consolidation.json"
        if cons_path.is_file():
            slot["consolidation"] = _load_json(cons_path)
        gate_path = run_dir / f"al_iter_{cycle}" / "anchor_gate.json"
        if gate_path.is_file():
            g = _load_json(gate_path)
            slot["anchor_gate"] = {
                "accepted": g.get("accepted"),
                "reason": g.get("reason"),
                "drop": g.get("drop"),
                "mcnemar_p_one_sided": g.get("mcnemar_p_one_sided"),
                "candidate_source": g.get("candidate_source"),
                "front_rank": g.get("front_rank"),
            }
        pf_path = run_dir / f"al_iter_{cycle}" / "pareto_front.json"
        if pf_path.is_file():
            pf = _load_json(pf_path)
            slot["pareto"] = {
                "n_candidates": pf.get("n_candidates"),
                "front_size": pf.get("size"),
                "front_sources": pf.get("sources"),
                "heir_source": pf.get("heir_source"),
                "heir_select_fitness": next(
                    (
                        m.get("fitness")
                        for m in (pf.get("members") or [])
                        if m.get("source") == pf.get("heir_source")
                    ),
                    None,
                ),
                "champion_slots": pf.get("champion_slots"),
            }
        info_path = run_dir / f"al_iter_{cycle}" / "best_program_info.json"
        if info_path.is_file():
            info = _load_json(info_path)
            slot["best_evo_score"] = info.get("best_score", slot.get("best_evo_score"))
            slot["oe_seed_mode"] = info.get("seed_mode", slot.get("oe_seed_mode"))
        out.append(slot)
    return out


def _eval_final_test(run_dir: Path, prompt: str) -> Dict[str, Any]:
    from prime.config import load_config
    from prime.data.wilds_loader import load_amazon_splits
    from prime.fitness.metrics import compute_metrics
    from prime.workers.ensemble import (
        build_workers,
        load_dotenv_if_present,
        parallel_predict,
    )

    load_dotenv_if_present()
    cfg = load_config(run_dir / "config_used.yaml")
    splits = load_amazon_splits(cfg.dataset, seed=cfg.active_learning.seed)
    test = splits["test"]
    # Cap to the same users as the interrupted run.
    assign = _load_json(run_dir / "cluster_assign_test.json")
    # assign format: user_id -> cluster_id (or richer); tolerate both.
    user_to_cluster: Dict[str, int] = {}
    if "assignments" in assign:
        raw = assign["assignments"]
    elif "user_to_cluster" in assign:
        raw = assign["user_to_cluster"]
    else:
        raw = assign
    for k, v in raw.items():
        if k.startswith("_"):
            continue
        user_to_cluster[str(k)] = int(v if not isinstance(v, dict) else v.get("cluster", 0))

    # Restrict to users present in the assignment artifact (the run's capped test).
    keep_users = set(user_to_cluster)
    idxs = [i for i, u in enumerate(test.user_ids) if str(u) in keep_users]
    if not idxs:
        raise RuntimeError("cluster_assign_test.json did not match any test users")
    texts = [test.texts[i] for i in idxs]
    labels = [test.labels[i] for i in idxs]
    user_ids = [test.user_ids[i] for i in idxs]
    cluster_ids = [user_to_cluster[str(u)] for u in user_ids]

    workers = build_workers(cfg.ensemble)
    print(
        f"[finish] evaluating final prompt on test "
        f"(n_examples={len(texts)}, n_users={len(set(map(str, user_ids)))})",
        flush=True,
    )
    ensemble, wp = parallel_predict(
        workers,
        texts,
        prompt,
        max_parallel=cfg.ensemble.max_parallel,
        tie_break=cfg.ensemble.tie_break,
        aggregation=cfg.ensemble.aggregation,
    )
    return compute_metrics(
        np.array(ensemble),
        np.array(labels),
        np.array(user_ids),
        worker_predictions=[np.array(w) for w in wp],
        cluster_ids=np.array(cluster_ids),
        cvar_quantile=cfg.fitness.cvar_quantile,
        beta_a=cfg.fitness.beta_a,
        beta_b=cfg.fitness.beta_b,
        tail_quantile=cfg.active_learning.tail_quantile,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        print(f"missing run dir: {run_dir}", file=sys.stderr)
        return 1
    if (run_dir / "summary.json").is_file():
        print(f"summary.json already exists in {run_dir}; nothing to do")
        return 0

    cycles = sorted(run_dir.glob("al_iter_*"))
    if len(cycles) < 3:
        print(
            f"run only has {len(cycles)} cycles — cannot finish; need a full AL loop",
            file=sys.stderr,
        )
        return 2

    prompt, best_key, best_cycle = _pick_final_prompt(run_dir)
    (run_dir / "final_prompt.txt").write_text(prompt, encoding="utf-8")
    print(
        f"[finish] selected prompt from cycle {best_cycle} "
        f"(val key={best_key}, chars={len(prompt)})",
        flush=True,
    )

    cycle_summaries = _rebuild_cycle_summaries(run_dir)
    final_test = _eval_final_test(run_dir, prompt)

    # Prefer last cycle's pareto_front as the "final front" snapshot.
    last_front = _load_json(cycles[-1] / "pareto_front.json")
    summary: Dict[str, Any] = {
        "final_test": final_test,
        "run_dir": str(run_dir),
        "use_mock": False,
        "inference_backend": "openrouter",
        "cluster_geometry": "pred_profile",
        "actual_n_clusters": len((last_front.get("cluster_axis") or [])),
        "fitness_mode": "cvar_lex",
        "best_selection_key": best_key,
        "best_selection_cycle": best_cycle,
        "finished_by": "finish_interrupted_e1_run.py",
        "cycles": cycle_summaries,
        "pareto": {
            "enabled": True,
            "mode": "champion_archive",
            "final_front": {
                "size": last_front.get("size"),
                "sources": last_front.get("sources"),
                "best_fitness": last_front.get("best_fitness"),
                "cluster_champions": last_front.get("cluster_champions"),
                "members": last_front.get("members"),
            },
            "final_champion_slots": last_front.get("champion_slots"),
            "heir_sources": [
                (c.get("pareto") or {}).get("heir_source") for c in cycle_summaries
            ],
            "consolidation_deltas": [
                (c.get("consolidation") or {}).get("delta_vs_evolved")
                for c in cycle_summaries
            ],
        },
        "anchor_gate": {
            "n_decisions": sum(
                1 for c in cycle_summaries if c.get("anchor_gate") is not None
            ),
            "n_accepted": sum(
                1
                for c in cycle_summaries
                if (c.get("anchor_gate") or {}).get("accepted")
            ),
            "n_rejected": sum(
                1
                for c in cycle_summaries
                if (c.get("anchor_gate") or {}).get("accepted") is False
            ),
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "best_selection_cycle": best_cycle,
                "final_test": {
                    k: final_test.get(k)
                    for k in [
                        "R_global",
                        "R_worst",
                        "R_tail",
                        "CVaR_cluster",
                        "mae",
                        "num_users",
                    ]
                },
            },
            indent=2,
        ),
        flush=True,
    )

    # Package FINDINGS like the pair launcher.
    from scripts.run_e1_pred_profile_pair import _package_run

    cfg_path = PKG_ROOT / "experiments" / "E1_pred_profile_cvar_vs_global" / "config_cvar_live.yaml"
    _package_run(run_dir, cfg_path, summary)
    print(f"[finish] wrote summary.json + FINDINGS.md under {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

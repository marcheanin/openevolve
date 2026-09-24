#!/usr/bin/env python
"""S8 pilot: Seed + APE + APO + GPO (+ optional PRIME) on fixed Regime-Shift sets.

ROADMAP_PHASE3 §6–§7 S8. Shared Scorer / D_dev / test_fixed / mutator LLM.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _metrics_block(m: dict) -> dict:
    return {
        k: m.get(k)
        for k in (
            "R_worst_gba",
            "R_gba_mean",
            "R_soft_min_gba",
            "toxic_recall",
            "R_global",
            "fitness",
        )
    }


def _load_prime_prompt(prime_run: Path) -> str:
    for rel in (
        "final_prompt.txt",
        "best_prompt.txt",
        "analysis/final_prompt.txt",
        "selected_prompt.txt",
    ):
        p = prime_run / rel
        if p.is_file():
            return p.read_text(encoding="utf-8")
    # Fallback: latest al_iter_*/best_prompt.txt
    cands = sorted(prime_run.glob("al_iter_*/best_prompt.txt"))
    if cands:
        return cands[-1].read_text(encoding="utf-8")
    raise FileNotFoundError(f"No prompt file under {prime_run}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/E5_civilcomments/config_prime_main.yaml",
    )
    parser.add_argument(
        "--fixed-dir",
        type=Path,
        default=ROOT / "experiments/E5_civilcomments/fixed_sets",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "results/E5_s8_pilot")
    parser.add_argument(
        "--methods",
        default="seed,ape,apo,gpo",
        help="Comma list: seed,ape,apo,gpo,prime",
    )
    parser.add_argument(
        "--prime-run",
        type=Path,
        default=None,
        help="Existing PRIME run dir (for method=prime)",
    )
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--u-target-n", type=int, default=4000)
    parser.add_argument("--apo-rounds", type=int, default=6)
    args = parser.parse_args()

    from baselines.ape import APEOptimizer, save_result
    from baselines.apo import APOOptimizer
    from baselines.api import LabeledSet, Task, UnlabeledSet, score_prompt_on_set
    from baselines.gpo import GPOOptimizer
    from baselines.optimizer_llm import build_optimizer_llm
    from baselines.regime_sets import load_or_build_regime_sets
    from prime.config import load_config
    from prime.data.civilcomments_loader import load_civilcomments_splits
    from prime.data.fixed_sets import FixedSet, materialize_split
    from prime.experiment.budget import TokenTracker
    from prime.workers.scorer import Scorer

    cfg = load_config(args.config)
    cfg.dataset.max_val_users = max(int(cfg.dataset.max_val_users or 0), 50_000)
    cfg.dataset.max_test_users = max(int(cfg.dataset.max_test_users or 0), 80_000)
    splits = load_civilcomments_splits(cfg.dataset, seed=cfg.active_learning.seed)
    d_dev = FixedSet.load(args.fixed_dir / "d_dev.json")
    mat_dev = materialize_split(splits["validation"], d_dev.indices)
    seed_prompt = (ROOT / "prompts/initial_prompt_civilcomments.txt").read_text(encoding="utf-8")

    regime = load_or_build_regime_sets(
        splits["train"],
        args.fixed_dir,
        seed=int(cfg.active_learning.seed),
        s_per_cell=45,
        u_n=int(args.u_target_n),
    )
    train_set: LabeledSet = regime["s_source"]  # type: ignore[assignment]
    u_set: UnlabeledSet = regime["u_target"]  # type: ignore[assignment]
    dev_set = LabeledSet(
        texts=mat_dev["texts"],
        labels=mat_dev["labels"],
        group_ids=mat_dev["cluster_ids"],
    )

    scorer = Scorer(cfg.ensemble, label_space="binary", use_mock=args.mock)
    # S8: report native spend honestly; do not abort mid-method on soft caps.
    cfg.budget.on_exhausted = "warn"
    budget = TokenTracker.from_cfg(cfg.budget)
    opt_llm = None if args.mock else build_optimizer_llm(cfg, temperature=0.0)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {"methods": {}}
    else:
        summary = {"methods": {}}
    summary.setdefault("methods", {})
    summary["regime"] = {
        "s_source_n": len(train_set.texts),
        "u_target_n": len(u_set.texts),
        "d_dev_n": len(dev_set.texts),
        "manifest": str(regime.get("manifest")),
    }
    summary["optimizer_model"] = None if opt_llm is None else opt_llm.model_name
    summary["scorer_model"] = (
        cfg.ensemble.workers[0].name if cfg.ensemble.workers else None
    )

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for method in methods:
        mdir = out / method
        mdir.mkdir(parents=True, exist_ok=True)
        print(f"=== S8 method={method} ===", flush=True)

        if method == "seed":
            best = seed_prompt
            trace = {"method": "seed"}
        elif method == "prime":
            if args.prime_run is None:
                print("skip prime: pass --prime-run DIR", flush=True)
                continue
            best = _load_prime_prompt(Path(args.prime_run))
            trace = {"method": "prime", "prime_run": str(args.prime_run)}
            (mdir / "best_prompt.txt").write_text(best, encoding="utf-8")
        elif method == "ape":
            # APE generation is T=0 (native); rebuild optimizer if needed.
            ape_llm = None if args.mock else build_optimizer_llm(cfg, temperature=0.0)
            task = Task(
                train=train_set,
                dev=dev_set,
                unlabeled=UnlabeledSet(texts=[]),
                label_budget=0,
                scorer=scorer,
                optimizer_llm=ape_llm,
                budget=budget,
                seed_prompt=seed_prompt,
                meta={"seed": cfg.active_learning.seed},
            )
            result = APEOptimizer(k_prompts=6, n_shots=36, temperature=0.0).run(task)
            save_result(result, mdir / "optimizer_result.json")
            best = result.best_prompt
            trace = result.trace
        elif method == "gpo":
            gpo_llm = None if args.mock else build_optimizer_llm(cfg, temperature=0.0)
            task = Task(
                train=train_set,
                dev=dev_set,
                unlabeled=u_set,
                label_budget=0,
                scorer=scorer,
                optimizer_llm=gpo_llm,
                budget=budget,
                seed_prompt=seed_prompt,
                meta={"seed": cfg.active_learning.seed},
            )
            result = GPOOptimizer(
                k_prompts=6, n_shots=36, conf_threshold=0.83, temperature=0.0
            ).run(task)
            save_result(result, mdir / "optimizer_result.json")
            best = result.best_prompt
            trace = result.trace
        elif method == "apo":
            # ProTeGi uses temperature 0.7 inside gradient/rewrite calls.
            apo_llm = None if args.mock else build_optimizer_llm(cfg, temperature=0.7)
            task = Task(
                train=train_set,
                dev=dev_set,
                unlabeled=UnlabeledSet(texts=[]),
                label_budget=0,
                scorer=scorer,
                optimizer_llm=apo_llm,
                budget=budget,
                seed_prompt=seed_prompt,
                meta={"seed": cfg.active_learning.seed},
            )
            result = APOOptimizer(rounds=int(args.apo_rounds), beam_size=4).run(task)
            save_result(result, mdir / "optimizer_result.json")
            best = result.best_prompt
            trace = result.trace
        else:
            print(f"skip unknown method {method}", flush=True)
            continue

        (mdir / "best_prompt.txt").write_text(best, encoding="utf-8")
        dev_metrics = score_prompt_on_set(scorer, best, dev_set)
        budget.charge("val", n_calls=len(dev_set.texts), kind="scorer", note=f"{method}_dev_report")
        block = {"dev": _metrics_block(dev_metrics), "trace": trace}
        if not args.skip_test and (args.fixed_dir / "test_fixed.json").is_file():
            tf = FixedSet.load(args.fixed_dir / "test_fixed.json")
            mat_t = materialize_split(splits["test"], tf.indices)
            test_set = LabeledSet(
                texts=mat_t["texts"],
                labels=mat_t["labels"],
                group_ids=mat_t["cluster_ids"],
            )
            tm = score_prompt_on_set(scorer, best, test_set)
            budget.charge("audit", n_calls=len(test_set.texts), kind="scorer", note=f"{method}_test")
            block["test_fixed"] = _metrics_block(tm)
        summary["methods"][method] = block
        (mdir / "metrics.json").write_text(json.dumps(block, indent=2), encoding="utf-8")
        print(method, json.dumps(block, indent=2), flush=True)

    # Merge budget spend across partial runs when summary already had a budget.
    snap = budget.snapshot()
    prev_b = summary.get("budget") if isinstance(summary.get("budget"), dict) else None
    if prev_b:
        for k in ("total_calls", "scorer_calls", "optimizer_calls"):
            if k in snap and k in prev_b and isinstance(snap[k], int):
                snap[k] = int(prev_b[k]) + int(snap[k])
        snap["note"] = "cumulative across partial S8 invocations"
    summary["budget"] = snap
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # Human-readable table
    lines = [
        "# E5 S8 pilot",
        "",
        f"- optimizer: {summary.get('optimizer_model')}",
        f"- scorer: {summary.get('scorer_model')}",
        f"- S_source n={summary['regime']['s_source_n']}, U_target n={summary['regime']['u_target_n']}",
        "",
        "| method | D_dev worst-GBA | D_dev softmin | test_fixed worst-GBA | test_fixed softmin |",
        "|---|---:|---:|---:|---:|",
    ]
    for m, block in summary["methods"].items():
        d = block.get("dev") or {}
        t = block.get("test_fixed") or {}
        lines.append(
            f"| {m} | {d.get('R_worst_gba')} | {d.get('R_soft_min_gba')} | "
            f"{t.get('R_worst_gba')} | {t.get('R_soft_min_gba')} |"
        )
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote", summary_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

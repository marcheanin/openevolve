#!/usr/bin/env python
"""S9 matrix: R0–R12 on CivilComments Regime-Shift (ROADMAP §7.1).

Usage examples:
  python scripts/run_e5_s9_matrix.py --seed 42 --methods seed,ape,ape_k48,ape_ut
  python scripts/run_e5_s9_matrix.py --seed 42 --methods gpo,random_al,oracle
  python scripts/run_e5_s9_matrix.py --seed 42 --methods prime --prime-run results/.../seed42_...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_METHODS = (
    "seed,ape,ape_k48,ape_ut,apo,gpo,evoprompt_ga,evoprompt_de,"
    "gepa,random_al,oracle,prime"
)


def _metrics_block(m: dict) -> dict:
    return {
        k: m.get(k)
        for k in (
            "R_worst_gba",
            "R_worst_group",
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
        "evals/stable_test_fixed/final_prompt.txt",
        "analysis/final_prompt.txt",
        "selected_prompt.txt",
    ):
        p = prime_run / rel
        if p.is_file():
            return p.read_text(encoding="utf-8")
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
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--prime-run", type=Path, default=None)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--u-target-n", type=int, default=4000)
    parser.add_argument("--apo-rounds", type=int, default=6)
    parser.add_argument("--oracle-max-target", type=int, default=800,
                        help="Cap fully-labeled target for Oracle APO cost")
    parser.add_argument("--evo-generations", type=int, default=10)
    parser.add_argument("--evo-pop", type=int, default=10)
    parser.add_argument("--gepa-reflections", type=int, default=12)
    args = parser.parse_args()

    from baselines.ape import APEOptimizer, save_result
    from baselines.ape_ut import APEUTOptimizer
    from baselines.apo import APOOptimizer
    from baselines.api import LabeledSet, Task, UnlabeledSet, score_prompt_on_set
    from baselines.evoprompt import EvoPromptOptimizer
    from baselines.gepa_baseline import GEPAOptimizer
    from baselines.gpo import GPOOptimizer
    from baselines.optimizer_llm import build_optimizer_llm
    from baselines.random_al import OracleALOptimizer, RandomALOptimizer
    from baselines.regime_sets import load_or_build_regime_sets
    from prime.config import load_config
    from prime.data.civilcomments_loader import load_civilcomments_splits
    from prime.data.fixed_sets import FixedSet, materialize_split
    from prime.experiment.budget import TokenTracker
    from prime.workers.scorer import Scorer

    cfg = load_config(args.config)
    # Optimizer / S_source / U_target sampling seed (varies across matrix).
    cfg.active_learning.seed = int(args.seed)
    cfg.dataset.max_val_users = max(int(cfg.dataset.max_val_users or 0), 50_000)
    cfg.dataset.max_test_users = max(int(cfg.dataset.max_test_users or 0), 80_000)
    # CRITICAL: D_dev / test_fixed indices were fingerprinted under loader seed=42.
    # Never re-cap splits with a different seed or indices map to different comments.
    splits = load_civilcomments_splits(cfg.dataset, seed=42)
    d_dev = FixedSet.load(args.fixed_dir / "d_dev.json")
    mat_dev = materialize_split(splits["validation"], d_dev.indices)
    seed_prompt = (ROOT / "prompts/initial_prompt_civilcomments.txt").read_text(encoding="utf-8")

    regime = load_or_build_regime_sets(
        splits["train"],
        args.fixed_dir,
        seed=int(args.seed),
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
    cfg.budget.on_exhausted = "warn"
    budget = TokenTracker.from_cfg(cfg.budget)

    out = Path(args.out) if args.out else ROOT / f"results/E5_s9_matrix/seed{int(args.seed)}"
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
    summary["seed"] = int(args.seed)
    summary["regime"] = {
        "s_source_n": len(train_set.texts),
        "u_target_n": len(u_set.texts),
        "d_dev_n": len(dev_set.texts),
        "manifest": str(regime.get("manifest")),
    }
    summary["scorer_model"] = (
        cfg.ensemble.workers[0].name if cfg.ensemble.workers else None
    )

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    # Oracle only on seed 42 per preregistration.
    if int(args.seed) != 42 and "oracle" in methods:
        print("skip oracle: preregistered for seed 42 only", flush=True)
        methods = [m for m in methods if m != "oracle"]

    default_prime = (
        ROOT
        / "results/E5_civilcomments_prime_main_v2/seed42_20260809_080918"
    )

    for method in methods:
        mdir = out / method
        mdir.mkdir(parents=True, exist_ok=True)
        # Resume: skip if metrics already present.
        if (mdir / "metrics.json").is_file() and method in summary.get("methods", {}):
            print(f"=== S9 skip existing method={method} ===", flush=True)
            continue
        print(f"=== S9 seed={args.seed} method={method} ===", flush=True)

        def _task(train, unlabeled, temp: float) -> Task:
            llm = None if args.mock else build_optimizer_llm(cfg, temperature=temp)
            return Task(
                train=train,
                dev=dev_set,
                unlabeled=unlabeled,
                label_budget=int(getattr(args, "label_budget", 240) or 240),
                scorer=scorer,
                optimizer_llm=llm,
                budget=budget,
                seed_prompt=seed_prompt,
                meta={"seed": int(args.seed)},
            )

        if method == "seed":
            best = seed_prompt
            trace = {"method": "seed"}
        elif method == "prime":
            pr = Path(args.prime_run) if args.prime_run else default_prime
            if int(args.seed) != 42 and args.prime_run is None:
                print(
                    f"skip prime seed={args.seed}: pass --prime-run for non-42",
                    flush=True,
                )
                continue
            best = _load_prime_prompt(pr)
            trace = {"method": "prime", "prime_run": str(pr), "protocol": "E5v2_artifact"}
        elif method == "ape":
            result = APEOptimizer(k_prompts=6, n_shots=36, temperature=0.0).run(
                _task(train_set, UnlabeledSet(texts=[]), 0.0)
            )
            save_result(result, mdir / "optimizer_result.json")
            best, trace = result.best_prompt, result.trace
        elif method == "ape_k48":
            result = APEOptimizer(k_prompts=48, n_shots=288, temperature=0.0).run(
                _task(train_set, UnlabeledSet(texts=[]), 0.0)
            )
            save_result(result, mdir / "optimizer_result.json")
            best, trace = result.best_prompt, result.trace
        elif method == "ape_ut":
            result = APEUTOptimizer(k_prompts=6, n_shots=36, n_unlabeled=12).run(
                _task(train_set, u_set, 0.0)
            )
            save_result(result, mdir / "optimizer_result.json")
            best, trace = result.best_prompt, result.trace
        elif method == "gpo":
            result = GPOOptimizer(
                k_prompts=6, n_shots=36, conf_threshold=0.83, temperature=0.0
            ).run(_task(train_set, u_set, 0.0))
            save_result(result, mdir / "optimizer_result.json")
            best, trace = result.best_prompt, result.trace
        elif method == "apo":
            result = APOOptimizer(rounds=int(args.apo_rounds), beam_size=4).run(
                _task(train_set, UnlabeledSet(texts=[]), 0.7)
            )
            save_result(result, mdir / "optimizer_result.json")
            best, trace = result.best_prompt, result.trace
        elif method == "evoprompt_ga":
            result = EvoPromptOptimizer(
                mode="ga",
                population_size=int(args.evo_pop),
                generations=int(args.evo_generations),
            ).run(_task(train_set, UnlabeledSet(texts=[]), 0.8))
            save_result(result, mdir / "optimizer_result.json")
            best, trace = result.best_prompt, result.trace
        elif method == "evoprompt_de":
            result = EvoPromptOptimizer(
                mode="de",
                population_size=int(args.evo_pop),
                generations=int(args.evo_generations),
            ).run(_task(train_set, UnlabeledSet(texts=[]), 0.8))
            save_result(result, mdir / "optimizer_result.json")
            best, trace = result.best_prompt, result.trace
        elif method == "gepa":
            result = GEPAOptimizer(
                max_metric_calls=20_000,
                reflections=int(args.gepa_reflections),
            ).run(_task(train_set, UnlabeledSet(texts=[]), 0.7))
            save_result(result, mdir / "optimizer_result.json")
            best, trace = result.best_prompt, result.trace
        elif method == "random_al":
            opt = RandomALOptimizer(
                label_budget=240,
                apo_rounds=int(args.apo_rounds),
                train_split=splits["train"],
            )
            result = opt.run(_task(train_set, u_set, 0.7))
            save_result(result, mdir / "optimizer_result.json")
            best, trace = result.best_prompt, result.trace
        elif method == "oracle":
            opt = OracleALOptimizer(
                apo_rounds=int(args.apo_rounds),
                train_split=splits["train"],
                max_target_labels=int(args.oracle_max_target),
            )
            result = opt.run(_task(train_set, u_set, 0.7))
            save_result(result, mdir / "optimizer_result.json")
            best, trace = result.best_prompt, result.trace
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
        summary["budget"] = budget.snapshot()
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(method, json.dumps(_metrics_block(block.get("test_fixed") or block["dev"]), indent=2), flush=True)

    summary["budget"] = budget.snapshot()
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# E5 S9 matrix", "", f"seed={args.seed}", ""]
    for m, block in summary.get("methods", {}).items():
        tf = block.get("test_fixed") or {}
        lines.append(
            f"- {m}: worst-GBA={tf.get('R_worst_gba')} softmin={tf.get('R_soft_min_gba')}"
        )
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""S8 pilot scaffold: Seed + APE on D_dev / test_fixed (one seed).

GPO/APO bodies land next; this wires the shared harness end-to-end.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/E5_civilcomments/config_prime_main.yaml")
    parser.add_argument("--fixed-dir", type=Path, default=ROOT / "experiments/E5_civilcomments/fixed_sets")
    parser.add_argument("--out", type=Path, default=ROOT / "results/E5_s8_pilot")
    parser.add_argument("--methods", default="seed,ape")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()

    from baselines.ape import APEOptimizer, save_result
    from baselines.api import LabeledSet, Task, UnlabeledSet, score_prompt_on_set
    from prime.config import load_config
    from prime.data.civilcomments_loader import load_civilcomments_splits
    from prime.data.fixed_sets import FixedSet, materialize_split
    from prime.experiment.budget import TokenTracker
    from prime.workers.ensemble import build_workers
    from prime.workers.scorer import Scorer

    cfg = load_config(args.config)
    cfg.dataset.max_val_users = max(int(cfg.dataset.max_val_users or 0), 50_000)
    cfg.dataset.max_test_users = max(int(cfg.dataset.max_test_users or 0), 80_000)
    splits = load_civilcomments_splits(cfg.dataset, seed=cfg.active_learning.seed)
    d_dev = FixedSet.load(args.fixed_dir / "d_dev.json")
    mat_dev = materialize_split(splits["validation"], d_dev.indices)
    seed_prompt = (ROOT / "prompts/initial_prompt_civilcomments.txt").read_text(encoding="utf-8")

    # Cheap S_source: first 360 of D_select-like cells from train head (labeled).
    train = splits["train"]
    # Use a stratified head of train as labeled source for APE.
    src_n = min(360, len(train.texts))
    train_set = LabeledSet(
        texts=list(train.texts[:src_n]),
        labels=[int(x) for x in train.labels[:src_n]],
        group_ids=[int(x) for x in (train.example_cluster_ids or [0] * src_n)[:src_n]],
    )
    dev_set = LabeledSet(
        texts=mat_dev["texts"],
        labels=mat_dev["labels"],
        group_ids=mat_dev["cluster_ids"],
    )

    scorer = Scorer(cfg.ensemble, label_space="binary", use_mock=args.mock)
    budget = TokenTracker.from_cfg(cfg.budget)
    opt_llm = None
    if not args.mock:
        workers = build_workers(cfg.ensemble)  # placeholder; mutator wired later
        opt_llm = workers[0] if workers else None

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"methods": {}}

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for method in methods:
        mdir = out / method
        mdir.mkdir(parents=True, exist_ok=True)
        if method == "seed":
            best = seed_prompt
            trace = {"method": "seed"}
        elif method == "ape":
            task = Task(
                train=train_set,
                dev=dev_set,
                unlabeled=UnlabeledSet(texts=[]),
                label_budget=0,
                scorer=scorer,
                optimizer_llm=opt_llm,
                budget=budget,
                seed_prompt=seed_prompt,
                meta={"seed": cfg.active_learning.seed},
            )
            result = APEOptimizer(k_prompts=6, n_shots=36).run(task)
            save_result(result, mdir / "optimizer_result.json")
            best = result.best_prompt
            trace = result.trace
        else:
            print(f"skip unknown method {method}")
            continue

        (mdir / "best_prompt.txt").write_text(best, encoding="utf-8")
        dev_metrics = score_prompt_on_set(scorer, best, dev_set)
        budget.charge("val", n_calls=len(dev_set.texts), kind="scorer", note=f"{method}_dev")
        block = {
            "dev": {
                k: dev_metrics.get(k)
                for k in (
                    "R_worst_gba",
                    "R_gba_mean",
                    "R_soft_min_gba",
                    "toxic_recall",
                    "R_global",
                    "fitness",
                )
            },
            "trace": trace,
        }
        if not args.skip_test and (args.fixed_dir / "test_fixed.json").is_file():
            tf = FixedSet.load(args.fixed_dir / "test_fixed.json")
            mat_t = materialize_split(splits["test"], tf.indices)
            test_set = LabeledSet(
                texts=mat_t["texts"], labels=mat_t["labels"], group_ids=mat_t["cluster_ids"]
            )
            tm = score_prompt_on_set(scorer, best, test_set)
            budget.charge("audit", n_calls=len(test_set.texts), kind="scorer", note=f"{method}_test")
            block["test_fixed"] = {
                k: tm.get(k)
                for k in (
                    "R_worst_gba",
                    "R_gba_mean",
                    "R_soft_min_gba",
                    "toxic_recall",
                    "R_global",
                    "fitness",
                )
            }
        summary["methods"][method] = block
        (mdir / "metrics.json").write_text(json.dumps(block, indent=2), encoding="utf-8")
        print(method, json.dumps(block, indent=2), flush=True)

    summary["budget"] = budget.snapshot()
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote", out / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

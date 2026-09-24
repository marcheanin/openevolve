#!/usr/bin/env python
"""E6 Phase B: top-3 CivilComments methods on Amazon category-shift.

Methods (S9 ranking): GPO, EvoPrompt-DE, PRIME.
Primary report metric: cvar25 of macro-within-cluster on test_fixed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def _load_prime_prompt(prime_run: Path) -> str:
    for rel in (
        "final_prompt.txt",
        "best_prompt.txt",
        "selected_prompt.txt",
    ):
        p = prime_run / rel
        if p.is_file():
            return p.read_text(encoding="utf-8")
    cands = sorted(prime_run.glob("al_iter_*/best_prompt.txt"))
    if cands:
        return cands[-1].read_text(encoding="utf-8")
    raise FileNotFoundError(f"No prompt under {prime_run}")


def _score_fixed(scorer, prompt: str, mat: dict, cache: Path) -> dict:
    from e6_metrics import all_metrics

    texts = mat["texts"]
    y = np.asarray(mat["labels"])
    c = np.asarray(mat["cluster_ids"])
    if cache.is_file() and len(np.load(cache)) == len(texts):
        preds = np.load(cache)
    else:
        preds = np.asarray(
            scorer.predict_batch(texts, prompt, labels_for_mock=list(y)).preds,
            dtype=np.int16,
        )
        np.save(cache, preds)
    return all_metrics(preds, y, c, np.arange(len(y)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/E6_amazon_category_controls/config.yaml",
    )
    ap.add_argument(
        "--fixed-dir",
        type=Path,
        default=ROOT / "experiments/E6_amazon_category_controls/fixed_sets",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--methods", default="seed,gpo,evoprompt_de,prime")
    ap.add_argument("--prime-run", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--u-target-n", type=int, default=400)
    ap.add_argument("--s-source-n", type=int, default=360)
    ap.add_argument("--evo-generations", type=int, default=8)
    ap.add_argument("--evo-pop", type=int, default=8)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--skip-test", action="store_true")
    args = ap.parse_args()

    from baselines.api import LabeledSet, Task, UnlabeledSet
    from baselines.evoprompt import EvoPromptOptimizer
    from baselines.gpo import GPOOptimizer
    from baselines.optimizer_llm import build_optimizer_llm
    from e6_score_prompt import load_mat
    from prime.config import load_config
    from prime.data.wilds_loader import load_amazon_splits
    from prime.experiment.budget import TokenTracker
    from prime.workers.ensemble import load_dotenv_if_present
    from prime.workers.scorer import Scorer

    load_dotenv_if_present()
    cfg = load_config(args.config)
    cfg.active_learning.seed = int(args.seed)
    cfg.budget.on_exhausted = "warn"

    splits = load_amazon_splits(cfg.dataset, seed=42)
    train, val = splits["train"], splits["validation"]
    rng = np.random.RandomState(int(args.seed))

    # S_source: subsample Books train, stratified by collapsed rating if possible.
    s_n = min(int(args.s_source_n), len(train.texts))
    s_idx = sorted(int(i) for i in rng.choice(len(train.texts), size=s_n, replace=False))
    train_set = LabeledSet(
        texts=[train.texts[i] for i in s_idx],
        labels=[int(train.labels[i]) for i in s_idx],
    )

    # U_target: unlabeled non-Books from validation (eval split already excludes Books).
    u_n = min(int(args.u_target_n), len(val.texts))
    u_idx = sorted(int(i) for i in rng.choice(len(val.texts), size=u_n, replace=False))
    u_set = UnlabeledSet(texts=[val.texts[i] for i in u_idx])

    d_dev = load_mat(args.fixed_dir, "d_dev")
    test_mat = None if args.skip_test else load_mat(args.fixed_dir, "test_fixed")
    dev_set = LabeledSet(
        texts=d_dev["texts"],
        labels=[int(x) for x in d_dev["labels"]],
        group_ids=[int(x) for x in d_dev["cluster_ids"]],
    )

    seed_prompt = (ROOT / "prompts/initial_prompt.txt").read_text(encoding="utf-8")
    scorer = Scorer(cfg.ensemble, label_space="ordinal5", use_mock=args.mock)
    budget = TokenTracker.from_cfg(cfg.budget)

    out = Path(args.out) if args.out else ROOT / f"results/E6_top3_matrix/seed{int(args.seed)}"
    out.mkdir(parents=True, exist_ok=True)
    preds_dir = out / "preds"
    preds_dir.mkdir(exist_ok=True)
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {"methods": {}}
    summary.setdefault("methods", {})
    summary["seed"] = int(args.seed)
    summary["regime"] = {
        "s_source_n": len(train_set.texts),
        "u_target_n": len(u_set.texts),
        "d_dev_n": len(dev_set.texts),
        "test_n": None if test_mat is None else len(test_mat["texts"]),
        "label_space": "ordinal5",
        "primary": "cvar25_macro_within_cluster",
    }

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    meta = {
        "seed": int(args.seed),
        "label_space": "ordinal5",
        "dev_score_key": "R_global",
    }

    for method in methods:
        mdir = out / method
        mdir.mkdir(parents=True, exist_ok=True)
        if (mdir / "metrics.json").is_file() and method in summary.get("methods", {}):
            print(f"=== E6 skip existing method={method} ===", flush=True)
            continue
        print(f"=== E6 top3 seed={args.seed} method={method} ===", flush=True)

        def _task(train_ls, unlabeled, temp: float) -> Task:
            llm = None if args.mock else build_optimizer_llm(cfg, temperature=temp)
            return Task(
                train=train_ls,
                dev=dev_set,
                unlabeled=unlabeled,
                label_budget=240,
                scorer=scorer,
                optimizer_llm=llm,
                budget=budget,
                seed_prompt=seed_prompt,
                meta=dict(meta),
            )

        if method == "seed":
            best = seed_prompt
            trace = {"method": "seed"}
        elif method == "prime":
            if args.prime_run is None:
                print("skip prime: pass --prime-run <results/...>", flush=True)
                continue
            best = _load_prime_prompt(Path(args.prime_run))
            trace = {"method": "prime", "prime_run": str(args.prime_run)}
        elif method == "gpo":
            from baselines.ape import save_result

            result = GPOOptimizer(
                k_prompts=6, n_shots=36, conf_threshold=0.83, temperature=0.0
            ).run(_task(train_set, u_set, 0.0))
            save_result(result, mdir / "optimizer_result.json")
            best, trace = result.best_prompt, result.trace
        elif method == "evoprompt_de":
            from baselines.ape import save_result

            result = EvoPromptOptimizer(
                mode="de",
                population_size=int(args.evo_pop),
                generations=int(args.evo_generations),
            ).run(_task(train_set, UnlabeledSet(texts=[]), 0.8))
            save_result(result, mdir / "optimizer_result.json")
            best, trace = result.best_prompt, result.trace
        else:
            print(f"skip unknown method {method}", flush=True)
            continue

        (mdir / "best_prompt.txt").write_text(best, encoding="utf-8")
        block = {
            "dev": _score_fixed(scorer, best, d_dev, preds_dir / f"{method}__d_dev.npy"),
            "trace": trace,
        }
        budget.charge("val", n_calls=len(d_dev["texts"]), kind="scorer", note=f"{method}_dev")
        if test_mat is not None:
            block["test_fixed"] = _score_fixed(
                scorer, best, test_mat, preds_dir / f"{method}__test_fixed.npy"
            )
            budget.charge(
                "audit", n_calls=len(test_mat["texts"]), kind="scorer", note=f"{method}_test"
            )
        summary["methods"][method] = block
        (mdir / "metrics.json").write_text(json.dumps(block, indent=2), encoding="utf-8")
        summary["budget"] = budget.snapshot()
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        tf = block.get("test_fixed") or block["dev"]
        print(
            f"{method}: cvar25={tf.get('cvar25'):.4f} global={tf.get('R_global'):.4f} "
            f"op={tf.get('op_shift'):.3f}",
            flush=True,
        )

    summary["budget"] = budget.snapshot()
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# E6 top-3 matrix (Amazon category-shift)",
        "",
        f"seed={args.seed}",
        "primary=cvar25 macro-within-cluster",
        "",
    ]
    for m, block in summary.get("methods", {}).items():
        tf = block.get("test_fixed") or {}
        lines.append(
            f"- {m}: cvar25={tf.get('cvar25')} global={tf.get('R_global')} "
            f"op_shift={tf.get('op_shift')}"
        )
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

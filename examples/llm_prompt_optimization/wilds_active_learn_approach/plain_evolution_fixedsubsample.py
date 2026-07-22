"""
Plain (non-AL) evolution ablation with fixed splits.

Runs N_AL "cycles" only as a logging / comparability device:
- Each cycle runs OpenEvolve for N_EVOLVE iterations on the SAME fixed train subset.
- No DataManager, no pool refresh/expansion, no batch reclassification.

We still compute the same metrics (ensemble + unified metrics) on fixed val/test
after each cycle, so curves can be overlaid with AL runs using existing plotters.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
WILDS_EXPERIMENT = SCRIPT_DIR.parent / "wilds_experiment"
EXPERIMENTS_ROOT = WILDS_EXPERIMENT / "experiments"

# Make imports consistent with active_loop.py execution model (works without editable install).
for p in (PROJECT_ROOT, WILDS_EXPERIMENT, EXPERIMENTS_ROOT, SCRIPT_DIR):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)


from fixed_splits import build_fixed_splits, load_fixed_splits


def _load_yaml(path: Path) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _parallel_predict(workers, texts: List[str], prompt_template: str, max_parallel: int):
    # Reuse the same helper as AL evaluator to keep behavior consistent.
    return _AL_EVAL._parallel_predict(workers, texts, prompt_template, max_parallel=max_parallel)


def _build_workers(cfg: dict):
    return _AL_EVAL._build_workers(cfg)


def _import_al_evaluator():
    """
    Import wilds_active_learn_approach/evaluator.py by absolute path to avoid
    collisions with wilds_experiment/evaluator.py (same module name).
    """
    import importlib.util

    p = (SCRIPT_DIR / "evaluator.py").resolve()
    spec = importlib.util.spec_from_file_location("wilds_active_learn_evaluator", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_AL_EVAL = _import_al_evaluator()


def _evaluate_fixed_split(
    *,
    cfg: dict,
    fixed,
    split_name: str,
    prompt: str,
) -> Dict[str, Any]:
    """
    Compute traditional + Hard/Anchor metrics on a fixed split.
    Hard/Anchor classification uses the same rule as in active_loop:
      Hard if (incorrect) OR (disagreement_score > uncertainty_threshold)
    """
    from experiments.metrics import compute_metrics, compute_combined_score_unified
    from data_manager import disagreement_score

    texts = fixed.texts_by_split[split_name]
    labels = np.asarray(fixed.labels_by_split[split_name], dtype=np.int64)
    user_ids = np.asarray(fixed.user_ids_by_split[split_name], dtype=np.int64)

    workers = _build_workers(cfg)
    max_parallel = cfg.get("worker_defaults", {}).get("max_parallel", 8)
    predictions, worker_preds = _parallel_predict(workers, list(texts), prompt, max_parallel=max_parallel)

    pred_arr = np.asarray(predictions, dtype=np.int64)
    wp_arr = [np.asarray(wp, dtype=np.int64) for wp in worker_preds]

    metrics = compute_metrics(pred_arr, labels, user_ids, worker_predictions=wp_arr)
    metrics["combined_score"] = compute_combined_score_unified(metrics, is_ensemble=True)

    # Hard/Anchor metrics (for compatibility with existing plots/logs)
    uncertainty_threshold = float(cfg.get("active_learning", {}).get("uncertainty_threshold", 0.0) or 0.0)
    hard_correct = hard_total = 0
    anchor_correct = anchor_total = 0

    for i in range(len(pred_arr)):
        gold = int(labels[i])
        pred = int(pred_arr[i])
        votes = [int(wp[i]) for wp in wp_arr]
        ds = float(disagreement_score(votes))
        is_hard = (pred != gold) or (ds > uncertainty_threshold)
        is_correct = (pred == gold)
        if is_hard:
            hard_total += 1
            if is_correct:
                hard_correct += 1
        else:
            anchor_total += 1
            if is_correct:
                anchor_correct += 1

    metrics["Acc_Hard"] = (hard_correct / hard_total) if hard_total else 0.0
    metrics["Acc_Anchor"] = (anchor_correct / anchor_total) if anchor_total else 0.0
    metrics["n_hard"] = int(hard_total)
    metrics["n_anchor"] = int(anchor_total)
    metrics["n_total"] = int(len(pred_arr))
    return metrics


def _append_active_style_log(results_dir: Path, entry: dict) -> None:
    path = results_dir / "active_loop_log.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = []
    data.append(entry)
    _write_json(path, data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_all_categories.yaml", help="Config YAML (workers + caps)")
    ap.add_argument("--results-dir", default="results_all_categories_plain_evolve_fixedsubsample")
    ap.add_argument("--n-al", type=int, default=8)
    ap.add_argument("--n-evolve", type=int, default=15)
    ap.add_argument(
        "--train-size",
        type=int,
        default=80,
        help="Fixed TRAIN subset size (stratified by category inside the pooled train). Match AL evolve batch (~80).",
    )
    ap.add_argument(
        "--val-size",
        type=int,
        default=225,
        help="Fixed VALIDATION subset size inside the pooled val (default 225 = old capped config). Use 450 with uncapped config.",
    )
    ap.add_argument(
        "--test-size",
        type=int,
        default=225,
        help="Fixed TEST subset size inside the pooled test (default 225). Use 600 with uncapped config (~40 users × 15).",
    )
    ap.add_argument(
        "--manifest-name",
        default="fixed_splits_allcats_v1.json",
        help="JSON manifest filename under results-dir (different name per experiment to avoid overwriting).",
    )
    ap.add_argument("--checkpoint-interval", type=int, default=10, help="Write metrics every K cycles (default 10 -> same as config checkpoint_interval, but per-cycle here).")
    ap.add_argument("--reuse-fixed-splits", action="store_true", help="If set, reuse existing manifest JSON under results dir (--manifest-name).")
    args = ap.parse_args()

    config_path = (SCRIPT_DIR / args.config).resolve()
    cfg = _load_yaml(config_path)

    results_dir = (SCRIPT_DIR / args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = results_dir / args.manifest_name
    sizes = {
        "train": int(args.train_size),
        "validation": int(args.val_size),
        "test": int(args.test_size),
    }
    if args.reuse_fixed_splits and manifest_path.exists():
        fixed = load_fixed_splits(manifest_path, config_path=str(config_path))
    else:
        fixed = build_fixed_splits(config_path=str(config_path), sizes=sizes, out_path=manifest_path)

    # Seed prompt
    prompt_path = cfg.get("prompt_path") or cfg.get("prompt_path", "initial_prompt_all_categories.txt")
    prompt_file = (SCRIPT_DIR / prompt_path).resolve()
    if not prompt_file.exists():
        raise SystemExit(f"Missing seed prompt: {prompt_file}")
    current_prompt = prompt_file.read_text(encoding="utf-8")

    # Prepare OpenEvolve run
    from openevolve.config import load_config as _oe_load_config
    from openevolve import run_evolution

    # OpenEvolve config is "open-evolve config", not our YAML. We load it from the same file
    # path that active_loop uses (config_all_categories.yaml / config.yaml).
    evolve_config = _oe_load_config(str(config_path))
    # Force full run length: no early stopping
    evolve_config.early_stopping_patience = None

    evaluator_path = str((SCRIPT_DIR / "plain_evaluator_fixed.py").resolve())

    # OpenEvolve runs evaluator in-process; pass context via env vars (like AL evaluator does).
    os.environ["WILDS_ACTIVE_LEARN_CONFIG"] = str(config_path)
    os.environ["WILDS_FIXED_SPLITS_PATH"] = str(manifest_path)

    # Baseline metrics (on fixed test)
    base_test = _evaluate_fixed_split(cfg=cfg, fixed=fixed, split_name="test", prompt=current_prompt)
    _write_json(results_dir / "baseline_test_metrics.json", base_test)

    best_val_score = -1.0
    best_val_prompt = current_prompt

    total_cycles = int(args.n_al)
    n_evolve = int(args.n_evolve)

    for al_iter in range(total_cycles):
        out_dir = results_dir / f"al_iter_{al_iter}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "start_prompt.txt").write_text(current_prompt, encoding="utf-8")

        # Fresh DB per "cycle" (mirrors AL structure, but data is fixed)
        db_dir = out_dir / "database"
        evolve_config.database.db_path = str(db_dir)
        oe_output_dir = str(out_dir / "openevolve_output")

        t0 = time.time()
        result = run_evolution(
            initial_program=current_prompt,
            evaluator=evaluator_path,
            config=evolve_config,
            iterations=n_evolve,
            output_dir=oe_output_dir,
            cleanup=False,
        )
        cycle_time = time.time() - t0

        best_code = current_prompt
        best_score = -1.0
        if result:
            if getattr(result, "best_program", None) is not None and hasattr(result.best_program, "code"):
                best_code = result.best_program.code
                best_score = float(result.best_score)
            elif getattr(result, "best_code", None):
                best_code = result.best_code
                best_score = float(result.best_score)

        current_prompt = best_code
        (out_dir / "best_prompt.txt").write_text(current_prompt, encoding="utf-8")

        # Evaluate val/test fixed splits for curves
        val_m = _evaluate_fixed_split(cfg=cfg, fixed=fixed, split_name="validation", prompt=current_prompt)
        test_m = _evaluate_fixed_split(cfg=cfg, fixed=fixed, split_name="test", prompt=current_prompt)

        entry = {
            "al_iter": al_iter,
            "expanded": False,
            "consolidated": False,
            "cycle_time_s": round(cycle_time, 2),
            "evo_best_score": best_score,
            "seed_val_score": float(val_m.get("combined_score", 0.0)),
            "prompt_tokens": int(len(current_prompt) // 4),
            # val_*
            "val_R_global": float(val_m.get("R_global", 0.0)),
            "val_R_worst": float(val_m.get("R_worst", 0.0)),
            "val_mae": float(val_m.get("mae", 0.0)),
            "val_mean_kappa": float(val_m.get("mean_kappa", 0.0)),
            "val_combined_score": float(val_m.get("combined_score", 0.0)),
            "val_Acc_Hard": float(val_m.get("Acc_Hard", 0.0)),
            "val_Acc_Anchor": float(val_m.get("Acc_Anchor", 0.0)),
            "val_n_hard": int(val_m.get("n_hard", 0)),
            "val_n_anchor": int(val_m.get("n_anchor", 0)),
            # test_* (for graphs only)
            "test_R_global": float(test_m.get("R_global", 0.0)),
            "test_R_worst": float(test_m.get("R_worst", 0.0)),
            "test_mae": float(test_m.get("mae", 0.0)),
            "test_mean_kappa": float(test_m.get("mean_kappa", 0.0)),
            "test_combined_score": float(test_m.get("combined_score", 0.0)),
            "test_Acc_Hard": float(test_m.get("Acc_Hard", 0.0)),
            "test_Acc_Anchor": float(test_m.get("Acc_Anchor", 0.0)),
            "test_n_hard": int(test_m.get("n_hard", 0)),
            "test_n_anchor": int(test_m.get("n_anchor", 0)),
        }

        if entry["val_combined_score"] > best_val_score:
            best_val_score = float(entry["val_combined_score"])
            best_val_prompt = current_prompt

        entry["global_best_val_score"] = best_val_score
        _append_active_style_log(results_dir, entry)

        if (al_iter + 1) % int(max(1, args.checkpoint_interval)) == 0:
            (results_dir / "best_val_prompt.txt").write_text(best_val_prompt, encoding="utf-8")
            _write_json(results_dir / "best_val_meta.json", {"best_val_score": best_val_score, "best_val_cycle": al_iter})

    # Final artifacts
    (results_dir / "final_prompt.txt").write_text(current_prompt, encoding="utf-8")
    (results_dir / "best_val_prompt.txt").write_text(best_val_prompt, encoding="utf-8")
    _write_json(results_dir / "best_val_meta.json", {"best_val_score": best_val_score})

    final_test = _evaluate_fixed_split(cfg=cfg, fixed=fixed, split_name="test", prompt=best_val_prompt)
    _write_json(results_dir / "final_test_metrics.json", final_test)

    print("Done.")
    print("Results:", results_dir)


if __name__ == "__main__":
    main()


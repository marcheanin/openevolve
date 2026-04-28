import argparse
import json
import os
import sys
import time
from pathlib import Path
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def run_full_test(
    prompt_path: Path,
    config_path: Path,
    results_dir: Path,
    max_parallel: int = None,
):
    """
    Runs the ensemble evaluation on the FULL test split of all categories
    (bypassing any max_val_users or max_samples_test limits).
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    progress_dir = results_dir / "predict_progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WILDS_PREDICT_DUMP_DIR"] = str(progress_dir.resolve())
    os.environ.setdefault("WILDS_PREDICT_DUMP_ON_FATAL", "1")
    os.environ.setdefault("WILDS_PREDICT_PERIODIC_EVERY", "5000")
    os.environ.setdefault("WILDS_PREDICT_PERIODIC_MIN_SEC", "30")

    # NOTE: Imports are intentionally inside this function so we can print
    # immediate progress before heavy dependencies (sklearn/scipy/openai) load.
    print(f"Loading config from {config_path}", flush=True)
    from active_loop import _evaluate_split_full, _print_holdout_eval_shapes

    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    # Force evaluator to use this config
    os.environ["WILDS_ACTIVE_LEARN_CONFIG"] = str(config_path)

    # Disable all limits for the test split
    if "dataset" not in cfg_dict:
        cfg_dict["dataset"] = {}
    
    cfg_dict["dataset"]["max_samples"] = None
    cfg_dict["dataset"]["max_samples_test"] = None
    cfg_dict["dataset"]["max_val_users"] = None
    cfg_dict["dataset"]["max_reviews_per_user"] = None
    
    # Optionally override max_parallel if we want to run faster
    if max_parallel is not None:
        if "worker_defaults" not in cfg_dict:
            cfg_dict["worker_defaults"] = {}
        cfg_dict["worker_defaults"]["max_parallel"] = max_parallel

    print("\n" + "=" * 60, flush=True)
    print("Full Uncapped Test Evaluation (All Test Users & Reviews)", flush=True)
    print("=" * 60, flush=True)
    _print_holdout_eval_shapes(cfg_dict)

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_text = f.read()

    print(
        f"\nStarting evaluation of prompt: {prompt_path.name}\n"
        f"  Grid checkpoints: {progress_dir} "
        f"(latest + periodic; resume with WILDS_PREDICT_RESUME_PATH=<path to predict_progress_latest.json>)",
        flush=True,
    )
    t0 = time.time()
    
    full_test_metrics = _evaluate_split_full(prompt_text, cfg_dict, split_name="test")
    
    dur = time.time() - t0
    print(f"  [timing] full_test: {dur:.1f}s")
    
    out_path = results_dir / "full_uncapped_test_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_test_metrics, f, indent=2)
        
    print(f"\nFinal Uncapped Test: R_global={full_test_metrics['R_global']:.2%}  R_worst={full_test_metrics['R_worst']:.2%}  combined={full_test_metrics['combined_score']:.4f}")
    print(f"Final Uncapped Test: Acc_Hard={full_test_metrics['Acc_Hard']:.2%}  Acc_Anchor={full_test_metrics['Acc_Anchor']:.2%}")
    print(f"Saved results to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a prompt on the full uncapped WILDS test set.")
    parser.add_argument("--prompt", type=str, required=True, help="Path to the prompt file to evaluate.")
    parser.add_argument("--config", type=str, default="config_all_categories.yaml", help="Path to YAML config.")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory to save full_uncapped_test_metrics.json")
    parser.add_argument("--max-parallel", type=int, default=15, help="Override max_parallel for faster eval.")
    
    args = parser.parse_args()
    
    prompt_path = Path(args.prompt).resolve()
    config_path = Path(args.config).resolve()
    results_dir = Path(args.results_dir).resolve()
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    run_full_test(prompt_path, config_path, results_dir, args.max_parallel)


if __name__ == "__main__":
    main()
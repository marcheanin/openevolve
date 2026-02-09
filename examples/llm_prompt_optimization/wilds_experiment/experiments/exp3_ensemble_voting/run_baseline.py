import json
import sys
from datetime import datetime
from pathlib import Path

# Ensure parent directory (with the experiments package) is on sys.path,
# but do not override the current working directory (which contains this evaluator).
EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]  # .../experiments
PROJECT_ROOT = EXPERIMENTS_ROOT.parent                  # .../wilds_experiment
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from experiments.metrics import compute_combined_score_unified  # noqa: E402

import evaluator as exp_evaluator  # noqa: E402

load_experiment_config = getattr(exp_evaluator, "load_experiment_config")
save_results = getattr(exp_evaluator, "save_results")
_evaluate_split = getattr(exp_evaluator, "_evaluate_split")


def main() -> None:
    config = load_experiment_config()
    prompt_path = Path(__file__).resolve().parent / config.get("prompt_path", "initial_prompt.txt")
    prompt_template = prompt_path.read_text(encoding="utf-8")

    dataset_cfg = config.get("dataset", {})
    split_name = dataset_cfg.get("split", "test")
    max_samples = dataset_cfg.get("max_samples_test", dataset_cfg.get("max_samples"))

    metrics, predictions, gold_labels, user_ids, worker_predictions = _evaluate_split(
        split_name,
        prompt_template,
        config,
        max_samples,
    )
    metrics["combined_score"] = compute_combined_score_unified(metrics, is_ensemble=True)

    output = {
        "metrics": metrics,
        "predictions": predictions,
        "gold_labels": gold_labels,
        "user_ids": user_ids,
        "worker_predictions": worker_predictions,
        "metadata": {
            "timestamp": datetime.utcnow().isoformat(),
            "prompt_path": str(prompt_path),
            "dataset_config": dataset_cfg,
            "workers": config.get("workers", []),
            "worker_defaults": config.get("worker_defaults", {}),
        },
    }

    output_dir = Path(__file__).resolve().parent / config.get("output_dir", "results")
    save_results(output, str(output_dir))

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(output["metrics"], f, ensure_ascii=False, indent=2)

    print("\nEnsemble baseline evaluation complete.")
    print(json.dumps(output["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()



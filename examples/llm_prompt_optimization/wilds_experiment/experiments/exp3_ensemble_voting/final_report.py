import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from evaluator import _evaluate_split, load_experiment_config
from experiments.metrics import compute_combined_score_unified


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final test report for exp3.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="openevolve_output/best/best_program.txt",
        help="Path to prompt file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="openevolve_output/final_report.json",
        help="Output report path",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit test samples for quick run",
    )
    args = parser.parse_args()

    config = load_experiment_config()
    prompt_path = Path(__file__).resolve().parent / args.prompt
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_path}")

    prompt_template = prompt_path.read_text(encoding="utf-8")

    dataset_cfg = config.get("dataset", {})
    max_samples_test = (
        args.max_samples
        if args.max_samples is not None
        else dataset_cfg.get("max_samples_test", dataset_cfg.get("max_samples"))
    )

    test_metrics, test_preds, test_gold, test_users, worker_predictions = _evaluate_split(
        "test",
        prompt_template,
        config,
        max_samples_test,
    )
    test_metrics["combined_score"] = compute_combined_score_unified(test_metrics, is_ensemble=True)

    report: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat(),
        "prompt_path": str(prompt_path),
        "workers": config.get("workers", []),
        "worker_defaults": config.get("worker_defaults", {}),
        "test_metrics": test_metrics,
        "num_predictions": len(test_preds),
    }

    output_path = Path(__file__).resolve().parent / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved final report to {output_path}")


if __name__ == "__main__":
    main()



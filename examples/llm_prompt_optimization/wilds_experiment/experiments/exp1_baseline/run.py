import importlib
import json
from datetime import datetime
from pathlib import Path

exp_evaluator = importlib.import_module("evaluator")
evaluate = getattr(exp_evaluator, "evaluate")
load_experiment_config = getattr(exp_evaluator, "load_experiment_config")
save_results = getattr(exp_evaluator, "save_results")


def main() -> None:
    config = load_experiment_config()
    output = evaluate()

    output["metadata"] = {
        "timestamp": datetime.utcnow().isoformat(),
        "prompt_path": config.get("prompt_path"),
        "dataset_config": config.get("dataset", {}),
        "model": config.get("model", {}),
    }

    output_dir = Path(__file__).resolve().parent / config.get("output_dir", "results")
    save_results(output, str(output_dir))

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(output["metrics"], f, ensure_ascii=False, indent=2)

    print("\nBaseline evaluation complete.")
    print(json.dumps(output["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


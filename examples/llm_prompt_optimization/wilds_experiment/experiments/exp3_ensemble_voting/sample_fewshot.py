"""
Sample 2 reviews from the training set (Home and Kitchen) for few-shot prompt.
Uses WILDS dataset directly so it does not require OPENAI_API_KEY or evaluator.

Usage (from exp3_ensemble_voting directory):
  python sample_fewshot.py

Or from wilds_experiment with data in ./data:
  python -c "
  import sys
  sys.path.insert(0, 'experiments')
  import os
  os.chdir('experiments/exp3_ensemble_voting')
  exec(open('sample_fewshot.py').read())
  "
"""
import sys
from pathlib import Path

import yaml

# Paths: exp3_ensemble_voting/sample_fewshot.py -> wilds_experiment/
SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent  # wilds_experiment
DATA_ROOT = PROJECT_ROOT / "data"


def load_dataset_config():
    with open(SCRIPT_DIR / "dataset.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_dataset_config()
    category_id = cfg.get("category_id", 24)
    data_path = str(DATA_ROOT)
    if not DATA_ROOT.exists():
        print(f"Data root not found: {DATA_ROOT}", file=sys.stderr)
        print("Run from wilds_experiment or set data path. Using placeholder examples.", file=sys.stderr)
        return

    try:
        from wilds import get_dataset
    except ImportError:
        print("wilds not installed. Using placeholder examples.", file=sys.stderr)
        return

    print("Loading WILDS Amazon dataset (this may be slow on first run)...", flush=True)
    dataset = get_dataset(dataset="amazon", download=False, root_dir=data_path)
    train_data = dataset.get_subset("train")
    metadata_fields = dataset.metadata_fields
    user_idx = metadata_fields.index("user")
    category_idx = metadata_fields.index("category")
    metadata_array = train_data.metadata_array
    y_array = train_data.y_array

    category_mask = metadata_array[:, category_idx] == category_id
    indices = [i for i in range(len(metadata_array)) if category_mask[i]]
    if len(indices) < 2:
        print("Not enough train examples in category.", file=sys.stderr)
        return

    # Labels in WILDS are 0-4 -> we use 1-5
    low_idx = None
    high_idx = None
    for i in indices:
        y = int(y_array[i].item() if hasattr(y_array[i], "item") else y_array[i]) + 1
        if y <= 2 and low_idx is None:
            low_idx = i
        if y >= 4 and high_idx is None:
            high_idx = i
        if low_idx is not None and high_idx is not None:
            break
    if low_idx is None:
        for i in indices:
            y = int(y_array[i].item() if hasattr(y_array[i], "item") else y_array[i]) + 1
            if y == 1:
                low_idx = i
                break
        low_idx = low_idx if low_idx is not None else indices[0]
    if high_idx is None:
        for i in indices:
            y = int(y_array[i].item() if hasattr(y_array[i], "item") else y_array[i]) + 1
            if y == 5:
                high_idx = i
                break
        high_idx = high_idx if high_idx is not None else indices[1]

    low_text = train_data[low_idx][0].strip()
    low_label = int(y_array[low_idx].item() if hasattr(y_array[low_idx], "item") else y_array[low_idx]) + 1
    high_text = train_data[high_idx][0].strip()
    high_label = int(y_array[high_idx].item() if hasattr(y_array[high_idx], "item") else y_array[high_idx]) + 1

    print("Example 1 (low rating):")
    print(f"  Stars: {low_label}")
    print(f"  Text: {low_text[:400]}{'...' if len(low_text) > 400 else ''}")
    print()
    print("Example 2 (high rating):")
    print(f"  Stars: {high_label}")
    print(f"  Text: {high_text[:400]}{'...' if len(high_text) > 400 else ''}")
    print()
    print("--- Copy into initial_prompt.txt (few-shot block) ---")
    print(f"Example 1:\nReview: {low_text}\nStar Rating: {low_label}\n")
    print(f"Example 2:\nReview: {high_text}\nStar Rating: {high_label}")


if __name__ == "__main__":
    main()

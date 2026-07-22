"""
Logging utilities for Active Prompt Evolution.

Tracks per AL iteration: Acc_Hard, Acc_Anchor, R_global, R_worst, MAE, etc.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent


def log_iteration(
    log_path: Path,
    al_iter: int,
    metrics: Dict[str, Any],
    append: bool = True,
) -> None:
    """
    Append a log entry for one AL iteration.

    Args:
        log_path: Path to JSON log file
        al_iter: AL iteration number
        metrics: Dict with keys like R_global, R_worst, mae, mean_kappa,
                 Acc_Hard, Acc_Anchor, n_hard, n_anchor, val_combined_score
        append: If True, append to existing log; else overwrite
    """
    entry = {"al_iter": al_iter, **metrics}
    entries: List[Dict] = []
    if append and log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        # Replace or append
        found = False
        for i, e in enumerate(entries):
            if e.get("al_iter") == al_iter:
                entries[i] = entry
                found = True
                break
        if not found:
            entries.append(entry)
    else:
        entries = [entry]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def load_log(log_path: Path) -> List[Dict[str, Any]]:
    """Load log entries from JSON file."""
    if not log_path.exists():
        return []
    with open(log_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_metric_series(
    log_path: Path,
    metric: str,
) -> List[float]:
    """Extract a metric series across iterations."""
    entries = load_log(log_path)
    entries.sort(key=lambda e: e.get("al_iter", 0))
    return [e.get(metric, 0) for e in entries]

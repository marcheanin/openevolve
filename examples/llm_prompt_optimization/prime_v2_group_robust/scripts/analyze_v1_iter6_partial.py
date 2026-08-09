"""Recover partial full-test metrics for v1's abandoned al_iter_6 evaluation.

Context: `results_all_categories_evolve_subsample` reported its headline on
al_iter_5 (full_uncapped_test_metrics.json, R_global 0.7541), while the
by-protocol selection (best val combined, cycle 6) had its full-test run
started and abandoned at 55k/103.6k worker cells. This script rebuilds the
v1 custom test split (same code path / seed), verifies alignment against the
completed iter-5 grid, then compares iter-5 vs iter-6 paired on the subset of
examples where all three iter-6 worker predictions exist.

Usage:  python analyze_v1_iter6_partial.py
Output: JSON report to stdout and analysis file next to the run dir.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
LLM_OPT = SCRIPT_DIR.parent.parent  # llm_prompt_optimization/
V1_DIR = LLM_OPT / "wilds_active_learn_approach"
WILDS_EXPERIMENT = LLM_OPT / "wilds_experiment"
RUN_DIR = V1_DIR / "results_all_categories_evolve_subsample"

GRID5 = RUN_DIR / "fulltest_al_iter_5" / "predict_progress" / "predict_progress_complete.json"
GRID6 = RUN_DIR / "fulltest_al_iter_6" / "predict_progress" / "predict_progress_latest.json"
METRICS5 = RUN_DIR / "fulltest_al_iter_5" / "full_uncapped_test_metrics.json"


def import_base_evaluator():
    spec = importlib.util.spec_from_file_location(
        "wilds_base_evaluator", WILDS_EXPERIMENT / "evaluator.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def majority_vote(votes):
    # v1 MajorityVoteAggregator: Counter insertion order breaks ties
    return Counter(votes).most_common(1)[0][0]


def r_worst_p10(correct: np.ndarray, user_ids: np.ndarray) -> float:
    accs = [float(np.mean(correct[user_ids == u])) for u in np.unique(user_ids)]
    return float(np.percentile(accs, 10))


def paired_user_bootstrap(delta_by_user: dict, n_boot: int = 10000, seed: int = 0):
    rng = np.random.RandomState(seed)
    deltas = np.array(list(delta_by_user.values()))
    n = len(deltas)
    means = np.array([deltas[rng.randint(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return float(deltas.mean()), (float(lo), float(hi)), float(min(1.0, p))


def main() -> None:
    base = import_base_evaluator()
    dataset_cfg = yaml.safe_load((V1_DIR / "dataset_all_categories.yaml").read_text(encoding="utf-8"))
    preprocessed = base.load_preprocessed_data(dataset_cfg)
    assert preprocessed is not None, "preprocessed cache catall_v2 not found"

    splits = base.create_splits_from_preprocessed(
        preprocessed,
        train_ratio=dataset_cfg.get("train_ratio", 0.7),
        val_ratio=dataset_cfg.get("validation_ratio", 0.15),
        test_ratio=dataset_cfg.get("test_ratio", 0.15),
        seed=int(dataset_cfg.get("split_seed", 42)),
    )
    idx = list(splits["test"]["indices"])
    gold = np.array([int(preprocessed["labels"][i]) for i in idx])
    users = np.array([int(preprocessed["user_ids"][i]) for i in idx])
    n = len(gold)
    print(f"test split rebuilt: {n} examples, {len(np.unique(users))} users", file=sys.stderr)

    g5 = json.loads(GRID5.read_text(encoding="utf-8"))
    g6 = json.loads(GRID6.read_text(encoding="utf-8"))
    assert g5["n_texts"] == n and g6["n_texts"] == n

    grid5 = [np.array([v if v is not None else -1 for v in row]) for row in g5["grid"]]
    grid6 = [np.array([v if v is not None else -1 for v in row]) for row in g6["grid"]]

    # --- verify alignment: recompute iter5 headline ---
    ens5 = np.array([majority_vote([grid5[w][i] for w in range(3)]) for i in range(n)])
    correct5 = ens5 == gold
    ref = json.loads(METRICS5.read_text(encoding="utf-8"))
    r5 = float(np.mean(correct5))
    rw5 = r_worst_p10(correct5, users)
    print(f"iter5 recomputed: R_global {r5:.6f} (ref {ref['R_global']:.6f}), "
          f"R_worst {rw5:.6f} (ref {ref['R_worst']:.6f})", file=sys.stderr)
    aligned = abs(r5 - ref["R_global"]) < 5e-3

    # --- iter6 partial subset: all 3 workers answered ---
    mask = (grid6[0] >= 0) & (grid6[1] >= 0) & (grid6[2] >= 0)
    m = int(mask.sum())
    ens6 = np.array([
        majority_vote([grid6[w][i] for w in range(3)]) if mask[i] else -1 for i in range(n)
    ])
    correct6 = ens6 == gold

    sub = np.where(mask)[0]
    acc5_sub = float(np.mean(correct5[sub]))
    acc6_sub = float(np.mean(correct6[sub]))

    # paired stats on the common subset
    u_sub = users[sub]
    delta_by_user = {}
    for u in np.unique(u_sub):
        um = u_sub == u
        delta_by_user[int(u)] = float(np.mean(correct6[sub][um]) - np.mean(correct5[sub][um]))
    mean_d, ci, p = paired_user_bootstrap(delta_by_user)

    b01 = int(np.sum(correct5[sub] & ~correct6[sub]))  # iter5-only correct
    b10 = int(np.sum(~correct5[sub] & correct6[sub]))  # iter6-only correct

    # R_worst on fully-covered users only
    full_users = [u for u in np.unique(u_sub) if np.sum(u_sub == u) == np.sum(users == u)]
    fu_mask = np.isin(users, full_users) & mask
    rw5_sub = r_worst_p10(correct5[fu_mask], users[fu_mask]) if len(full_users) else None
    rw6_sub = r_worst_p10(correct6[fu_mask], users[fu_mask]) if len(full_users) else None

    per_class = {}
    for c in range(1, 6):
        cm = gold[sub] == c
        if cm.sum():
            per_class[c] = {
                "n": int(cm.sum()),
                "acc_iter5": float(np.mean(correct5[sub][cm])),
                "acc_iter6": float(np.mean(correct6[sub][cm])),
            }

    report = {
        "alignment_check_passed": bool(aligned),
        "iter5_recomputed": {"R_global": r5, "R_worst": rw5},
        "iter5_reference": {"R_global": ref["R_global"], "R_worst": ref["R_worst"]},
        "subset": {
            "n_examples": m,
            "n_users": int(len(np.unique(u_sub))),
            "n_fully_covered_users": len(full_users),
        },
        "paired_on_subset": {
            "acc_iter5": acc5_sub,
            "acc_iter6": acc6_sub,
            "delta_iter6_minus_iter5": acc6_sub - acc5_sub,
            "user_bootstrap": {"mean_delta": mean_d, "ci95": ci, "p": p},
            "mcnemar_counts": {"iter5_only_correct": b01, "iter6_only_correct": b10},
        },
        "r_worst_fully_covered_users": {"iter5": rw5_sub, "iter6": rw6_sub},
        "per_class_on_subset": per_class,
    }
    out = RUN_DIR / "fulltest_al_iter_6" / "partial_metrics_recovered.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nsaved -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()

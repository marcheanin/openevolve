#!/usr/bin/env python
"""Pre-flight: does gemma-3 respond differently to six deliberate prompts? (ROADMAP §3.3)

Gate: spread of worst-GBA ≥ 0.05 and toxic_recall ≥ 0.10 on D_dev.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROMPTS = {
    "seed": None,  # filled from file
    "strict_identity": (
        "You are a strict toxicity classifier. Flag as toxic (1) any insult, "
        "threat, or demeaning remark about a person or identity group. "
        "Benign identity mentions stay 0. Output only 'Label: 0' or 'Label: 1'.\n"
        "Comment: {review}"
    ),
    "conservative": (
        "Mark toxic (1) ONLY for clear abusive attacks. Prefer 0 when unsure. "
        "Output Label: 0 or Label: 1.\nComment: {review}"
    ),
    "one_liner": "Toxic=1 else 0. Comment: {review}\nAnswer:",
    "identity_aware": (
        "Pay special attention to race, religion, gender, LGBTQ attacks. "
        "Identity terms alone are not toxic. Label: 0 or 1.\nComment: {review}"
    ),
    "fewshot": (
        "Examples:\n"
        "Comment: Everyone deserves equal rights.\nLabel: 0\n"
        "Comment: Wipe those people out.\nLabel: 1\n"
        "Now classify. Output Label: 0 or Label: 1.\nComment: {review}"
    ),
}


def _to_template(prompt: str) -> str:
    """Workers format with ``{review}`` (see ensemble.LLMWorker.predict)."""
    p = prompt.replace("{text}", "{review}")
    if "{review}" not in p:
        p = p.rstrip() + "\nComment: {review}"
    return p


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=0, help="0 = full D_dev (900)")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/E5_civilcomments/config_prime_main.yaml",
    )
    parser.add_argument(
        "--fixed-dir",
        type=Path,
        default=ROOT / "experiments/E5_civilcomments/fixed_sets",
    )
    args = parser.parse_args()

    from prime.config import load_config
    from prime.data.civilcomments_loader import load_civilcomments_splits
    from prime.data.fixed_sets import FixedSet, build_d_dev, materialize_split
    from prime.fitness.metrics import compute_metrics
    from prime.workers.ensemble import build_workers, mock_predict, parallel_predict

    seed_path = ROOT / "prompts" / "initial_prompt_civilcomments.txt"
    PROMPTS["seed"] = seed_path.read_text(encoding="utf-8")

    cfg = load_config(args.config)
    cfg.dataset.max_val_users = max(int(cfg.dataset.max_val_users or 0), 50_000)
    splits = load_civilcomments_splits(cfg.dataset, seed=cfg.active_learning.seed)
    val = splits["validation"]
    d_dev_path = args.fixed_dir / "d_dev.json"
    fs = FixedSet.load(d_dev_path) if d_dev_path.is_file() else build_d_dev(val, seed=42)
    mat = materialize_split(val, fs.indices)
    if args.n and args.n < len(mat["texts"]):
        rng = np.random.RandomState(0)
        pick = sorted(rng.choice(len(mat["texts"]), size=args.n, replace=False).tolist())
        mat = {k: ([v[i] for i in pick] if isinstance(v, list) else v) for k, v in mat.items()}

    y = np.asarray(mat["labels"], dtype=np.int16)
    cids = np.asarray(mat["cluster_ids"], dtype=np.int16)
    uids = np.asarray(mat["user_ids"])
    results: Dict[str, Any] = {}

    for name, raw in PROMPTS.items():
        prompt = _to_template(str(raw))
        print(f"=== sensitivity: {name} n={len(mat['texts'])} ===", flush=True)
        if args.mock:
            ens, _ = mock_predict(mat["texts"], mat["labels"], 1, 0, label_space="binary")
        else:
            workers = build_workers(cfg.ensemble)
            ens, _ = parallel_predict(
                workers,
                mat["texts"],
                prompt,
                max_parallel=cfg.ensemble.max_parallel,
                aggregation=cfg.ensemble.aggregation,
                label_space="binary",
                fail_closed=True,
            )
        m = compute_metrics(
            np.asarray(ens),
            y,
            uids,
            cluster_ids=cids,
            class_balanced=True,
            gba_min_pos=10,
            gba_min_neg=10,
        )
        results[name] = {
            "R_worst_gba": m.get("R_worst_gba"),
            "R_gba_mean": m.get("R_gba_mean"),
            "toxic_recall": m.get("toxic_recall"),
            "specificity": m.get("specificity"),
            "R_global": m.get("R_global"),
            "pred_pos_rate": m.get("pred_pos_rate"),
            "invalid_rate": m.get("invalid_rate"),
        }
        print(json.dumps({name: results[name]}, indent=2), flush=True)

    worst = [float(v["R_worst_gba"] or 0) for v in results.values()]
    recall = [float(v["toxic_recall"] or 0) for v in results.values()]
    spread_w = max(worst) - min(worst)
    spread_r = max(recall) - min(recall)
    gate_ok = spread_w >= 0.05 and spread_r >= 0.10
    out = {
        "status": "ok" if gate_ok else "fail_gate",
        "n": len(mat["texts"]),
        "d_dev_fingerprint": fs.fingerprint,
        "results": results,
        "spread_worst_gba": spread_w,
        "spread_toxic_recall": spread_r,
        "gate_worst_gba_spread": 0.05,
        "gate_recall_spread": 0.10,
        "gate_passed": gate_ok,
    }
    out_path = args.out or (ROOT / "experiments/E5_civilcomments/preflight_sensitivity.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Wrote", out_path, "gate_passed=", gate_ok)
    return 0 if gate_ok else 3


if __name__ == "__main__":
    raise SystemExit(main())

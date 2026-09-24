#!/usr/bin/env python
"""E6 F9: selection regret uniform d_dev vs d_dev_targeted (same candidate pool)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


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
    ap.add_argument(
        "--pool",
        type=Path,
        default=ROOT / "experiments/E6_amazon_category_controls/pools/harshness_sweep",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results/E6_amazon_category_controls/f9_realloc",
    )
    ap.add_argument("--rule", default="hard_min")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    from e6_metrics import all_metrics
    from e6_score_prompt import load_mat
    from prime.config import load_config
    from prime.workers.ensemble import load_dotenv_if_present
    from prime.workers.scorer import Scorer

    load_dotenv_if_present()
    cfg = load_config(args.config)
    scorer = Scorer(cfg.ensemble, label_space="ordinal5", use_mock=args.mock)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    preds_dir = out / "preds"
    preds_dir.mkdir(exist_ok=True)

    man = json.loads((args.pool / "manifest.json").read_text(encoding="utf-8"))
    entries = [
        (c["name"], args.pool / c["file"]) for c in sorted(man["candidates"], key=lambda x: x["rank"])
    ]
    mats = {
        "d_dev": load_mat(args.fixed_dir, "d_dev"),
        "d_dev_targeted": load_mat(args.fixed_dir, "d_dev_targeted"),
        "test_fixed": load_mat(args.fixed_dir, "test_fixed"),
    }

    def score(name: str, prompt: str, set_name: str) -> dict:
        mat = mats[set_name]
        cache = preds_dir / f"{name}__{set_name}.npy"
        texts, y = mat["texts"], np.asarray(mat["labels"])
        c = np.asarray(mat["cluster_ids"])
        if cache.is_file() and len(np.load(cache)) == len(texts):
            preds = np.load(cache)
        else:
            print(f"[e6f9] {name} {set_name} n={len(texts)}", flush=True)
            preds = np.asarray(
                scorer.predict_batch(texts, prompt, labels_for_mock=list(y)).preds,
                dtype=np.int16,
            )
            np.save(cache, preds)
        return all_metrics(preds, y, c)

    rows = []
    for name, path in entries:
        prompt = path.read_text(encoding="utf-8")
        row = {"name": name}
        for sn in mats:
            row[sn] = score(name, prompt, sn)
        rows.append(row)
        print(
            f"[e6f9] {name:18} uni={row['d_dev'][args.rule]:.4f} "
            f"tgt={row['d_dev_targeted'][args.rule]:.4f} "
            f"test_cvar={row['test_fixed']['cvar25']:.4f}",
            flush=True,
        )

    oracle = max(rows, key=lambda r: r["test_fixed"]["cvar25"])
    pick_u = max(rows, key=lambda r: r["d_dev"][args.rule])
    pick_t = max(rows, key=lambda r: r["d_dev_targeted"][args.rule])
    regret_u = float(oracle["test_fixed"]["cvar25"] - pick_u["test_fixed"]["cvar25"])
    regret_t = float(oracle["test_fixed"]["cvar25"] - pick_t["test_fixed"]["cvar25"])
    report = {
        "rule": args.rule,
        "oracle": oracle["name"],
        "oracle_test_cvar25": oracle["test_fixed"]["cvar25"],
        "pick_uniform": pick_u["name"],
        "pick_targeted": pick_t["name"],
        "regret_uniform": regret_u,
        "regret_targeted": regret_t,
        "delta_regret": regret_u - regret_t,
        "rows": rows,
    }
    (out / "f9_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""E6 Phase A: score harshness pool + existing optimizer finals; selection shootout."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

OPTIMIZER_PROMPTS = [
    ("seed", ROOT / "prompts/initial_prompt.txt"),
    (
        "e2_category_heir",
        ROOT
        / "results/E2_category_shift_books/seed42_20260802_052137/al_iter_2/best_prompt.txt",
    ),
    (
        "e1_global_heir",
        ROOT / "results/E1_constraint_global/seed42_20260801_002853/al_iter_3/best_prompt.txt",
    ),
]


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
        default=ROOT / "results/E6_amazon_category_controls/selection_control",
    )
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--seed-repeats", type=int, default=3)
    args = ap.parse_args()

    from e6_metrics import all_metrics, spearman
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
    entries: list[tuple[str, Path]] = [
        (c["name"], args.pool / c["file"]) for c in sorted(man["candidates"], key=lambda x: x["rank"])
    ]
    for name, path in OPTIMIZER_PROMPTS:
        if path.is_file():
            entries.append((f"opt:{name}", path))

    mats = {s: load_mat(args.fixed_dir, s) for s in ("d_dev", "test_fixed")}
    rows = []

    def score_one(name: str, prompt: str, set_name: str, repeats: int = 1) -> np.ndarray:
        mat = mats[set_name]
        texts, y = mat["texts"], mat["labels"]
        reps = []
        for r in range(repeats):
            cache = preds_dir / f"{name.replace(':', '__')}__{set_name}__r{r}.npy"
            if cache.is_file() and len(np.load(cache)) == len(texts):
                preds = np.load(cache)
            else:
                print(f"[e6sel] {name} {set_name} r{r} n={len(texts)}", flush=True)
                preds = np.asarray(
                    scorer.predict_batch(texts, prompt, labels_for_mock=y).preds, dtype=np.int16
                )
                np.save(cache, preds)
            reps.append(preds)
        if len(reps) == 1:
            return reps[0]
        stacked = np.stack(reps)
        return np.apply_along_axis(
            lambda col: int(np.bincount(col.astype(int), minlength=6)[1:].argmax() + 1),
            0,
            stacked,
        ).astype(np.int16)

    for name, path in entries:
        prompt = path.read_text(encoding="utf-8")
        reps = args.seed_repeats if name in ("seed_anchor", "opt:seed") else 1
        pd_ = score_one(name, prompt, "d_dev", 1)
        pt_ = score_one(name, prompt, "test_fixed", reps)
        yd = np.asarray(mats["d_dev"]["labels"])
        cd = np.asarray(mats["d_dev"]["cluster_ids"])
        yt = np.asarray(mats["test_fixed"]["labels"])
        ct = np.asarray(mats["test_fixed"]["cluster_ids"])
        md = all_metrics(pd_, yd, cd)
        mt = all_metrics(pt_, yt, ct)
        rows.append({"name": name, "dev": md, "test": mt, "prompt_path": str(path)})
        print(
            f"[e6sel] {name:22} dev_cvar25={md['cvar25']:.4f} "
            f"test_cvar25={mt['cvar25']:.4f} global={mt['R_global']:.4f}",
            flush=True,
        )
        (out / "rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    rules = ["mean_macro", "worst_class", "cvar25", "hard_min", "softmin_shrunk", "R_tail", "R_global"]
    picks = {}
    print("\n=== Selection rules (pick on d_dev, report test) ===")
    for rule in rules:
        best = max(rows, key=lambda r: r["dev"][rule])
        picks[rule] = {
            "picked": best["name"],
            "dev": best["dev"][rule],
            "test_cvar25": best["test"]["cvar25"],
            "test_global": best["test"]["R_global"],
        }
        print(
            f"  {rule:16} -> {best['name']:22} "
            f"test_cvar25={best['test']['cvar25']:.4f} global={best['test']['R_global']:.4f}"
        )

    # Spearman: each rule's ranking vs test cvar25
    names = [r["name"] for r in rows]
    test_c = [r["test"]["cvar25"] for r in rows]
    corr = {}
    for rule in rules:
        corr[rule] = spearman([r["dev"][rule] for r in rows], test_c)

    report = {
        "n_candidates": len(rows),
        "rows": rows,
        "picks": picks,
        "spearman_dev_vs_test_cvar25": corr,
        "oracle_test": max(rows, key=lambda r: r["test"]["cvar25"])["name"],
    }
    (out / "selection_control_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out / 'selection_control_report.json'}")
    print("spearman:", {k: round(v, 3) for k, v in corr.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

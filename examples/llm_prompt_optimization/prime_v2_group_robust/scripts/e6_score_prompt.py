#!/usr/bin/env python
"""Score prompts on E6 fixed sets (d_dev / test_fixed) with caching."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def load_mat(fixed_dir: Path, name: str) -> dict:
    p = fixed_dir / f"{name}_materialized.json"
    if not p.is_file():
        raise SystemExit(f"missing {p} — run build_e6_amazon_fixed_sets.py first")
    return json.loads(p.read_text(encoding="utf-8"))


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
    ap.add_argument("--prompt", type=Path, required=True)
    ap.add_argument("--name", type=str, default=None, help="cache key (default: prompt stem)")
    ap.add_argument("--sets", default="d_dev,test_fixed")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    from e6_metrics import all_metrics
    from prime.config import load_config
    from prime.workers.ensemble import load_dotenv_if_present
    from prime.workers.scorer import Scorer

    load_dotenv_if_present()
    cfg = load_config(args.config)
    scorer = Scorer(cfg.ensemble, label_space="ordinal5", use_mock=args.mock)
    prompt = Path(args.prompt).read_text(encoding="utf-8")
    name = args.name or Path(args.prompt).stem
    out = args.out_dir or (ROOT / "results/E6_amazon_category_controls/scores")
    out.mkdir(parents=True, exist_ok=True)

    report = {"name": name, "prompt_path": str(args.prompt), "sets": {}}
    for set_name in [s.strip() for s in args.sets.split(",") if s.strip()]:
        mat = load_mat(args.fixed_dir, set_name)
        y = np.asarray(mat["labels"], dtype=np.int16)
        c = np.asarray(mat["cluster_ids"], dtype=np.int16)
        u = np.asarray(mat["user_ids"])
        texts = mat["texts"]
        reps = []
        for r in range(args.repeats):
            cache = out / f"{name}__{set_name}__r{r}_preds.npy"
            if cache.is_file() and len(np.load(cache)) == len(texts):
                preds = np.load(cache)
                print(f"[e6score] {name} {set_name} r{r} cached", flush=True)
            else:
                print(f"[e6score] {name} {set_name} r{r} scoring n={len(texts)}…", flush=True)
                preds = np.asarray(
                    scorer.predict_batch(texts, prompt, labels_for_mock=list(y)).preds,
                    dtype=np.int16,
                )
                np.save(cache, preds)
            reps.append(preds)
            m = all_metrics(preds, y, c, np.arange(len(y)))
            print(
                f"         cvar25={m['cvar25']:.4f} global={m['R_global']:.4f} "
                f"tail={m['R_tail']:.4f} op={m['op_shift']:.3f}",
                flush=True,
            )
        maj = (np.stack(reps).mean(axis=0) >= 0.5).astype(np.int16) if False else reps[0]
        # ordinal majority: mode across repeats
        if len(reps) > 1:
            stacked = np.stack(reps)
            maj = np.apply_along_axis(
                lambda col: int(np.bincount(col.astype(int), minlength=6)[1:].argmax() + 1),
                0,
                stacked,
            ).astype(np.int16)
        else:
            maj = reps[0]
        report["sets"][set_name] = {
            "n": int(len(y)),
            "fingerprint": mat.get("fingerprint"),
            "metrics": all_metrics(maj, y, c, np.arange(len(y))),
            "repeat_metrics": [all_metrics(p, y, c, np.arange(len(y))) for p in reps],
        }
    (out / f"{name}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out / f'{name}_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

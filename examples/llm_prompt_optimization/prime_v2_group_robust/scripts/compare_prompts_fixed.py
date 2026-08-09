#!/usr/bin/env python
"""Same-session rescore of prompts on D_dev + test_fixed with bootstrap / McNemar."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _materialize(cfg, fixed_name: str, fixed_dir: Path, seed: int) -> Dict[str, Any]:
    from prime.data.civilcomments_loader import load_civilcomments_splits
    from prime.data.fixed_sets import FixedSet, build_d_dev, build_test_fixed, materialize_split

    path = fixed_dir / f"{fixed_name}.json"
    splits = load_civilcomments_splits(cfg.dataset, seed=seed)
    if fixed_name == "d_dev":
        split = splits["validation"]
        fs = FixedSet.load(path) if path.is_file() else build_d_dev(split, seed=seed)
    elif fixed_name == "test_fixed":
        split = splits["test"]
        fs = FixedSet.load(path) if path.is_file() else build_test_fixed(split, seed=seed)
    else:
        raise ValueError(fixed_name)
    if not path.is_file():
        fs.save(path)
    mat = materialize_split(split, fs.indices)
    mat["fingerprint"] = fs.fingerprint
    mat["name"] = fs.name
    return mat


def _score(
    texts: List[str],
    labels: List[int],
    cluster_ids: List[int],
    user_ids: List[Any],
    prompt: str,
    cfg,
    use_mock: bool,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    from prime.fitness.metrics import compute_metrics
    from prime.fitness.objective import soft_min_accuracies
    from prime.workers.ensemble import build_workers, mock_predict, parallel_predict

    if use_mock:
        ens, _wp = mock_predict(
            texts, labels, 1, seed=0, aggregation="majority", label_space="binary"
        )
    else:
        workers = build_workers(cfg.ensemble)
        ens, _wp = parallel_predict(
            workers,
            texts,
            prompt,
            max_parallel=cfg.ensemble.max_parallel,
            tie_break=cfg.ensemble.tie_break,
            aggregation=cfg.ensemble.aggregation,
            label_space="binary",
            fail_closed=True,
        )
    preds = np.asarray(ens, dtype=np.int16)
    y = np.asarray(labels, dtype=np.int16)
    cids = np.asarray(cluster_ids, dtype=np.int16)
    uids = np.asarray(user_ids)
    m = compute_metrics(
        preds,
        y,
        uids,
        cluster_ids=cids,
        shrink_prior_weight=40.0,
        class_balanced=True,
        gba_min_pos=10,
        gba_min_neg=10,
    )
    gba = m.get("cluster_gba_shrunk") or m.get("cluster_gba") or {}
    m["R_soft_min_gba"] = float(
        soft_min_accuracies({int(k): float(v) for k, v in gba.items()}, 0.10)
    ) if gba else float(m.get("R_gba_mean", 0.0))
    return preds, m


def _mcnemar(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    from prime.experiment.anchor_gate import mcnemar_one_sided_p

    a_ok = a == y
    b_ok = b == y
    n_b_only = int(np.sum(~a_ok & b_ok))  # b improved vs a
    n_a_only = int(np.sum(a_ok & ~b_ok))  # a better
    # one-sided: is b worse than a? (worsened for b = a_only)
    p_b_worse = mcnemar_one_sided_p(n_b_only, n_a_only)
    return {
        "a_only_ok": n_a_only,
        "b_only_ok": n_b_only,
        "agreement": float(np.mean(a == b)),
        "mcnemar_p_b_worse_than_a": round(p_b_worse, 6),
        "net_b_minus_a": n_b_only - n_a_only,
    }


def _bootstrap_worst_gba(
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    cids: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, Any]:
    from prime.fitness.metrics import group_balanced_accuracies

    rng = np.random.RandomState(seed)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        ga = group_balanced_accuracies(a[idx], y[idx], cids[idx], min_pos=5, min_neg=5)
        gb = group_balanced_accuracies(b[idx], y[idx], cids[idx], min_pos=5, min_neg=5)
        # Align on intersection of eligible groups
        keys = set(ga) & set(gb)
        if not keys:
            continue
        wa = min(ga[k] for k in keys)
        wb = min(gb[k] for k in keys)
        deltas.append(wb - wa)
    if not deltas:
        return {"n_boot": 0}
    arr = np.asarray(deltas)
    return {
        "n_boot": len(arr),
        "delta_mean": float(np.mean(arr)),
        "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
        "p_positive": float(np.mean(arr > 0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/E5_civilcomments/config_prime_main.yaml")
    parser.add_argument(
        "--fixed-dir",
        type=Path,
        default=ROOT / "experiments/E5_civilcomments/fixed_sets",
    )
    parser.add_argument("--seed-prompt", type=Path, required=True)
    parser.add_argument("--candidate-prompt", type=Path, required=True)
    parser.add_argument("--seed-name", default="seed")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--sets", default="d_dev,test_fixed")
    args = parser.parse_args()

    from prime.config import load_config

    cfg = load_config(args.config)
    cfg.dataset.max_val_users = max(int(cfg.dataset.max_val_users or 0), 50_000)
    cfg.dataset.max_test_users = max(int(cfg.dataset.max_test_users or 0), 80_000)

    seed_p = _load_prompt(args.seed_prompt)
    cand_p = _load_prompt(args.candidate_prompt)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "seed_prompt.txt").write_text(seed_p, encoding="utf-8")
    (out / "candidate_prompt.txt").write_text(cand_p, encoding="utf-8")

    report: Dict[str, Any] = {
        "seed_name": args.seed_name,
        "candidate_name": args.candidate_name,
        "sets": {},
    }

    for set_name in [s.strip() for s in args.sets.split(",") if s.strip()]:
        print(f"=== materialize {set_name} ===", flush=True)
        mat = _materialize(cfg, set_name, args.fixed_dir, seed=cfg.active_learning.seed)
        print(f"=== score {args.seed_name} on {set_name} n={len(mat['texts'])} ===", flush=True)
        pred_a, met_a = _score(
            mat["texts"], mat["labels"], mat["cluster_ids"], mat["user_ids"], seed_p, cfg, args.mock
        )
        print(f"=== score {args.candidate_name} on {set_name} ===", flush=True)
        pred_b, met_b = _score(
            mat["texts"], mat["labels"], mat["cluster_ids"], mat["user_ids"], cand_p, cfg, args.mock
        )
        y = np.asarray(mat["labels"], dtype=np.int16)
        cids = np.asarray(mat["cluster_ids"], dtype=np.int16)
        pair = _mcnemar(y, pred_a, pred_b)
        boot = _bootstrap_worst_gba(y, pred_a, pred_b, cids, n_boot=args.n_boot)
        keys = [
            "R_worst_gba",
            "R_gba_mean",
            "R_soft_min_gba",
            "toxic_recall",
            "specificity",
            "R_global",
            "R_macro",
            "pred_pos_rate",
            "invalid_rate",
        ]
        deltas = {k: float(met_b.get(k, 0) or 0) - float(met_a.get(k, 0) or 0) for k in keys}
        block = {
            "fingerprint": mat["fingerprint"],
            "n": len(y),
            args.seed_name: {k: met_a.get(k) for k in keys},
            args.candidate_name: {k: met_b.get(k) for k in keys},
            "delta_candidate_minus_seed": deltas,
            "paired": pair,
            "bootstrap_delta_worst_gba": boot,
        }
        report["sets"][set_name] = block
        tag = f"{set_name}"
        np.save(out / f"{tag}_{args.seed_name}_preds.npy", pred_a)
        np.save(out / f"{tag}_{args.candidate_name}_preds.npy", pred_b)
        np.save(out / f"{tag}_labels.npy", y)
        np.save(out / f"{tag}_cluster_ids.npy", cids)
        print(json.dumps({set_name: {"deltas": deltas, "paired": pair, "boot": boot}}, indent=2), flush=True)

    (out / "compare_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Markdown summary
    lines = ["# Prompt compare (same-session)", ""]
    for sn, block in report["sets"].items():
        lines.append(f"## {sn} (n={block['n']}, fp={block['fingerprint']})")
        lines.append("")
        lines.append("| metric | seed | candidate | Δ |")
        lines.append("|---|---:|---:|---:|")
        for k, d in block["delta_candidate_minus_seed"].items():
            a = block[args.seed_name].get(k)
            b = block[args.candidate_name].get(k)
            if a is None:
                continue
            lines.append(f"| {k} | {float(a):.4f} | {float(b):.4f} | {d:+.4f} |")
        boot = block["bootstrap_delta_worst_gba"]
        if boot.get("n_boot"):
            lines.append("")
            lines.append(
                f"Bootstrap ΔR_worst_gba mean={boot['delta_mean']:+.4f} "
                f"CI95=[{boot['ci95'][0]:+.4f}, {boot['ci95'][1]:+.4f}] "
                f"P(Δ>0)={boot['p_positive']:.3f}"
            )
        lines.append("")
        lines.append(
            f"McNemar: candidate_only_ok={block['paired']['b_only_ok']} "
            f"seed_only_ok={block['paired']['a_only_ok']} "
            f"agreement={block['paired']['agreement']:.4f}"
        )
        lines.append("")
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", out / "RESULTS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Selection-control harness: does the *selection rule* explain the gains?

Idrissi et al. (CLeaR 2022) and Yang et al. (ICML 2023) report that on
worst-group benchmarks most of the apparent algorithmic progress comes from
selecting on a group-labelled validation set. This harness isolates that: it
takes ONE candidate pool, scores every candidate once on D_dev and once on the
test set, and then lets three selection rules pick a winner from the same pool:

  mean_gba      average group-balanced accuracy   (no group robustness)
  worst_class   worst of {class-0 acc, class-1 acc}  (group-free control)
  cvar25        CVaR@25% over the 8 identity GBAs   (our primary metric)

It also bootstraps D_dev to report how *stable* each rule's pick is, which is
the quantity that actually matters when the dev set is small.

Budget note (POWER_AUDIT F2): repeats are worth ~1/250 of the variance of extra
examples, so the default is a single repeat.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

RULES = ["mean_gba", "worst_class", "cvar25"]


def _load_fixed(cfg, fixed_dir: Path, name: str, split_name: str) -> dict[str, Any]:
    from prime.data.civilcomments_loader import load_civilcomments_splits
    from prime.data.fixed_sets import FixedSet, materialize_split

    path = fixed_dir / f"{name}.json"
    if not path.is_file():
        raise SystemExit(f"missing fixed set {path}")
    fs = FixedSet.load(path)
    splits = load_civilcomments_splits(cfg.dataset, seed=cfg.active_learning.seed)
    mat = materialize_split(splits[split_name], fs.indices)
    mat["fingerprint"] = fs.fingerprint
    mat["name"] = name
    return mat


def _metrics(preds: np.ndarray, y: np.ndarray, cids: np.ndarray) -> dict[str, float]:
    groups = [g for g in sorted(set(cids.tolist())) if g != 0]
    gba = []
    for g in groups:
        m = cids == g
        pos, neg = m & (y == 1), m & (y == 0)
        tpr = float((preds[pos] == 1).mean()) if pos.any() else 0.5
        tnr = float((preds[neg] == 0).mean()) if neg.any() else 0.5
        gba.append(0.5 * (tpr + tnr))
    gba = np.sort(np.asarray(gba))
    k25 = max(1, int(round(0.25 * len(gba))))
    tau = 0.10
    recall = float((preds[y == 1] == 1).mean())
    spec = float((preds[y == 0] == 0).mean())
    return {
        "hard_min": float(gba[0]),
        "cvar25": float(gba[:k25].mean()),
        "softmin": float(-tau * np.log(np.mean(np.exp(-gba / tau)))),
        "mean_gba": float(gba.mean()),
        "worst_class": float(min(recall, spec)),
        "spread_gba": float(gba[-1] - gba[0]),
        "recall": recall,
        "specificity": spec,
        "global_acc": float((preds == y).mean()),
        "pred_pos_rate": float(preds.mean()),
    }


def _score(prompt_text: str, mat: dict, cfg, cache: Path, use_mock: bool) -> np.ndarray:
    if cache.is_file():
        arr = np.load(cache)
        if len(arr) == len(mat["texts"]):
            return arr
    from e5_stable_eval import _score_once

    preds, _ = _score_once(
        mat["texts"],
        mat["labels"],
        mat["cluster_ids"],
        mat["user_ids"],
        prompt_text,
        cfg,
        use_mock=use_mock,
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, preds)
    return preds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pool",
        type=Path,
        default=ROOT / "experiments/E5_civilcomments/pools/strictness_sweep",
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/E5_civilcomments/config_prime_main.yaml",
    )
    ap.add_argument(
        "--fixed-dir",
        type=Path,
        default=ROOT / "experiments/E5_civilcomments/fixed_sets",
    )
    ap.add_argument("--dev-set", default="d_dev")
    ap.add_argument("--test-set", default="test_fixed_large")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    from prime.config import load_config
    from prime.workers.ensemble import load_dotenv_if_present

    load_dotenv_if_present()
    cfg = load_config(args.config)
    cfg.dataset.max_val_users = max(int(cfg.dataset.max_val_users or 0), 50_000)
    cfg.dataset.max_test_users = max(int(cfg.dataset.max_test_users or 0), 80_000)

    pool_manifest = json.loads((args.pool / "manifest.json").read_text(encoding="utf-8"))
    cands = sorted(pool_manifest["candidates"], key=lambda c: c["rank"])
    out = args.out_dir or (ROOT / "results/E5_selection_control" / args.pool.name)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[sel] pool={args.pool.name} candidates={len(cands)}")
    dev = _load_fixed(cfg, args.fixed_dir, args.dev_set, "validation")
    test = _load_fixed(cfg, args.fixed_dir, args.test_set, "test")
    print(f"[sel] dev n={len(dev['texts'])} fp={dev['fingerprint']}")
    print(f"[sel] test n={len(test['texts'])} fp={test['fingerprint']}")

    y_dev = np.asarray(dev["labels"], dtype=np.int16)
    c_dev = np.asarray(dev["cluster_ids"], dtype=np.int16)
    y_test = np.asarray(test["labels"], dtype=np.int16)
    c_test = np.asarray(test["cluster_ids"], dtype=np.int16)

    # PREREGISTRATION_S10: the 1800-example subset informed the metric choice, so
    # every headline number is mirrored on the never-touched complement.
    clean_mask = None
    reuse_path = args.fixed_dir / f"{args.test_set}_reuse_map.json"
    if reuse_path.is_file():
        rm = json.loads(reuse_path.read_text(encoding="utf-8"))
        contaminated = np.asarray(rm["rows_in_large_for_each_small_row"], dtype=int)
        clean_mask = np.ones(len(y_test), dtype=bool)
        clean_mask[contaminated] = False
        print(f"[sel] clean confirmatory subset n={int(clean_mask.sum())}")

    rows: list[dict] = []
    dev_preds: dict[str, np.ndarray] = {}
    for c in cands:
        text = (args.pool / c["file"]).read_text(encoding="utf-8")
        print(f"[sel] {c['rank']:2d} {c['name']:16} dev...", flush=True)
        pd_ = _score(text, dev, cfg, out / f"preds/{c['name']}__dev.npy", args.mock)
        print(f"[sel] {c['rank']:2d} {c['name']:16} test...", flush=True)
        pt_ = _score(text, test, cfg, out / f"preds/{c['name']}__test.npy", args.mock)
        dev_preds[c["name"]] = pd_
        md = _metrics(pd_, y_dev, c_dev)
        mt = _metrics(pt_, y_test, c_test)
        row = {"name": c["name"], "rank": c["rank"], "dev": md, "test": mt}
        if clean_mask is not None:
            row["test_clean"] = _metrics(pt_[clean_mask], y_test[clean_mask], c_test[clean_mask])
        rows.append(row)
        print(
            f"     dev  cvar25={md['cvar25']:.3f} meanGBA={md['mean_gba']:.3f} "
            f"wclass={md['worst_class']:.3f} rec={md['recall']:.3f} spec={md['specificity']:.3f}"
        )
        print(
            f"     test cvar25={mt['cvar25']:.3f} meanGBA={mt['mean_gba']:.3f} "
            f"rec={mt['recall']:.3f} spec={mt['specificity']:.3f}"
        )
        (out / "rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    by_name = {r["name"]: r for r in rows}
    seed_row = next((r for r in rows if r["name"] == "seed_anchor"), None)

    print()
    print("=== Selection rules (pick on dev, report on test) ===")
    picks = {}
    for rule in RULES:
        best = max(rows, key=lambda r: r["dev"][rule])
        picks[rule] = {
            "picked": best["name"],
            "dev_value": best["dev"][rule],
            "test_cvar25": best["test"]["cvar25"],
            "test_hard_min": best["test"]["hard_min"],
            "test_clean_cvar25": (best.get("test_clean") or {}).get("cvar25"),
        }
        clean = picks[rule]["test_clean_cvar25"]
        print(
            f"  {rule:12} -> {best['name']:16} "
            f"dev={best['dev'][rule]:.4f}  test_cvar25={best['test']['cvar25']:.4f}"
            + (f"  clean={clean:.4f}" if clean is not None else "")
        )
    oracle_pick = max(rows, key=lambda r: r["test"]["cvar25"])
    print(
        f"  {'ORACLE(test)':12} -> {oracle_pick['name']:16} "
        f"test_cvar25={oracle_pick['test']['cvar25']:.4f}   <- unreachable upper bound"
    )
    if seed_row:
        print(f"  {'seed anchor':12}    test_cvar25={seed_row['test']['cvar25']:.4f}")

    # Selection stability: resample dev, re-pick, look at the test value obtained.
    print()
    print("=== Selection-rule stability (bootstrap over D_dev) ===")
    rng = np.random.default_rng(0)
    cells = [
        np.where((c_dev == g) & (y_dev == lab))[0]
        for g in sorted(set(c_dev.tolist()))
        for lab in (0, 1)
    ]
    stability: dict[str, Any] = {}
    obtained_by_rule = {r: np.empty(args.n_boot) for r in RULES}
    chosen_by_rule: dict[str, dict[str, int]] = {r: {} for r in RULES}
    for b in range(args.n_boot):
        idx = np.concatenate([rng.choice(v, size=len(v), replace=True) for v in cells])
        yb, cb = y_dev[idx], c_dev[idx]
        boot_m = {r["name"]: _metrics(dev_preds[r["name"]][idx], yb, cb) for r in rows}
        for rule in RULES:
            best_name = max(boot_m, key=lambda n: boot_m[n][rule])
            chosen_by_rule[rule][best_name] = chosen_by_rule[rule].get(best_name, 0) + 1
            obtained_by_rule[rule][b] = by_name[best_name]["test"]["cvar25"]

    for rule in RULES:
        obtained = obtained_by_rule[rule]
        chosen = chosen_by_rule[rule]
        stability[rule] = {
            "mean_test_cvar25": float(obtained.mean()),
            "sd_test_cvar25": float(obtained.std(ddof=1)),
            "ci95": [float(np.percentile(obtained, 2.5)), float(np.percentile(obtained, 97.5))],
            "pick_distribution": dict(sorted(chosen.items(), key=lambda kv: -kv[1])),
        }
        top = list(stability[rule]["pick_distribution"].items())[:3]
        print(
            f"  {rule:12} E[test cvar25]={obtained.mean():.4f} "
            f"(sd {obtained.std(ddof=1):.4f})  top picks: "
            + ", ".join(f"{k}:{v / args.n_boot:.0%}" for k, v in top)
        )

    report = {
        "pool": args.pool.name,
        "dev_set": {"name": args.dev_set, "n": int(len(y_dev)), "fingerprint": dev["fingerprint"]},
        "test_set": {"name": args.test_set, "n": int(len(y_test)), "fingerprint": test["fingerprint"]},
        "repeats": 1,
        "rows": rows,
        "picks": picks,
        "oracle_pick": {"name": oracle_pick["name"], "test_cvar25": oracle_pick["test"]["cvar25"]},
        "seed_anchor_test_cvar25": seed_row["test"]["cvar25"] if seed_row else None,
        "stability": stability,
    }
    (out / "selection_control_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print()
    print(f"wrote {out / 'selection_control_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

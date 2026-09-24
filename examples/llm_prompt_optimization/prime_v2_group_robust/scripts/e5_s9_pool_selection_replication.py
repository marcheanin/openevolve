#!/usr/bin/env python
"""Replicate the R16 selection-rule finding on the S9 optimizer pool.

R15/R16 used a pool of 12 one-line edits. If the conclusion "min-based dev rules
are anti-correlated with test worst-group score" is real, it must also hold on a
pool of genuinely different prompts produced by the optimizers themselves.

Test-side predictions already exist (majority-of-3 on the 1800-row set), so the
only new cost is one D_dev pass per unique final prompt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

MATRIX = ROOT / "results/E5_s9_matrix"
SS = MATRIX / "stable_session"
METHODS = [
    "ape", "ape_k48", "ape_ut", "apo", "gpo", "evoprompt_ga",
    "evoprompt_de", "gepa", "prime", "random_al", "oracle",
]


def majority(job: Path, kind: str):
    ps = [job / f"{kind}_repeat{r}_preds.npy" for r in range(3)]
    ps = [p for p in ps if p.is_file()]
    if not ps:
        return None
    return (np.stack([np.load(p) for p in ps]).mean(axis=0) >= 0.5).astype(np.int8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/E5_civilcomments/config_prime_main.yaml",
    )
    ap.add_argument(
        "--fixed-dir", type=Path, default=ROOT / "experiments/E5_civilcomments/fixed_sets"
    )
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/E5_selection_control/s9_pool")
    ap.add_argument("--shrink-weight", type=float, default=40.0)
    ap.add_argument("--tau", type=float, default=0.10)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    from e5_selection_rule_shootout import group_gba, rule_values
    from e5_stable_eval import _score_once
    from prime.config import load_config
    from prime.data.civilcomments_loader import load_civilcomments_splits
    from prime.data.fixed_sets import FixedSet, materialize_split
    from prime.workers.ensemble import load_dotenv_if_present

    load_dotenv_if_present()
    cfg = load_config(args.config)
    cfg.dataset.max_val_users = max(int(cfg.dataset.max_val_users or 0), 50_000)
    cfg.dataset.max_test_users = max(int(cfg.dataset.max_test_users or 0), 80_000)

    # --- assemble the pool: seed + every final that has test predictions -------
    entries: list[tuple[str, str]] = []
    seed_txt = (ROOT / "prompts/initial_prompt_civilcomments.txt").read_text(encoding="utf-8")
    entries.append(("seed", seed_txt))
    for s in ("42", "43", "44"):
        for m in METHODS:
            bp = MATRIX / f"seed{s}" / m / "best_prompt.txt"
            if not bp.is_file() or not (SS / f"seed{s}_{m}" / "stable_report.json").is_file():
                continue
            entries.append((f"{s}_{m}", bp.read_text(encoding="utf-8")))

    # Dedupe: several methods returned the byte-identical seed prompt.
    by_hash: dict[str, str] = {}
    alias: dict[str, str] = {}
    pool: list[tuple[str, str]] = []
    for name, txt in entries:
        h = hashlib.sha256(txt.encode("utf-8")).hexdigest()[:12]
        if h in by_hash:
            alias[name] = by_hash[h]
            continue
        by_hash[h] = name
        pool.append((name, txt))
    print(f"[pool] {len(entries)} artifacts -> {len(pool)} unique prompts")
    if alias:
        print("[pool] duplicates collapsed:")
        for k, v in alias.items():
            print(f"        {k} == {v}")

    # --- dev side: score once per unique prompt --------------------------------
    fs = FixedSet.load(args.fixed_dir / "d_dev.json")
    splits = load_civilcomments_splits(cfg.dataset, seed=cfg.active_learning.seed)
    dev = materialize_split(splits["validation"], fs.indices)
    yd = np.asarray(dev["labels"])
    cd = np.asarray(dev["cluster_ids"])
    print(f"[dev] n={len(yd)} fp={fs.fingerprint}")

    preds_dir = args.out_dir / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    dev_preds: dict[str, np.ndarray] = {}
    for i, (name, txt) in enumerate(pool):
        cache = preds_dir / f"{name}__dev.npy"
        if cache.is_file() and len(np.load(cache)) == len(yd):
            dev_preds[name] = np.load(cache)
            print(f"[dev] {i + 1:2d}/{len(pool)} {name:16} cached")
            continue
        print(f"[dev] {i + 1:2d}/{len(pool)} {name:16} scoring...", flush=True)
        p, _ = _score_once(
            dev["texts"], dev["labels"], dev["cluster_ids"], dev["user_ids"],
            txt, cfg, use_mock=args.mock,
        )
        np.save(cache, p)
        dev_preds[name] = p

    # --- test side: reuse S9 majority-of-3 on the 1800-row set -----------------
    small = json.loads(
        (args.fixed_dir / "test_fixed_materialized.json").read_text(encoding="utf-8")
    )
    yt = np.asarray(small["labels"])
    ct = np.asarray(small["cluster_ids"])
    test_preds: dict[str, np.ndarray] = {}
    for name, _ in pool:
        if name == "seed":
            p = majority(SS / "_shared_seed42", "seed")
        else:
            s, m = name.split("_", 1)
            p = majority(SS / f"seed{s}_{m}", "final")
        if p is None:
            raise SystemExit(f"missing test predictions for {name}")
        test_preds[name] = p

    names = [n for n, _ in pool]
    dev_v = {
        n: rule_values(dev_preds[n], yd, cd, shrink_w=args.shrink_weight, tau=args.tau)
        for n in names
    }
    test_cvar = {
        n: rule_values(test_preds[n], yt, ct, shrink_w=args.shrink_weight, tau=args.tau)["cvar25"]
        for n in names
    }

    rules = [
        "mean_gba", "worst_class", "global_acc", "cvar25", "cvar50",
        "hard_min", "softmin", "softmin_shrunk", "hard_min_shrunk",
    ]
    best_test = max(test_cvar.values())
    order_true = np.argsort([test_cvar[n] for n in names])
    rt = np.empty(len(names))
    rt[order_true] = np.arange(len(names))

    print()
    print(f"=== S9 POOL: {len(names)} prompts, test CVaR@25% range "
          f"[{min(test_cvar.values()):.4f}, {best_test:.4f}] ===")
    print(f"{'dev rule':16} {'pick':16} {'test cvar25':>11} {'regret':>9} {'rank_corr':>10}")
    summary = {}
    for rule in rules:
        pick = max(names, key=lambda n: dev_v[n][rule])
        order_dev = np.argsort([dev_v[n][rule] for n in names])
        rd = np.empty(len(names))
        rd[order_dev] = np.arange(len(names))
        rho = float(np.corrcoef(rd, rt)[0, 1])
        summary[rule] = {
            "pick": pick,
            "test_cvar25": test_cvar[pick],
            "regret": best_test - test_cvar[pick],
            "spearman": rho,
        }
        print(f"{rule:16} {pick:16} {test_cvar[pick]:11.4f} "
              f"{best_test - test_cvar[pick]:+9.4f} {rho:10.2f}")

    print()
    print("=== dev statistic precision on this pool's seed prompt ===")
    rng = np.random.default_rng(0)
    cells = [
        np.where((cd == g) & (yd == lab))[0]
        for g in sorted(set(cd.tolist()))
        for lab in (0, 1)
    ]
    samples = {r: [] for r in rules}
    g8 = []
    for _ in range(args.n_boot):
        idx = np.concatenate([rng.choice(v, size=len(v), replace=True) for v in cells])
        v = rule_values(dev_preds["seed"][idx], yd[idx], cd[idx],
                        shrink_w=args.shrink_weight, tau=args.tau)
        for r in rules:
            samples[r].append(v[r])
        g8.append(group_gba(dev_preds["seed"][idx], yd[idx], cd[idx]).get(8, np.nan))
    prec = {r: float(np.std(samples[r], ddof=1)) for r in rules}
    for r in rules:
        print(f"  {r:16} {prec[r]:.4f}")
    print(f"  {'single group g8':16} {float(np.nanstd(g8, ddof=1)):.4f}")

    out = args.out_dir / "s9_pool_selection_replication.json"
    out.write_text(
        json.dumps(
            {
                "n_artifacts": len(entries),
                "n_unique": len(pool),
                "duplicates": alias,
                "rules": summary,
                "dev_statistic_sd": prec,
                "dev_values": dev_v,
                "test_cvar25": test_cvar,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

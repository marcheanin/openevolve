#!/usr/bin/env python
"""Does concentrating D_dev on the known-worst groups make a worst-group rule work?

This is the one intervention F8 left open. Both pools are re-scored on
`d_dev_targeted` (groups 8/3/5, 150 per cell, same 900-example budget), and every
selection statistic is then evaluated twice on the *same three groups*:

  uniform  D_dev, n=100 per group   (what we have been using)
  targeted D_dev, n=300 per group   (same cost, 3x the per-group precision)

Both predict the same target, test CVaR@25%, so the difference isolates
precision. If minimum-based rules are still anti-correlated at 3x precision, the
question is settled and the claim has to move off worst-group optimization.

The targeted set is a superset of the uniform set on those groups, so 300 rows
were already scored; they are re-scored here and compared, which doubles as an
end-to-end consistency check on the scoring path.
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

SEL = ROOT / "results/E5_selection_control"
SS = ROOT / "results/E5_s9_matrix/stable_session"
MATRIX = ROOT / "results/E5_s9_matrix"
METHODS = [
    "ape", "ape_k48", "ape_ut", "apo", "gpo", "evoprompt_ga",
    "evoprompt_de", "gepa", "prime", "random_al", "oracle",
]
TARGET_GROUPS = (3, 5, 8)


def collect_prompts() -> tuple[list[tuple[str, str]], dict[str, list[str]]]:
    """Unique prompt texts across both pools, plus which pool names map to each."""
    entries: list[tuple[str, str]] = []
    pool_of: dict[str, str] = {}

    pm = ROOT / "experiments/E5_civilcomments/pools/strictness_sweep"
    man = json.loads((pm / "manifest.json").read_text(encoding="utf-8"))
    for c in sorted(man["candidates"], key=lambda c: c["rank"]):
        name = f"r15:{c['name']}"
        entries.append((name, (pm / c["file"]).read_text(encoding="utf-8")))
        pool_of[name] = "r15_strictness"

    entries.append(
        ("s9:seed", (ROOT / "prompts/initial_prompt_civilcomments.txt").read_text(encoding="utf-8"))
    )
    pool_of["s9:seed"] = "s9_optimizers"
    for s in ("42", "43", "44"):
        for m in METHODS:
            bp = MATRIX / f"seed{s}" / m / "best_prompt.txt"
            if not bp.is_file() or not (SS / f"seed{s}_{m}" / "stable_report.json").is_file():
                continue
            name = f"s9:{s}_{m}"
            entries.append((name, bp.read_text(encoding="utf-8")))
            pool_of[name] = "s9_optimizers"

    by_hash: dict[str, str] = {}
    aliases: dict[str, list[str]] = {}
    unique: list[tuple[str, str]] = []
    for name, txt in entries:
        h = hashlib.sha256(txt.encode("utf-8")).hexdigest()[:12]
        if h in by_hash:
            aliases[by_hash[h]].append(name)
            continue
        by_hash[h] = name
        aliases[name] = []
        unique.append((name, txt))
    return unique, aliases


def group_gba(preds, y, c, groups=None) -> dict[int, float]:
    out = {}
    for g in sorted(set(c.tolist())):
        if g == 0 or (groups is not None and g not in groups):
            continue
        m = c == g
        pos, neg = m & (y == 1), m & (y == 0)
        if not pos.any() or not neg.any():
            continue
        out[int(g)] = 0.5 * (float((preds[pos] == 1).mean()) + float((preds[neg] == 0).mean()))
    return out


def shrink(g, counts, w):
    if w <= 0:
        return dict(g)
    pooled = float(np.mean(list(g.values())))
    return {k: (counts[k] * v + w * pooled) / (counts[k] + w) for k, v in g.items()}


def cvar(vals, k):
    v = np.sort(np.asarray(list(vals)))
    return float(v[: max(1, min(k, len(v)))].mean())


def softmin(vals, tau):
    v = np.asarray(list(vals), dtype=float)
    return float(-tau * np.log(np.mean(np.exp(-v / tau))))


def spearman(a, b):
    ra = np.empty(len(a)); ra[np.argsort(a)] = np.arange(len(a))
    rb = np.empty(len(b)); rb[np.argsort(b)] = np.arange(len(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def majority(job: Path, kind: str):
    ps = [job / f"{kind}_repeat{r}_preds.npy" for r in range(3)]
    ps = [p for p in ps if p.is_file()]
    return None if not ps else (np.stack([np.load(p) for p in ps]).mean(axis=0) >= 0.5).astype(np.int8)


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
    ap.add_argument("--out-dir", type=Path, default=SEL / "dev_targeted")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    from e5_stable_eval import _score_once
    from prime.config import load_config
    from prime.data.civilcomments_loader import load_civilcomments_splits
    from prime.data.fixed_sets import FixedSet, materialize_split
    from prime.workers.ensemble import load_dotenv_if_present

    load_dotenv_if_present()
    cfg = load_config(args.config)
    cfg.dataset.max_val_users = max(int(cfg.dataset.max_val_users or 0), 50_000)
    cfg.dataset.max_test_users = max(int(cfg.dataset.max_test_users or 0), 80_000)

    unique, aliases = collect_prompts()
    print(f"[pool] {len(unique)} unique prompts across both pools")
    dupes = {k: v for k, v in aliases.items() if v}
    for k, v in dupes.items():
        print(f"        {k} == {', '.join(v)}")
    # Dedup spans both pools, so e.g. `s9:seed` collapses into `r15:seed_anchor`.
    # Aliases must still appear in their own pool's analysis, pointing at the
    # canonical prompt's predictions.
    canonical: dict[str, str] = {n: n for n, _ in unique}
    for canon, alias_list in aliases.items():
        for a in alias_list:
            canonical[a] = canon

    fs = FixedSet.load(args.fixed_dir / "d_dev_targeted.json")
    splits = load_civilcomments_splits(cfg.dataset, seed=cfg.active_learning.seed)
    tdev = materialize_split(splits["validation"], fs.indices)
    yT = np.asarray(tdev["labels"])
    cT = np.asarray(tdev["cluster_ids"])
    print(f"[dev] targeted n={len(yT)} fp={fs.fingerprint} groups={sorted(set(cT.tolist()))}")

    preds_dir = args.out_dir / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    tgt_preds: dict[str, np.ndarray] = {}
    todo = [(n, t) for n, t in unique if not (preds_dir / f"{n.replace(':', '__')}__tdev.npy").is_file()]
    print(f"[dev] {len(unique) - len(todo)} cached, {len(todo)} to score")
    for i, (name, txt) in enumerate(unique):
        cache = preds_dir / f"{name.replace(':', '__')}__tdev.npy"
        if cache.is_file() and len(np.load(cache)) == len(yT):
            tgt_preds[name] = np.load(cache)
            continue
        if args.analyze_only:
            raise SystemExit(f"--analyze-only but {name} is not scored yet")
        print(f"[dev] {i + 1:2d}/{len(unique)} {name:24} scoring...", flush=True)
        p, _ = _score_once(
            tdev["texts"], tdev["labels"], tdev["cluster_ids"], tdev["user_ids"],
            txt, cfg, use_mock=args.mock,
        )
        np.save(cache, p)
        tgt_preds[name] = p

    # ---- consistency check on the 300 rows shared with the uniform dev --------
    shared = json.loads(
        (args.fixed_dir / "d_dev_targeted_shared_rows.json").read_text(encoding="utf-8")
    )
    u_rows = np.asarray([p["uniform_row"] for p in shared["pairs"]], dtype=int)
    t_rows = np.asarray([p["targeted_row"] for p in shared["pairs"]], dtype=int)
    udev = json.loads((args.fixed_dir / "d_dev_materialized.json").read_text(encoding="utf-8"))
    yU = np.asarray(udev["labels"])
    cU = np.asarray(udev["cluster_ids"])
    if not np.array_equal(yU[u_rows], yT[t_rows]):
        raise SystemExit("shared-row map misaligned: labels differ")

    agree = []
    uni_preds: dict[str, np.ndarray] = {}
    for name in canonical:
        pool, short = name.split(":", 1)
        cand = (
            SEL / "strictness_sweep/preds" / f"{short}__dev.npy"
            if pool == "r15"
            else SEL / "s9_pool/preds" / f"{short}__dev.npy"
        )
        if not cand.is_file():
            continue
        pu = np.load(cand)
        uni_preds[name] = pu
        agree.append(float((pu[u_rows] == tgt_preds[canonical[name]][t_rows]).mean()))
    print()
    print(
        f"[check] re-scored agreement on the {len(u_rows)} shared rows: "
        f"mean {np.mean(agree):.4f}, min {np.min(agree):.4f} over {len(agree)} prompts"
    )

    # ---- analysis: same statistic, same groups, two dev precisions ------------
    test_targets: dict[str, dict] = {}
    small = json.loads((args.fixed_dir / "test_fixed_materialized.json").read_text(encoding="utf-8"))
    big = json.loads(
        (args.fixed_dir / "test_fixed_large_materialized.json").read_text(encoding="utf-8")
    )
    r15_rep = json.loads(
        (SEL / "strictness_sweep/selection_control_report.json").read_text(encoding="utf-8")
    )
    r15_names = [f"r15:{r['name']}" for r in sorted(r15_rep["rows"], key=lambda r: r["rank"])]
    test_targets["r15_strictness"] = {
        "names": r15_names,
        "y": np.asarray(big["labels"]),
        "c": np.asarray(big["cluster_ids"]),
        "preds": {
            n: np.load(SEL / f"strictness_sweep/preds/{n.split(':', 1)[1]}__test.npy")
            for n in r15_names
        },
    }
    s9_names = [n for n in canonical if n.startswith("s9:")]
    s9_test = {}
    for n in s9_names:
        short = n.split(":", 1)[1]
        job = SS / ("_shared_seed42" if short == "seed" else f"seed{short.split('_', 1)[0]}_{short.split('_', 1)[1]}")
        s9_test[n] = majority(job, "seed" if short == "seed" else "final")
    test_targets["s9_optimizers"] = {
        "names": s9_names,
        "y": np.asarray(small["labels"]),
        "c": np.asarray(small["cluster_ids"]),
        "preds": s9_test,
    }

    stats: list[tuple[str, object]] = []
    for k in (1, 2, 3):
        stats.append((f"cvar_k{k}", lambda g, k=k: cvar(g.values(), k)))
    for tau in (0.02, 0.10, 0.50):
        stats.append((f"softmin_t{tau:g}", lambda g, t=tau: softmin(g.values(), t)))

    cU_counts = {g: int((cU == g).sum()) for g in TARGET_GROUPS}
    cT_counts = {g: int((cT == g).sum()) for g in TARGET_GROUPS}
    print(f"[check] per-group n: uniform {cU_counts}, targeted {cT_counts}")

    report = {
        "targeted_fingerprint": fs.fingerprint,
        "target_groups": list(TARGET_GROUPS),
        "shared_row_agreement": {
            "n_rows": int(len(u_rows)),
            "mean": float(np.mean(agree)),
            "min": float(np.min(agree)),
        },
        "pools": {},
    }

    for pool, P in test_targets.items():
        names = [n for n in P["names"] if canonical.get(n) in tgt_preds and n in uni_preds]
        if len(names) < 4:
            print(f"\n[skip] pool {pool}: only {len(names)} prompts with both dev scorings")
            continue
        tgt_cvar = [cvar(group_gba(P["preds"][n], P["y"], P["c"]).values(), 2) for n in names]
        print()
        print(f"===== POOL {pool} (n={len(names)}) — target: test CVaR@25% =====")
        print(f"{'statistic (on groups 3,5,8)':30} {'uniform n=100':>14} {'targeted n=300':>15} {'delta':>8}")
        block = {"n": len(names), "rules": {}}
        for sname, fn in stats:
            for w in (0.0, 40.0):
                du = [fn(shrink(group_gba(uni_preds[n], yU, cU, TARGET_GROUPS), cU_counts, w)) for n in names]
                dt = [
                    fn(shrink(group_gba(tgt_preds[canonical[n]], yT, cT, TARGET_GROUPS), cT_counts, w))
                    for n in names
                ]
                ru, rt = spearman(du, tgt_cvar), spearman(dt, tgt_cvar)
                best = max(tgt_cvar)
                reg_u = best - tgt_cvar[int(np.argmax(du))]
                reg_t = best - tgt_cvar[int(np.argmax(dt))]
                label = f"{sname}, w={w:g}"
                block["rules"][label] = {
                    "spearman_uniform": ru,
                    "spearman_targeted": rt,
                    "regret_uniform": reg_u,
                    "regret_targeted": reg_t,
                    "pick_uniform": names[int(np.argmax(du))],
                    "pick_targeted": names[int(np.argmax(dt))],
                }
                print(f"{label:30} {ru:+14.2f} {rt:+15.2f} {rt - ru:+8.2f}")
        report["pools"][pool] = block

        deltas = [v["spearman_targeted"] - v["spearman_uniform"] for v in block["rules"].values()]
        n_pos = sum(1 for v in block["rules"].values() if v["spearman_targeted"] > 0)
        print(
            f"  mean change in rank corr: {np.mean(deltas):+.3f}; "
            f"statistics now positive: {n_pos}/{len(block['rules'])}"
        )
        block["mean_spearman_delta"] = float(np.mean(deltas))
        block["n_positive_targeted"] = n_pos

    out = args.out_dir / "dev_budget_reallocation.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

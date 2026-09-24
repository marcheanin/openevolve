#!/usr/bin/env python
"""Soft-minimum sweep over TEMPERATURE, which `statistic_reproducibility.py` never varies.

That script hard-codes tau = 0.1 and sweeps the shrinkage weight instead. With eight groups
inside a corridor of 0.05-0.15, the spread-to-temperature ratio stays below 1.5, so every
`softmin_w*` is algebraically a mean (rho 0.96-0.99 against mean_gba, 0.60-0.69 against
hard_min). The family therefore never passes between the minimum and the mean, and the
IV.8 claim -- no statistic is at once faithful to the worst group and reproducible -- was
checked on a set that contains no intermediate candidate. This sweeps tau instead.

Faithfulness: rank correlation against hard_min and against mean_gba, over the full test.
Reproducibility: of the pairs a statistic calls resolvable on one half of the test, the share
whose DIRECTION holds on the disjoint half -- the same criterion `statistic_reproducibility.py
--sweep` uses, so the numbers sit in the same table as the ones IV.8 cites. `flipped` counts
pairs the second half declares resolvable the other way round.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_s11 import IDS, load_preds, load_set, valid_rows  # noqa: E402
from dataset_config import cfg  # noqa: E402

TEMPS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.2, 0.4]


def statistics(gv: np.ndarray) -> dict:
    """Every statistic from one array of per-group GBAs; last axis is the group."""
    s = np.sort(gv, axis=-1)
    out = {"hard_min": s[..., 0],
           "cvar25": s[..., : max(1, gv.shape[-1] // 4)].mean(axis=-1),
           "mean_gba": gv.mean(axis=-1)}
    for t in TEMPS:
        # -t log mean exp(-a/t): the max is subtracted first so exp cannot underflow at small t
        z = -gv / t
        m = z.max(axis=-1, keepdims=True)
        out[f"softmin_t{t}"] = -t * (m[..., 0] + np.log(np.exp(z - m).mean(axis=-1)))
    return out


ORDER = ["hard_min", "cvar25"] + [f"softmin_t{t}" for t in TEMPS] + ["mean_gba"]


def group_gba(pred, pos, neg):
    return np.array([0.5 * ((pred[p] == 1).mean() + (pred[q] == 0).mean())
                     for p, q in zip(pos, neg)])


def boot_groups(preds, names, pos, neg, n_boot, rng):
    """(prompt, boot, group) GBAs on one shared resample, so contrasts stay paired."""
    draws = [(rng.integers(0, len(p), size=(n_boot, len(p))),
              rng.integers(0, len(q), size=(n_boot, len(q)))) for p, q in zip(pos, neg)]
    out = np.empty((len(names), n_boot, len(pos)))
    for i, n in enumerate(names):
        pr = preds[n]
        for k, ((dp, dq), p, q) in enumerate(zip(draws, pos, neg)):
            out[i, :, k] = 0.5 * ((pr[p][dp] == 1).mean(axis=1) + (pr[q][dq] == 0).mean(axis=1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--n-split", type=int, default=12)
    ap.add_argument("--pool", default="finals", choices=["finals", "all"],
                    help="finals = optimizer finals + seed, the pool statistic_reproducibility.py "
                         "uses, so the numbers are comparable with IV.8; all = every prompt")
    args = ap.parse_args()

    y, c, _ = load_set(cfg.test_set)
    P = load_preds(cfg.test_set, len(y))
    names = sorted(n for n in P if not n.startswith("CONTROL:"))
    if args.pool == "finals":
        names = [n for n in names if n.startswith(cfg.final_prefix + ":") or n == "seed"]
    keep = valid_rows(P, len(y), names)
    pos = [np.flatnonzero((c == g) & (y == 1) & keep) for g in IDS]
    neg = [np.flatnonzero((c == g) & (y == 0) & keep) for g in IDS]
    print(f"пул '{args.pool}': {len(names)} промптов, {len(IDS)} групп, ячейки по "
          f"{min(len(p) for p in pos)}-{max(len(p) for p in pos)} строк\n")

    obs = {n: statistics(group_gba(P[n], pos, neg)) for n in names}
    hm = np.array([obs[n]["hard_min"] for n in names])
    mg = np.array([obs[n]["mean_gba"] for n in names])
    print("=== верность цели: за чем следует статистика ===")
    print(f"{'статистика':16s} {'rho с hard_min':>15s} {'rho со средним':>15s}  следует за")
    faith = {}
    for s in ORDER:
        v = np.array([obs[n][s] for n in names])
        a, b = spearmanr(v, hm).statistic, spearmanr(v, mg).statistic
        faith[s] = (float(a), float(b))
        tag = "худшей группой" if a > b + 0.1 else ("средним" if b > a + 0.1 else "обоими поровну")
        print(f"{s:16s} {a:15.3f} {b:15.3f}  {tag}")

    print(f"\n=== воспроизводимость: объявили на половине A, подтвердилось ли на B "
          f"({args.n_split} расщеплений) ===", flush=True)
    rng = np.random.default_rng(0)
    decl = {s: [0, 0, 0] for s in ORDER}  # объявлено, направление совпало, перевернулось
    for sp in range(args.n_split):
        perm_p = [rng.permutation(p) for p in pos]
        perm_n = [rng.permutation(q) for q in neg]
        half, obs_half = [], []
        for h in (0, 1):
            hp = [p[: len(p) // 2] if h == 0 else p[len(p) // 2:] for p in perm_p]
            hn = [q[: len(q) // 2] if h == 0 else q[len(q) // 2:] for q in perm_n]
            half.append(statistics(boot_groups(P, names, hp, hn, args.n_boot, rng)))
            g = np.stack([group_gba(P[n], hp, hn) for n in names])
            obs_half.append(statistics(g))  # наблюдённое на этой половине, без ресэмпла
        for s in ORDER:
            A, B, oB = half[0][s], half[1][s], obs_half[1][s]
            for i in range(len(names)):
                loA, hiA = np.percentile(A[i][None, :] - A[i + 1:], [2.5, 97.5], axis=1)
                res = np.flatnonzero((loA > 0) | (hiA < 0))
                if not len(res):
                    continue
                loB, hiB = np.percentile(B[i][None, :] - B[i + 1:], [2.5, 97.5], axis=1)
                for j in res:
                    sign = 1.0 if loA[j] > 0 else -1.0
                    decl[s][0] += 1
                    decl[s][1] += int(np.sign(oB[i] - oB[i + 1 + j]) == sign)
                    if (loB[j] > 0 or hiB[j] < 0) and (1.0 if loB[j] > 0 else -1.0) != sign:
                        decl[s][2] += 1
        print(f"  расщепление {sp + 1}/{args.n_split}", flush=True)

    pairs = len(names) * (len(names) - 1) // 2 * args.n_split
    print(f"\n{'статистика':16s} {'объявлено':>10s} {'из всех пар':>12s} "
          f"{'направление совпало':>20s} {'перевернулось':>14s} {'rho с hard_min':>15s}")
    rows = {}
    for s in ORDER:
        d, k, f = decl[s]
        share = k / d if d else float("nan")
        rows[s] = {"declared": d, "reproduced": k, "share": share, "flipped": f,
                   "declared_share_of_pairs": d / pairs,
                   "rho_hard_min": faith[s][0], "rho_mean": faith[s][1]}
        print(f"{s:16s} {d:10d} {100 * d / pairs:11.1f}% {100 * share:19.1f}% "
              f"{f:14d} {faith[s][0]:15.3f}")

    print("\n=== вывод ===")
    good = [s for s in ORDER if rows[s]["rho_hard_min"] >= 0.90 and rows[s]["share"] >= 0.90]
    if good:
        print("Есть статистика, ОДНОВРЕМЕННО верная худшей группе (rho >= 0.90) и "
              f"воспроизводимая (>= 0.90): {', '.join(good)}.")
        print("IV.8 в нынешней формулировке не устоял и требует сужения.")
    else:
        print("Ни одна температура не даёт одновременно rho >= 0.90 с hard_min и "
              "воспроизводимость >= 0.90: вывод IV.8 устоял на расширенном семействе.")

    out = cfg.outputs / f"temperature_sweep_{args.pool}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"temps": TEMPS, "n_boot": args.n_boot,
                               "n_split": args.n_split, "stats": rows}, indent=2),
                   encoding="utf-8")
    print(f"\nзаписано {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

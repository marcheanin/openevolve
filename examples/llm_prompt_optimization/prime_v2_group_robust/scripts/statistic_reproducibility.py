"""Which worst-group statistic survives being run twice?

IV.7 diagnosed why a study on small group cells declares more differences than a larger one
finds: a statistic built from the minimum of eight noisy group estimates picks its group by
the noise, which inflates the point difference between two systems while paired resampling
keeps the interval honest. The prediction that follows is specific -- the more a statistic
leans on the single worst group, the less its verdicts should reproduce -- and the obvious
remedy is to stop taking a bare minimum: soften it, and shrink each group estimate toward
the pooled mean first.

Run it with --faithfulness before believing any remedy. The result there is that every
smoothed variant, including the mildest one, ranks prompts by the group mean rather than by
the worst group. The remedy works by abandoning the measurement.

The design is the same split-half as `ci_calibration_check.py`, at 100 rows per cell, but
every statistic is scored on the identical splits and the identical resamples, so the
comparison between statistics is paired. Two numbers matter together and neither alone:

  declared   - how many of the 325 pairs the statistic calls resolvable at 95%
  reproduced - of those, the share whose direction holds on the disjoint half

A statistic that declares nothing reproduces perfectly and is useless. A statistic that
declares a lot and reproduces poorly is worse than useless, because every one of those
declarations is a claim someone would publish.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_config import cfg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IDS = cfg.ids
STATS = ["hard_min", "cvar25", "mean_gba", "softmin_w40", "worst_class"]
# Shrinkage weights for the sweep, in units of observations: w is how many pooled-mean
# observations each group estimate is mixed with before the soft minimum is taken.
WEIGHTS = [0, 10, 20, 40, 80, 160, 320]
SWEEP_STATS = ["hard_min", "cvar25"] + [f"softmin_w{w}" for w in WEIGHTS]
N_BOOT = 300
N_SPLIT = 12


def all_stats(pred, pos, neg):
    """Every statistic from one pass over the group rates, so they share the resample."""
    g, n = [], []
    for p, q in zip(pos, neg):
        if len(p) and len(q):
            g.append(0.5 * ((pred[p] == 1).mean() + (pred[q] == 0).mean()))
            n.append(len(p) + len(q))
    g, n = np.asarray(g), np.asarray(n)
    v = np.sort(g)
    pooled = g.mean()
    allp = np.concatenate(pos) if len(pos) else np.array([], int)
    alln = np.concatenate(neg) if len(neg) else np.array([], int)
    out = {
        "hard_min": float(v[0]),
        "cvar25": float(v[: max(1, len(v) // 4)].mean()),
        "mean_gba": float(g.mean()),
        "worst_class": float(min((pred[allp] == 1).mean(), (pred[alln] == 0).mean())),
    }
    for w in WEIGHTS:
        sm = (n * g + w * pooled) / (n + w)
        out[f"softmin_w{w}"] = float(-0.1 * np.log(np.mean(np.exp(-sm / 0.1))))
    return out


def cells_of(y, c, rows):
    pos = [np.flatnonzero((c[rows] == g) & (y[rows] == 1)) for g in IDS]
    neg = [np.flatnonzero((c[rows] == g) & (y[rows] == 0)) for g in IDS]
    return pos, neg


def boot_all(preds, names, rows, y, c, rng, stats):
    """{stat: (prompts, N_BOOT)} on one half, one shared resampling scheme throughout."""
    flat = [np.flatnonzero((c[rows] == g) & (y[rows] == lab)) for g in IDS for lab in (0, 1)]
    flat = [x for x in flat if len(x)]
    idx = np.concatenate([rng.choice(x, size=(N_BOOT, len(x)), replace=True) for x in flat], axis=1)
    yy, cc = y[rows][idx], c[rows][idx]
    cells = [cells_of(yy[b], cc[b], slice(None)) for b in range(N_BOOT)]
    out = {s: np.empty((len(names), N_BOOT)) for s in stats}
    for i, n in enumerate(names):
        take = preds[n][rows][idx]
        for b in range(N_BOOT):
            vals = all_stats(take[b], *cells[b])
            for s in stats:
                out[s][i, b] = vals[s]
    return out


def observed_all(preds, names, rows, y, c):
    pos, neg = cells_of(y, c, rows)
    return {n: all_stats(preds[n][rows], pos, neg) for n in names}


def test_cap():
    """Rows per group x label cell the test set was built with (200 for truth_large)."""
    rec = json.loads(cfg.set_path(cfg.test_set).read_text(encoding="utf-8"))
    return int(rec["cap_per_cell"])


def load():
    rec = json.loads(cfg.set_path(cfg.test_set).read_text(encoding="utf-8"))
    y, c = np.array(rec["labels"]), np.array(rec["cluster_ids"])
    preds = {}
    for f in sorted((cfg.preds_dir / cfg.test_set).glob("*.npy")):
        if f.name.endswith(".partial.npy") or f.stem.startswith("_"):
            continue
        if f.name.endswith(".lo.npy"):
            continue  # log-odds companion of a prediction file, not a prompt of its own
        # First "__" only: `s13__42_ape__soft_min` -> `s13:42_ape__soft_min`.
        preds[f.stem.replace("__", ":", 1)] = np.load(f)
    keep = ~np.any(np.stack([p < 0 for p in preds.values()]), axis=0)
    names = [n for n in sorted(preds) if n.startswith(cfg.final_prefix + ":") or n == "seed"]
    return names, {n: preds[n][keep] for n in names}, y[keep], c[keep]


def faithfulness() -> int:
    """What is each statistic actually ranking prompts by?

    Reproducibility is worthless on its own: a statistic that quietly reports the group mean
    will reproduce beautifully and answer a different question than the one asked. This
    ranks the prompts by each statistic on the full 200-per-cell test and rank-correlates
    that ordering with hard-min GBA (the worst-group goal) and with mean GBA.
    """
    from scipy.stats import spearmanr

    names, preds, y, c = load()
    pos, neg = cells_of(y, c, slice(None))
    vals = {n: all_stats(preds[n], pos, neg) for n in names}
    order = ["hard_min", "cvar25"] + [f"softmin_w{w}" for w in WEIGHTS] + ["mean_gba", "worst_class"]
    hm = np.array([vals[n]["hard_min"] for n in names])
    mg = np.array([vals[n]["mean_gba"] for n in names])
    print(f"{len(names)} prompts on the full {test_cap()}-per-cell test")
    print(f"{'statistic':14s} {'rho vs hard_min':>16s} {'rho vs mean_gba':>16s}   tracking")
    for st in order:
        v = np.array([vals[n][st] for n in names])
        a, b = spearmanr(v, hm).statistic, spearmanr(v, mg).statistic
        tag = "worst group" if a > b + 0.1 else ("group mean" if b > a + 0.1 else "both equally")
        print(f"{st:14s} {a:16.3f} {b:16.3f}   {tag}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", action="store_true",
                    help="vary the shrinkage weight instead of comparing statistic families")
    ap.add_argument("--faithfulness", action="store_true",
                    help="rank-correlate each statistic with hard-min and with the group mean")
    # hard_min declares so few pairs that 12 splits leave ~23 of them, and the reproduced
    # share swings by 20 points between split counts. Raise this to pin the number down;
    # the default keeps the output identical to what the earlier runs produced.
    ap.add_argument("--n-split", type=int, default=N_SPLIT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    stats = SWEEP_STATS if args.sweep else STATS
    if args.faithfulness:
        return faithfulness()

    n_split = args.n_split
    names, preds, y, c = load()
    rng = np.random.default_rng(args.seed)
    k = len(names)
    pairs = k * (k - 1) // 2
    tally = {s: [0, 0, 0] for s in stats}  # declared, reproduced, opposite-declared

    print(f"{k} prompts, {pairs} pairs, {n_split} half-splits at {test_cap() // 2} rows per cell, "
          f"{N_BOOT} resamples per half\n")
    for sp in range(n_split):
        a_rows, b_rows = [], []
        for g in IDS:
            for lab in (0, 1):
                cell = np.flatnonzero((c == g) & (y == lab))
                perm = rng.permutation(cell)
                a_rows.append(perm[: len(perm) // 2])
                b_rows.append(perm[len(perm) // 2:])
        A, B = np.concatenate(a_rows), np.concatenate(b_rows)
        MA = boot_all(preds, names, A, y, c, rng, stats)
        MB = boot_all(preds, names, B, y, c, rng, stats)
        ob = observed_all(preds, names, B, y, c)
        for s in stats:
            for i in range(k):
                for j in range(i + 1, k):
                    lo, hi = np.percentile(MA[s][i] - MA[s][j], [2.5, 97.5])
                    if not (lo > 0 or hi < 0):
                        continue
                    sign = 1.0 if lo > 0 else -1.0
                    tally[s][0] += 1
                    tally[s][1] += int(np.sign(ob[names[i]][s] - ob[names[j]][s]) == sign)
                    blo, bhi = np.percentile(MB[s][i] - MB[s][j], [2.5, 97.5])
                    if (blo > 0 or bhi < 0) and (1.0 if blo > 0 else -1.0) != sign:
                        tally[s][2] += 1
        print(f"  split {sp + 1:2d} done")

    print(f"\n{'statistic':13s} {'declared':>9s} {'of all pairs':>13s} {'reproduced':>11s} "
          f"{'flipped':>8s}   verdict")
    for s in stats:
        d, r, f = tally[s]
        share = 100 * d / (pairs * n_split)
        rep = 100 * r / d if d else float("nan")
        note = "declares little" if share < 1 else ("reproduces" if rep >= 90 else "lottery")
        print(f"{s:13s} {d:9d} {share:12.1f}% {rep:10.1f}% {f:8d}   {note}")

    print("\nRead the table as a pair, then read it against --faithfulness. A statistic is")
    print("useful only if it declares a workable number of differences, those differences")
    print("hold up on independent data of the same size, AND it still ranks prompts by the")
    # The last sentence is the CivilComments result; it is not a property of the method, so it is
    # not printed for another dataset.
    print("worst group rather than by the group mean."
          + (" On this benchmark nothing does all three." if cfg.key == "civil" else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

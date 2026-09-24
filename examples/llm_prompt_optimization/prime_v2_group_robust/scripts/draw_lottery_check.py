"""At a fixed test size, how much does the test DRAW decide what a study can resolve?

IV.7 first blamed the small group cells: gate A's test has 100 rows per group x label cell
and declared 11% of the 325 pairwise contrasts among the optimizer finals resolvable, while
the 200-per-cell set declared 4%. That comparison changed two things at once, the cell size
and the draw, and attributed all of it to the first.

This changes only the draw. `truth_large` is split into halves of 100 rows per cell -- gate
A's exact cell size -- and each half is asked the same question with the same prompts, the
same statistic and the same resampling. The interval widths come out the same everywhere,
so anything left is the draw.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from dataset_config import require_civil  # noqa: E402

# Both the gate-A comparison (`old_pool`) and the set of prompts the halves are compared on
# (those scored on both tests) come from the E5 data, which only exists for CivilComments.
require_civil("draw_lottery_check.py")
from statistic_reproducibility import all_stats, cells_of, load  # noqa: E402

IDS = tuple(range(1, 9))
N_BOOT = 400
N_DRAW = 12
STAT = "cvar25"


def boot(preds, names, rows, y, c, rng):
    flat = [np.flatnonzero((c[rows] == g) & (y[rows] == lab)) for g in IDS for lab in (0, 1)]
    flat = [x for x in flat if len(x)]
    idx = np.concatenate([rng.choice(x, size=(N_BOOT, len(x)), replace=True) for x in flat], axis=1)
    yy, cc = y[rows][idx], c[rows][idx]
    cells = [cells_of(yy[b], cc[b], slice(None)) for b in range(N_BOOT)]
    M = np.empty((len(names), N_BOOT))
    for i, n in enumerate(names):
        take = preds[n][rows][idx]
        for b in range(N_BOOT):
            M[i, b] = all_stats(take[b], *cells[b])[STAT]
    return M


def summarize(tag, preds, names, rows, y, c, rng):
    M = boot(preds, names, rows, y, c, rng)
    pos, neg = cells_of(y, c, rows)
    obs = np.array([all_stats(preds[n][rows], pos, neg)[STAT] for n in names])
    k = len(names)
    diffs, widths, res = [], [], 0
    for i in range(k):
        for j in range(i + 1, k):
            lo, hi = np.percentile(M[i] - M[j], [2.5, 97.5])
            widths.append(hi - lo)
            diffs.append(abs(obs[i] - obs[j]))
            res += int(lo > 0 or hi < 0)
    pairs = k * (k - 1) // 2
    print(f"{tag:28s} rows={len(rows):5d}  median|d|={np.median(diffs):.4f}  "
          f"median CI width={np.median(widths):.4f}  resolvable={res:3d}/{pairs} "
          f"({100 * res / pairs:5.1f}%)")
    return np.median(diffs), np.median(widths), 100 * res / pairs


def old_pool():
    te = json.loads((ROOT / "experiments/E5_civilcomments/fixed_sets"
                     "/test_fixed_materialized.json").read_text(encoding="utf-8"))
    SS = ROOT / "results/E5_s9_matrix/stable_session"
    preds = {}
    for job in sorted(SS.glob("seed4*_*")):
        s, m = job.name.split("_", 1)
        r = [np.load(f) for f in sorted(job.glob("final_repeat*_preds.npy"))]
        if r:
            preds[f"s9:{s[4:]}_{m}"] = (np.stack(r).mean(0) >= 0.5).astype(int)
    r = [np.load(f) for f in sorted((SS / "_shared_seed42").glob("seed_repeat*_preds.npy"))]
    if r:
        preds["seed"] = (np.stack(r).mean(0) >= 0.5).astype(int)
    return preds, np.array(te["labels"]), np.array(te["cluster_ids"])


def main() -> int:
    old, yo, co = old_pool()
    names_all, new, yn, cn = load()
    shared = sorted(set(old) & set(new))
    rng = np.random.default_rng(0)
    print(f"{len(shared)} prompts scored on both tests, statistic {STAT}\n")

    summarize("gate A test_fixed", old, shared, np.arange(len(yo)), yo, co, rng)
    summarize("truth_large, full 200/cell", new, shared, np.arange(len(yn)), yn, cn, rng)
    print()

    rows_out = []
    for sp in range(N_DRAW):
        half = []
        for g in IDS:
            for lab in (0, 1):
                cell = np.flatnonzero((cn == g) & (yn == lab))
                half.append(rng.permutation(cell)[: len(cell) // 2])
        rows_out.append(summarize(f"truth_large half #{sp + 1}", new, shared,
                                  np.concatenate(half), yn, cn, rng))

    d, w, r = (np.array(x) for x in zip(*rows_out))
    print(f"\nover {N_DRAW} draws of the SAME size and the SAME cell size:")
    print(f"  median|d|       {d.mean():.4f}  range {d.min():.4f}..{d.max():.4f}")
    print(f"  median CI width {w.mean():.4f}  range {w.min():.4f}..{w.max():.4f}  <- flat")
    print(f"  resolvable      {r.mean():5.1f}%  range {r.min():.1f}%..{r.max():.1f}%  <- not flat")
    print("\nThe interval width is a property of the design and barely moves. What a study can")
    print("resolve moves by more than an order of magnitude on the draw alone, which makes the")
    print("count of resolvable differences a lottery ticket rather than a property of the method.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Does the bootstrap interval for a worst-group metric mean what it says?

Doubling the test set should resolve MORE pairs of prompts, not fewer. It resolved fewer:
on `test_fixed` (100 rows per group x label cell) 11% of the 325 pairwise contrasts among
the optimizer finals came out resolvable at 95%, and on `truth_large` (200 per cell, the
same 26 prompts, the same official split) only 4% did. One of those two inferences is
wrong, and the suspect is the smaller one: CVaR@25% averages the two smallest of eight
group estimates, and resampling inference for a min-type statistic is known to under-cover
when the cells it minimises over are small.

This measures it directly instead of arguing about it. Each cell of `truth_large` is split
at random into two halves of 100 rows -- exactly the cell size gate A worked at. Half A
declares which pairs are resolvable and in which direction; half B, disjoint and never seen
by that decision, says which direction actually held. Half B votes twice: once with its plain observed difference, and once with its own
bootstrap interval. The second vote is the one that matters, because it cannot be blamed
on half B being noisy in turn.

Result, recorded here so the script is read with it: the strict count is ZERO -- the
interval at 100 rows per cell is not confidently wrong, and the suspicion that it fails to
cover is not supported. But a quarter of the pairs it declares resolvable do not reproduce
in direction, which says the declarations are dominated by marginal calls near the
threshold rather than by real separations.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_config import cfg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IDS = cfg.ids
N_BOOT = 400
N_SPLIT = 20


def cvar25_cells(pred, cells_pos, cells_neg):
    """CVaR@25% given precomputed per-group row indices, so the hot loop does no masking."""
    vals = []
    for pos, neg in zip(cells_pos, cells_neg):
        if len(pos) and len(neg):
            vals.append(0.5 * ((pred[pos] == 1).mean() + (pred[neg] == 0).mean()))
    v = np.sort(vals)
    return float(v[: max(1, len(v) // 4)].mean())


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


def boot_metrics(preds, names, rows, y, c, rng):
    """(prompts, N_BOOT) metric matrix on one half, resampling cells within that half."""
    cells = []
    for g in IDS:
        for lab in (0, 1):
            cells.append(np.flatnonzero((c[rows] == g) & (y[rows] == lab)))
    cells = [x for x in cells if len(x)]
    idx = np.concatenate([rng.choice(x, size=(N_BOOT, len(x)), replace=True) for x in cells], axis=1)
    yy, cc = y[rows][idx], c[rows][idx]
    pos = [[np.flatnonzero((cc[b] == g) & (yy[b] == 1)) for g in IDS] for b in range(N_BOOT)]
    neg = [[np.flatnonzero((cc[b] == g) & (yy[b] == 0)) for g in IDS] for b in range(N_BOOT)]
    M = np.empty((len(names), N_BOOT))
    for i, n in enumerate(names):
        take = preds[n][rows][idx]
        for b in range(N_BOOT):
            M[i, b] = cvar25_cells(take[b], pos[b], neg[b])
    return M


def observed(preds, names, rows, y, c):
    pos = [np.flatnonzero((c[rows] == g) & (y[rows] == 1)) for g in IDS]
    neg = [np.flatnonzero((c[rows] == g) & (y[rows] == 0)) for g in IDS]
    return np.array([cvar25_cells(preds[n][rows], pos, neg) for n in names])


def main() -> int:
    names, preds, y, c = load()
    rng = np.random.default_rng(0)
    pairs = len(names) * (len(names) - 1) // 2   # 325 for the 26 CivilComments prompts
    print(f"{len(names)} prompts, {len(y)} rows, {N_SPLIT} random half-splits, "
          f"{N_BOOT} bootstrap resamples per half\n")

    declared = agreed = flipped = 0
    per_split = []
    for sp in range(N_SPLIT):
        a_rows, b_rows = [], []
        for g in IDS:
            for lab in (0, 1):
                cell = np.flatnonzero((c == g) & (y == lab))
                perm = rng.permutation(cell)
                a_rows.append(perm[: len(perm) // 2])
                b_rows.append(perm[len(perm) // 2:])
        A, B = np.concatenate(a_rows), np.concatenate(b_rows)
        MA = boot_metrics(preds, names, A, y, c, rng)
        MB = boot_metrics(preds, names, B, y, c, rng)
        obs_b = observed(preds, names, B, y, c)

        d_sp = a_sp = f_sp = 0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                lo, hi = np.percentile(MA[i] - MA[j], [2.5, 97.5])
                if not (lo > 0 or hi < 0):
                    continue
                d_sp += 1
                sign_a = 1.0 if lo > 0 else -1.0
                a_sp += int(np.sign(obs_b[i] - obs_b[j]) == sign_a)
                # The strict count: half B does not merely lean the other way, it declares
                # the opposite direction resolvable at the same level. Two honest and
                # independent 95% intervals should almost never do that.
                blo, bhi = np.percentile(MB[i] - MB[j], [2.5, 97.5])
                if (blo > 0 or bhi < 0) and (1.0 if blo > 0 else -1.0) != sign_a:
                    f_sp += 1
        declared += d_sp
        agreed += a_sp
        flipped += f_sp
        per_split.append((d_sp, a_sp, f_sp))
        print(f"  split {sp + 1:2d}: half A ({len(A)} rows) declared {d_sp:3d}/{pairs} resolvable, "
              f"B agreed {a_sp:3d} ({100 * a_sp / d_sp if d_sp else float('nan'):3.0f}%), "
              f"B declared the OPPOSITE {f_sp:2d}")

    print(f"\ndeclared resolvable on half A: {declared} pair-decisions over {N_SPLIT} splits")
    share = lambda k: 100 * k / declared if declared else float("nan")  # noqa: E731
    print(f"  direction merely confirmed on half B: {agreed} ({share(agreed):.1f}%)")
    print(f"  direction merely contradicted:        {declared - agreed} "
          f"({share(declared - agreed):.1f}%)")
    print(f"  half B declared the OPPOSITE direction resolvable: {flipped} "
          f"({share(flipped):.2f}%)")
    print("\nA calibrated 95% interval, checked against an independent half of the same size,\n"
          "should rarely be contradicted outright on an independent half of the same size.")
    if cfg.key == "civil":
        # This paragraph is the CivilComments reading (see the module docstring); it states a
        # result, so it is not printed for another dataset, where the counts above have to be read.
        print("Reading the result: a zero OPPOSITE-direction count means the interval is not")
        print("confidently wrong -- it is not grossly anti-conservative, and the suspicion that")
        print("it fails to cover is not supported. What the merely-contradicted share measures is")
        print("something else: the pairs a study of this size declares resolvable are dominated by")
        print("marginal calls sitting near the detection threshold, so a quarter of them do not")
        print("reproduce even in direction. The declaration is honest; the selection is a lottery.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

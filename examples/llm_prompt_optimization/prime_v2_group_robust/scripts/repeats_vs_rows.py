"""Does scoring each prompt three times and voting buy more resolving power than rows?

Gate A's test set (`test_fixed`, 100 rows per group x label cell) declared 11% of the 325
pairwise contrasts among the optimizer finals resolvable at 95%. A half of `truth_large`
at the same 100 rows per cell declares 1.3%, and the full 200-per-cell set declares 4%.
The test sets are both official-split and both cell-balanced, so size cannot explain a gap
that runs the wrong way. The remaining difference is how the predictions were produced:
gate A's are the majority of three scoring passes, S11's are a single pass.

The old run kept all three passes, so this is directly testable on identical rows and
identical prompts: score once, or score three times and vote, and count what each can
resolve. If voting wins, then the number of scoring repeats -- which papers in this area do
not report -- decides how many differences a study can see, at the same number of test rows.
That makes it another protocol dimension, and a cheap lever for anyone designing one.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_config import require_civil  # noqa: E402

require_civil("repeats_vs_rows.py")  # the three scoring passes exist only for the old E5 runs

ROOT = Path(__file__).resolve().parents[1]
FX = ROOT / "experiments/E5_civilcomments/fixed_sets"
SS = ROOT / "results/E5_s9_matrix/stable_session"
IDS = tuple(range(1, 9))
N_BOOT = 1000


def cvar25(pred, pos, neg):
    vals = [0.5 * ((pred[p] == 1).mean() + (pred[n] == 0).mean())
            for p, n in zip(pos, neg) if len(p) and len(n)]
    v = np.sort(vals)
    return float(v[: max(1, len(v) // 4)].mean())


def load_repeats():
    te = json.loads((FX / "test_fixed_materialized.json").read_text(encoding="utf-8"))
    y, c = np.array(te["labels"]), np.array(te["cluster_ids"])
    reps = {}
    for job in sorted(SS.glob("seed4*_*")):
        s, m = job.name.split("_", 1)
        r = [np.load(f) for f in sorted(job.glob("final_repeat*_preds.npy"))]
        if len(r) == 3:
            reps[f"s9:{s[4:]}_{m}"] = np.stack(r)
    r = [np.load(f) for f in sorted((SS / "_shared_seed42").glob("seed_repeat*_preds.npy"))]
    if len(r) == 3:
        reps["seed"] = np.stack(r)
    return reps, y, c


def resolvable(preds, names, y, c, rng):
    cells = [np.flatnonzero((c == g) & (y == lab)) for g in IDS for lab in (0, 1)]
    cells = [x for x in cells if len(x)]
    idx = np.concatenate([rng.choice(x, size=(N_BOOT, len(x)), replace=True) for x in cells], axis=1)
    yy, cc = y[idx], c[idx]
    pos = [[np.flatnonzero((cc[b] == g) & (yy[b] == 1)) for g in IDS] for b in range(N_BOOT)]
    neg = [[np.flatnonzero((cc[b] == g) & (yy[b] == 0)) for g in IDS] for b in range(N_BOOT)]
    M = np.empty((len(names), N_BOOT))
    for i, n in enumerate(names):
        take = preds[n][idx]
        for b in range(N_BOOT):
            M[i, b] = cvar25(take[b], pos[b], neg[b])
    k, res = len(names), 0
    for i in range(k):
        for j in range(i + 1, k):
            lo, hi = np.percentile(M[i] - M[j], [2.5, 97.5])
            res += int(lo > 0 or hi < 0)
    pairs = k * (k - 1) // 2
    return res, pairs, float(np.mean(M.var(axis=1)))


def main() -> int:
    reps, y, c = load_repeats()
    names = sorted(reps)
    print(f"{len(names)} prompts with three scoring passes each, {len(y)} rows "
          f"(100 per group x label cell)\n")

    flip = np.mean([(reps[n][0] != reps[n][1]).mean() for n in names])
    unstable = np.mean([(reps[n].sum(0) % 3 != 0).mean() for n in names])
    print(f"model stochasticity: two passes of the same prompt disagree on {100 * flip:.2f}% of rows; "
          f"{100 * unstable:.2f}% of rows are not unanimous across the three")

    variants = {
        "single pass (repeat 0)": {n: reps[n][0] for n in names},
        "single pass (repeat 1)": {n: reps[n][1] for n in names},
        "majority of three": {n: (reps[n].mean(0) >= 0.5).astype(int) for n in names},
    }
    print(f"\n{'scoring':26s} {'API calls':>10s} {'resolvable pairs':>18s} {'per-prompt noise sd':>21s}")
    for tag, preds in variants.items():
        rng = np.random.default_rng(0)  # same resampling scheme for every variant
        res, pairs, var = resolvable(preds, names, y, c, rng)
        calls = len(y) * (3 if "majority" in tag else 1)
        print(f"{tag:26s} {calls:10d} {res:8d}/{pairs:<9d} {np.sqrt(var):21.4f}")

    print("\nIf majority voting resolves materially more pairs than a single pass on the very\n"
          "same rows, the repeat count is buying resolving power that rows alone would have to\n"
          "be multiplied to match -- and it is not reported by anyone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

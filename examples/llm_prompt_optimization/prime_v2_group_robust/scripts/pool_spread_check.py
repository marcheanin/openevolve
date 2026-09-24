"""How much of the candidate pool's spread is real, and how much is test-set noise?

IV.6 established that the protocol-induced spread scales with how far apart the candidates
already sit on the test set. That makes the pool spread a load-bearing quantity, and a
pool spread measured on a small test is inflated: every prompt's score carries sampling
noise, and the variance of noisy scores is the variance of the true scores plus the noise.

The same optimizer finals were scored twice -- on `test_fixed` (n=1800, majority of three
repeats, used by gate A) and on `truth_large` (n=3584, single pass, S11). This script
reports, for each test set, the raw spread of the pool and the spread after subtracting
the per-prompt sampling variance estimated by resampling within group x label cells.

If the corrected spreads agree while the raw ones do not, gate A's protocol variance was
inflated through the pool, and section 5 of the paper needs rewriting.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from dataset_config import require_civil  # noqa: E402

require_civil("pool_spread_check.py")  # compares against the old E5 test_fixed scoring

FX = ROOT / "experiments/E5_civilcomments/fixed_sets"
SS = ROOT / "results/E5_s9_matrix/stable_session"
S11 = ROOT / "results/S11_protocol_matrix"
IDS = tuple(range(1, 9))
N_BOOT = 1000


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def cvar25(pred, y, c, rows=None):
    vals = []
    for g in IDS:
        m = (c == g) if rows is None else ((c == g) & rows)
        pos, neg = m & (y == 1), m & (y == 0)
        if pos.any() and neg.any():
            vals.append(0.5 * ((pred[pos] == 1).mean() + (pred[neg] == 0).mean()))
    v = np.sort(vals)
    return float(v[: max(1, len(v) // 4)].mean())


def boot_matrix(preds, y, c, rng):
    """Metric per prompt on each resample, all prompts read on the SAME resampled rows.

    Sharing the rows matters. Every prompt is scored on one test set, so their errors are
    positively correlated; subtracting each prompt's marginal noise variance from the
    across-prompt variance double-counts what the shared rows already cancel, and drives
    any pool to an apparent spread of zero. The shared index keeps the correlation intact.
    """
    cells = [np.flatnonzero((c == g) & (y == lab)) for g in IDS for lab in (0, 1)]
    cells = [x for x in cells if len(x)]
    idx = np.concatenate([rng.choice(x, size=(N_BOOT, len(x)), replace=True) for x in cells], axis=1)
    yy, cc = y[idx], c[idx]
    names = list(preds)
    M = np.empty((len(names), N_BOOT))
    for i, n in enumerate(names):
        take = preds[n][idx]
        for b in range(N_BOOT):
            M[i, b] = cvar25(take[b], yy[b], cc[b])
    return names, M


def report(tag, preds, y, c, rng):
    truth = {n: cvar25(p, y, c) for n, p in preds.items()}
    names, M = boot_matrix(preds, y, c, rng)
    obs = float(np.var([truth[n] for n in names]))
    # Standard bootstrap bias correction: the resampled across-prompt variance carries the
    # noise we want gone, so V_true ~= 2 * V_observed - E[V_resampled].
    exp_boot = float(np.mean(M.var(axis=0)))
    corrected = max(0.0, 2 * obs - exp_boot)
    marginal = float(np.mean(M.var(axis=1)))
    # How many pairwise contrasts survive a paired 95% interval -- the assumption-free version.
    k = len(names)
    res = 0
    for i in range(k):
        for j in range(i + 1, k):
            lo, hi = np.percentile(M[i] - M[j], [2.5, 97.5])
            res += int(lo > 0 or hi < 0)
    print(f"{tag:34s} n={len(y):5d} prompts={k:3d} | raw sd {np.sqrt(obs):.4f}  "
          f"per-prompt noise sd {np.sqrt(marginal):.4f}  corrected pool sd {np.sqrt(corrected):.4f} | "
          f"resolvable pairs {res:5d}/{k * (k - 1) // 2}  ({100 * res / (k * (k - 1) / 2):3.0f}%)")
    return truth, corrected


def old_pool():
    te = jload(FX / "test_fixed_materialized.json")
    y, c = np.array(te["labels"]), np.array(te["cluster_ids"])
    preds = {}
    for job in sorted(SS.glob("seed4*_*")):
        s, m = job.name.split("_", 1)
        reps = [np.load(f) for f in sorted(job.glob("final_repeat*_preds.npy"))]
        if reps:
            preds[f"s9:{s[4:]}_{m}"] = (np.stack(reps).mean(0) >= 0.5).astype(int)
    reps = [np.load(f) for f in sorted((SS / "_shared_seed42").glob("seed_repeat*_preds.npy"))]
    if reps:
        preds["seed"] = (np.stack(reps).mean(0) >= 0.5).astype(int)
    return preds, y, c


def new_pool():
    rec = jload(S11 / "fixed_sets/truth_large.json")
    y, c = np.array(rec["labels"]), np.array(rec["cluster_ids"])
    preds = {}
    for f in sorted((S11 / "preds/truth_large").glob("*.npy")):
        if f.name.endswith(".partial.npy") or f.stem.startswith("_"):
            continue
        if f.name.endswith(".lo.npy"):
            continue  # log-odds companion of a prediction file, not a prompt of its own
        preds[f.stem.replace("__", ":", 1)] = np.load(f)
    keep = ~np.any(np.stack([p < 0 for p in preds.values()]), axis=0)
    return {n: p for n, p in preds.items()}, y, c, keep


def main() -> int:
    rng = np.random.default_rng(0)
    po, yo, co = old_pool()
    pn, yn, cn, keep = new_pool()
    pn = {n: p[keep] for n, p in pn.items()}
    yn, cn = yn[keep], cn[keep]

    shared = sorted(set(po) & set(pn))
    print(f"prompts present on both tests: {len(shared)}\n")
    report("old test_fixed, all its prompts", po, yo, co, rng)
    report("new truth_large, all its prompts", pn, yn, cn, rng)
    print()
    _, co_c = report("old test_fixed, shared prompts", {n: po[n] for n in shared}, yo, co, rng)
    _, cn_c = report("new truth_large, shared prompts", {n: pn[n] for n in shared}, yn, cn, rng)

    print(f"\ncorrected pool sd: old {np.sqrt(co_c):.4f} vs new {np.sqrt(cn_c):.4f} "
          f"(ratio {np.sqrt(co_c / cn_c) if cn_c else float('inf'):.2f})")
    print("If these agree, the pool really is that spread out and gate A's protocol variance\n"
          "stands. If old >> new, part of gate A's pool spread was test noise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

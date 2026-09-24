"""H4 (draw lottery of the test set), for any dataset of `dataset_config`.

`draw_lottery_check.py` asked, for CivilComments, how much the DRAW of a test decides what a study
can resolve at a fixed test size: `truth_large` (200 rows per group x label cell) is cut into
random halves of 100 rows per cell (gate A's exact cell size) and every half is asked the same
question with the same prompts, statistic and resampling. That script needs the old E5 data
(gate A's `test_fixed`, the E5 predictions), so it cannot run for MultiNLI. This one does the same
thing with the test set of `dataset_config` only (`truth_large` for civil, `truth_mnli` for mnli).

What is computed, exactly as in `draw_lottery_check.py`:

  half        per group x label cell (group in cfg.ids) a random subset of cap_per_cell // 2 rows
              (cap_per_cell is read from the set JSON; a cell with fewer rows gives len // 2, which
              is what the original does after it dropped unparsable rows). 12 halves by default,
              one rng, fixed seed.
  resampling  N_BOOT = 400 paired bootstrap draws INSIDE each cell of the half (`Bootstrap` of
              analyze_s11.py), the same draws for every prompt.
  pairs       every pair of prompts of the pool (default: all prompts with predictions on the test
              set except CONTROL:*).
  resolvable  a pair is "distinguishable at 95%" if the [2.5, 97.5] percentile interval of the
              bootstrap difference of the statistic (default cvar25, as in the original) excludes 0.
  width       97.5th minus 2.5th percentile of that difference; mean and median over the pairs.

The pre-registered H4 says: the ratio max/min of the resolvable share over the halves is >= 3 while
the ratio max/min of the mean interval width is <= 1.15 (the draw decides what is resolvable, the
width does not move). It is UNTESTABLE when the mean share over halves is < 2%, REFUTED when the
ratio of shares is < 2.

CivilComments acceptance: `--pool s9 --match-draw-lottery-check` selects the 26 prompts the original
compares (s9 finals + seed) and advances the rng through the reference summary the original computes
first (gate A's test_fixed, cells read from the E5 data) so that "half #k" is the same random half
and the numbers are those of results/S11_protocol_matrix/final/draw_lottery.txt. Without
`--match-draw-lottery-check` the halves are a different (equally valid) draw of the same design.
The line "full set" is a reference for the whole test set (the original's second line); the original's
first line (gate A's test_fixed, E5 data) has no counterpart here.

Usage:  python scripts/draw_lottery_halves.py [--metric cvar25] [--pool all|<prefix>|r15]
        S11_DATASET=mnli python scripts/draw_lottery_halves.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_config import cfg  # noqa: E402
from analyze_s11 import Bootstrap, cell_index, load_preds, load_set, valid_rows  # noqa: E402

METRICS = ["hard_min", "cvar25", "mean_gba", "worst_class"]
N_BOOT = 400          # as draw_lottery_check.N_BOOT
N_HALVES = 12         # as draw_lottery_check.N_DRAW
SEED = 0
# Pre-registered thresholds of H4.
MIN_SHARE_TESTABLE = 0.02
RATIO_REFUTED_BELOW = 2.0
RATIO_SHARE_MIN = 3.0
RATIO_WIDTH_MAX = 1.15
EPS = 1e-9            # ratios of shares are ratios of counts: 0.15 / 0.05 is 2.9999999999999996 in floats


def group_cells(y, c, keep):
    """{(group, label): row indices} for the groups the metrics are computed over, in the order
    `for g in cfg.ids for label in (0, 1)`; the group of CivilComments that the metrics ignore
    (id 0) and empty cells are left out, as in draw_lottery_check.py."""
    full = cell_index(y, c, keep)
    return {(g, lab): full[(g, lab)] for g in cfg.ids for lab in (0, 1) if len(full.get((g, lab), ()))}


def draw_half(cells, per_cell, rng):
    """One random half: per cell a random subset of min(per_cell, len // 2) rows. The order of the
    permuted rows is kept on purpose (the bootstrap draws positions inside the array it is given)."""
    return {k: rng.permutation(rows)[: min(per_cell, len(rows) // 2)] for k, rows in cells.items()}


def summarize(P, names, cells, y, c, n_boot, rng, metric):
    """Pairwise paired-bootstrap resolvability of the prompts `names` on the rows `cells`."""
    bs = Bootstrap(y, c, n_boot, rng, cells=cells)
    obs, boot = {}, {}
    for n in names:
        o, b = bs.metrics(P[n])
        obs[n], boot[n] = o[metric], b[metric]
    diffs, widths, res = [], [], 0
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            lo, hi = np.percentile(boot[a] - boot[b], [2.5, 97.5])
            widths.append(hi - lo)
            diffs.append(abs(obs[a] - obs[b]))
            res += int(lo > 0 or hi < 0)
    pairs = len(widths)
    sizes = [len(v) for v in bs.cells.values()]
    return {"rows": int(sum(sizes)), "cell_min": min(sizes), "cell_max": max(sizes), "pairs": pairs,
            "resolvable": res, "share": res / pairs, "median_abs_d": float(np.median(diffs)),
            "width_mean": float(np.mean(widths)), "width_median": float(np.median(widths))}


def line(tag, s):
    return (f"{tag:30s} rows={s['rows']:5d}  median|d|={s['median_abs_d']:.4f}  "
            f"CI width mean={s['width_mean']:.4f} median={s['width_median']:.4f}  "
            f"resolvable={s['resolvable']:4d}/{s['pairs']} ({100 * s['share']:5.1f}%)")


def burn_gate_a_stream(rng, n_boot):
    """Advance `rng` exactly as the first summary of draw_lottery_check.py does (bootstrap of gate
    A's test_fixed, whose cells are read from the E5 data). Only for the civil acceptance."""
    p = ROOT / "experiments/E5_civilcomments/fixed_sets/test_fixed_materialized.json"
    te = json.loads(p.read_text(encoding="utf-8"))
    y, c = np.asarray(te["labels"]), np.asarray(te["cluster_ids"])
    for g in cfg.ids:
        for lab in (0, 1):
            m = int(((c == g) & (y == lab)).sum())
            if m:
                rng.choice(np.arange(m), size=(n_boot, m), replace=True)


def ratio(hi, lo):
    return float("inf") if lo <= 0 else hi / lo


def verdict(shares, wmeans, wmedians):
    """H4 verdict from the per-half shares and widths. Returns (verdict, printable lines)."""
    mean_share = float(np.mean(shares))
    r_share = ratio(max(shares), min(shares))
    r_w, r_wmed = ratio(max(wmeans), min(wmeans)), ratio(max(wmedians), min(wmedians))
    lines = [f"rule: UNTESTABLE if mean share over halves < {100 * MIN_SHARE_TESTABLE:.0f}%; "
             f"REFUTED if max/min share < {RATIO_REFUTED_BELOW:g}; CONFIRMED if max/min share >= "
             f"{RATIO_SHARE_MIN:g} AND max/min mean CI width <= {RATIO_WIDTH_MAX:g}",
             f"values: mean share {100 * mean_share:.2f}%; max/min share = "
             f"{100 * max(shares):.1f}% / {100 * min(shares):.1f}% = "
             + ("inf (min share is 0)" if min(shares) <= 0 else f"{r_share:.2f}")
             + f"; max/min mean CI width = {max(wmeans):.4f} / {min(wmeans):.4f} = {r_w:.3f} "
             f"(with medians: {r_wmed:.3f})"]
    if mean_share < MIN_SHARE_TESTABLE:
        v = "НЕПРОВЕРЯЕМО (средняя доля различимых пар < 2%)"
    elif r_share < RATIO_REFUTED_BELOW - EPS:
        v = "ОПРОВЕРГНУТО (отношение долей < 2)"
    elif r_share >= RATIO_SHARE_MIN - EPS and r_w <= RATIO_WIDTH_MAX + EPS:
        v = "ПОДТВЕРЖДЕНО (отношение долей >= 3 при отношении ширин <= 1.15)"
    else:
        why = []
        if r_share < RATIO_SHARE_MIN - EPS:
            why.append(f"отношение долей {r_share:.2f} в [2, 3)")
        if r_w > RATIO_WIDTH_MAX + EPS:
            why.append(f"отношение ширин {r_w:.3f} > 1.15")
        v = "ПРОМЕЖУТОЧНО: не подтверждено и не опровергнуто по заданным правилам (" + "; ".join(why) + ")"
    return v, lines


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", default="cvar25", choices=METRICS)
    ap.add_argument("--pool", default="all", choices=cfg.pools,
                    help=f"prompts compared: all = every non-CONTROL prompt scored on the test set; "
                         f"{cfg.final_prefix} = optimizer finals + seed; r15 = one-line edits + seed")
    ap.add_argument("--halves", type=int, default=N_HALVES)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--valid-over", default="all", choices=["all", "pool"],
                    help="unparsable (-1) rows are dropped for every prompt at once: over every scored "
                         "prompt (default, as draw_lottery_check.py) or over the compared pool only")
    ap.add_argument("--match-draw-lottery-check", action="store_true",
                    help="civil only: advance the rng through the reference summary computed first by "
                         "draw_lottery_check.py so that the halves are the same draws")
    args = ap.parse_args()

    y, c, rec = load_set(cfg.test_set)
    if "cap_per_cell" not in rec:
        raise SystemExit(f"{cfg.set_path(cfg.test_set)} has no 'cap_per_cell'")
    per_cell = int(rec["cap_per_cell"]) // 2
    P = load_preds(cfg.test_set, controls=False)
    if not P:
        raise SystemExit(f"no predictions in {cfg.preds_dir / cfg.test_set}")
    bad = sorted(n for n, p in P.items() if len(p) != len(y))
    if bad:
        raise SystemExit(f"prediction length != set size {len(y)}: {bad[:5]}")
    names = sorted(n for n in P if args.pool == "all" or n.startswith(args.pool + ":") or n == "seed")
    if len(names) < 2:
        raise SystemExit(f"pool {args.pool!r}: {len(names)} prompt(s), need at least 2")
    keep = valid_rows(P, len(y), None if args.valid_over == "all" else names)
    cells = group_cells(y, c, keep)
    rng = np.random.default_rng(args.seed)

    print(f"dataset {cfg.key}, {cfg.test_set}: n={len(y)} ({int(keep.sum())} valid), "
          f"cap_per_cell={rec['cap_per_cell']} -> {per_cell} rows per cell in a half, "
          f"{len(cells)} cells of {len(cfg.ids)} groups")
    print(f"{len(names)} prompts (pool '{args.pool}', CONTROL:* excluded), "
          f"{len(names) * (len(names) - 1) // 2} pairs, statistic {args.metric}, "
          f"{args.n_boot} bootstrap resamples per half, seed {args.seed}\n")

    if args.match_draw_lottery_check:
        if cfg.key != "civil":
            raise SystemExit("--match-draw-lottery-check is only defined for S11_DATASET=civil")
        burn_gate_a_stream(rng, args.n_boot)
    print(line(f"{cfg.test_set}, full set", summarize(P, names, cells, y, c, args.n_boot, rng, args.metric)))
    print()

    out = []
    for k in range(args.halves):
        half = draw_half(cells, per_cell, rng)
        s = summarize(P, names, half, y, c, args.n_boot, rng, args.metric)
        out.append(s)
        print(line(f"{cfg.test_set} half #{k + 1}", s) + f"  cell rows {s['cell_min']}..{s['cell_max']}")

    sh = np.array([s["share"] for s in out])
    wm = np.array([s["width_mean"] for s in out])
    wd = np.array([s["width_median"] for s in out])
    md = np.array([s["median_abs_d"] for s in out])
    print(f"\nover {len(out)} draws of the SAME size and the SAME cell size:")
    print(f"  median|d|        {md.mean():.4f}  range {md.min():.4f}..{md.max():.4f}")
    print(f"  mean CI width    {wm.mean():.4f}  range {wm.min():.4f}..{wm.max():.4f}")
    print(f"  median CI width  {wd.mean():.4f}  range {wd.min():.4f}..{wd.max():.4f}")
    print(f"  resolvable       {100 * sh.mean():5.1f}%  range {100 * sh.min():.1f}%..{100 * sh.max():.1f}%")
    v, lines = verdict(sh, wm, wd)
    print("\n=== H4 (draw lottery): verdict ===")
    for ln in lines:
        print(ln)
    print("verdict:", v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

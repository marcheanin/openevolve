"""H6 (how thin the cells are), MultiNLI only: the same predictions, three partitions into groups.

The worst-group metrics of the S11 study are computed over groups, and a bootstrap interval of a
minimum-type statistic gets wider as the cells (group x label) it is built from get thinner. This
script varies ONLY the partition of the test rows into groups and keeps everything else fixed: the
labels, the predictions, the rows that survive `valid_rows`, the metric, the number of resamples.

  A  genre                        10 groups (the main design; 250 rows per genre x label cell)
  B  genre x neg_broad            up to 20 groups (thin cells: as few as 17 rows in truth_mnli)
  C  neg_broad                    2 groups (very thick cells)
  B' genre x neg_sagawa, C' neg_sagawa      the narrow negation list, EXPLORATORY: not part of H6

`neg_broad` / `neg_sagawa` are boolean vectors of the set JSON (`truth_mnli.json`).

Contrasts: every optimizer final (prompts named `<final_prefix>:...`, i.e. `s13:...`) against `seed`,
on hard-min GBA (`--metric`, default hard_min, the pre-registered one). Per partition: the paired
bootstrap 95% interval (`Bootstrap` of analyze_s11.py: rows resampled inside each group x label cell,
the same draws for every prompt), its width, the number of contrasts resolvable at 95% without a
correction and after Holm-Bonferroni (alpha 0.05, the family is all contrasts of that partition).

Pre-registered H6: the mean interval width is ordered C < A < B, and the number of contrasts resolvable
after Holm does not decrease along B -> A -> C. REFUTED if the order of widths is violated.

Details that matter:
  * `Bootstrap.metrics(pred, groups=...)` takes the groups explicitly (its default is `cfg.ids`, the
    ten genres, which is wrong for B and C). Groups that lack rows of one label (after valid_rows) are
    dropped BEFORE the resampling: hard-min etc. only look at groups with both labels, and
    `Bootstrap.metrics` raises KeyError if a one-label group is left among its cells.
  * The size of the smallest group x label cell is printed for every partition.
  * Each partition has its own rng, `default_rng([seed, partition index])`, so the output does not
    depend on which partitions are run.

Usage:  S11_DATASET=mnli python scripts/partition_thinness.py [--n-boot 4000] [--detail]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_config import cfg  # noqa: E402
from analyze_s11 import (Bootstrap, boot_p, cell_index, delta_ci, holm, load_preds,  # noqa: E402
                         load_set, valid_rows)

METRICS = ["hard_min", "cvar25", "mean_gba"]
N_BOOT = 4000
SEED = 0
ALPHA = 0.05
# (key, title, neg field or None, exploratory). Order fixes the rng stream index of each partition.
PARTITIONS = [("A", "genre", None, False),
              ("B", "genre x neg_broad", "neg_broad", False),
              ("C", "neg_broad", "neg_broad", False),
              ("B'", "genre x neg_sagawa", "neg_sagawa", True),
              ("C'", "neg_sagawa", "neg_sagawa", True)]


def build_partition(key, rec, c):
    """(group id vector, {id: name}) of one partition; ids are 1..K."""
    genre = cfg.group_names(rec)
    if key == "A":
        return np.asarray(c), {int(g): str(n) for g, n in genre.items()}
    field = dict((k, f) for k, _, f, _ in PARTITIONS)[key]
    if field not in rec:
        raise SystemExit(f"{cfg.set_path(cfg.test_set)} has no field {field!r}")
    neg = np.asarray(rec[field], dtype=bool)
    if len(neg) != len(c):
        raise SystemExit(f"field {field!r} has {len(neg)} entries, the set has {len(c)} rows")
    nm = {0: "non-neg", 1: "neg"}
    if key.startswith("C"):
        return neg.astype(int) + 1, {1: nm[0], 2: nm[1]}
    raw = np.asarray(c) * 2 + neg.astype(int)            # genre id and negation flag in one integer
    uniq, inv = np.unique(raw, return_inverse=True)
    names = {i + 1: f"{genre.get(int(u) // 2, int(u) // 2)}|{nm[int(u) % 2]}" for i, u in enumerate(uniq)}
    return inv + 1, names


def bootstrap_partition(P, names, y, gid, gnames, keep, n_boot, rng):
    """Resample one partition once and evaluate the metrics of every prompt in `names` on it.

    Returns (info, obs, boot): obs[name] and boot[name] are the dicts of `Bootstrap.metrics`.
    """
    full = cell_index(y, gid, keep)
    defined = sorted(set(gid.tolist()))
    used = [g for g in defined if len(full.get((g, 0), ())) and len(full.get((g, 1), ()))]
    if not used:
        raise SystemExit("no group has rows of both labels")
    cells = {(g, lab): full[(g, lab)] for g in used for lab in (0, 1)}
    bs = Bootstrap(y, gid, n_boot, rng, cells=cells)
    obs, boot = {}, {}
    for n in names:
        obs[n], boot[n] = bs.metrics(P[n], groups=used)
    sizes = {k: len(v) for k, v in cells.items()}
    smallest = min(sizes, key=sizes.get)
    info = {"groups_defined": len(defined), "groups_used": len(used),
            "dropped": [gnames.get(g, g) for g in defined if g not in used],
            "cells": len(cells), "cell_min": sizes[smallest],
            "cell_min_at": f"{gnames.get(smallest[0], smallest[0])} label {smallest[1]}",
            "cell_median": float(np.median(list(sizes.values())))}
    return info, obs, boot


def evaluate(obs, boot, contrasts, metric, alpha=ALPHA):
    """Interval, p-value, 95% and Holm decisions for a list of (name_a, name_b) contrasts a - b."""
    rows = []
    for a, b in contrasts:
        d, lo, hi = delta_ci(boot[a], boot[b], obs[a], obs[b], metric)
        rows.append((a, b, d, lo, hi, boot_p(boot[a], boot[b], metric)))
    d, lo, hi, p = (np.array(x) for x in zip(*[r[2:] for r in rows]))
    res95 = (lo > 0) | (hi < 0)
    rej = holm(p, alpha)
    return {"rows": rows, "n": len(rows), "width_mean": float((hi - lo).mean()),
            "width_median": float(np.median(hi - lo)), "abs_d_mean": float(np.abs(d).mean()),
            "res95": int(res95.sum()), "holm": int(rej.sum()), "res95_vec": res95, "holm_vec": rej}


def run_partitions(P, contrasts, y, rec, c, keep, n_boot, seed, metric, keys=None):
    """{key: (info, evaluation)} for the requested partitions (all five by default)."""
    names = sorted({n for pair in contrasts for n in pair})
    out = {}
    for idx, (key, title, _, expl) in enumerate(PARTITIONS):
        if keys is not None and key not in keys:
            continue
        gid, gnames = build_partition(key, rec, c)
        rng = np.random.default_rng([seed, idx])
        info, obs, boot = bootstrap_partition(P, names, y, gid, gnames, keep, n_boot, rng)
        out[key] = (info, evaluate(obs, boot, contrasts, metric))
        del obs, boot
    return out


def table(rows, title):
    print(f"\n{title}")
    print(f"{'part':4s} {'partition':20s} {'groups':>7s} {'cells':>5s} {'min cell':>8s} {'med cell':>8s} "
          f"{'contr.':>6s} {'mean width':>10s} {'med width':>9s} {'mean|d|':>8s} "
          f"{'res@95':>7s} {'Holm':>5s}")
    for key, title_, info, ev in rows:
        print(f"{key:4s} {title_:20s} {info['groups_used']:3d}/{info['groups_defined']:<3d} "
              f"{info['cells']:5d} {info['cell_min']:8d} {info['cell_median']:8.0f} {ev['n']:6d} "
              f"{ev['width_mean']:10.4f} {ev['width_median']:9.4f} {ev['abs_d_mean']:8.4f} "
              f"{ev['res95']:7d} {ev['holm']:5d}")
    for key, _, info, ev in rows:
        drop = f"; groups dropped for lack of a label: {', '.join(info['dropped'])}" if info["dropped"] else ""
        print(f"  {key}: smallest cell = {info['cell_min']} rows ({info['cell_min_at']}){drop}")


def h6_verdict(wA, wB, wC, nA, nB, nC, n_contrasts):
    order = wC < wA < wB
    mono = nB <= nA <= nC
    lines = [f"rule: REFUTED if the mean widths are not ordered C < A < B; CONFIRMED if they are AND the "
             f"number of contrasts resolvable after Holm does not decrease B -> A -> C",
             f"values: mean width C {wC:.4f} < A {wA:.4f} < B {wB:.4f}: {'holds' if order else 'VIOLATED'}; "
             f"Holm-resolved B {nB} -> A {nA} -> C {nC} (of {n_contrasts}): "
             f"{'non-decreasing' if mono else 'DECREASES'}"]
    if not order:
        v = "ОПРОВЕРГНУТО (порядок ширин C < A < B нарушен)"
    elif mono:
        v = "ПОДТВЕРЖДЕНО (ширины упорядочены C < A < B, число различимых после Holm не убывает B -> A -> C)"
        if nA == nB == nC:
            v += ". Осторожно: число различимых одинаково во всех разбиениях"
            v += " (0 или все контрасты), монотонность выполнена тривиально"
    else:
        v = ("НЕ ПОДТВЕРЖДЕНО: порядок ширин верен, но число различимых после Holm убывает при B -> A -> C "
             "(правило опровержения этим не сработало)")
    return v, lines


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if cfg.key != "mnli":
        print(f"partition_thinness.py: not applicable with S11_DATASET={cfg.key!r}. It needs the negation "
              "flags (neg_broad, neg_sagawa) of the MultiNLI set truth_mnli; run it with S11_DATASET=mnli.")
        return 0
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--metric", default="hard_min", choices=METRICS,
                    help="H6 is pre-registered for hard_min; another metric is reported as non-registered")
    ap.add_argument("--valid-over", default="all", choices=["all", "pool"],
                    help="unparsable (-1) rows are dropped for every prompt at once: over every scored prompt "
                         "(default, as analyze_s11.py --part power) or over seed + finals only")
    ap.add_argument("--detail", action="store_true", help="print every contrast of every partition")
    ap.add_argument("--no-exploratory", action="store_true", help="skip B' and C' (neg_sagawa)")
    args = ap.parse_args()

    y, c, rec = load_set(cfg.test_set)
    P = load_preds(cfg.test_set, controls=False)
    if "seed" not in P:
        raise SystemExit("seed predictions not scored yet")
    bad = sorted(n for n, p in P.items() if len(p) != len(y))
    if bad:
        raise SystemExit(f"prediction length != set size {len(y)}: {bad[:5]}")
    finals = sorted(n for n in P if n.startswith(cfg.final_prefix + ":"))
    if len(finals) < 2:
        raise SystemExit(f"{len(finals)} finals with prefix {cfg.final_prefix!r}: not enough contrasts")
    contrasts = [(f, "seed") for f in finals]
    keep = valid_rows(P, len(y), None if args.valid_over == "all" else finals + ["seed"])

    print(f"dataset {cfg.key}, {cfg.test_set}: n={len(y)} ({int(keep.sum())} rows valid for every prompt), "
          f"fingerprint {rec.get('fingerprint')}")
    print(f"{len(finals)} contrasts (each '{cfg.final_prefix}:*' final vs seed), metric {args.metric}"
          f"{'' if args.metric == 'hard_min' else '  [NOT the pre-registered metric]'}, "
          f"{args.n_boot} bootstrap resamples, seed {args.seed}, Holm alpha {ALPHA}")

    keys = [k for k, _, _, e in PARTITIONS if not (e and args.no_exploratory)]
    res = run_partitions(P, contrasts, y, rec, c, keep, args.n_boot, args.seed, args.metric, keys)
    title = {k: t for k, t, _, _ in PARTITIONS}
    table([(k, title[k], *res[k]) for k in ("A", "B", "C")], "=== H6: genre x label cells, three partitions ===")

    wA, wB, wC = (res[k][1]["width_mean"] for k in ("A", "B", "C"))
    nA, nB, nC = (res[k][1]["holm"] for k in ("A", "B", "C"))
    v, lines = h6_verdict(wA, wB, wC, nA, nB, nC, len(contrasts))
    print("\n=== H6 verdict ===")
    for ln in lines:
        print(ln)
    print("verdict:", v + ("" if args.metric == "hard_min" else " [метрика не предрегистрирована]"))

    if "B'" in res:
        table([(k, title[k], *res[k]) for k in ("A", "B'", "C'")],
              "=== разведочно (не входит в H6): neg_sagawa в ролях B и C ===")
        w1, w2, w3 = (res[k][1]["width_mean"] for k in ("A", "B'", "C'"))
        n1, n2, n3 = (res[k][1]["holm"] for k in ("A", "B'", "C'"))
        print(f"  exploratory: width C' {w3:.4f} < A {w1:.4f} < B' {w2:.4f}: {'holds' if w3 < w1 < w2 else 'VIOLATED'}; "
              f"Holm B' {n2} -> A {n1} -> C' {n3}: {'non-decreasing' if n2 <= n1 <= n3 else 'DECREASES'}")

    if args.detail:
        for key in keys:
            ev = res[key][1]
            print(f"\n--- contrasts, partition {key} ({title[key]}) ---")
            print(f"{'final':34s} {'delta':>8s} {'CI95':>19s} {'p':>8s}  res95 Holm")
            for (a, _, d, lo, hi, p), r95, hm in zip(ev["rows"], ev["res95_vec"], ev["holm_vec"]):
                print(f"{a:34s} {d:+8.4f} [{lo:+.4f},{hi:+.4f}] {p:8.4f}  {'yes' if r95 else ' no':>5s} "
                      f"{'yes' if hm else ' no':>4s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

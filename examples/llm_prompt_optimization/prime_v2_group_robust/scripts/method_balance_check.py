#!/usr/bin/env python
"""Does the unbalanced S9 pool distort the protocol-versus-method comparison?

The headline decomposition of IV.6 divides the spread a protocol produces by the spread a
method produces, and reports 0.75-1.00: comparable, not larger. The method spread is the sd
of per-method means, each averaged over the seeds that method happens to have. But the pool
is a legacy collection, not a factorial: seed 42 carries 10 of the 11 methods, seed 43 only 6,
seed 44 nine. A method represented by a single seed has a mean whose sampling noise is the
noise of one run, which inflates the sd of the method means, which in turn understates the
protocol/method ratio -- the conservative direction, but it should be measured rather than
assumed.

This recomputes the method spread three ways on the same truth set: over all methods, over
only the methods that have all three seeds, and with the single-seed methods dropped. It also
estimates how much of the all-methods spread is seed noise, by comparing the between-method
variance with the within-method variance across seeds.
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_s11 import load_preds, load_set, metric, valid_rows  # noqa: E402
from dataset_config import cfg  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="cvar25",
                    choices=["hard_min", "cvar25", "mean_gba", "worst_class", "global_acc"])
    args = ap.parse_args()

    y, c, _ = load_set(cfg.test_set)
    P = load_preds(cfg.test_set, len(y))
    names = sorted(n for n in P if not n.startswith("CONTROL:"))
    keep = valid_rows(P, len(y), names)
    truth = {n: metric(args.target, P[n], y, c, rows=keep) for n in names}

    per = collections.defaultdict(dict)
    for n in names:
        m = cfg.method_of(n)
        if m is None:
            continue
        per[m][n.split(":", 1)[1].split("_", 1)[0]] = truth[n]

    seeds = sorted({s for v in per.values() for s in v})
    print(f"цель {args.target}; методов {len(per)}, сидов {len(seeds)} ({', '.join(seeds)})\n")
    print(f"{'метод':16s} {'сидов':>6s}  " + "  ".join(f"{s:>8s}" for s in seeds) + f" {'среднее':>9s}")
    for m in sorted(per, key=lambda m: -len(per[m])):
        row = "  ".join(f"{per[m].get(s, float('nan')):8.4f}" for s in seeds)
        print(f"{m:16s} {len(per[m]):6d}  {row} {np.mean(list(per[m].values())):9.4f}")

    full = [m for m in per if len(per[m]) == len(seeds)]
    multi = [m for m in per if len(per[m]) >= 2]
    variants = {
        "все методы (как в IV.6)": list(per),
        "только с полными тремя сидами": full,
        "без методов с одним сидом": multi,
    }
    print(f"\n{'вариант':34s} {'методов':>8s} {'sd средних':>12s} {'размах':>9s}")
    sds = {}
    for tag, ms in variants.items():
        v = np.array([np.mean(list(per[m].values())) for m in ms])
        sds[tag] = float(v.std())
        print(f"{tag:34s} {len(ms):8d} {v.std():12.4f} {np.ptp(v):9.4f}")

    # Сколько разброса средних по методу — это шум сида, а не различие методов
    within = [np.var(list(per[m].values()), ddof=1) for m in multi if len(per[m]) >= 2]
    sw = float(np.mean(within))
    v_full = np.array([np.mean(list(per[m].values())) for m in full])
    between_raw = float(v_full.var(ddof=1))
    corrected = max(0.0, between_raw - sw / len(seeds))
    print(f"\nвнутри метода (между сидами): дисперсия {sw:.6f}, sd {np.sqrt(sw):.4f} "
          f"(по {len(within)} методам с >=2 сидами)")
    print(f"между методами (полные сиды): дисперсия {between_raw:.6f}, sd {np.sqrt(between_raw):.4f}")
    print(f"с поправкой на шум сида (минус within/{len(seeds)}): sd {np.sqrt(corrected):.4f}")
    if corrected <= 0:
        print("  -> различия между методами неотличимы от шума сидов: "
              "знаменатель отношения протокол/метод держится на шуме, и отношение — нижняя оценка")

    print("\nчто это значит для отношения протокол/метод:")
    base = sds["все методы (как в IV.6)"]
    for tag, s in sds.items():
        if s > 0:
            print(f"  знаменатель «{tag}»: sd {s:.4f} -> отношение изменится в {base / s:.2f} раза "
                  f"против опубликованного")
    print(f"  знаменатель с поправкой на шум сида: sd {np.sqrt(corrected):.4f}"
          + (f" -> в {base / np.sqrt(corrected):.2f} раза" if corrected > 0 else " -> не определено"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

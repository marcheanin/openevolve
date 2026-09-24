#!/usr/bin/env python
"""Откуда берётся прирост hard-min: движение порога, подъём кривой или подмена худшей группы.

Три механизма, которые в отчётах обычно неразличимы:

  1. ДВИЖЕНИЕ ПОРОГА. Промпт стал строже или мягче, рабочая точка проехала по той же кривой ROC.
     Это калибровка, а не улучшение: размен между пропусками и ложными тревогами.
  2. ПОДЪЁМ КРИВОЙ. Модель под этим промптом реально лучше различает классы в связывающей
     группе — точка ушла ВЫШЕ кривой.
  3. ПОДМЕНА СВЯЗЫВАЮЩЕЙ ГРУППЫ. Худшей стала другая группа, и hard-min вырос не потому, что
     прежней худшей стало лучше, а потому что её место заняла группа повыше.

Почему нельзя взять для этого threshold_frontier.py. Тот скрипт строит кривую группы по
операционным точкам ВСЕГО пула, включая сами финалы. Поэтому «идеальная развёртка общего
порога» у него по построению не ниже лучшего промпта пула, и разность «развёртка минус seed»
равна приросту лучшего промпта, а не запасу, доступному одним лишь порогом. Число circular.

Как здесь. Кривая каждой группы строится ТОЛЬКО по однострочным правкам строгости (`r15:*`) и
стартовому промпту. Эти промпты по построению и есть развёртка порога над одной и той же
моделью: они меняют требование строгости и ничего больше. Тогда

    запас по порогу  = max_tau min_g GBA_g(tau) по кривой правок  -  hard-min стартового промпта
    подъём кривой    = hard-min лучшего финала  -  max_tau min_g GBA_g(tau) по кривой правок

и первое слагаемое не зависит от финалов, а второе измеряет ровно то, что финалы добавили
сверх достижимого порогом. Развёртка ограничена диапазоном строгости, покрытым правками; выход
за него — экстраполяция, и он отчитывается отдельно.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_s11 import IDS, load_preds, load_set, valid_rows  # noqa: E402
from dataset_config import cfg  # noqa: E402


def operating_points(P, names, y, c, keep, groups):
    tpr = np.zeros((len(names), len(groups)))
    fpr = np.zeros((len(names), len(groups)))
    for j, g in enumerate(groups):
        pos, neg = (c == g) & (y == 1) & keep, (c == g) & (y == 0) & keep
        for i, n in enumerate(names):
            tpr[i, j] = (P[n][pos] == 1).mean()
            fpr[i, j] = (P[n][neg] == 1).mean()
    return tpr, fpr


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", type=int, default=2001)
    ap.add_argument("--sweep-prefix", default="r15:",
                    help="префикс промптов, образующих развёртку порога (правки строгости)")
    args = ap.parse_args()

    y, c, _ = load_set(cfg.test_set)
    P = load_preds(cfg.test_set, len(y))
    names = sorted(n for n in P if not n.startswith("CONTROL:"))
    keep = valid_rows(P, len(y), names)
    groups = list(IDS)
    gnames = cfg.group_names()
    tpr, fpr = operating_points(P, names, y, c, keep, groups)
    gba = 0.5 * (tpr + (1 - fpr))
    idx = {n: i for i, n in enumerate(names)}
    hard = gba.min(axis=1)
    argmin_g = gba.argmin(axis=1)

    sweep = [n for n in names if n.startswith(args.sweep_prefix)] + (["seed"] if "seed" in idx else [])
    finals = [n for n in names if n.startswith(cfg.final_prefix + ":")]
    if len(sweep) < 3:
        raise SystemExit(f"развёртку строить не из чего: промптов с префиксом {args.sweep_prefix!r} "
                         f"найдено {len(sweep) - 1}")
    print(f"стенд {cfg.key}, {cfg.test_set}: промптов {len(names)}, групп {len(groups)}, "
          f"строк в анализе {int(keep.sum())}")
    print(f"развёртка порога строится по {len(sweep)} промптам ({args.sweep_prefix}* и seed); "
          f"финалов {len(finals)}")

    # --- 1. запас, достижимый одним порогом -------------------------------------------------
    si = [idx[n] for n in sweep]
    strict_sweep = fpr[si].mean(axis=1)
    o = np.argsort(strict_sweep)
    lo, hi = float(strict_sweep[o][0]), float(strict_sweep[o][-1])
    grid = np.linspace(lo, hi, args.grid)
    curve = np.stack([np.interp(grid, strict_sweep[o], gba[si][o, j]) for j in range(len(groups))], axis=1)
    shared = curve.min(axis=1)
    k = int(np.argmax(shared))
    best_sweep = float(shared[k])
    seed_hard = float(hard[idx["seed"]]) if "seed" in idx else float("nan")
    print("\n=== 1. что достижимо ОДНИМ ПОРОГОМ (кривая построена без финалов) ===")
    print(f"  диапазон строгости правок: FPR {lo:.3f}..{hi:.3f}")
    print(f"  лучшая точка развёртки: hard-min {best_sweep:.4f} при строгости {grid[k]:.3f}, "
          f"связывающая группа {gnames.get(groups[int(np.argmin(curve[k]))])}")
    print(f"  стартовый промпт:       hard-min {seed_hard:.4f}")
    print(f"  ЗАПАС ПО ПОРОГУ:        {best_sweep - seed_hard:+.4f}")

    # --- 2. что финалы добавили сверх порога ------------------------------------------------
    print("\n=== 2. что финалы добавили СВЕРХ достижимого порогом ===")
    rows = []
    for n in finals:
        i = idx[n]
        s = float(fpr[i].mean())
        inside = lo <= s <= hi
        rows.append((n, float(hard[i]), float(hard[i]) - best_sweep, s, inside,
                     gnames.get(groups[int(argmin_g[i])])))
    rows.sort(key=lambda r: -r[1])
    print(f"  {'финал':32s} {'hard-min':>9s} {'сверх развёртки':>16s} {'строгость':>10s} "
          f"{'в диапазоне':>12s} {'связывающая':>14s}")
    for n, h, up, s, inside, g in rows:
        print(f"  {n:32s} {h:9.4f} {up:+16.4f} {s:10.3f} {'да' if inside else 'ЭКСТРАП':>12s} {str(g):>14s}")
    best = rows[0]
    above = [r for r in rows if r[2] > 0]
    print(f"\n  лучший финал {best[0]}: hard-min {best[1]:.4f}")
    print(f"  ПОДЪЁМ КРИВОЙ (лучший финал минус развёртка): {best[2]:+.4f}")
    print(f"  финалов выше развёртки: {len(above)} из {len(rows)}")
    print(f"  финалов вне диапазона строгости правок (экстраполяция): "
          f"{sum(not r[4] for r in rows)} из {len(rows)}")

    # --- 3. подмена связывающей группы ------------------------------------------------------
    print("\n=== 3. подмена связывающей группы ===")
    seed_g = gnames.get(groups[int(argmin_g[idx['seed']])]) if "seed" in idx else None
    changed = [r for r in rows if r[5] != seed_g]
    print(f"  связывающая группа стартового промпта: {seed_g}")
    from collections import Counter
    cnt = Counter(r[5] for r in rows)
    print("  связывающая группа финалов: " + ", ".join(f"{k}:{v}" for k, v in cnt.most_common()))
    print(f"  ДОЛЯ ФИНАЛОВ СО СМЕНОЙ СВЯЗЫВАЮЩЕЙ ГРУППЫ: {len(changed)} из {len(rows)} "
          f"({100 * len(changed) / max(1, len(rows)):.0f}%)")
    cnt_all = Counter(gnames.get(groups[int(argmin_g[i])]) for i in range(len(names)))
    print("  для справки, по всем промптам пула: " + ", ".join(f"{k}:{v}" for k, v in cnt_all.most_common()))

    # --- 4. насколько связывающая группа отделена от остальных ------------------------------
    i_seed = idx["seed"]
    srt = np.sort(gba[i_seed])
    print(f"\n  отрыв худшей группы от второй снизу у стартового промпта: {srt[1] - srt[0]:+.4f} "
          f"(худшая {srt[0]:.4f}, вторая {srt[1]:.4f})")
    print("  Чем меньше отрыв, тем чаще argmin перескакивает при ресэмпле и тем шумнее hard-min.")

    out = cfg.outputs / "mechanism_decompose.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "stand": cfg.key, "sweep_prompts": sweep, "sweep_fpr_range": [lo, hi],
        "best_sweep_hard_min": best_sweep, "seed_hard_min": seed_hard,
        "threshold_headroom": best_sweep - seed_hard,
        "best_final": best[0], "best_final_hard_min": best[1], "curve_lift": best[2],
        "finals_above_sweep": len(above), "finals_total": len(rows),
        "finals_extrapolated": sum(not r[4] for r in rows),
        "seed_binding_group": str(seed_g),
        "binding_changed": len(changed), "gap_worst_to_second": float(srt[1] - srt[0]),
        "per_final": [{"name": r[0], "hard_min": r[1], "above_sweep": r[2], "strictness": r[3],
                       "in_range": r[4], "binding": str(r[5])} for r in rows]}, indent=1,
        ensure_ascii=False), encoding="utf-8")
    print(f"\nзаписано {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

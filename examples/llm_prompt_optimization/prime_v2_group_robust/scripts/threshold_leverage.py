#!/usr/bin/env python
"""Рычаг порога: величину можно посчитать ДО трат на оптимизатор.

Зачем. Оптимизатор стоит денег (обращения к API), а есть дешёвая величина, которую можно
посчитать заранее по уже собранным однострочным правкам строгости (`r15:*`, см.
`mechanism_decompose.py`) и которая говорит, сколько вообще есть места для содержательного
улучшения группового профиля, а не только для калибровки порога.

Как считается. Для каждого промпта берётся Δ = метрика(промпт) - метрика(seed) по трём метрикам
`analyze_s11.metric`:
  hard_min      минимум GBA по ГРУППАМ -- то, что работа улучшает.
  worst_class   min(TPR, TNR) -- минимум по ДВУМ КЛАССАМ, то есть чувствительность к тому, где
                стоит порог принятия решения (строже/мягче), а не к тому, как модель различает
                классы внутри группы.
  global_acc    для масштаба.

РЫЧАГ ПОРОГА = средний |Δ| по worst_class / средний |Δ| по hard_min. Если он большой, то то, до
чего дотягивается промпт (по величине типичного сдвига), -- это в основном размен между классами
(порог), а не групповой профиль (hard_min). Промпт, который просто ужесточает или смягчает ответ,
даёт большой worst_class-эффект почти бесплатно; hard_min-эффект -- это то, что остаётся сверх
одной лишь калибровки.

Считается ОТДЕЛЬНО по двум пулам:
  r15:*   однострочные правки строгости. Они по построению меняют только порог (это ровно
          развёртка, на которой строится `mechanism_decompose.py`), поэтому средний |Δ| здесь --
          оценка рычага порога ДО того, как потрачен бюджет на сам оптимизатор.
  finals  готовые финалы оптимизаторов (`cfg.final_prefix + ":"`). Если оптимизация даёт что-то
          сверх калибровки порога, рычаг порога у финалов должен быть МЕНЬШЕ, чем у голых правок
          строгости.

Отдельно печатается запас по порогу: максимум Δ hard_min среди правок r15 (лучшее, что дала одна
лишь развёртка строгости) и его отношение к полуширине доверительного интервала контраста
hard_min на том же пуле -- если запас меньше полуширины, дальше двигать порог бессмысленно: шум
самого контраста больше остатка запаса.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_s11 import IDS, load_preds, load_set, metric, valid_rows  # noqa: E402
from regroup_contrasts import contrasts, summarize  # noqa: E402
from dataset_config import cfg  # noqa: E402

METRICS = ["hard_min", "worst_class", "global_acc"]


def pool_leverage(P: dict, names: list, y, c, keep, groups) -> tuple[dict, dict, dict]:
    """seed-значение, средний |Δ| и максимум Δ по каждой метрике для одного пула промптов."""
    seed_pred = P["seed"]
    seed_val = {m: metric(m, seed_pred, y, c, keep, groups) for m in METRICS}
    delta = {m: np.array([metric(m, P[n], y, c, keep, groups) - seed_val[m] for n in names])
             for m in METRICS}
    mean_abs = {m: float(np.abs(delta[m]).mean()) for m in METRICS}
    max_delta = {m: float(delta[m].max()) for m in METRICS}
    return seed_val, mean_abs, max_delta


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=4000,
                    help="бутстреп для полуширины интервала контраста hard_min (пул r15)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    y, c, _ = load_set(cfg.test_set)
    P = load_preds(cfg.test_set, len(y))
    all_names = sorted(n for n in P if not n.startswith("CONTROL:"))
    keep = valid_rows(P, len(y), all_names)
    groups = tuple(IDS)
    if "seed" not in P:
        raise SystemExit("в матрице нет предсказаний seed -- рычаг порога считать не от чего")

    pools = {
        "r15": [n for n in all_names if n.startswith("r15:")],
        cfg.final_prefix: [n for n in all_names if n.startswith(cfg.final_prefix + ":")],
    }
    print(f"стенд {cfg.key}, {cfg.test_set}: строк в анализе {int(keep.sum())}, групп {len(groups)}")

    result: dict = {"stand": cfg.key, "pools": {}}
    for pool_name, names in pools.items():
        if len(names) < 2:
            print(f"\nпул {pool_name}:* -- промптов {len(names)}, пропущено (мало для среднего)")
            continue
        seed_val, mean_abs, max_delta = pool_leverage(P, names, y, c, keep, groups)
        ratio = mean_abs["worst_class"] / mean_abs["hard_min"]
        print(f"\n=== пул {pool_name}:* ({len(names)} промптов) ===")
        for m in METRICS:
            print(f"  {m:12s} seed {seed_val[m]:.4f}  средний |Δ| {mean_abs[m]:.4f}  "
                  f"макс Δ {max_delta[m]:+.4f}")
        print(f"  РЫЧАГ ПОРОГА (worst_class / hard_min по среднему |Δ|): {ratio:.2f}")
        result["pools"][pool_name] = {"n": len(names), "seed": seed_val,
                                       "mean_abs_delta": mean_abs, "max_delta": max_delta,
                                       "leverage_ratio": ratio}

    # --- запас по порогу против шума контраста (только пул r15) ----------------------------
    r15_names = pools["r15"]
    if len(r15_names) >= 2:
        rows, rej, seed_hm, _ = contrasts(P, r15_names, P["seed"], y, c, groups, "hard_min",
                                           args.n_boot, np.random.default_rng(args.seed), keep)
        s = summarize(rows, rej)
        headroom = max(r[1] for r in rows)
        half_width = s["mean_width"] / 2
        print("\n=== запас по порогу против шума контраста (пул r15, метрика hard_min) ===")
        print(f"  seed hard_min: {seed_hm:.4f}")
        print(f"  максимум Δ hard_min среди правок r15: {headroom:+.4f}")
        print(f"  полуширина интервала контраста hard_min (средняя по r15): {half_width:.4f}")
        print(f"  ОТНОШЕНИЕ запас/полуширина: {headroom / half_width:.2f}")
        result["threshold_headroom_r15"] = {"seed_hard_min": seed_hm, "max_delta_hard_min": headroom,
                                             "half_width_hard_min": half_width,
                                             "headroom_over_half_width": headroom / half_width}
    else:
        print("\nпул r15 слишком мал -- запас по порогу не считается")

    out = cfg.outputs / "threshold_leverage.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nзаписано {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

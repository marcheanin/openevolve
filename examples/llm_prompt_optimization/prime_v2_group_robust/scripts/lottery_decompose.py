"""Какое именно измерение протокола двигает исход отбора.

«Лотерея протокола» до сих пор подавалась как одно число — размах доли значимых побед по
72 протоколам. Это скрывает механизм: протокол задаётся четырьмя независимыми решениями
(аллокация бюджета, статистика отбора, штраф за длину, правило ничьей), и они не равны по
влиянию. План полнофакторный и сбалансированный (2 x 6 x 2 x 3 = 72), поэтому главные
эффекты раскладываются суммами квадратов без остатка на несбалансированность.

Печатаем для каждого отклика (точечная победа, значимая при 95%, значимая после Холма,
средняя опубликованная величина) долю дисперсии на каждое измерение и разницу средних
между уровнями — чтобы «протокол решает исход» можно было заменить на «решает вот это».
"""
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

FIELDS = [("win_point", "точечная победа"), ("win_sig", "значимая при 95%"),
          ("win_holm", "значимая после Холма"), ("mean_published", "опубликованная величина")]
DIMS = ["аллокация", "статистика", "штраф за длину", "ничья"]


def load(path):
    d = json.load(open(path, encoding="utf-8"))
    keys = list(d["protocols"])
    parts = [k.split("|") for k in keys]
    # Берём только поля, записанные ПО КАЖДОМУ протоколу; одноимённые сводки верхнего
    # уровня (медиана, IQR) для разложения не годятся.
    resp = {f: np.array([d["protocols"][k][f] for k in keys], float)
            for f, _ in FIELDS if all(f in d["protocols"][k] for k in keys)}
    return d, keys, parts, resp


def decompose(vals, parts):
    """Доля общей суммы квадратов на каждое измерение (главные эффекты)."""
    grand = vals.mean()
    sst = ((vals - grand) ** 2).sum()
    out = []
    for j, dim in enumerate(DIMS):
        levels = sorted({p[j] for p in parts})
        ss = 0.0
        means = {}
        for lv in levels:
            m = np.array([p[j] == lv for p in parts])
            means[lv] = vals[m].mean()
            ss += m.sum() * (means[lv] - grand) ** 2
        out.append((dim, ss / sst if sst > 0 else 0.0, means))
    return out, sst


def main(path):
    d, keys, parts, resp = load(path)
    print("=" * 96)
    print("%s   цель %s, пул %s (%s промптов), протоколов %d"
          % (Path(path).name, d.get("target"), d.get("pool"), d.get("pool_size"), len(keys)))
    print("=" * 96)
    for f, title in FIELDS:
        if f not in resp:
            print("\n%s: нет в файле" % title)
            continue
        v = resp[f]
        rows, sst = decompose(v, parts)
        print("\n--- %s ---" % title)
        print("  по всем протоколам: медиана %.3f  IQR [%.3f, %.3f]  размах [%.3f, %.3f]"
              % (np.median(v), np.percentile(v, 25), np.percentile(v, 75), v.min(), v.max()))
        print("  %-16s %10s   %s" % ("измерение", "доля SS", "средние по уровням"))
        for dim, share, means in sorted(rows, key=lambda t: -t[1]):
            txt = "  ".join("%s %.3f" % (k, val) for k, val in sorted(means.items(), key=lambda t: -t[1]))
            print("  %-16s %9.1f%%   %s" % (dim, 100 * share, txt))
        explained = sum(s for _, s, _ in rows)
        print("  главные эффекты объясняют %.1f%%, остальное — взаимодействия" % (100 * explained))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)

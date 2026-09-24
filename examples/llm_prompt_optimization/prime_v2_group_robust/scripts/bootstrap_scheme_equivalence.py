"""Где расходятся две реализации бутстрепа для hard_min.

statistic_reproducibility.py при 40 расщеплениях объявляет 56 пар и воспроизводит 66.1%;
statistic_temperature_sweep.py на тех же данных объявляет 38 и воспроизводит 44.7%. На cvar25
и сглаженных обе согласуются. Меньше объявлений = шире интервал = больше дисперсия бутстрепа.
Здесь обе схемы ресэмпла прогоняются на ОДНОЙ И ТОЙ ЖЕ половине и сравниваются напрямую.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"c:/Users/march/things/mipt/AlphaEvolveProject/openevolve/examples/llm_prompt_optimization/prime_v2_group_robust")
sys.path.insert(0, str(ROOT / "scripts"))
import statistic_reproducibility as sr  # noqa: E402

B = 2000
names, preds, y, c = sr.load()
rng = np.random.default_rng(7)

# одна половина, построенная как в оригинале
a_rows = []
for g in sr.IDS:
    for lab in (0, 1):
        cell = np.flatnonzero((c == g) & (y == lab))
        a_rows.append(rng.permutation(cell)[: len(cell) // 2])
A = np.concatenate(a_rows)

# --- схема 1: как в statistic_reproducibility.py
M1 = sr.boot_all(preds, names, A, y, c, np.random.default_rng(1), ["hard_min", "cvar25", "mean_gba"])

# --- схема 2: как в statistic_temperature_sweep.py (ресэмпл по ячейкам «группа x метка»)
pos = [np.flatnonzero((c[A] == g) & (y[A] == 1)) for g in sr.IDS]
neg = [np.flatnonzero((c[A] == g) & (y[A] == 0)) for g in sr.IDS]
r2 = np.random.default_rng(1)
draws = [(r2.integers(0, len(p), size=(B, len(p))), r2.integers(0, len(q), size=(B, len(q))))
         for p, q in zip(pos, neg)]
G = np.empty((len(names), B, len(sr.IDS)))
for i, n in enumerate(names):
    pr = preds[n][A]
    for k, ((dp, dq), p, q) in enumerate(zip(draws, pos, neg)):
        G[i, :, k] = 0.5 * ((pr[p][dp] == 1).mean(axis=1) + (pr[q][dq] == 0).mean(axis=1))
S = np.sort(G, axis=2)
M2 = {"hard_min": S[:, :, 0], "cvar25": S[:, :, :2].mean(axis=2), "mean_gba": G.mean(axis=2)}

print(f"{len(names)} промптов, половина {len(A)} строк, B={sr.N_BOOT} против {B}\n")
print(f"{'метрика':10s} {'sd схемы 1':>12s} {'sd схемы 2':>12s} {'отношение':>10s} "
      f"{'sd разности пар 1':>18s} {'sd разности пар 2':>18s}")
for m in ("hard_min", "cvar25", "mean_gba"):
    s1 = M1[m].std(axis=1).mean()
    s2 = M2[m].std(axis=1).mean()
    d1, d2 = [], []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d1.append((M1[m][i] - M1[m][j]).std())
            d2.append((M2[m][i] - M2[m][j]).std())
    print(f"{m:10s} {s1:12.5f} {s2:12.5f} {s2 / s1:10.3f} {np.mean(d1):18.5f} {np.mean(d2):18.5f}")

print("\nЕсли sd по промпту совпадает, а sd РАЗНОСТИ различается — расходится связанность пар\n"
      "(в одной схеме контрасты спарены сильнее, чем в другой).")

# сколько пар объявлено каждой схемой на этой половине
for m in ("hard_min", "cvar25"):
    n1 = n2 = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            lo, hi = np.percentile(M1[m][i] - M1[m][j], [2.5, 97.5])
            n1 += int(lo > 0 or hi < 0)
            lo, hi = np.percentile(M2[m][i] - M2[m][j], [2.5, 97.5])
            n2 += int(lo > 0 or hi < 0)
    print(f"{m:10s} объявлено: схема 1 = {n1}, схема 2 = {n2} из {len(names)*(len(names)-1)//2}")

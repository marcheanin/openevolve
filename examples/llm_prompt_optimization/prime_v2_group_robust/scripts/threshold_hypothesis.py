"""H_thr прямо из матрицы S11, без единого обращения к API.

Гипотеза: промпт двигает только скалярный порог на неизменном внутреннем скоре модели.
Следствия, проверяемые по бинарным предсказаниям 37 промптов x 8 групп:

  (1) внутри каждой группы точки (FPR, TPR) 37 промптов монотонны и лежат на ОДНОЙ кривой;
  (2) порядок промптов по строгости ОДИН И ТОТ ЖЕ во всех группах (один скаляр на промпт);
  (3) отклонение от кривой не больше биномиального шума ячейки.

Если (2) нарушено — промпт меняет ранжирование внутри группы, а не только порог, и
"скалярная развёртка порога" не заменяет поиск.
"""
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(r"c:/Users/march/things/mipt/AlphaEvolveProject/openevolve/examples/llm_prompt_optimization/prime_v2_group_robust")
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_s11 import IDS, load_preds, load_set, valid_rows  # noqa: E402
from dataset_config import cfg  # noqa: E402

y, c, rec = load_set(cfg.test_set)
P = load_preds(cfg.test_set, len(y))
# Имена групп: у CivilComments фиксированный список, у MultiNLI они в JSON множества.
GN = cfg.group_names(rec)
NAMES = {g: str(GN.get(g, g)) for g in IDS}
names = sorted(n for n in P if not n.startswith("CONTROL:"))
keep = valid_rows(P, len(y), names)
print(f"промптов {len(names)}, строк в анализе {int(keep.sum())}\n")

# --- рабочая точка каждого промпта в каждой группе
tpr = np.zeros((len(names), len(IDS)))
fpr = np.zeros((len(names), len(IDS)))
npos = np.zeros(len(IDS), int)
nneg = np.zeros(len(IDS), int)
for j, g in enumerate(IDS):
    pos = (c == g) & (y == 1) & keep
    neg = (c == g) & (y == 0) & keep
    npos[j], nneg[j] = pos.sum(), neg.sum()
    for i, n in enumerate(names):
        tpr[i, j] = (P[n][pos] == 1).mean()
        fpr[i, j] = (P[n][neg] == 1).mean()

print("ячейки: положительных", npos.tolist(), "\n         отрицательных", nneg.tolist(), "\n")

# --- (1) монотонность внутри группы
print("=== (1) внутри группы: растёт ли TPR вместе с FPR по 37 промптам ===")
print(f"{'группа':18s} {'rho(FPR,TPR)':>13s} {'разброс FPR':>12s} {'разброс TPR':>12s}")
for j, g in enumerate(IDS):
    r = stats.spearmanr(fpr[:, j], tpr[:, j]).statistic
    print(f"{NAMES[g]:18s} {r:13.3f} {np.ptp(fpr[:, j]):12.3f} {np.ptp(tpr[:, j]):12.3f}")

# --- (2) один ли скаляр строгости: согласие порядков между группами
print("\n=== (2) один скаляр на промпт? согласие порядка промптов по FPR между группами ===")
rs = [stats.spearmanr(fpr[:, a], fpr[:, b]).statistic
      for a in range(len(IDS)) for b in range(a + 1, len(IDS))]
rs_t = [stats.spearmanr(tpr[:, a], tpr[:, b]).statistic
        for a in range(len(IDS)) for b in range(a + 1, len(IDS))]
print(f"  по FPR: медиана rho {np.median(rs):.3f}, размах [{min(rs):.3f}, {max(rs):.3f}] ({len(rs)} пар групп)")
print(f"  по TPR: медиана rho {np.median(rs_t):.3f}, размах [{min(rs_t):.3f}, {max(rs_t):.3f}]")

# потолок согласия из-за шума: расщепляем каждую ячейку пополам и меряем rho сам с собой
rng = np.random.default_rng(0)
half_r = []
for _ in range(20):
    f1 = np.zeros_like(fpr); f2 = np.zeros_like(fpr)
    for j, g in enumerate(IDS):
        idx = np.flatnonzero((c == g) & (y == 0) & keep)
        rng.shuffle(idx)
        a, b = idx[: len(idx) // 2], idx[len(idx) // 2:]
        for i, n in enumerate(names):
            f1[i, j] = (P[n][a] == 1).mean()
            f2[i, j] = (P[n][b] == 1).mean()
    half_r += [stats.spearmanr(f1[:, j], f2[:, j]).statistic for j in range(len(IDS))]
rel = float(np.median(half_r))
print(f"  потолок из-за шума (расщепление ячейки пополам, та же группа): медиана rho {rel:.3f}")
print(f"  -> согласие между группами с поправкой на ослабление: {np.median(rs) / rel:.3f}")

# --- (3) отклонение от общей кривой против биномиального шума
print("\n=== (3) лежат ли промпты на одной кривой ROC внутри группы ===")
print(f"{'группа':18s} {'медиана |остаток| TPR':>22s} {'биномиальный шум':>18s} {'отношение':>10s}")
ratios = []
for j, g in enumerate(IDS):
    o = np.argsort(fpr[:, j])
    x, yy = fpr[o, j], tpr[o, j]
    res = []
    for k in range(1, len(x) - 1):
        m = np.ones(len(x), bool); m[k] = False
        if x[m].min() <= x[k] <= x[m].max():
            res.append(abs(yy[k] - np.interp(x[k], x[m], yy[m])))
    res = np.array(res)
    noise = np.sqrt(tpr[:, j].mean() * (1 - tpr[:, j].mean()) / npos[j])
    ratios.append(np.median(res) / noise)
    print(f"{NAMES[g]:18s} {np.median(res):22.4f} {noise:18.4f} {ratios[-1]:10.2f}")
print(f"  медиана отношения по группам: {np.median(ratios):.2f}  (около 1 = кривая объясняет всё)")

# --- что это значит для GBA: разложение на порог и на форму
print("\n=== (4) сколько разброса GBA по промптам объясняет один общий порог ===")
gba = 0.5 * (tpr + (1 - fpr))
strict = fpr.mean(axis=1)  # общая строгость промпта
for j, g in enumerate(IDS):
    sl, ic, r, _, _ = stats.linregress(strict, gba[:, j])
    print(f"{NAMES[g]:18s} R^2 GBA по общей строгости = {r ** 2:.3f}")
sl, ic, r, _, _ = stats.linregress(strict, gba.min(axis=1))
print(f"{'hard-min GBA':18s} R^2 по общей строгости = {r ** 2:.3f}")

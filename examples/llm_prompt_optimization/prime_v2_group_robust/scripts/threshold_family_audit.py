"""Ревизия утверждения «финалы поиска поднимаются над общей кривой, а правки нет».

Первая версия (threshold_who_is_above.py) считала остаток каждого промпта от кривой,
построенной остальными, и сравнивала два семейства. Разбор показал три дыры, которые
здесь закрываются:

  1. ПЕРЕКРЫТИЕ. Утверждение «худший финал выше лучшей правки» неверно; печатаем границы
     семейств явным числом и считаем, сколько финалов лежит ниже лучшей правки.
  2. ПСЕВДОРЕПЛИКИ. Правки thr_pXX дают почти одинаковые предсказания, поэтому n=11 в
     тесте Манна-Уитни раздут. Склеиваем промпты, согласные более чем на DEDUP долю строк,
     и повторяем тест на представителях.
  3. МЕТОД-КОНФАУНД. Все финалы ниже кривой относятся к одному оптимизатору. Считаем
     остатки по методам отдельно и повторяем сравнение без самого сильного промпта.

Плюс печатаем, почему часть промптов вообще выпадает из анализа: остаток определён лишь
там, где рабочая точка промпта лежит ВНУТРИ размаха FPR остальных; у самых крайних по
строгости промптов интерполировать не от чего, и они молча исчезали из таблицы.
"""
import sys
from itertools import combinations
from math import erfc, sqrt
from pathlib import Path

import numpy as np

ROOT = Path(r"c:/Users/march/things/mipt/AlphaEvolveProject/openevolve/examples/llm_prompt_optimization/prime_v2_group_robust")
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_s11 import IDS, load_preds, load_set, valid_rows  # noqa: E402
from dataset_config import cfg  # noqa: E402

DEDUP = 0.99      # доля совпадающих предсказаний, при которой промпты считаются одним
MIN_GROUPS = 4    # сколько групп должно дать определённый остаток

y, c, rec = load_set(cfg.test_set)
P = load_preds(cfg.test_set, len(y))
names = sorted(n for n in P if not n.startswith("CONTROL:"))
keep = valid_rows(P, len(y), names)
print("промптов %d, строк в анализе %d\n" % (len(names), int(keep.sum())))


def method_of(n):
    if n.startswith("r15:"):
        return "правка"
    if n == "seed":
        return "seed"
    tail = n.split(":", 1)[1] if ":" in n else n
    for m in ("evoprompt_de", "gepa", "ape", "prime"):
        if m in tail:
            return m
    return "?"


tpr = np.zeros((len(names), len(IDS)))
fpr = np.zeros((len(names), len(IDS)))
for j, g in enumerate(IDS):
    pos, neg = (c == g) & (y == 1) & keep, (c == g) & (y == 0) & keep
    for i, n in enumerate(names):
        tpr[i, j] = (P[n][pos] == 1).mean()
        fpr[i, j] = (P[n][neg] == 1).mean()

def residuals(ref):
    """Остаток от кривой, построенной по промптам из ref (сам промпт всегда исключается).

    Состав ref небезразличен: восемь почти одинаковых правок thr_pXX стоят в том же
    диапазоне FPR, что и финалы, и, попав в опорный набор, тянут кривую к себе. Поэтому
    считаем дважды — по всем промптам и только по представителям склеек.
    """
    out = np.full((len(names), len(IDS)), np.nan)
    for j in range(len(IDS)):
        for i in range(len(names)):
            m = ref.copy()
            m[i] = False
            if m.sum() < 3:
                continue
            o = np.argsort(fpr[m, j])
            x, yy = fpr[m, j][o], tpr[m, j][o]
            if x.min() <= fpr[i, j] <= x.max():
                out[i, j] = tpr[i, j] - np.interp(fpr[i, j], x, yy)
    return out


res = residuals(np.ones(len(names), bool))

ok = (~np.isnan(res)).sum(axis=1)
print("=== кто выпадает из анализа остатков и почему ===")
dropped = [(names[i], int(ok[i]), float(fpr[i].mean())) for i in range(len(names)) if ok[i] < MIN_GROUPS]
if not dropped:
    print("  никто")
for n, k, s in sorted(dropped, key=lambda t: t[1]):
    print("  %-34s остаток определён в %d/%d группах, средний FPR %.3f" % (n, k, len(IDS), s))
print("  Причина: рабочая точка вне размаха FPR остальных промптов -> интерполировать не от чего.")
print("  Это смещает сравнение семейств: выпадают самые крайние по строгости промпты.\n")

# --- склейка почти одинаковых промптов -------------------------------------------------
parent = {n: n for n in names}


def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


for a, b in combinations(names, 2):
    if float((P[a][keep] == P[b][keep]).mean()) >= DEDUP:
        parent[find(a)] = find(b)
clusters = {}
for n in names:
    clusters.setdefault(find(n), []).append(n)
multi = {k: v for k, v in clusters.items() if len(v) > 1}
print("=== склейка промптов, согласных не менее чем на %d%% строк ===" % round(DEDUP * 100))
if not multi:
    print("  дубликатов нет")
for k, v in sorted(multi.items(), key=lambda t: -len(t[1])):
    print("  %d промптов как один: %s" % (len(v), ", ".join(sorted(v))))
reps = {sorted(v)[0] for v in clusters.values()}
print("  различимых промптов: %d из %d\n" % (len(reps), len(names)))

rng = np.random.default_rng(0)
rows = []
for i, n in enumerate(names):
    r = res[i][~np.isnan(res[i])]
    if len(r) < MIN_GROUPS:
        continue
    obs = float(r.mean())
    null = np.abs((rng.choice([-1, 1], size=(20000, len(r))) * r).mean(axis=1))
    rows.append(dict(name=n, res=obs, p=float((null >= abs(obs)).mean()),
                     method=method_of(n), rep=n in reps, k=len(r)))
rows.sort(key=lambda d: -d["res"])

print("%-34s %12s %7s %13s %6s %13s" % ("промпт", "выше кривой", "p", "метод", "групп", "в тесте"))
for d in rows:
    mark = "да" if d["rep"] else "склеен"
    print("  %-32s %+12.4f %7.3f %13s %6d %13s" % (d["name"], d["res"], d["p"], d["method"], d["k"], mark))

fin = [d for d in rows if d["method"] not in ("правка", "seed")]
edt = [d for d in rows if d["method"] == "правка"]
print("\n=== границы семейств (проверка утверждения «без перекрытия») ===")
if fin and edt:
    best_e = max(edt, key=lambda d: d["res"])
    worst_f = min(fin, key=lambda d: d["res"])
    below = [d for d in fin if d["res"] < best_e["res"]]
    print("  лучшая правка     %-32s %+.4f" % (best_e["name"], best_e["res"]))
    print("  худший финал      %-32s %+.4f" % (worst_f["name"], worst_f["res"]))
    print("  финалов НИЖЕ лучшей правки: %d из %d" % (len(below), len(fin)))
    for d in below:
        print("      %-32s %+.4f  (%s)" % (d["name"], d["res"], d["method"]))
    print("  ПЕРЕКРЫТИЕ СЕМЕЙСТВ: %s" % ("есть" if below else "нет"))


def mw(a, b):
    """Манн-Уитни U двусторонний, нормальное приближение со связями."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan"), float("nan")
    allv = np.concatenate([a, b])
    order = allv.argsort()
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    for k in np.where(cnt > 1)[0]:
        ranks[inv == k] = ranks[inv == k].mean()
    u = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    tie = sum(t ** 3 - t for t in cnt if t > 1)
    sd = np.sqrt(n1 * n2 / 12 * ((n1 + n2 + 1) - tie / ((n1 + n2) * (n1 + n2 - 1))))
    z = (u - mu) / sd if sd > 0 else 0.0
    return float(u), float(erfc(abs(z) / sqrt(2)))


def report(tag, fin_, edt_):
    if len(fin_) < 2 or len(edt_) < 2:
        print("  %-46s слишком мало промптов (%d против %d)" % (tag, len(fin_), len(edt_)))
        return
    u, p = mw([d["res"] for d in fin_], [d["res"] for d in edt_])
    print("  %-46s n=%2d против %2d  медианы %+.4f / %+.4f  U=%6.1f  p=%.2g"
          % (tag, len(fin_), len(edt_),
             np.median([d["res"] for d in fin_]), np.median([d["res"] for d in edt_]), u, p))


print("\n=== финалы против правок: как меняется вывод от того, что считать наблюдением ===")
report("все промпты, как в первой версии", fin, edt)
report("только представители (псевдореплики склеены)", [d for d in fin if d["rep"]], [d for d in edt if d["rep"]])
strong = max(fin, key=lambda d: d["res"])["name"] if fin else None
report("без самого сильного финала (%s)" % strong, [d for d in fin if d["name"] != strong], edt)
report("представители и без самого сильного финала",
       [d for d in fin if d["rep"] and d["name"] != strong], [d for d in edt if d["rep"]])

print("\n=== по оптимизаторам отдельно ===")
for m in sorted({d["method"] for d in fin}):
    sub = [d for d in fin if d["method"] == m]
    print("  %-14s медиана %+.4f  выше нуля %d/%d  значимо выше %d  значимо ниже %d"
          % (m, np.median([d["res"] for d in sub]), sum(d["res"] > 0 for d in sub), len(sub),
             sum(d["res"] > 0 and d["p"] < 0.05 for d in sub),
             sum(d["res"] < 0 and d["p"] < 0.05 for d in sub)))
    report("    %s против всех правок" % m, sub, edt)

# --- кривая без псевдореплик -----------------------------------------------------------
print("\n=== то же, но опорная кривая построена только по представителям ===")
print("Восемь клонов thr_pXX стоят в том же диапазоне FPR, что и финалы. Пока они в опорном")
print("наборе, кривая в этой области проходит через них, и остаток финалов завышен.")
ref = np.array([n in reps for n in names])
res2 = residuals(ref)
rng2 = np.random.default_rng(0)
rows2 = []
for i, n in enumerate(names):
    r = res2[i][~np.isnan(res2[i])]
    if len(r) < MIN_GROUPS:
        continue
    obs = float(r.mean())
    null = np.abs((rng2.choice([-1, 1], size=(20000, len(r))) * r).mean(axis=1))
    rows2.append(dict(name=n, res=obs, p=float((null >= abs(obs)).mean()),
                      method=method_of(n), rep=n in reps))
by = {d["name"]: d for d in rows2}
fin2 = [d for d in rows2 if d["method"] not in ("правка", "seed")]
edt2 = [d for d in rows2 if d["method"] == "правка"]
print("  промптов с определённым остатком: %d" % len(rows2))
print("  заметно вне кривой (p<0.05): %d из %d" % (sum(d["p"] < 0.05 for d in rows2), len(rows2)))
for tag, s in (("финалы", fin2), ("правки", edt2)):
    if s:
        print("  %-8s медиана %+.4f, размах [%+.4f, %+.4f]"
              % (tag, np.median([d["res"] for d in s]),
                 min(d["res"] for d in s), max(d["res"] for d in s)))
if "seed" in by:
    print("  %-8s %+.4f" % ("seed", by["seed"]["res"]))
report("финалы против правок, опора = представители", fin2, edt2)
report("то же, правки тоже только представители", fin2, [d for d in edt2 if d["rep"]])
for m in sorted({d["method"] for d in fin2}):
    sub = [d for d in fin2 if d["method"] == m]
    print("  %-14s медиана %+.4f  выше нуля %d/%d  значимо выше %d  значимо ниже %d"
          % (m, np.median([d["res"] for d in sub]), sum(d["res"] > 0 for d in sub), len(sub),
             sum(d["res"] > 0 and d["p"] < 0.05 for d in sub),
             sum(d["res"] < 0 and d["p"] < 0.05 for d in sub)))

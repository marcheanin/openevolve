"""Кто из промптов реально различает лучше, а кто просто стоит в другой точке порога.

Под H_thr все промпты лежат на одной кривой ROC внутри каждой группы. Промпт, который
систематически ВЫШЕ кривой, построенной остальными, действительно поднимает различающую
способность; промпт на кривой лишь сдвинул рабочую точку. Остаток усредняется по 8 группам,
значимость — перестановкой знаков остатков по группам.
"""
import sys
from pathlib import Path

import numpy as np

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

tpr = np.zeros((len(names), len(IDS)))
fpr = np.zeros((len(names), len(IDS)))
for j, g in enumerate(IDS):
    pos, neg = (c == g) & (y == 1) & keep, (c == g) & (y == 0) & keep
    for i, n in enumerate(names):
        tpr[i, j] = (P[n][pos] == 1).mean()
        fpr[i, j] = (P[n][neg] == 1).mean()

res = np.full((len(names), len(IDS)), np.nan)
for j in range(len(IDS)):
    for i in range(len(names)):
        m = np.ones(len(names), bool); m[i] = False
        o = np.argsort(fpr[m, j])
        x, yy = fpr[m, j][o], tpr[m, j][o]
        if x.min() <= fpr[i, j] <= x.max():
            res[i, j] = tpr[i, j] - np.interp(fpr[i, j], x, yy)

rng = np.random.default_rng(0)
gba = 0.5 * (tpr + (1 - fpr))
rows = []
for i, n in enumerate(names):
    r = res[i][~np.isnan(res[i])]
    if len(r) < 4:
        continue
    obs = r.mean()
    null = np.abs((rng.choice([-1, 1], size=(20000, len(r))) * r).mean(axis=1))
    rows.append((n, obs, float((null >= abs(obs)).mean()), fpr[i].mean(), gba[i].min()))

rows.sort(key=lambda t: -t[1])
print(f"{'промпт':22s} {'выше кривой':>12s} {'p (знаки)':>10s} {'строгость FPR':>14s} {'hard-min GBA':>13s}")
for n, obs, p, s, hm in rows:
    star = " <-" if n in ("seed", "s9:44_prime", "s9:42_prime", "r15:nl_lenient_max") else ""
    print(f"{n:22s} {obs:+12.4f} {p:10.3f} {s:14.3f} {hm:13.4f}{star}")

sig = [r for r in rows if r[2] < 0.05]
print(f"\nпромптов заметно выше/ниже общей кривой: {len(sig)} из {len(rows)} (без поправки на множественность)")
print("Ожидание при чистом H_thr и 37 промптах: около 2 ложных при alpha=0.05.")

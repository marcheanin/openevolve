"""Насколько устойчив вывод IV.4 «однострочные правки бьют финалы оптимизаторов».

На полном тесте (gemma, truth_large): правок выше seed 64%, финалов 20%. На валидации
(dev_universe, тот же скорер, те же промпты) — 0% и 4%. Но это РАЗНЫЕ сплиты CivilComments
(test и validation), поэтому расхождение может быть свойством сплита, а не розыгрыша.
Здесь вывод пересчитывается на 12 случайных половинах ВНУТРИ каждого множества: если доля
скачет и внутри одного сплита, дело в розыгрыше; если внутри устойчива, а между множествами
различается — дело в сплите.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"c:/Users/march/things/mipt/AlphaEvolveProject/openevolve/examples/llm_prompt_optimization/prime_v2_group_robust")
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_s11 import IDS, load_preds, load_set, metric, valid_rows  # noqa: E402

N_HALF = 12
for set_name in ("truth_large", "dev_universe"):
    y, c, rec = load_set(set_name)
    P = load_preds(set_name, len(y))
    names = sorted(n for n in P if not n.startswith("CONTROL:"))
    keep = valid_rows(P, len(y), names)
    edits = [n for n in names if n.startswith("r15:")]
    finals = [n for n in names if n.startswith("s9:")]
    rng = np.random.default_rng(0)
    cells = [np.flatnonzero((c == g) & (y == lab) & keep) for g in IDS for lab in (0, 1)]

    def shares(mask):
        sv = metric("cvar25", P["seed"], y, c, rows=mask)
        e = np.mean([metric("cvar25", P[n], y, c, rows=mask) > sv for n in edits])
        f = np.mean([metric("cvar25", P[n], y, c, rows=mask) > sv for n in finals])
        return 100 * e, 100 * f, sv

    fe, ff, fs = shares(keep)
    print(f"\n=== {set_name} ({rec['source_split']}), {int(keep.sum())} строк ===")
    print(f"полное множество: правок выше seed {fe:.0f}% ({len(edits)}), "
          f"финалов {ff:.0f}% ({len(finals)}), seed cvar25 {fs:.4f}")
    E, F = [], []
    for h in range(N_HALF):
        m = np.zeros(len(y), bool)
        for cell in cells:
            p = rng.permutation(cell)
            m[p[: len(p) // 2]] = True
        e, f, s = shares(m)
        E.append(e); F.append(f)
        print(f"  половина #{h + 1:2d}: правок {e:5.0f}%  финалов {f:5.0f}%  seed {s:.4f}")
    E, F = np.array(E), np.array(F)
    print(f"  по 12 половинам: правки медиана {np.median(E):.0f}% размах {E.min():.0f}..{E.max():.0f}%; "
          f"финалы медиана {np.median(F):.0f}% размах {F.min():.0f}..{F.max():.0f}%")
    print(f"  правки выше финалов в {100 * np.mean(E > F):.0f}% половин")

#!/usr/bin/env python3
"""Несёт ли вес ρ между скорерами по эффектам на целевую группу (CivilComments, 25 финалов)?

Ранговая корреляция эффектов между двумя скорерами ограничена сверху надёжностью оценки эффекта у
каждого из них: ρ_набл ≈ ρ_истин · sqrt(r1 · r2). Если у одного скорера надёжность около нуля, то
ρ ≈ 0 получится и при полном переносе, и такое ρ не свидетельствует ни о чём.

Надёжность — как в compare_scorers.py: строки теста делятся пополам внутри ячеек группа × метка,
эффекты «финал − seed» считаются на каждой половине, берётся ρ Спирмена между половинами (по 25
финалам), среднее по 500 разбиениям, затем Спирмен–Браун до полной длины. Отрицательное
половинное ρ означает нулевую надёжность; Спирмен–Браун тогда не применяется.

Строки: исключаются те, где хоть один из 37 промптов у хоть одного скорера дал INVALID (16 из 3600).
ρ — scipy со средними рангами при совпадениях. Опубликованное 0,011 было посчитано ранговой
корреляцией без учёта совпадений; со средними рангами то же вычисление даёт 0,036.

Выход: results/S11_protocol_matrix/target_rho_reliability.json и таблица в stdout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_s11 import IDS, cell_index, metric  # noqa: E402
from compare_scorers import OUT, load_first, load_second  # noqa: E402

REPS = 500
KINDS = ("target", "hard_min", "worst_class", "mean_gba")


def main() -> int:
    rec = json.loads((OUT / "fixed_sets" / "truth_large.json").read_text(encoding="utf-8"))
    p2, _ = load_second("gpt4omini", "truth_large")
    L = min(len(v) for v in p2.values())
    p1 = load_first("truth_large", L)
    names = sorted(set(p1) & set(p2))
    y, c = np.asarray(rec["labels"])[:L], np.asarray(rec["cluster_ids"])[:L]
    keep = np.ones(L, bool)
    for n in names:
        keep &= (p1[n][:L] >= 0) & (p2[n][:L] >= 0)
    finals = [n for n in names if n.startswith("s9:")]

    def gba(pred, g, rows):
        pos, neg = rows & (c == g) & (y == 1), rows & (c == g) & (y == 0)
        return 0.5 * ((pred[pos] == 1).mean() + (pred[neg] == 0).mean())

    seed_g = {g: gba(p1["seed"][:L], g, keep) for g in IDS}
    target = min(seed_g, key=seed_g.get)

    def effects(P, kind, rows):
        f = (lambda p: gba(p[:L], target, rows)) if kind == "target" else \
            (lambda p: metric(kind, p[:L], y, c, rows=rows))
        s = f(P["seed"])
        return np.array([f(P[n]) - s for n in finals])

    rng = np.random.default_rng(0)
    cells = cell_index(y, c, keep)
    halves = []
    for _ in range(REPS):
        a, b = np.zeros(L, bool), np.zeros(L, bool)
        for rows in cells.values():
            perm = rng.permutation(rows)
            a[perm[: len(perm) // 2]] = True
            b[perm[len(perm) // 2:]] = True
        halves.append((a, b))

    def reliability(P, kind):
        r = float(np.mean([spearmanr(effects(P, kind, a), effects(P, kind, b)).statistic for a, b in halves]))
        return r, (2 * r / (1 + r) if r > 0 else 0.0)

    res = {"rows": int(keep.sum()), "finals": len(finals), "target_group_id": int(target), "reps": REPS, "rows_by": {}}
    print(f"строк {int(keep.sum())} из {L}; финалов {len(finals)}; целевая группа id {target} "
          f"(GBA seed на gemma {seed_g[target]:.4f})\n")
    print(f"{'величина':12s} {'ρ кросс':>8s} {'r½ gemma':>9s} {'r gemma':>8s} {'r½ gpt':>8s} {'r gpt':>6s} {'потолок ρ':>10s}")
    for kind in KINDS:
        rho = float(spearmanr(effects(p1, kind, keep), effects(p2, kind, keep)).statistic)
        h1, r1 = reliability(p1, kind)
        h2, r2 = reliability(p2, kind)
        ceil = float(np.sqrt(r1 * r2))
        res["rows_by"][kind] = {"rho_cross": rho, "half_rho_gemma": h1, "rel_gemma": r1,
                                "half_rho_gpt": h2, "rel_gpt": r2, "rho_ceiling": ceil}
        print(f"{kind:12s} {rho:8.3f} {h1:9.3f} {r1:8.3f} {h2:8.3f} {r2:6.3f} {ceil:10.3f}")
    print("\nпотолок ρ = sqrt(r gemma · r gpt): наибольшее ρ, которое можно увидеть даже при полном переносе")

    out = ROOT / "results" / "S11_protocol_matrix" / "target_rho_reliability.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

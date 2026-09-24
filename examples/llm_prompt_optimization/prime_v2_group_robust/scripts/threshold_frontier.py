#!/usr/bin/env python
"""What a single shared threshold can reach, against what per-group thresholds could.

`threshold_hypothesis.py` shows the prompts of the pool are one knob: their operating points
lie on one ROC curve per group and their strictness order is the same in every group. The
question this asks is what that costs. A prompt sets ONE threshold for every group at once,
so the reachable worst-group accuracy is

    max_tau  min_g  GBA_g(tau),

whereas a system allowed to threshold each group separately reaches

    min_g  max_tau_g  GBA_g(tau_g).

The gap between the two is the price of the control having the wrong dimensionality: it is
not a search failure and no amount of prompt optimisation closes it. Both quantities are
computed from the same matrix, by interpolating each group's ROC through the pool's own
operating points, so the answer is about what THIS pool can reach.
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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", type=int, default=400, help="точек развёртки общего порога")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    y, c, _ = load_set(cfg.test_set)
    P = load_preds(cfg.test_set, len(y))
    names = sorted(n for n in P if not n.startswith("CONTROL:"))
    keep = valid_rows(P, len(y), names)
    groups = list(IDS)
    tpr, fpr = operating_points(P, names, y, c, keep, groups)
    gba = 0.5 * (tpr + (1 - fpr))
    gnames = cfg.group_names()

    # Общая ось строгости: у каждого промпта одно число, поэтому берём среднюю по группам
    # долю ложных тревог. Проверено отдельно (threshold_hypothesis.py), что порядок промптов
    # по этой оси один и тот же во всех группах.
    strict = fpr.mean(axis=1)
    o = np.argsort(strict)
    grid = np.linspace(strict[o].min(), strict[o].max(), args.grid)
    # GBA каждой группы вдоль общей оси: кривая группы, прочитанная в точках развёртки
    curve = np.stack([np.interp(grid, strict[o], gba[o, j]) for j in range(len(groups))], axis=1)

    shared_min = curve.min(axis=1)
    k = int(np.argmax(shared_min))
    best_shared = float(shared_min[k])
    per_group_best = gba.max(axis=0)
    best_per_group = float(per_group_best.min())

    print(f"{len(names)} промптов, {len(groups)} групп, строк в анализе {int(keep.sum())}\n")
    print("=== один общий порог против отдельного порога на группу ===")
    print(f"{'группа':18s} {'GBA в лучшей общей точке':>25s} {'лучшее своим порогом':>22s} {'потеря':>9s}")
    loss = []
    for j, g in enumerate(groups):
        a, b = curve[k, j], per_group_best[j]
        loss.append(b - a)
        print(f"{gnames.get(g, g):18s} {a:25.4f} {b:22.4f} {b - a:9.4f}")
    print(f"\nхудшая группа при одном общем пороге:     {best_shared:.4f}  "
          f"(строгость {grid[k]:.3f}, худшая — {gnames.get(groups[int(np.argmin(curve[k]))])})")
    print(f"худшая группа при своём пороге на группу: {best_per_group:.4f}")
    print(f"РАЗРЫВ (цена одномерности контроля):      {best_per_group - best_shared:.4f}")

    # Лучший реальный промпт пула — для сравнения с идеализированной развёрткой
    real = gba.min(axis=1)
    i_best = int(np.argmax(real))
    print(f"\nлучший промпт пула по худшей группе: {names[i_best]} {real[i_best]:.4f}")
    print(f"идеальная развёртка общего порога:   {best_shared:.4f} "
          f"(выигрыш развёртки над лучшим промптом {best_shared - real[i_best]:+.4f})")
    print(f"медиана пула:                        {np.median(real):.4f} "
          f"(выигрыш развёртки над медианой {best_shared - np.median(real):+.4f})")

    # Доверительный интервал разрыва: ресэмпл строк внутри ячеек «группа x метка»
    rng = np.random.default_rng(0)
    cells = [(np.flatnonzero((c == g) & (y == 1) & keep),
              np.flatnonzero((c == g) & (y == 0) & keep)) for g in groups]
    draws = [(rng.integers(0, len(p), size=(args.n_boot, len(p))),
              rng.integers(0, len(q), size=(args.n_boot, len(q)))) for p, q in cells]
    gaps = np.empty(args.n_boot)
    T = np.empty((len(names), args.n_boot, len(groups)))
    F = np.empty((len(names), args.n_boot, len(groups)))
    for i, n in enumerate(names):
        for j, ((dp, dq), (p, q)) in enumerate(zip(draws, cells)):
            T[i, :, j] = (P[n][p][dp] == 1).mean(axis=1)
            F[i, :, j] = (P[n][q][dq] == 1).mean(axis=1)
    Gb = 0.5 * (T + (1 - F))
    for b in range(args.n_boot):
        st = F[:, b, :].mean(axis=1)
        ob = np.argsort(st)
        gr = np.linspace(st[ob].min(), st[ob].max(), args.grid)
        cv = np.stack([np.interp(gr, st[ob], Gb[ob, b, j]) for j in range(len(groups))], axis=1)
        gaps[b] = Gb[:, b, :].max(axis=0).min() - cv.min(axis=1).max()
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    print(f"\nразрыв, 95%: [{lo:+.4f}, {hi:+.4f}] (парный ресэмпл внутри ячеек)")
    # Разрыв неотрицателен по построению (максимум по группе не меньше максимума общей точки),
    # поэтому «нижняя граница выше нуля» ничего не проверяет. Сравниваем с тем, что вообще
    # различимо на этом тесте: ширина 95%-го интервала контраста по hard-min около 0.05.
    resolvable = 0.05
    verdict = ("ПРЕНЕБРЕЖИМ" if hi < resolvable / 2 else
               "сопоставим с разрешающей способностью теста" if hi < resolvable else "ЗАМЕТЕН")
    print(f"сравнение: ширина интервала контраста по hard-min на этом тесте ≈ {resolvable:.3f}")
    print(f"вывод: разрыв {verdict}.")
    if hi < resolvable:
        print("То есть на этом бенчмарке один общий порог достигает практически того же, что "
              "дали бы отдельные пороги на группу. Утверждение «worst-group робастность\n"
              "структурно недостижима промптом из-за одномерности контроля» этими данными\n"
              "НЕ подтверждается и писать его нельзя. Потолок создаёт не размерность ручки,\n"
              "а форма кривой в связывающей группе: она низкая при любом пороге.")

    out = cfg.outputs / "threshold_frontier.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"best_shared": best_shared, "best_per_group": best_per_group,
         "gap": best_per_group - best_shared, "gap_ci95": [float(lo), float(hi)],
         "best_prompt": names[i_best], "best_prompt_value": float(real[i_best]),
         "pool_median": float(np.median(real)),
         "per_group_best": {str(g): float(v) for g, v in zip(groups, per_group_best)},
         "loss_per_group": {str(g): float(v) for g, v in zip(groups, loss)}}, indent=2),
        encoding="utf-8")
    print(f"\nзаписано {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

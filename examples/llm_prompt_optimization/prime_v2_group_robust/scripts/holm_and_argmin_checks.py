#!/usr/bin/env python
"""Two small robustness checks that `06_OPEN.md` section 5 lists as owed.

1. Holm family sensitivity. The study corrects a family of 36 contrasts, every final against
   the seed. A reader can fairly ask what happens if the family is every pair of the pool
   instead -- 666 of them -- since nothing about the seed makes it the privileged comparison.
   A larger family corrects harder, so the count of surviving differences can only drop; this
   measures by how much, for every metric.

2. Which group wins the minimum, and how stable that is. `hard_min` is the minimum over eight
   noisy group estimates, so the group it lands on is itself a random variable, and the
   optimism of the minimum was so far bounded rather than measured. Resampling gives both:
   how often the observed argmin group stays the argmin, and the gap between the observed
   minimum and the average minimum over resamples.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_s11 import (IDS, Bootstrap, cell_index, holm, load_preds,  # noqa: E402
                         load_set, valid_rows)
from dataset_config import cfg  # noqa: E402

METRICS = ["hard_min", "cvar25", "mean_gba", "worst_class", "global_acc"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=4000)
    # 666 тестов требуют p до 7.5e-5, то есть не меньше ~13 300 повторов только чтобы порог
    # стал достижим; берём с запасом.
    ap.add_argument("--pairs-boot", type=int, default=40000)
    args = ap.parse_args()

    y, c, _ = load_set(cfg.test_set)
    P = load_preds(cfg.test_set, len(y))
    names = sorted(n for n in P if not n.startswith("CONTROL:"))
    keep = valid_rows(P, len(y), names)
    bs = Bootstrap(y, c, args.n_boot, np.random.default_rng(1),
                   cells=cell_index(y, c, keep))
    obs, boot = {}, {}
    for n in names:
        obs[n], boot[n] = bs.metrics(P[n])
    gnames = cfg.group_names()
    print(f"{len(names)} промптов, строк {int(keep.sum())}, повторов бутстрепа {args.n_boot}\n")

    # ---------------- 1. семейство Холма: 36 против seed или все 666 пар
    print("=== 1. чувствительность к выбору семейства для поправки Холма ===")
    others = [n for n in names if n != "seed"]
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    # Бутстреп-оценка p не бывает меньше 1/B, а первый порог Холма при 666 тестах равен
    # 0.05/666 = 7.5e-5. При B = 4000 пол оценки 2.5e-4 выше порога, и НИ ОДИН контраст не
    # может быть отвергнут по построению — это артефакт разрешения, а не свойство данных.
    need = int(np.ceil(len(pairs) / 0.05))
    print(f"пар {len(pairs)}; первый порог Холма {0.05 / len(pairs):.2e}; "
          f"пол бутстреп-оценки p при B={args.n_boot} равен {1 / args.n_boot:.2e}")
    if args.n_boot <= need:
        print(f"  ВНИМАНИЕ: чтобы отвергнуть хоть что-то в семье из {len(pairs)} тестов, нужно "
              f"B > {need}. Большая семья считается на B={args.pairs_boot}.")
    print(f"{'метрика':12s} {'семья 36 против seed':>22s} {'семья 666: против seed':>24s} "
          f"{'все 666 пар':>13s}")
    out = {}
    # Общая схема ресэмпла для большой семьи: индексы не храним, а порождаем заново с тем же
    # зерном для каждого промпта — так пары остаются связанными, а память не растёт с B.
    cells_pn = [(np.flatnonzero((c == g) & (y == 1) & keep),
                 np.flatnonzero((c == g) & (y == 0) & keep)) for g in IDS]

    def big_boot(name):
        r = np.random.default_rng(12345)
        G = np.empty((args.pairs_boot, len(IDS)))
        for k, (p, q) in enumerate(cells_pn):
            dp = r.integers(0, len(p), size=(args.pairs_boot, len(p)))
            dq = r.integers(0, len(q), size=(args.pairs_boot, len(q)))
            G[:, k] = 0.5 * ((P[name][p][dp] == 1).mean(axis=1)
                             + (P[name][q][dq] == 0).mean(axis=1))
        s = np.sort(G, axis=1)
        pos = np.concatenate([p for p, _ in cells_pn])
        neg = np.concatenate([q for _, q in cells_pn])
        rp = r.integers(0, len(pos), size=(args.pairs_boot, len(pos)))
        rn = r.integers(0, len(neg), size=(args.pairs_boot, len(neg)))
        tp = (P[name][pos][rp] == 1).mean(axis=1)
        tn = (P[name][neg][rn] == 0).mean(axis=1)
        return {"hard_min": s[:, 0], "cvar25": s[:, : max(1, len(IDS) // 4)].mean(axis=1),
                "mean_gba": G.mean(axis=1), "worst_class": np.minimum(tp, tn),
                "global_acc": 0.5 * (tp + tn)}

    big = {n: big_boot(n) for n in names}
    for m in METRICS:
        p_seed = []
        for n in others:
            d = boot[n][m] - boot["seed"][m]
            p_seed.append(max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0 / args.n_boot))
        rej36 = holm(np.array(p_seed)).sum()

        p_all, is_seed_pair = [], []
        for a, b in pairs:
            d = big[a][m] - big[b][m]
            p_all.append(max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0 / args.pairs_boot))
            is_seed_pair.append(a == "seed" or b == "seed")
        rej_all = holm(np.array(p_all))
        seed_in_big = int(rej_all[np.array(is_seed_pair)].sum())
        out[m] = {"holm36": int(rej36), "holm666_seed_pairs": seed_in_big,
                  "holm666_all": int(rej_all.sum()), "n_pairs": len(pairs),
                  "pairs_boot": args.pairs_boot}
        print(f"{m:12s} {f'{int(rej36)}/36':>22s} {f'{seed_in_big}/36':>24s} "
              f"{f'{int(rej_all.sum())}/{len(pairs)}':>13s}")
    print("\nСемья из 666 пар корректирует сильнее, поэтому число выживших контрастов против seed\n"
          "может только упасть. Если оно не падает, вывод не зависит от привилегированности seed.")

    # ---------------- 2. какая группа даёт минимум и насколько это устойчиво
    print("\n=== 2. устойчивость argmin: какая группа даёт hard-min ===")
    gvals = {}
    for n in names:
        g = np.array([0.5 * ((P[n][(c == gg) & (y == 1) & keep] == 1).mean()
                             + (P[n][(c == gg) & (y == 0) & keep] == 0).mean()) for gg in IDS])
        gvals[n] = g
    rng = np.random.default_rng(2)
    cells = [(np.flatnonzero((c == g) & (y == 1) & keep),
              np.flatnonzero((c == g) & (y == 0) & keep)) for g in IDS]
    draws = [(rng.integers(0, len(p), size=(args.n_boot, len(p))),
              rng.integers(0, len(q), size=(args.n_boot, len(q)))) for p, q in cells]
    stay, bias, winners = [], [], []
    for n in names:
        G = np.empty((args.n_boot, len(IDS)))
        for k, ((dp, dq), (p, q)) in enumerate(zip(draws, cells)):
            G[:, k] = 0.5 * ((P[n][p][dp] == 1).mean(axis=1) + (P[n][q][dq] == 0).mean(axis=1))
        am = G.argmin(axis=1)
        obs_am = int(gvals[n].argmin())
        stay.append(float((am == obs_am).mean()))
        bias.append(float(gvals[n].min() - G.min(axis=1).mean()))
        winners.append(obs_am)
    stay, bias = np.array(stay), np.array(bias)
    from collections import Counter
    cnt = Counter(IDS[w] for w in winners)
    print(f"наблюдённая группа-минимум остаётся минимумом при ресэмпле: медиана "
          f"{100 * np.median(stay):.0f}%, размах {100 * stay.min():.0f}..{100 * stay.max():.0f}%")
    print(f"смещение минимума (наблюдённый минус средний по ресэмплам): медиана {np.median(bias):+.4f}, "
          f"размах {bias.min():+.4f}..{bias.max():+.4f}")
    print("группа, дающая минимум (по промптам): "
          + ", ".join(f"{gnames.get(g, g)} {k}" for g, k in cnt.most_common()))
    print(f"разных групп в роли минимума: {len(cnt)} из {len(IDS)}")
    print("\nСмещение положительно: наблюдённый минимум систематически оптимистичнее среднего\n"
          "по ресэмплам, потому что минимум восьми шумных оценок ловит благоприятный шум.\n"
          "Величина смещения — цена того, что метрика выбирает группу по данным.")

    res = {"holm": out, "argmin_stay_median": float(np.median(stay)),
           "argmin_bias_median": float(np.median(bias)),
           "argmin_groups": {str(g): k for g, k in cnt.items()}}
    (cfg.outputs / "holm_and_argmin_checks.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nзаписано {cfg.outputs / 'holm_and_argmin_checks.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

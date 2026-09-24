#!/usr/bin/env python
"""Контрасты промптов против стартового промпта по НАСТОЯЩИМ смысловым надгруппам.

Зачем. Возражение против всей worst-group-части работы звучит так: «число групп k -- это
свободный рычаг экспериментатора, значит весь результат об отрыве худшей группы -- это просто
выбор удобного k». `regroup_contrasts.py --mode pseudo` отвечает на соседний вопрос (разрушает
разнородность групп, сохраняя k), но не на этот: псевдогруппы при укрупнении СЛУЧАЙНО перемешивают
строки, и вместе с k меняется ещё и то, насколько группы вообще разные по трудности -- отрыв
худшей группы от следующей размывается заодно с числом групп, и нельзя сказать, что из двух это
сделало.

Здесь k меняется, а состав групп -- нет: несколько надгрупп настоящих групп стенда, каждая
надгруппа -- объединение исходных категорий по ВНЕШНЕЙ таксономии бенчмарка (разметка Jigsaw,
официальное разбиение MultiNLI, регистр текста). ЭТО ПРИНЦИПИАЛЬНО: надгруппы задаются
таксономией, известной до всякого прогона модели, а НЕ тем, где ошибается стартовый промпт.
Группировка «худшая надгруппа -- там, где seed ошибается» гарантированно даст отрыв, потому что
худшая группа отобрана по результату; это ровно тот класс ошибки post-hoc-группировки, который
работа ловит в чужих текстах, и здесь он запрещён так же, как в regroup_contrasts.py.

Предсказания, модель, точность и данные те же, что и в родном стенде; меняется только то, какие
из настоящих категорий считаются одной надгруппой при вычислении hard-min (или другой метрики).

Схемы (SCHEMES, ключ верхнего уровня -- cfg.key):
  civil real8       None = настоящие 8 групп (контроль, k как в остальной работе).
  civil sem3        три семьи идентичностей в разметке Jigsaw: пол/ориентация, религия, раса.
  civil sem2_relig  религия против всех остальных семей идентичности.
  mnli  real10      None = настоящие 10 жанров (контроль).
  mnli  matched2    официальное разбиение MultiNLI на matched и mismatched.
  mnli  spoken2     транскрипты речи (facetoface, telephone) против письменных жанров.
  mnli  reg5        пятёрки: жанры сгруппированы по регистру, худший (verbatim) не растворяется
                    в четырёх остальных.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_s11 import IDS, load_preds, load_set, valid_rows, gba_by_group  # noqa: E402
from regroup_contrasts import contrasts, summarize  # noqa: E402
from dataset_config import cfg  # noqa: E402

SCHEMES = {
    "civil": {
        "real8": None,  # настоящие 8 групп стенда (контроль)
        # три семьи идентичностей в разметке Jigsaw: пол/ориентация | религия | раса
        "sem3": {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3},
        # религия против всего остального
        "sem2_relig": {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 1, 8: 1},
    },
    "mnli": {
        "real10": None,  # настоящие 10 жанров стенда (контроль)
        # matched против mismatched -- разбиение задано самим бенчмарком MultiNLI
        "matched2": {2: 1, 3: 1, 7: 1, 8: 1, 9: 1, 1: 2, 4: 2, 5: 2, 6: 2, 10: 2},
        # устная речь против письменной: facetoface и telephone -- транскрипты
        "spoken2": {1: 1, 8: 1, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 9: 2, 10: 2},
        # пятёрки: жанры сгруппированы по регистру, худший (verbatim) не растворяется в четырёх других
        "reg5": {1: 1, 8: 1, 2: 2, 4: 2, 3: 3, 6: 3, 7: 4, 9: 4, 5: 5, 10: 5},
    },
    "toxlang": {
        "real6": None,  # настоящие 6 языков стенда (контроль)
        # ресурсный уровень языка -- внешний признак, не зависящий от того, где ошибается
        # модель: высокий (en, de, ru) | средний (ar, hi) | низкий (am)
        "tier3": {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3},
        # письменность: латиница (en, de) против всего остального
        "script2": {1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 2},
    },
}


def run(scheme_name: str, mapping: dict | None, pool: str, metric_name: str, n_boot: int, seed: int) -> dict:
    """Один прогон: одна схема группировки x один пул. Печатает и возвращает сводку."""
    y, c, _ = load_set(cfg.test_set)
    P = load_preds(cfg.test_set, len(y))
    every = sorted(n for n in P if not n.startswith("CONTROL:"))
    keep = valid_rows(P, len(y), every)
    names = [n for n in every if n != "seed"]
    if pool == "finals":
        names = [n for n in names if n.startswith(cfg.final_prefix + ":")]
    if mapping is None:
        cc, groups = c, tuple(IDS)
    else:
        cc = np.zeros_like(c)
        for src, dst in mapping.items():
            cc[c == src] = dst
        groups = tuple(sorted(set(mapping.values())))

    print("=" * 96)
    print(f"стенд {cfg.key} | группировка {scheme_name} | k={len(groups)} | пул {pool} N={len(names)} "
          f"| метрика {metric_name}")
    g = gba_by_group(P["seed"], y, cc, keep, groups)
    vals = np.sort(np.array(list(g.values()), float))
    ncell = {int(gg): int(((cc == gg) & keep).sum()) for gg in groups}
    print("  строк в надгруппе: " + ", ".join(f"{k}:{v}" for k, v in ncell.items()))
    print("  GBA seed по надгруппам: " + ", ".join(f"{k}:{v:.4f}" for k, v in g.items()))
    print(f"  худшая {vals[0]:.4f}  вторая снизу {vals[1]:.4f}  ОТРЫВ {vals[1] - vals[0]:+.4f}  "
          f"размах {vals[-1] - vals[0]:.4f}")

    rows, rej, seed_m, bs = contrasts(P, names, P["seed"], y, cc, groups, metric_name, n_boot,
                                       np.random.default_rng(seed), keep)
    s = summarize(rows, rej)
    print(f"  seed {metric_name} {seed_m:.4f} | ширина {s['mean_width']:.4f} "
          f"(полуширина {s['mean_width'] / 2:.4f}) | средний |Δ| {s['mean_abs_d']:.4f}")
    print(f"  различимо на 95%: {s['res95']} из {s['n']} | побед: {s['wins95']} | "
          f"ХОЛМ: {s['holm']} (побед {s['wins_holm']})")

    # шумовой потолок «лучший из N» под ЭТОЙ ЖЕ группировкой (см. best_of_n_ceiling.py)
    obs_s, boot_s = bs.metrics(P["seed"][keep], groups=groups)
    obs, boot = [], []
    for n in names:
        o, b = bs.metrics(P[n][keep], groups=groups)
        obs.append(o[metric_name] - obs_s[metric_name])
        boot.append(b[metric_name] - boot_s[metric_name])
    obs = np.array(obs)
    boot = np.stack(boot, 0)
    noise = boot - obs[:, None]
    mx = noise.max(axis=0)
    i = int(np.argmax(obs))
    best = float(obs[i])
    p_max = float((mx >= best).mean())
    print(f"  лучший {names[i]} Δ={best:+.4f} | потолок «лучший из {len(names)}»: "
          f"среднее {mx.mean():+.4f}, 95-й проц. {np.percentile(mx, 95):+.4f}")
    print(f"  p(шум даёт не меньше) = {p_max:.3f}  ->  {'ШУМ' if p_max > 0.05 else 'СИГНАЛ'}")

    return dict(stand=cfg.key, scheme=scheme_name, metric=metric_name, k=len(groups), pool=pool,
                n=s["n"], ncell=ncell, gba_seed={str(k): float(v) for k, v in g.items()},
                worst=float(vals[0]), second_worst=float(vals[1]), gap=float(vals[1] - vals[0]),
                spread=float(vals[-1] - vals[0]), seed_metric=float(seed_m),
                width=s["mean_width"], half_width=s["mean_width"] / 2, mean_abs_d=s["mean_abs_d"],
                res95=s["res95"], wins95=s["wins95"], holm=s["holm"], wins_holm=s["wins_holm"],
                best=best, best_name=names[i], ceiling_mean=float(mx.mean()),
                ceiling_p95=float(np.percentile(mx, 95)), p_max=p_max)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    schemes_here = list(SCHEMES[cfg.key])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scheme", default="all", choices=schemes_here + ["all"],
                    help="схема группировки этого стенда; all (по умолчанию) -- пройти все по очереди")
    ap.add_argument("--metric", default="hard_min", help="метрика analyze_s11.metric/Bootstrap.metrics")
    ap.add_argument("--pool", default="both", choices=["all", "finals", "both"],
                    help="all: весь пул кроме seed; finals: только финалы оптимизаторов; "
                         "both (по умолчанию) -- напечатать оба пула")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    scheme_names = schemes_here if args.scheme == "all" else [args.scheme]
    pools = ["all", "finals"] if args.pool == "both" else [args.pool]

    out = []
    for nm in scheme_names:
        mapping = SCHEMES[cfg.key][nm]
        for pool in pools:
            out.append(run(nm, mapping, pool, args.metric, args.n_boot, args.seed))

    dest = cfg.outputs / f"semantic_regroup_{args.metric}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nзаписано {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

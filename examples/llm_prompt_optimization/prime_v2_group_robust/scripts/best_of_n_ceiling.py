#!/usr/bin/env python
"""Шумовой потолок «лучшего из N»: сколько даст выбор максимума, если улучшения нет вовсе.

Зачем. Работа сравнивает прирост ЛУЧШЕГО финала пула с разрешающей способностью теста и делает
вывод «видно / не видно». Но лучший из N промптов выбран по тому же шумному тесту, на котором
измеряется, поэтому он смещён вверх просто отбором максимума — тот самый winner's curse,
который работа описывает у других. Порог сравнения поэтому не единица: под нулевой гипотезой
«ни один промпт не лучше стартового» максимум из N контрастов уже положителен и тем больше,
чем больше N.

Как считается. Берётся та же парная бутстреп-схема, что и везде: строки ресэмплируются внутри
ячеек «группа x метка», индексы общие для всех промптов. Для каждой реплики b и промпта i
считается контраст d_i^b, затем он центрируется на наблюдённом значении:

    e_i^b = d_i^b - d_i^набл

Вектор e^b — это оценка шума контрастов с ПРАВИЛЬНОЙ корреляционной структурой: все контрасты
делят один и тот же стартовый промпт и одни и те же строки, поэтому независимыми они не
являются, и аналитическая формула для максимума N независимых нормальных величин завысила бы
потолок. Распределение max_i e_i^b и есть распределение «лучшего из N» под нулём.

ПРОВЕРКА НА СМЕЩЕНИЕ ЦЕНТРИРОВАНИЯ (--center). Бутстреп-распределение статистики-минимума (а
hard_min и родственные метрики — это минимум по группам) как правило смещено: бутстреп-среднее
контраста НЕ равно наблюдённому значению даже без всякого прироста, потому что минимум k величин
у ресэмплов систематически ведёт себя не так, как у исходной выборки. Центрирование на
наблюдённом значении (--center observed, поведение по умолчанию, как в формуле выше) оставляет
у остатков e_i^b ненулевое среднее — часть этого смещения бутстрепа просачивается в оценку шума.
Альтернатива — центрировать на бутстреп-СРЕДНЕМ того же контраста:

    e_i^b = d_i^b - mean_b(d_i^b)     (--center bootmean)

Это вычитает именно то смещение, которым обладает процедура бутстрепа для ДАННОГО промпта, и не
зависит от того, где относительно этого среднего лежит наблюдённое значение. Скрипт печатает ОБА
варианта ВСЕГДА, как проверку чувствительности: если вывод («шум» или «сигнал») зависит от
соглашения о центрировании, ему нельзя доверять. JSON хранит оба p-value (`p_max_observed`,
`p_max_bootmean`) и само смещение (`contrast_boot_bias`); --center выбирает только то, какой из
двух вариантов используется ниже в развёртке потолка по N и в разделе «требуемый подъём».

Что печатается: наблюдённый максимум против этого распределения, доля реплик, в которых чистый
шум дал бы столько же или больше (эмпирический p для максимума), зависимость потолка от N, и
сравнение обоих способов центрирования.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_s11 import IDS, Bootstrap, load_preds, load_set, valid_rows  # noqa: E402
from dataset_config import cfg  # noqa: E402


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", default="hard_min")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool", choices=["all", "finals"], default="all",
                    help="all: все промпты кроме seed; finals: только финалы оптимизаторов")
    ap.add_argument("--center", choices=["observed", "bootmean"], default="observed",
                    help="чем центрировать шум бутстрепа для развёртки по N и вывода: "
                         "observed (по умолчанию, прежнее поведение) или bootmean (см. docstring); "
                         "оба p-value печатаются и пишутся в JSON независимо от выбора")
    args = ap.parse_args()

    y, c, _ = load_set(cfg.test_set)
    P = load_preds(cfg.test_set, len(y))
    every = sorted(n for n in P if not n.startswith("CONTROL:"))
    keep = valid_rows(P, len(y), every)
    names = [n for n in every if n != "seed"]
    if args.pool == "finals":
        names = [n for n in names if n.startswith(cfg.final_prefix + ":")]
    yy, cc = y[keep], c[keep]
    bs = Bootstrap(yy, cc, args.n_boot, np.random.default_rng(args.seed))
    obs_s, boot_s = bs.metrics(P["seed"][keep], groups=tuple(IDS))
    m = args.metric

    obs, boot = [], []
    for n in names:
        o, b = bs.metrics(P[n][keep], groups=tuple(IDS))
        obs.append(o[m] - obs_s[m])
        boot.append(b[m] - boot_s[m])
    obs = np.array(obs)                      # (N,)
    boot = np.stack(boot, axis=0)            # (N, n_boot)
    # два способа центрировать бутстреп-контраст (см. docstring, --center):
    #   observed -- вычитает наблюдённое значение контраста;
    #   bootmean -- вычитает собственное бутстреп-среднее того же контраста.
    noise_by_center = {
        "observed": boot - obs[:, None],
        "bootmean": boot - boot.mean(axis=1, keepdims=True),
    }
    contrast_boot_bias = float(np.mean(boot.mean(axis=1) - obs))

    N = len(names)
    print(f"стенд {cfg.key}, {cfg.test_set}: пул {args.pool}, контрастов против seed N={N}, "
          f"метрика {m}, бутстреп {args.n_boot}, --center {args.center}")
    print(f"строк в анализе {int(keep.sum())}; seed {obs_s[m]:.4f}")
    print(f"смещение бутстрепа контраста (бутстреп-среднее минус наблюдённое, усреднено по "
          f"{N} промптам пула): {contrast_boot_bias:+.5f}")

    i_best = int(np.argmax(obs))
    best = float(obs[i_best])
    half = float(np.mean([np.percentile(boot[i], 97.5) - np.percentile(boot[i], 2.5)
                          for i in range(N)]) / 2)

    print("\n=== чувствительность к центрированию бутстрепа (см. docstring) ===")
    p_max_by_center = {}
    for center, nz in noise_by_center.items():
        mx_c = nz.max(axis=0)
        p_max_by_center[center] = float((mx_c >= best).mean())
        print(f"  --center {center:9s}: потолок «лучший из {N}» среднее {mx_c.mean():+.4f}, "
              f"95-й проц. {np.percentile(mx_c, 95):+.4f}, "
              f"p(шум даёт не меньше) = {p_max_by_center[center]:.3f}")
    p_max_observed, p_max_bootmean = p_max_by_center["observed"], p_max_by_center["bootmean"]

    noise = noise_by_center[args.center]     # используется ниже: развёртка по N, требуемый подъём
    mx = noise.max(axis=0)
    p_max = p_max_by_center[args.center]
    print(f"\n=== наблюдённый максимум против шумового потолка (--center {args.center}) ===")
    print(f"  лучший промпт пула: {names[i_best]}  Δ = {best:+.4f}")
    print(f"  средняя полуширина интервала одного контраста: {half:.4f}")
    print(f"  ШУМОВОЙ ПОТОЛОК «лучший из {N}»: среднее {mx.mean():+.4f}, медиана {np.median(mx):+.4f}, "
          f"95-й процентиль {np.percentile(mx, 95):+.4f}")
    print(f"  наблюдённый максимум / шумовой потолок (среднее): {best / mx.mean():.2f}")
    print(f"  доля реплик, где ЧИСТЫЙ ШУМ дал бы столько же или больше: {p_max:.3f}")
    verdict = ("максимум НЕ отличим от выбора лучшего из N под нулём" if p_max > 0.05
               else "максимум превышает то, что даёт отбор лучшего из N под нулём")
    print(f"  ВЫВОД: {verdict}")
    verdict_obs, verdict_boot = p_max_observed > 0.05, p_max_bootmean > 0.05
    agree = ("ВЫВОД ОДИНАКОВ при обоих способах центрирования" if verdict_obs == verdict_boot
             else "ВЫВОД РАСХОДИТСЯ между способами центрирования -- нужна осторожность")
    print(f"  {agree} (observed: {'шум' if verdict_obs else 'сигнал'} p={p_max_observed:.3f}, "
          f"bootmean: {'шум' if verdict_boot else 'сигнал'} p={p_max_bootmean:.3f})")

    print("\n=== как потолок зависит от размера пула (те же реплики, случайные подпулы) ===")
    rng = np.random.default_rng(args.seed + 7)
    print(f"  {'N':>4s} {'потолок (среднее)':>18s} {'95-й процентиль':>16s}")
    curve = {}
    for k in sorted({2, 5, 10, 15, 20, 25, 30, N} & set(range(2, N + 1))):
        vals = []
        for _ in range(200):
            sub = rng.choice(N, size=k, replace=False)
            vals.append(noise[sub].max(axis=0).mean())
        curve[k] = float(np.mean(vals))
        sub_all = noise[rng.choice(N, size=k, replace=False)].max(axis=0)
        print(f"  {k:4d} {np.mean(vals):+18.4f} {np.percentile(sub_all, 95):+16.4f}")

    print("\n=== требуемый подъём: сколько поиск обязан добавить, чтобы это было не шумом ===")
    print(f"  чтобы максимум пула из {N} промптов не объяснялся отбором, он должен превышать")
    print(f"  95-й процентиль шумового потолка, то есть {np.percentile(mx, 95):+.4f};")
    print(f"  наблюдено {best:+.4f} -> {'ДА' if best > np.percentile(mx, 95) else 'НЕТ'}")

    out = cfg.outputs / f"best_of_n_ceiling_{m}_{args.pool}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"stand": cfg.key, "metric": m, "pool": args.pool, "N": N,
                               "best_name": names[i_best], "best_delta": best,
                               "ceiling_mean": float(mx.mean()),
                               "ceiling_p95": float(np.percentile(mx, 95)),
                               "p_max": p_max, "half_width": half,
                               "ceiling_by_N": curve,
                               # проверка на смещение центрирования (см. docstring, --center):
                               # существующие ключи выше не менялись, эти -- добавлены поверх них.
                               "center": args.center,
                               "p_max_observed": p_max_observed, "p_max_bootmean": p_max_bootmean,
                               "contrast_boot_bias": contrast_boot_bias},
                              indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nзаписано {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

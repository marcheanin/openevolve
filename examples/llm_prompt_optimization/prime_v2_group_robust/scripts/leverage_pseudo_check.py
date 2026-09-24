#!/usr/bin/env python
"""Проверка рычага порога на псевдогруппах: меряет ли он групповую структуру или шум.

Зачем. Рычаг порога `L = средний |Δ worst_class| / средний |Δ hard_min|` по правкам строгости был
предложен как величина, считаемая до трат на оптимизатор, и как ведущий вход решающего правила
(B-3 и H-2 в `paper/09_PAPER_SPEC.md`). Смысл, который ему приписывался: насколько всё, до чего
дотягивается промпт, есть одна ручка строгости, то есть насколько задача одноосевая ОТНОСИТЕЛЬНО
ГРУПП.

Проверка. Если L действительно про групповую структуру, то на псевдогруппах — случайном
переназначении внутри метки, где настоящей структуры нет, а баланс ячеек тот же — он обязан стать
заметно другим. Если же L на псевдогруппах такой же, значит он не про группы вообще.

Устройство проверки. Числитель `|Δ worst_class|` от группировки НЕ ЗАВИСИТ вовсе: worst_class —
минимум по двум меткам, групп в нём нет. Поэтому вся разница между настоящим и псевдо-L сидит в
знаменателе — в том, насколько сильно правка строгости двигает hard-min. Знаменатель на
псевдогруппах и есть «шумовой пол» отклика hard-min: сколько минимум по k случайным подмножествам
шевелится просто от того, что промпт что-то поменял.

Что печатается: настоящий и псевдо знаменатель, их отношение (отклик назначенной худшей группы
против отклика случайной группы того же размера), и настоящий с псевдо L.

Строки берутся ровно те, что входят в настоящие группы (у CivilComments это исключает группу 0
«none»), иначе у псевдогрупп оказалось бы больше строк и более узкий шумовой пол.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_s11 import IDS, load_preds, load_set, metric, valid_rows  # noqa: E402
from dataset_config import cfg  # noqa: E402


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=50, help="сколько случайных разбиений")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool-prefix", default="r15:",
                    help="пул, по которому считается отклик (правки строгости)")
    args = ap.parse_args()

    y, c, _ = load_set(cfg.test_set)
    P = load_preds(cfg.test_set, len(y))
    every = sorted(n for n in P if not n.startswith("CONTROL:"))
    # только строки, входящие в настоящие группы: у псевдогрупп должен быть тот же материал
    keep = valid_rows(P, len(y), every) & np.isin(c, list(IDS))
    pool = [n for n in every if n.startswith(args.pool_prefix)]
    if len(pool) < 3:
        raise SystemExit(f"пул {args.pool_prefix!r} слишком мал: {len(pool)}")
    k = len(IDS)

    def hard_min_response(cc, groups):
        base = metric("hard_min", P["seed"], y, cc, keep, groups)
        return float(np.mean([abs(metric("hard_min", P[n], y, cc, keep, groups) - base) for n in pool]))

    base_wc = metric("worst_class", P["seed"], y, c, keep, tuple(IDS))
    wc = float(np.mean([abs(metric("worst_class", P[n], y, c, keep, tuple(IDS)) - base_wc) for n in pool]))
    real = hard_min_response(c, tuple(IDS))

    rng = np.random.default_rng(args.seed)
    vals = []
    for _ in range(args.reps):
        pc = np.zeros_like(c)
        for lab in (0, 1):
            idx = np.flatnonzero((y == lab) & keep)
            pc[rng.permutation(idx)] = np.tile(np.arange(1, k + 1), len(idx) // k + 1)[:len(idx)]
        vals.append(hard_min_response(pc, tuple(range(1, k + 1))))
    vals = np.array(vals)
    pseudo = float(np.median(vals))

    print(f"стенд {cfg.key}, {cfg.test_set}: строк в анализе {int(keep.sum())}, групп {k}, "
          f"пул {args.pool_prefix}* — {len(pool)} промптов")
    print(f"\n  средний |Δ worst_class| (от группировки не зависит):      {wc:.4f}")
    print(f"  средний |Δ hard_min|, НАСТОЯЩИЕ группы:                   {real:.4f}")
    print(f"  средний |Δ hard_min|, псевдогруппы (медиана {args.reps}):        {pseudo:.4f} "
          f"[{np.percentile(vals, 5):.4f}–{np.percentile(vals, 95):.4f}]")
    print(f"\n  ОТКЛИК НАСТОЯЩЕЙ ХУДШЕЙ ГРУППЫ ПРОТИВ СЛУЧАЙНОЙ: {real / pseudo:.2f}")
    print(f"  рычаг порога L, настоящие группы: {wc / real:.2f}")
    print(f"  рычаг порога L, псевдогруппы:     {wc / pseudo:.2f}")
    verdict = ("L НЕ отличим от псевдогрупп — он не про групповую структуру"
               if wc / real > 0.8 * (wc / pseudo) else
               "L на настоящих группах заметно ниже псевдо — отклик групп есть")
    print(f"  ВЫВОД: {verdict}")

    out = cfg.outputs / "leverage_pseudo_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "stand": cfg.key, "k": k, "pool_prefix": args.pool_prefix, "pool_n": len(pool),
        "rows": int(keep.sum()), "worst_class_response": wc,
        "hard_min_response_real": real, "hard_min_response_pseudo_median": pseudo,
        "hard_min_response_pseudo_p5": float(np.percentile(vals, 5)),
        "hard_min_response_pseudo_p95": float(np.percentile(vals, 95)),
        "response_ratio_real_over_pseudo": real / pseudo,
        "leverage_real": wc / real, "leverage_pseudo": wc / pseudo,
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nзаписано {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

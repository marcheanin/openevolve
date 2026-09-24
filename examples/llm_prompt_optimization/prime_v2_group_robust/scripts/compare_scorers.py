"""Совпадают ли выводы на двух скорерах? Сравнение на одних и тех же строках матрицы S11.

Вопрос рецензента: не свойство ли всё это именно gemma-3-12b-it. Проверяется четыре вещи,
каждая предсказывает, что случится, если это свойство модели:

  1. Согласие рангов промптов между скорерами (Спирмен). Голое ρ читать нельзя: при малом
     числе строк оно ограничено шумом самой оценки. Поэтому рядом считается надёжность
     каждого скорера расщеплением строк пополам, экстраполированная по Спирмену–Брауну, и
     ρ с поправкой на ослабление: ρ_кросс / sqrt(r1 · r2). Если поправленное ρ близко к 1,
     скореры измеряют одно и то же с точностью до шума.
  2. Согласие знака контраста «промпт против seed».
  3. Семейства. Однострочные правки против финалов оптимизаторов: доля выше seed и медиана.
     Это то, что в основной матрице выглядело как результат про оптимизаторы.
  4. Соблюдение формата: доля неразобранных ответов по семействам. Если разрыв между
     семействами есть только у gemma, «оптимизаторы учат формат» — свойство gemma.

Строки, на которых хоть один промпт любого скорера дал неразобранный ответ, исключаются из
всех сравнений сразу для всех промптов, иначе парность нарушается.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_s11 import IDS, cell_index, metric  # noqa: E402

OUT = ROOT / "results/S11_protocol_matrix"
METRICS = ["cvar25", "hard_min", "worst_class", "mean_gba"]


def load_first(set_name: str, L: int | None) -> dict[str, np.ndarray]:
    d = {}
    for f in sorted((OUT / "preds" / set_name).glob("*.npy")):
        if f.stem.startswith("_") or f.name.endswith(".partial.npy"):
            continue
        if f.name.endswith(".lo.npy"):
            continue  # log-odds companion of a prediction file, not a prompt of its own
        a = np.load(f)
        d[f.stem.replace("__", ":", 1)] = a[:L] if L else a
    return d


def load_second(tag: str, set_name: str) -> tuple[dict, dict]:
    pdir = OUT / f"scorer2_{tag}" / "preds" / set_name
    preds, lo = {}, {}
    for f in sorted(pdir.glob("*.npy")):
        if f.name.endswith(".lo.npy"):
            continue
        name = f.stem.replace("__", ":", 1)
        preds[name] = np.load(f)
        g = pdir / f"{f.stem}.lo.npy"
        lo[name] = np.load(g)[: len(preds[name])] if g.is_file() else None
    return preds, lo


def family(name: str) -> str:
    return "seed" if name == "seed" else ("однострочные правки" if name.startswith("r15:") else "оптимизаторы")


def split_half_reliability(preds, names, y, c, keep, metric_name, rng, reps=100) -> float:
    """Надёжность ранжирования промптов при половинной длине; затем Спирмен–Браун до полной."""
    cells = cell_index(y, c, keep)
    rhos = []
    for _ in range(reps):
        a = np.zeros(len(y), bool)
        b = np.zeros(len(y), bool)
        for rows in cells.values():
            if len(rows) < 2:
                continue
            perm = rng.permutation(rows)
            a[perm[: len(perm) // 2]] = True
            b[perm[len(perm) // 2:]] = True
        va = [metric(metric_name, preds[n], y, c, rows=a) for n in names]
        vb = [metric(metric_name, preds[n], y, c, rows=b) for n in names]
        r = spearmanr(va, vb).statistic
        if np.isfinite(r):
            rhos.append(r)
    r_half = float(np.mean(rhos))
    return 2 * r_half / (1 + r_half) if r_half > -1 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--set", default="truth_large")
    ap.add_argument("--prefix", type=int, default=0, help="0 = самый длинный общий префикс")
    args = ap.parse_args()

    rec = json.loads((OUT / "fixed_sets" / f"{args.set}.json").read_text(encoding="utf-8"))
    y_all, c_all = np.asarray(rec["labels"]), np.asarray(rec["cluster_ids"])
    p2, lo2 = load_second(args.tag, args.set)
    if not p2:
        raise SystemExit("предсказаний второго скорера ещё нет")
    L = min(len(v) for v in p2.values())
    if args.prefix:
        L = min(L, args.prefix)
    p1 = load_first(args.set, L)
    names = sorted(set(p1) & set(p2))
    p2 = {n: p2[n][:L] for n in names}
    p1 = {n: p1[n] for n in names}
    y, c = y_all[:L], c_all[:L]

    bad = np.zeros(L, bool)
    for n in names:
        bad |= (p1[n] < 0) | (p2[n] < 0)
    keep = ~bad
    print(f"{args.set}: общий префикс {L} строк ({L // 18} на ячейку), промптов {len(names)}, "
          f"исключено {int(bad.sum())} строк с неразобранным ответом ({100 * bad.mean():.2f}%)\n")

    rng = np.random.default_rng(0)
    res: dict = {"prefix": L, "prompts": len(names), "excluded_rows": int(bad.sum())}

    # 1. согласие рангов
    print("=== 1. согласие рангов промптов между скорерами ===")
    print(f"{'метрика':12s} {'ρ кросс':>8s} {'надёжн. gemma':>14s} {'надёжн. 2-го':>13s} {'ρ с поправкой':>14s}")
    res["rank"] = {}
    for m in METRICS:
        v1 = [metric(m, p1[n], y, c, rows=keep) for n in names]
        v2 = [metric(m, p2[n], y, c, rows=keep) for n in names]
        rho = float(spearmanr(v1, v2).statistic)
        r1 = split_half_reliability(p1, names, y, c, keep, m, rng)
        r2 = split_half_reliability(p2, names, y, c, keep, m, rng)
        corr = rho / np.sqrt(max(r1, 1e-9) * max(r2, 1e-9)) if r1 > 0 and r2 > 0 else float("nan")
        res["rank"][m] = {"rho": rho, "reliab_first": r1, "reliab_second": r2, "rho_corrected": float(corr)}
        print(f"{m:12s} {rho:8.3f} {r1:14.3f} {r2:13.3f} {corr:14.3f}")
    print("Поправленное ρ около 1 означает: скореры ранжируют одинаково с точностью до шума.\n"
          "Если надёжность мала (< 0.5), сравнение на такой длине ничего не решает.\n")

    # 2. согласие знака контраста с seed
    print("=== 2. согласие знака «промпт − seed» ===")
    res["sign"] = {}
    for m in METRICS:
        s1 = metric(m, p1["seed"], y, c, rows=keep)
        s2 = metric(m, p2["seed"], y, c, rows=keep)
        d1 = np.array([metric(m, p1[n], y, c, rows=keep) - s1 for n in names if n != "seed"])
        d2 = np.array([metric(m, p2[n], y, c, rows=keep) - s2 for n in names if n != "seed"])
        agree = float(np.mean(np.sign(d1) == np.sign(d2)))
        res["sign"][m] = agree
        print(f"{m:12s} знак совпадает в {100 * agree:4.0f}% контрастов "
              f"(выше seed: gemma {int((d1 > 0).sum())}/{len(d1)}, второй {int((d2 > 0).sum())}/{len(d2)})")

    # 3. семейства
    print("\n=== 3. однострочные правки против финалов оптимизаторов (cvar25) ===")
    print(f"{'скорер':10s} {'семейство':20s} {'промптов':>9s} {'выше seed':>10s} {'медиана':>8s} {'seed':>8s}")
    res["family"] = {}
    for label, P in (("gemma", p1), ("второй", p2)):
        seed_v = metric("cvar25", P["seed"], y, c, rows=keep)
        for fam in ("однострочные правки", "оптимизаторы"):
            vals = [metric("cvar25", P[n], y, c, rows=keep) for n in names if family(n) == fam]
            above = int(sum(v > seed_v for v in vals))
            res["family"][f"{label}|{fam}"] = {"n": len(vals), "above_seed": above,
                                               "median": float(np.median(vals)), "seed": seed_v}
            print(f"{label:10s} {fam:20s} {len(vals):9d} {above:5d} ({100 * above / len(vals):3.0f}%) "
                  f"{np.median(vals):8.4f} {seed_v:8.4f}")

    # 4. формат ответа: доли считаются по ВСЕМ строкам префикса, без исключения
    print("\n=== 4. неразобранные ответы по семействам (доля строк) ===")
    res["invalid"] = {}
    raw1 = load_first(args.set, L)
    for label, P in (("gemma", raw1), ("второй", {n: p2raw for n, p2raw in load_second(args.tag, args.set)[0].items()})):
        for fam in ("однострочные правки", "оптимизаторы"):
            arr = [(P[n][:L] < 0).mean() for n in names if family(n) == fam and n in P]
            hit = int(sum(a > 0 for a in arr))
            res["invalid"][f"{label}|{fam}"] = {"mean_rate": float(np.mean(arr)), "prompts_with_any": hit,
                                                "of": len(arr)}
            print(f"{label:10s} {fam:20s} средняя доля {100 * np.mean(arr):5.2f}%  "
                  f"промптов с хотя бы одним сбоем {hit}/{len(arr)}")

    # 5. рабочая точка: доля «токсично» по промптам
    print("\n=== 5. доля ответов «токсично» (рабочая точка) ===")
    pos1 = np.array([p1[n][keep].mean() for n in names])
    pos2 = np.array([p2[n][keep].mean() for n in names])
    rho_pos = float(spearmanr(pos1, pos2).statistic)
    res["pos_rate_rho"] = rho_pos
    print(f"ранговая корреляция долей между скорерами: {rho_pos:.3f}; "
          f"диапазон gemma {pos1.min():.2f}..{pos1.max():.2f}, второй {pos2.min():.2f}..{pos2.max():.2f}")

    out_dir = OUT / f"scorer2_{args.tag}" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"compare_{args.set}_L{L}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                                           encoding="utf-8")
    print(f"\nзаписано: {out_dir / f'compare_{args.set}_L{L}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

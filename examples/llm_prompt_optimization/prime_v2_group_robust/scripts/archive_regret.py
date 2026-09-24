#!/usr/bin/env python3
"""Пункт 7: сожаление отбора на toxlang (план — experiments/S15_toxlang/PREREG_LOTTERY.md,
поправка — experiments/S15_toxlang/PREREG_LOTTERY_AMENDMENT.md).

Для каждого из 12 живых прогонов сравнивает выбранный по dev финал с лучшим кандидатом того же
архива на тесте (первые 1000 строк truth_tox):

  * наивное сожаление R = max_c T(c) - T(финал) — смещено проклятием победителя;
  * сожаление с поправкой на отбор — бутстреп внутри ячеек язык x метка, общие индексы,
    M* = max_c[(T*_c - T*_f) - (T_c - T_f)], p = доля повторов с M* >= R, Холм по прогонам;
  * кросс-фит — лучший кандидат выбирается на одной половине теста, сравнивается с финалом на
    другой (стратификация по ячейкам, обе стороны, 200 разбиений);
  * согласие dev и теста (Спирмен) отдельно для полного dev и для подвыборки;
  * где теряются кандидаты, превзошедшие финал сильнее C95 прогона.

ПОПРАВКА к предрегистрации (см. PREREG_LOTTERY_AMENDMENT.md): у EvoPrompt-DE и GEPA часть
кандидатов архива дают INVALID на КАЖДОЙ из 1000 строк — не шум скорера, а сами эти мутации
ломают формат ответа модели (правило "хоть одна строка невалидна — исключить строку у всех"
из плана обнулило бы анализ всех 12 прогонов разом). Правило исключения строк применяется не
глобально по всем 138 кандидатам, а внутри каждого прогона отдельно, и внутри прогона —
только среди кандидатов, чьё включение не роняет число общих валидных строк ниже MIN_KEEP_ROWS;
финал прогона из пула никогда не исключается (все 12 финалов дают <=2.8% INVALID). Исключённые
кандидаты перечислены в выводе отдельно как "негодные к оценке", а не молча выброшены.

Основной режим — INVALID трактуется как отсутствие данных (пул сокращается, см. выше).
Чувствительность (--invalid-as-error): INVALID считается ошибкой классификатора, ни один
кандидат и ни одна строка не исключаются — сравнивает, меняет ли обработка INVALID вывод.

Выход: results/S15_toxlang_matrix/archive_regret[_inv_err].json и таблица в stdout.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
os.environ["S11_DATASET"] = "toxlang"
os.environ.pop("S11_PREDS_DIR", None)

from analyze_s11 import IDS, holm, load_set  # noqa: E402

MATRIX = ROOT / "results" / "S15_toxlang_matrix"
MAIN_PREDS = MATRIX / "scorer_gemma" / "preds" / "truth_tox"
ARC_PREDS = MATRIX / "archive_scorer_gemma" / "preds" / "truth_tox"
METRICS = ("hard_min", "mean_gba", "worst_class")
N_ROWS = 1000
N_BOOT = 4000
N_SPLITS = 200
FULL_DEV = 900
# Пол ниже которого прогон теряет разрешающую способность бутстрепа; кандидаты, чьё включение
# уронило бы общее число валидных строк прогона ниже этого числа, из пула исключаются (см.
# PREREG_LOTTERY_AMENDMENT.md). 700 — тот же порядок, что и test_fixed CivilComments (720).
MIN_KEEP_ROWS = 700


def pred_path(name: str) -> Path:
    stem = name.replace(":", "__", 1)
    return (ARC_PREDS if name.startswith("arc:") else MAIN_PREDS) / f"{stem}.npy"


def val_info(evals: list[dict]) -> tuple[float, int]:
    """Оценка на dev по наибольшей выборке, на которой кандидат оценивался."""
    e = max(evals, key=lambda x: x["rows"])
    return float(e["selected_on"]), int(e["rows"])


def spearman(a, b) -> float | None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 4 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    for r, v in ((ra, a), (rb, b)):
        for u in np.unique(v):
            m = v == u
            if m.sum() > 1:
                r[m] = r[m].mean()
    return float(np.corrcoef(ra, rb)[0, 1])


def select_pool(names: list[str], final: str, P: dict, keep_all: np.ndarray,
                min_keep: int) -> tuple[list[str], list[dict], np.ndarray]:
    """Жадно строит пул кандидатов прогона, чьи строки без INVALID (в объединении с ``final``)
    остаются >= ``min_keep``. Финал не исключается никогда; остальные добавляются от наименее к
    наиболее ненадёжному (по доле INVALID), первый, кто уронил бы порог, и все последующие
    (более ненадёжные) — в «исключённые»."""
    rates = {n: float((P[n] < 0).mean()) for n in names}
    order = [final] + sorted((n for n in names if n != final), key=lambda n: rates[n])
    keep = keep_all.copy()
    pool: list[str] = []
    excluded: list[dict] = []
    for n in order:
        cand_keep = keep & (P[n] >= 0)
        if n == final or int(cand_keep.sum()) >= min_keep:
            pool.append(n)
            keep = cand_keep
        else:
            excluded.append({"name": n, "invalid_rate": rates[n]})
    return pool, excluded, keep


class HalfMetrics:
    """Метрики каждого промпта на каждой половине каждого разбиения, векторно (см. docstring
    исходной версии): K — матрица верных ответов (промпты x строки этого прогона)."""

    def __init__(self, K, y, c, rng):
        n = K.shape[1]
        cells = [(g, l) for g in IDS for l in (1, 0)]
        Cm = np.stack([(c == g) & (y == l) for g, l in cells]).astype(float)
        A = np.zeros((N_SPLITS, n), float)
        for s in range(N_SPLITS):
            for row in Cm.astype(bool):
                idx = rng.permutation(np.flatnonzero(row))
                A[s, idx[: len(idx) // 2]] = 1.0
        B = 1.0 - A
        self.half = {}
        for tag, H in (("A", A), ("B", B)):
            HC = H[:, None, :] * Cm[None]
            num = K @ HC.reshape(-1, n).T
            den = HC.sum(axis=2).reshape(-1)
            rate = (num / den).reshape(K.shape[0], N_SPLITS, len(cells))
            gba = 0.5 * (rate[:, :, 0::2] + rate[:, :, 1::2])
            pos_n = (H[:, None, :] * Cm[None, 0::2]).sum(axis=(1, 2))
            neg_n = (H[:, None, :] * Cm[None, 1::2]).sum(axis=(1, 2))
            pos = (num.reshape(K.shape[0], N_SPLITS, len(cells))[:, :, 0::2].sum(axis=2)) / pos_n
            neg = (num.reshape(K.shape[0], N_SPLITS, len(cells))[:, :, 1::2].sum(axis=2)) / neg_n
            self.half[tag] = {"hard_min": gba.min(axis=2), "mean_gba": gba.mean(axis=2),
                              "worst_class": np.minimum(pos, neg)}

    def crossfit(self, metric: str, fi: int, rng) -> tuple[float, float]:
        gains = []
        for sel, ev in (("A", "B"), ("B", "A")):
            S, E = self.half[sel][metric], self.half[ev][metric]
            for s in range(N_SPLITS):
                col = S[:, s]
                best = col.max()
                pick = fi if col[fi] >= best else int(rng.choice(np.flatnonzero(col == best)))
                gains.append(E[pick, s] - E[fi, s])
        g = np.asarray(gains)
        return float(g.mean()), float((g > 0).mean())


class Boot:
    """Тот же общий бутстреп, что analyze_s11.Bootstrap, но только по ячейкам, которые
    непусты на строках, оставшихся у этого прогона."""

    def __init__(self, y, c, n_boot, rng):
        self.cells = {(int(g), int(l)): np.flatnonzero((c == g) & (y == l))
                     for g in IDS for l in (0, 1)}
        self.cells = {k: v for k, v in self.cells.items() if len(v)}
        self.slices, start, cols = {}, 0, []
        for k, rows in self.cells.items():
            cols.append(rng.choice(rows, size=(n_boot, len(rows)), replace=True))
            self.slices[k] = slice(start, start + len(rows))
            start += len(rows)
        self.idx = np.concatenate(cols, axis=1)

    def metrics(self, pred, groups):
        obs_cell, boot_cell = {}, {}
        for k, rows in self.cells.items():
            g, l = k
            obs_cell[k] = float((pred[rows] == l).mean())
            boot_cell[k] = (pred[self.idx[:, self.slices[k]]] == l).mean(axis=1)

        def assemble(get):
            g_vals = {g: 0.5 * (get((g, 1)) + get((g, 0))) for g in groups
                      if (g, 1) in self.cells and (g, 0) in self.cells}
            arr = np.sort(np.stack(list(g_vals.values())), axis=0)
            pos = sum(get((g, 1)) * len(self.cells[(g, 1)]) for g in groups if (g, 1) in self.cells) / \
                sum(len(self.cells[(g, 1)]) for g in groups if (g, 1) in self.cells)
            neg = sum(get((g, 0)) * len(self.cells[(g, 0)]) for g in groups if (g, 0) in self.cells) / \
                sum(len(self.cells[(g, 0)]) for g in groups if (g, 0) in self.cells)
            return {"hard_min": arr[0], "mean_gba": arr.mean(axis=0),
                    "worst_class": np.minimum(pos, neg)}

        return assemble(lambda k: obs_cell[k]), assemble(lambda k: boot_cell[k])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--invalid-as-error", action="store_true",
                    help="чувствительность: INVALID = ошибка, ни строки, ни кандидаты не исключаются")
    args = ap.parse_args()

    amap = json.loads((MATRIX / "archive_map.json").read_text(encoding="utf-8"))
    y, c, _ = load_set("truth_tox")
    y, c = y[:N_ROWS], c[:N_ROWS]

    names = sorted({x["name"] for r in amap["runs"] for x in r["candidates"]} |
                   {r["final"] for r in amap["runs"]})
    P, missing = {}, []
    for n in names:
        p = pred_path(n)
        if not p.is_file() or len(np.load(p)) < N_ROWS:
            missing.append(n)
            continue
        P[n] = np.load(p)[:N_ROWS].astype(int)
    if missing:
        print(f"[!] нет полных предсказаний для {len(missing)} из {len(names)}: оценка не закончена")
        return 1

    inv = {n: float((v < 0).mean()) for n, v in P.items()}
    print(f"промптов {len(P)}; доля INVALID: 0%% у {sum(1 for r in inv.values() if r == 0)}, "
          f">=50%% у {sum(1 for r in inv.values() if r >= 0.5)}, "
          f"=100%% у {sum(1 for r in inv.values() if r == 1.0)}")
    if args.invalid_as_error:
        for n in P:
            P[n] = np.where(P[n] < 0, 1 - y, P[n])

    tie_rng = np.random.default_rng(2)
    rows_out: list[dict] = []
    pooled_full: list[tuple[float, float]] = []
    pooled_sub: list[tuple[float, float]] = []
    for r in amap["runs"]:
        cand, vinfo = [], {}
        for x in r["candidates"]:
            if x["name"] not in vinfo:
                cand.append(x["name"])
                vinfo[x["name"]] = val_info(x["evals"])
            else:
                v = val_info(x["evals"])
                if v[1] > vinfo[x["name"]][1]:
                    vinfo[x["name"]] = v
        f = r["final"]
        all_names = sorted(set(cand) | {f})

        if args.invalid_as_error:
            pool, excluded, keep = all_names, [], np.ones(N_ROWS, bool)
        else:
            pool, excluded, keep = select_pool(all_names, f, P, np.ones(N_ROWS, bool), MIN_KEEP_ROWS)

        yk, ck = y[keep], c[keep]
        bs = Boot(yk, ck, N_BOOT, np.random.default_rng(hash(r["run"]) % (2**31)))
        obs, boot = {}, {}
        for n in pool:
            o, b = bs.metrics(P[n][keep], groups=IDS)
            obs[n] = {m: float(o[m]) for m in METRICS}
            boot[n] = {m: b[m] for m in METRICS}
        K = np.stack([(P[n][keep] == yk).astype(float) for n in pool])
        half = HalfMetrics(K, yk, ck, np.random.default_rng(hash(r["run"] + "h") % (2**31)))
        fi = pool.index(f)

        full = [n for n in pool if vinfo.get(n, (0, 0))[1] >= FULL_DEV]
        argmax_full = max(full, key=lambda n: vinfo[n][0]) if full else None
        rec = {"run": r["run"], "method": r["method"], "seed": r["seed"], "final": f,
               "k": len(cand), "k_pool": len(pool), "k_excluded": len(excluded),
               "excluded": excluded, "rows_kept": int(keep.sum()),
               "final_is_dev_argmax": (argmax_full == f) if argmax_full else None,
               "final_dev": vinfo[f][0], "final_dev_rows": vinfo[f][1]}
        for m in METRICS:
            o = np.array([obs[n][m] for n in pool])
            b = np.stack([boot[n][m] for n in pool])
            R = float(o.max() - o[fi])
            best = pool[int(o.argmax())]
            Mstar = ((b - b[fi]) - (o - o[fi])[:, None]).max(axis=0)
            p = float(max((Mstar >= R - 1e-12).mean(), 1.0 / N_BOOT))
            c95 = float(np.percentile(Mstar, 95))
            cf, cf_pos = half.crossfit(m, fi, tie_rng)
            rec[m] = {"final_test": float(o[fi]), "best_test": float(o.max()), "best": best,
                      "best_dev": vinfo.get(best, (None, None))[0], "naive_regret": R, "c95": c95,
                      "p_sel": p, "crossfit_regret": cf, "crossfit_share_pos": cf_pos}
            if m == "hard_min":
                lost = [n for n in pool if o[pool.index(n)] - o[fi] > c95]
                rec["lost_beyond_c95"] = [{"name": n, "gain": float(o[pool.index(n)] - o[fi])}
                                          for n in lost]
                rec["spearman_full"] = spearman([vinfo[n][0] for n in full], [obs[n][m] for n in full])
                sub = [n for n in pool if n in vinfo and vinfo[n][1] < FULL_DEV]
                rec["spearman_sub"] = spearman([vinfo[n][0] for n in sub], [obs[n][m] for n in sub])
                pooled_full.extend((vinfo[n][0], obs[n][m]) for n in full)
                pooled_sub.extend((vinfo[n][0], obs[n][m]) for n in sub)
        rows_out.append(rec)

    rej = holm([x["hard_min"]["p_sel"] for x in rows_out])
    for x, rj in zip(rows_out, rej):
        x["hard_min"]["holm_reject"] = bool(rj)

    summary = {"rows_total": N_ROWS, "min_keep_rows": MIN_KEEP_ROWS,
               "invalid_mode": "as_error" if args.invalid_as_error else "excluded_by_floor",
               "n_prompts": len(P), "n_runs": len(rows_out),
               "total_excluded_candidates": sum(x["k_excluded"] for x in rows_out)}
    for m in METRICS:
        summary[m] = {
            "mean_naive_regret": float(np.mean([x[m]["naive_regret"] for x in rows_out])),
            "mean_crossfit_regret": float(np.mean([x[m]["crossfit_regret"] for x in rows_out])),
            "n_p_below_05": int(sum(x[m]["p_sel"] < 0.05 for x in rows_out)),
        }
    summary["hard_min"]["n_holm"] = int(sum(rej))
    summary["spearman_pooled_full"] = spearman([a for a, _ in pooled_full], [b for _, b in pooled_full])
    summary["n_full"] = len(pooled_full)
    summary["spearman_pooled_sub"] = spearman([a for a, _ in pooled_sub], [b for _, b in pooled_sub])
    summary["n_sub"] = len(pooled_sub)
    cf = summary["hard_min"]["mean_crossfit_regret"]
    summary["H_L1"] = "подтверждена" if summary["hard_min"]["n_holm"] >= 1 else "не подтверждена"
    summary["H_L2"] = ("подтверждена" if cf >= 0.02 else "промежуточно" if cf > 0 else "опровергнута")

    out = MATRIX / ("archive_regret_inv_err.json" if args.invalid_as_error else "archive_regret.json")
    out.write_text(json.dumps({"summary": summary, "runs": rows_out}, ensure_ascii=False, indent=1),
                   encoding="utf-8")

    print(f"\n{'прогон':<27} {'k':>3} {'пул':>4} {'искл':>4} {'строк':>5} {'финал=argmax':>12} "
          f"{'T финал':>8} {'T лучш.':>8} {'R наив':>7} {'C95':>6} {'p':>6} {'Холм':>5} "
          f"{'кросс-фит':>9} {'ρ full':>7}")
    for x in rows_out:
        h = x["hard_min"]
        rho = "—" if x["spearman_full"] is None else f"{x['spearman_full']:+.2f}"
        print(f"{x['run']:<27} {x['k']:>3} {x['k_pool']:>4} {x['k_excluded']:>4} {x['rows_kept']:>5} "
              f"{str(x['final_is_dev_argmax']):>12} {h['final_test']:>8.4f} {h['best_test']:>8.4f} "
              f"{h['naive_regret']:>+7.4f} {h['c95']:>6.4f} {h['p_sel']:>6.3f} "
              f"{('да' if h['holm_reject'] else 'нет'):>5} {h['crossfit_regret']:>+9.4f} {rho:>7}")
    print("\nсводка:")
    for m in METRICS:
        s = summary[m]
        print(f"  {m:<12} наивное {s['mean_naive_regret']:+.4f}  кросс-фит {s['mean_crossfit_regret']:+.4f}  "
              f"p<0,05 в {s['n_p_below_05']}/12" + (f"  после Холма {s['n_holm']}/12" if m == "hard_min" else ""))
    sf, ss = summary["spearman_pooled_full"], summary["spearman_pooled_sub"]
    print(f"  Спирмен dev–тест (hard_min), все прогоны: полный dev "
          f"{sf if sf is None else round(sf, 3)} (n={summary['n_full']}), "
          f"подвыборка {ss if ss is None else round(ss, 3)} (n={summary['n_sub']})")
    print(f"  исключено из пулов кандидатов (см. PREREG_LOTTERY_AMENDMENT.md): "
          f"{summary['total_excluded_candidates']}")
    for x in rows_out:
        if x["excluded"]:
            print(f"    {x['run']}: " + ", ".join(f"{e['name']} ({e['invalid_rate']:.0%} INVALID)"
                                                   for e in x["excluded"]))
    lost = [(x["run"], y_) for x in rows_out for y_ in x["lost_beyond_c95"]]
    print(f"  кандидатов лучше финала сильнее C95: {len(lost)}"
          + "".join(f"\n    {r}: {d['name']} +{d['gain']:.4f}" for r, d in lost))
    print(f"  H-L1: {summary['H_L1']};  H-L2: {summary['H_L2']}")
    print(f"\n→ {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

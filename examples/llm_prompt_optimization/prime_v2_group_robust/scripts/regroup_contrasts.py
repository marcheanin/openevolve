#!/usr/bin/env python
"""Пересчёт контрастов против стартового промпта при ДРУГОМ разбиении тех же строк на группы.

Зачем. Два стенда работы различаются сразу по трём осям: есть ли сигнал, однородны ли группы по
трудности, велик ли шум. Пока меняются все три сразу, сказать, что именно делает стенд измеримым,
нельзя. Этот скрипт меняет РОВНО ОДНУ ось — состав групп, — оставляя те же предсказания, ту же
модель, ту же точность и ту же толщину ячеек. Новых обращений к API не требуется.

Три режима (предрегистрация S14, гипотезы H-M2, H-M2b, H-M3):

  pseudo    строки случайно раскидываются на N псевдогрупп равного размера ВНУТРИ каждой метки,
            так что баланс ячеек «группа x метка» сохраняется. Разнородность групп исчезает,
            всё остальное — включая смещение оператора минимума — остаётся. Повторяется на
            --reps разбиениях, ведущее число — медиана по разбиениям.
  feature   группы задаются априорным признаком строки (квартили лексического пересечения или
            длины гипотезы, признак отрицания). Группы становятся разнородными по ТРУДНОСТИ при
            той же задаче и модели.
  real      контроль: настоящие группы стенда, чтобы убедиться, что воспроизводится известное.

ВАЖНОЕ ОГРАНИЧЕНИЕ. Признак группировки должен быть априорным, то есть вычисляться из текста, а
не из того, где ошибается стартовый промпт. Группировка «там, где seed ошибается» даёт худшую
группу, отобранную по результату, и гарантированно покажет улучшение. Это ровно тот класс
ошибки, который работа ловит в чужих текстах, поэтому здесь он запрещён явно.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_s11 import (Bootstrap, boot_p, delta_ci, holm, load_preds,  # noqa: E402
                         load_set, valid_rows)
from dataset_config import cfg  # noqa: E402

WORD = re.compile(r"[a-z0-9']+")


def split_premise_hypothesis(t: str):
    """Тексты множества хранятся как 'Premise: ...\\nHypothesis: ...'."""
    low = t.replace("\r\n", "\n")
    if "\nHypothesis:" in low:
        p, h = low.split("\nHypothesis:", 1)
        return p.replace("Premise:", "", 1), h
    return low, low


def feature_values(kind: str, rec: dict, n: int):
    """Значение признака для каждой строки; None, если признак на этом стенде недоступен."""
    if kind in rec and isinstance(rec.get(kind), list) and len(rec[kind]) == n:
        return np.array([1 if bool(v) else 0 for v in rec[kind]]), f"{kind} (из файла множества)"
    texts = rec.get("texts")
    if not (isinstance(texts, list) and len(texts) == n):
        return None, f"{kind}: в файле множества нет ни поля {kind!r}, ни texts"
    if kind == "overlap":
        vals = []
        for t in texts:
            p, h = split_premise_hypothesis(str(t))
            pw, hw = set(WORD.findall(p.lower())), WORD.findall(h.lower())
            vals.append(sum(w in pw for w in hw) / len(hw) if hw else 0.0)
        return np.array(vals, float), "доля слов гипотезы, встречающихся в премиссе"
    if kind == "hyp_len":
        vals = [len(WORD.findall(split_premise_hypothesis(str(t))[1].lower())) for t in texts]
        return np.array(vals, float), "число слов в гипотезе"
    return None, f"неизвестный признак {kind!r}"


def quantile_groups(vals: np.ndarray, q: int):
    """Разбиение на q групп по квантилям значения признака; группы нумеруются с 1."""
    edges = np.quantile(vals, np.linspace(0, 1, q + 1)[1:-1])
    return np.searchsorted(edges, vals, side="right") + 1


def pseudo_groups(y: np.ndarray, n_groups: int, rng) -> np.ndarray:
    """Случайные группы равного размера ВНУТРИ каждой метки: баланс ячеек не меняется."""
    g = np.zeros(len(y), int)
    for lab in (0, 1):
        idx = np.flatnonzero(y == lab)
        rng.shuffle(idx)
        g[idx] = np.arange(len(idx)) % n_groups + 1
    return g


def contrasts(pred_by_name, names, seed_pred, y, c, groups, metric_name, n_boot, rng, keep):
    """Δ каждого промпта против seed по metric_name: точечная оценка, интервал, p, Холм."""
    yy, cc = y[keep], c[keep]
    bs = Bootstrap(yy, cc, n_boot, rng)
    obs_s, boot_s = bs.metrics(seed_pred[keep], groups=groups)
    rows = []
    for nm in names:
        obs_a, boot_a = bs.metrics(pred_by_name[nm][keep], groups=groups)
        d, lo, hi = delta_ci(boot_a, boot_s, obs_a, obs_s, metric_name)
        rows.append((nm, d, lo, hi, boot_p(boot_a, boot_s, metric_name)))
    rej = holm(np.array([r[4] for r in rows]))
    return rows, rej, obs_s[metric_name], bs


def heterogeneity(seed_pred, pred_by_name, names, y, c, groups, keep):
    """Насколько группировка вообще разнородна и насколько «худшая группа» отличается от средней.

    Без этих двух чисел результат не читается: если новая группировка даёт группы, одинаковые по
    трудности, то она проверяет не разнородность, а только толщину ячейки.
    """
    from analyze_s11 import gba_by_group, metric
    g = gba_by_group(seed_pred, y, c, keep, groups)
    vals = np.array(list(g.values()), float)
    hm = np.array([metric("hard_min", pred_by_name[n], y, c, keep, groups) for n in names])
    mn = np.array([metric("mean_gba", pred_by_name[n], y, c, keep, groups) for n in names])
    def spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        ra -= ra.mean(); rb -= rb.mean()
        d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
        return float((ra * rb).sum() / d) if d else float("nan")
    return dict(seed_by_group={str(k): round(float(v), 4) for k, v in g.items()},
                spread=float(vals.max() - vals.min()), worst=float(vals.min()),
                rho_hardmin_mean=spearman(hm, mn))


def print_heterogeneity(h):
    print("  разнородность по стартовому промпту: худшая группа %.4f, лучшая %.4f, размах %.4f"
          % (h["worst"], h["worst"] + h["spread"], h["spread"]))
    print("  отдельность цели: rho(hard-min, среднее по группам) по пулу = %.3f" % h["rho_hardmin_mean"])
    print("  GBA seed по группам: " + ", ".join(f"{k}:{v}" for k, v in h["seed_by_group"].items()))


def summarize(rows, rej):
    d = np.array([r[1] for r in rows])
    res95 = sum(not (r[2] <= 0 <= r[3]) for r in rows)
    return dict(n=len(rows), mean_abs_d=float(np.abs(d).mean()), median_d=float(np.median(d)),
                mean_width=float(np.mean([r[3] - r[2] for r in rows])),
                res95=int(res95), holm=int(rej.sum()),
                wins95=int(sum(r[2] > 0 for r in rows)),
                wins_holm=int(sum(rej[i] and rows[i][1] > 0 for i in range(len(rows)))))


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["real", "pseudo", "feature"], required=True)
    ap.add_argument("--feature", default="overlap", help="overlap | hyp_len | neg_broad | neg_sagawa")
    ap.add_argument("--quantiles", type=int, default=4, help="сколько групп для непрерывного признака")
    ap.add_argument("--n-groups", type=int, default=0, help="число псевдогрупп; 0 = как настоящих")
    ap.add_argument("--reps", type=int, default=200, help="сколько случайных разбиений в режиме pseudo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--metric", default="hard_min")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--contrasts", choices=["all", "finals"], default="all",
                    help="finals: только финалы оптимизаторов (префикс стенда), как в H6")
    args = ap.parse_args()

    y, c, rec = load_set(cfg.test_set)
    P = load_preds(cfg.test_set, len(y))
    all_names = sorted(n for n in P if not n.startswith("CONTROL:"))
    keep = valid_rows(P, len(y), all_names)
    if "seed" not in P:
        raise SystemExit("в матрице нет предсказаний seed — контрасты невозможны")
    names = [n for n in all_names if n != "seed"]
    if args.contrasts == "finals":
        names = [n for n in names if n.startswith(cfg.final_prefix + ":")]
    n_real = len(set(int(x) for x in c))
    print(f"стенд {cfg.key}, множество {cfg.test_set}: строк {len(y)}, в анализе {int(keep.sum())}; "
          f"промптов {len(all_names)}, контрастов против seed {len(names)}; метрика {args.metric}; "
          f"бутстреп {args.n_boot}")

    if args.mode == "real":
        from analyze_s11 import IDS
        print_heterogeneity(heterogeneity(P["seed"], P, all_names, y, c, tuple(IDS), keep))
        rows, rej, seed_val, bs = contrasts(P, names, P["seed"], y, c, tuple(IDS),
                                            args.metric, args.n_boot,
                                            np.random.default_rng(args.seed), keep)
        s = summarize(rows, rej)
        print(f"\nНАСТОЯЩИЕ группы ({len(IDS)}): seed {seed_val:.4f}; "
              f"средний |Δ| {s['mean_abs_d']:.4f}; ср. ширина {s['mean_width']:.4f}; "
              f"различимо {s['res95']}/{s['n']}, после Холма {s['holm']}; "
              f"значимых побед {s['wins95']}, после Холма {s['wins_holm']}")
        return 0

    if args.mode == "feature":
        vals, how = feature_values(args.feature, rec, len(y))
        if vals is None:
            raise SystemExit(how)
        binary = set(np.unique(vals)) <= {0, 1}
        g = (vals.astype(int) + 1) if binary else quantile_groups(vals, args.quantiles)
        gids = tuple(sorted(set(int(x) for x in g)))
        print(f"\nпризнак {args.feature!r}: {how}; групп {len(gids)}"
              + ("" if binary else f" (квартили, границы {np.quantile(vals, np.linspace(0,1,args.quantiles+1)[1:-1]).round(4).tolist()})"))
        cells = [(int(gg), int(ll), int(((g == gg) & (y == ll) & keep).sum())) for gg in gids for ll in (0, 1)]
        print("  ячейки «группа x метка»: " + ", ".join(f"{a}/{b}:{n}" for a, b, n in cells)
              + f"; самая тонкая {min(n for _, _, n in cells)}")
        print_heterogeneity(heterogeneity(P["seed"], P, all_names, y, g, gids, keep))
        rows, rej, seed_val, bs = contrasts(P, names, P["seed"], y, g, gids, args.metric,
                                            args.n_boot, np.random.default_rng(args.seed), keep)
        s = summarize(rows, rej)
        print(f"  seed {seed_val:.4f}; средний |Δ| {s['mean_abs_d']:.4f}; ср. ширина {s['mean_width']:.4f}; "
              f"различимо {s['res95']}/{s['n']}, после Холма {s['holm']}; "
              f"значимых побед {s['wins95']}, после Холма {s['wins_holm']}")
        top = sorted(rows, key=lambda r: -r[1])[:5]
        print("  пять наибольших Δ: " + ", ".join(f"{nm} {d:+.4f}" for nm, d, *_ in top))
        return 0

    # --- pseudo -------------------------------------------------------------------------
    ng = args.n_groups or n_real
    rng = np.random.default_rng(args.seed)
    print(f"\nПСЕВДОГРУППЫ: {ng} групп равного размера внутри каждой метки, {args.reps} разбиений, "
          f"сид {args.seed}")
    print("Разнородность групп устранена; предсказания, точность, толщина ячеек и оператор "
          "минимума прежние.")
    print_heterogeneity(heterogeneity(P["seed"], P, all_names, y,
                                      pseudo_groups(y, ng, np.random.default_rng(args.seed)),
                                      tuple(range(1, ng + 1)), keep))
    acc = []
    for rep in range(args.reps):
        g = pseudo_groups(y, ng, rng)
        gids = tuple(range(1, ng + 1))
        rows, rej, seed_val, _ = contrasts(P, names, P["seed"], y, g, gids, args.metric,
                                           args.n_boot, np.random.default_rng(args.seed + 1000 + rep), keep)
        s = summarize(rows, rej)
        s["seed_val"] = seed_val
        acc.append(s)
        if (rep + 1) % 20 == 0:
            print(f"  разбиение {rep + 1}/{args.reps} готово")
    def col(k):
        return np.array([a[k] for a in acc], float)
    print(f"\nпо {args.reps} разбиениям (медиана [5-й, 95-й процентиль]):")
    for k, lab in (("holm", "различимо после Холма"), ("res95", "различимо при 95%"),
                   ("wins_holm", "значимых побед после Холма"), ("mean_abs_d", "средний |Δ|"),
                   ("mean_width", "средняя ширина интервала"), ("seed_val", "значение seed")):
        v = col(k)
        fmt = "%.4f" if k in ("mean_abs_d", "mean_width", "seed_val") else "%.1f"
        print(("  %-28s " + fmt + "  [" + fmt + ", " + fmt + "]  размах [" + fmt + ", " + fmt + "]")
              % (lab, np.median(v), np.percentile(v, 5), np.percentile(v, 95), v.min(), v.max()))
    out = cfg.outputs / f"regroup_pseudo_{args.metric}_{ng}g_{args.reps}rep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"mode": "pseudo", "n_groups": ng, "reps": args.reps,
                               "metric": args.metric, "contrasts": len(names),
                               "per_rep": acc}, indent=1), encoding="utf-8")
    print(f"\nзаписано {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

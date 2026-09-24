#!/usr/bin/env python3
"""Седьмая проверка на MultiNLI: переносятся ли Холм-значимые выигрыши на второй скорер.

План — experiments/S13_mnli/PREREG_CHECK7.md (захэширован до оценки). Основная метрика —
hard_min на предсказаниях gpt-4o-mini; парный бутстреп внутри ячеек жанр x метка, 4000 повторов,
общие индексы; строки, где хоть один из трёх промптов дал INVALID на gpt-4o-mini, исключаются для
всех трёх; Холм по двум контрастам «финал − seed». Вторичное: жанр, худший у seed на gemma
(verbatim); точность обоих скореров против разметки по жанрам у seed; согласие скореров.

Выход: results/S13_mnli_matrix/check7.json и таблица в stdout.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
os.environ["S11_DATASET"] = "mnli"
os.environ.pop("S11_PREDS_DIR", None)

from analyze_s11 import IDS, Bootstrap, boot_p, holm, load_set  # noqa: E402
from dataset_config import cfg  # noqa: E402

MATRIX = ROOT / "results" / "S13_mnli_matrix"
GEMMA = MATRIX / "scorer_gemma" / "preds" / "truth_mnli"
GPT = MATRIX / "scorer2_gpt4omini" / "preds" / "truth_mnli"
FINALS = ["s13:43_ape__global", "s13:42_gepa__hard_min"]
NAMES = ["seed"] + FINALS
N_BOOT = 4000


def load(d: Path, n: str, rows: int) -> np.ndarray:
    a = np.load(d / f"{n.replace(':', '__', 1)}.npy").astype(int)
    if len(a) < rows:
        raise SystemExit(f"{d.name}/{n}: {len(a)} строк из {rows} — оценка не закончена")
    return a[:rows]


def group_gba(pred, y, c, g):
    pos, neg = (c == g) & (y == 1), (c == g) & (y == 0)
    return 0.5 * ((pred[pos] == 1).mean() + (pred[neg] == 0).mean())


def contrasts(P: dict, y, c, keep, target: int, rng_seed: int = 0) -> dict:
    bs = Bootstrap(y[keep], c[keep], N_BOOT, np.random.default_rng(rng_seed))
    o_s, b_s = bs.metrics(P["seed"][keep], groups=IDS)
    cells = bs.cells

    def tgt(pred):
        pos, neg = cells[(target, 1)], cells[(target, 0)]
        ob = 0.5 * ((pred[pos] == 1).mean() + (pred[neg] == 0).mean())
        bt = 0.5 * ((pred[bs.idx[:, bs.slices[(target, 1)]]] == 1).mean(1)
                    + (pred[bs.idx[:, bs.slices[(target, 0)]]] == 0).mean(1))
        return ob, bt

    ts_o, ts_b = tgt(P["seed"][keep])
    out = {}
    for n in FINALS:
        o, b = bs.metrics(P[n][keep], groups=IDS)
        d = b["hard_min"] - b_s["hard_min"]
        lo, hi = np.percentile(d, [2.5, 97.5])
        to, tb = tgt(P[n][keep])
        td = tb - ts_b
        tlo, thi = np.percentile(td, [2.5, 97.5])
        out[n] = {"hard_min": {"delta": float(o["hard_min"] - o_s["hard_min"]), "ci": [float(lo), float(hi)],
                               "p": boot_p(b, b_s, "hard_min")},
                  "target": {"delta": float(to - ts_o), "ci": [float(tlo), float(thi)],
                             "p": float(max(2 * min((td <= 0).mean(), (td >= 0).mean()), 1 / N_BOOT))}}
    for key in ("hard_min", "target"):
        rej = holm([out[n][key]["p"] for n in FINALS])
        for n, r in zip(FINALS, rej):
            out[n][key]["holm"] = bool(r)
    return out


def main() -> int:
    y, c, rec = load_set(cfg.test_set)
    n = len(y)
    G = {k: load(GEMMA, k, n) for k in NAMES}
    Q = {k: load(GPT, k, n) for k in NAMES}
    names = {int(k): v for k, v in rec["group_names"].items()}

    keep_q = np.ones(n, bool)
    for v in Q.values():
        keep_q &= v >= 0
    keep_g = np.ones(n, bool)
    for v in G.values():
        keep_g &= v >= 0
    both = keep_q & keep_g

    # целевой жанр — худший у seed на gemma (как в Таблице 3)
    seed_g = {g: group_gba(G["seed"][keep_g], y[keep_g], c[keep_g], g) for g in IDS}
    target = min(seed_g, key=seed_g.get)

    res = {"rows": n, "rows_valid_gpt": int(keep_q.sum()), "rows_valid_both": int(both.sum()),
           "invalid_gpt": {k: int((v < 0).sum()) for k, v in Q.items()},
           "target_group": names[target],
           "gpt": contrasts(Q, y, c, keep_q, target),        # основной анализ по плану
           "gemma_same_rows": contrasts(G, y, c, both, target)}  # для сопоставления на тех же строках

    # точность скореров против разметки у seed и согласие между ними
    acc = {}
    for g in IDS:
        m = both & (c == g)
        acc[names[g]] = {"gemma": float(group_gba(G["seed"][both], y[both], c[both], g)),
                         "gpt": float(group_gba(Q["seed"][both], y[both], c[both], g)),
                         "agree": float((G["seed"][m] == Q["seed"][m]).mean())}
    res["seed_accuracy"] = acc
    res["seed_accuracy_mean"] = {k: float(np.mean([a[k] for a in acc.values()])) for k in ("gemma", "gpt", "agree")}

    n_transfer = sum(1 for f in FINALS if res["gpt"][f]["hard_min"]["holm"] and res["gpt"][f]["hard_min"]["delta"] > 0)
    res["n_transfer"] = n_transfer
    res["verdict"] = ("0 из 2 переносятся — третий слой держится на трёх стендах" if n_transfer == 0 else
                      f"{n_transfer} из 2 переносятся — у третьего слоя есть граница")
    (MATRIX / "check7.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"строк {n}; валидных на gpt-4o-mini {res['rows_valid_gpt']}; INVALID gpt: {res['invalid_gpt']}")
    print(f"целевой жанр (худший у seed на gemma): {res['target_group']}\n")
    print(f"{'финал':<26} {'скорер':<8} {'Δ hard_min':>10} {'95% CI':>20} {'p':>7} {'Холм':>5}   "
          f"{'Δ целевой':>9} {'p':>7} {'Холм':>5}")
    for f in FINALS:
        for tag, key in (("gemma", "gemma_same_rows"), ("gpt", "gpt")):
            h, t = res[key][f]["hard_min"], res[key][f]["target"]
            print(f"{f:<26} {tag:<8} {h['delta']:>+10.4f} [{h['ci'][0]:+.4f}; {h['ci'][1]:+.4f}] {h['p']:>7.4f} "
                  f"{('да' if h['holm'] else 'нет'):>5}   {t['delta']:>+9.4f} {t['p']:>7.4f} {('да' if t['holm'] else 'нет'):>5}")
    print("\nточность seed против разметки (GBA по жанрам) и согласие скореров:")
    for g, a in acc.items():
        print(f"  {g:<12} gemma {a['gemma']:.4f}  gpt {a['gpt']:.4f}  согласие {a['agree']:.1%}")
    mm = res["seed_accuracy_mean"]
    print(f"  {'в среднем':<12} gemma {mm['gemma']:.4f}  gpt {mm['gpt']:.4f}  согласие {mm['agree']:.1%}")
    print(f"\nвердикт по плану: {res['verdict']}\n→ {(MATRIX / 'check7.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

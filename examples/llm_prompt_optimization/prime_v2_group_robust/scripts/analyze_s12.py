#!/usr/bin/env python
"""Анализ S12: протокол отбора внутри живой петли (3 метода x 3 протокола x 3 сида = 27 прогонов).

Что проверяется. В S11 протокол отбора применялся к готовому пулу промптов и двигал заявленный
выигрыш не меньше метода. Здесь протокол встроен в петлю (подменяет число, по которому оптимизатор
выбирает лучшего), и вопрос тот же: что сильнее сдвигает истинное качество финала -- метод или
протокол? Приложенные к скрипту блоки:

  1. Инвентарь: сколько прогонов готово (признак -- файл done.json), какой финал совпал с seed или с
     финалом другого прогона, сколько финалов уже оценено на тесте.
  2. Значимость против seed: парный бутстреп-контраст каждого финала против seed по пяти метрикам
     S11, счёт различимых без поправки и после Холма-Бонферрони, счёт точечных побед.
  3. Разложение дисперсии истинной метрики на дизайне сид x метод x протокол: суммы квадратов,
     перестановочные критерии главных эффектов метода и протокола, отношение разброса
     «протокол / метод» с бутстреп-интервалом по сидам.
  4. Средние по протоколам и методам, доли точечных побед над seed.
  5. Только по архивам (тест не нужен): какой промпт выбрала бы другая статистика на том же архиве,
     сколько таких промптов не оценено на тесте и сколько будет стоить оценка; регрет смены статистики,
     если оценка на тесте есть.

Как сопоставляются прогон и предсказания. `collect_prompts_json.py` отбрасывает промпты, чей текст уже
встречался (порядок: seed, затем финалы `sorted(root.glob("seed*/*"))` готовых прогонов). Поэтому у
финала, вернувшего seed или совпавшего с финалом другого прогона, собственного файла предсказаний нет.
Здесь предсказания ищутся по sha256[:12] текста `best_prompt.txt`, а не по имени: прогон-дубликат
остаётся полноправным наблюдением (со своими сидом, методом и протоколом) с предсказаниями
представителя. Контраст такого финала против seed равен нулю по построению: это ничья, не победа.

Предупреждение об архиве. В `archive.json` поле `R_soft_min_gba` хранит ЧИСЛО ПРОТОКОЛА, а не
настоящий soft-min: у hard_min оно равно R_worst_gba, у global -- R_global. Настоящий soft-min есть
только у прогонов soft_min, поэтому именно для них он входит в блок 5.

Для тестового множества берётся `truth_large` из S11 (3600 строк, 9 групп 0..8, ячейки по 200; метрики
считаются по группам 1..8; строки с предсказанием -1 исключаются у всех промптов сразу -- так, что
в S11 рабочий размер получился 3584). Датасет всегда CivilComments. Платных обращений нет.

`--selftest` проверяет скрипт на синтетике во временном каталоге (см. selftest()).
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import io
import itertools
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_s11 import (METRICS, Bootstrap, boot_p, cell_index, gba_by_group, holm,  # noqa: E402
                         load_set, metric, valid_rows)
from dataset_config import cfg  # noqa: E402

SET = cfg.test_set
# Префикс имён прогонов. Он же префикс файлов предсказаний в матрице соответствующего стенда:
# живая петля CivilComments -- S12, живая петля MultiNLI -- S13, и их финалы лежат в матрице
# того же номера под именами вида "s13:42_ape__global".
RUN_PREFIX = {"civil": "s12", "mnli": "s13"}.get(cfg.key, cfg.key)
# Где по умолчанию лежат прогоны, предсказания финалов и текст стартового промпта каждого стенда.
STAND = {
    "civil": dict(runs="results/S12_live_loop",
                  preds="results/S12_live_loop/finals_scorer_gemma/preds/truth_large",
                  seed_prompt="prompts/initial_prompt_civilcomments.txt",
                  seed_fallback="results/S11_protocol_matrix/preds/truth_large/seed.npy"),
    # У MultiNLI отдельного каталога финалов нет: финалы живой петли S13 -- это и есть
    # пул матрицы S13, они уже оценены на truth_mnli, поэтому предсказания берутся оттуда.
    "mnli": dict(runs="results/S13_mnli_loop",
                 preds="results/S13_mnli_matrix/scorer_gemma/preds/truth_mnli",
                 seed_prompt="prompts/initial_prompt_mnli.txt",
                 seed_fallback="results/S13_mnli_matrix/scorer_gemma/preds/truth_mnli/seed.npy"),
}.get(cfg.key)
DEFAULT_METHODS = ("ape", "evoprompt_de", "gepa")
DEFAULT_PROTOCOLS = ("soft_min", "hard_min", "global")
DEFAULT_SEEDS = (42, 43, 44)
# протокол -> поле архива, по которому петля выбирала лучшего (см. ArchiveTap в run_s12_live_loop.py)
PROTOCOL_KEY = {"soft_min": "R_soft_min_gba", "hard_min": "R_worst_gba",
                "mean_gba": "R_gba_mean", "global": "R_global"}
# статистики, вычислимые из архива у ЛЮБОГО прогона; soft_min добавляется только протоколу soft_min
STAT_KEY = {"hard_min": "R_worst_gba", "global": "R_global", "mean_gba": "R_gba_mean"}
TIE_TOL = 1e-12          # разность метрик не больше этого -- ничья
BOOT_SEED, PERM_SEED, RATIO_SEED = 0, 1, 2
UNIT_PRICE_PER_1K = 0.029   # $ за 1000 вызовов gemma (измерено в S11)


# ---------------------------------------------------------------------------------------------
# общие мелочи
# ---------------------------------------------------------------------------------------------
def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def rp(p: Path) -> Path:
    """Относительный путь считается от корня проекта (как в dataset_config)."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def fmt(x, nd=4, sign=False, width=0) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        s = "nan"
    else:
        s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.rjust(width)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(o, Path):
        return str(o)
    return o


def header(title: str) -> None:
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


# ---------------------------------------------------------------------------------------------
# прогоны и архивы
# ---------------------------------------------------------------------------------------------
@dataclass
class Run:
    seed: int
    method: str
    protocol: str
    dir: Path
    done: bool
    in_plan: bool = True
    usable: bool = False              # done.json есть и архив с финалом прочитаны
    notes: list = field(default_factory=list)
    evaluated: list = field(default_factory=list)
    n_eval: int = 0
    final_text: str = ""
    final_hash: str = ""
    final_in_archive: bool = False
    seed_in_archive: bool = False
    seed_last: bool = False
    dup_of: str = ""                  # "seed" | имя прогона-представителя | ""
    pred_src: str = ""                # имя файла предсказаний, из которого взяты предсказания

    @property
    def name(self) -> str:
        return f"{RUN_PREFIX}:{self.seed}_{self.method}__{self.protocol}"

    @property
    def cell(self):
        return (self.seed, self.method, self.protocol)

    @property
    def words(self) -> int:
        return len(self.final_text.split())


def read_run(r: Run, seed_hash: str) -> None:
    """Читает archive.json и best_prompt.txt ГОТОВОГО прогона (done.json пишется последним)."""
    try:
        arch = json.loads((r.dir / "archive.json").read_text(encoding="utf-8"))
        best = (r.dir / "best_prompt.txt").read_text(encoding="utf-8")  # как collect_prompts_json
    except Exception as e:  # noqa: BLE001 -- любой сбой чтения превращаем в пометку, а не в падение
        r.notes.append(f"не прочитан: {type(e).__name__}: {e}")
        return
    for key, want in (("protocol", r.protocol), ("method", r.method), ("seed", r.seed)):
        if arch.get(key) != want:
            r.notes.append(f"архив: {key}={arch.get(key)!r}, а каталог говорит {want!r}")
    r.evaluated = list(arch.get("evaluated") or [])
    r.n_eval = len(r.evaluated)
    bad = sum(text_hash(rec.get("prompt", "")) != rec.get("hash") for rec in r.evaluated)
    if bad:
        r.notes.append(f"{bad} записей архива: hash не равен sha256[:12] текста")
    try:
        d = json.loads((r.dir / "done.json").read_text(encoding="utf-8"))
        if d.get("unique_prompts_evaluated") not in (None, r.n_eval):
            r.notes.append(f"done.json: {d['unique_prompts_evaluated']} промптов, в архиве {r.n_eval}")
    except Exception:  # noqa: BLE001
        pass
    r.final_text, r.final_hash = best, text_hash(best)
    hashes = [rec.get("hash") for rec in r.evaluated]
    r.final_in_archive = r.final_hash in hashes
    if not r.final_in_archive:
        raw = (r.dir / "best_prompt.txt").read_bytes().decode("utf-8", errors="replace")
        for tag, variant in (("сырые байты без перевода строк", raw), ("без крайних пробелов", best.strip())):
            if text_hash(variant) in hashes:
                r.notes.append(f"финал найден в архиве только как «{tag}»")
                break
        else:
            r.notes.append("финал НЕ найден в архиве по хэшу")
    r.seed_in_archive = seed_hash in hashes
    cands = arch.get("all_candidates") or []
    r.seed_last = bool(cands) and text_hash(cands[-1]) == seed_hash
    r.usable = True


def load_runs(root: Path, plan_cells, seed_hash: str):
    """Готовые прогоны (в порядке обхода, как в collect_prompts_json) и ещё не готовые ячейки плана."""
    plan = set(plan_cells)
    runs, seen = [], set()
    for d in sorted(root.glob("seed*/*")):
        if not (d / "done.json").is_file():
            continue  # недописанный прогон: его каталог сейчас пишется, не трогаем
        try:
            seed = int(d.parent.name.replace("seed", ""))
        except ValueError:
            continue
        method, _, protocol = d.name.partition("__")
        r = Run(seed, method, protocol, d, done=True, in_plan=(seed, method, protocol) in plan)
        read_run(r, seed_hash)
        runs.append(r)
        seen.add(r.cell)
    pending = [Run(s, m, p, root / f"seed{s}" / f"{m}__{p}", done=False)
               for (s, m, p) in plan_cells if (s, m, p) not in seen]
    # дубликаты текста: первый по порядку обхода остаётся представителем, seed идёт первым
    first = {seed_hash: "seed"}
    for r in runs:
        if not r.usable:
            continue
        if r.final_hash in first:
            r.dup_of = first[r.final_hash]
        else:
            first[r.final_hash] = r.name
    return runs, pending


def load_pred_files(preds_dir: Path, n_rows: int):
    """{имя: int16-вектор}; имя = имя файла с заменой ПЕРВОГО '__' на ':' (как load_preds в analyze_s11)."""
    out, skipped = {}, []
    if not preds_dir.is_dir():
        return out, skipped
    for f in sorted(preds_dir.glob("*.npy")):
        if f.stem.startswith("_") or f.name.endswith(".partial.npy"):
            continue  # промпт ещё считается: строк меньше, чем в множестве
        if f.name.endswith(".lo.npy"):
            continue  # спутник с логарифмами шансов, не отдельный промпт
        try:
            arr = np.load(f)
        except Exception as e:  # noqa: BLE001
            skipped.append((f.name, f"не прочитан: {type(e).__name__}"))
            continue
        if arr.ndim != 1 or not (0 < len(arr) <= n_rows):
            skipped.append((f.name, f"длина {arr.shape}, ожидалось до ({n_rows},)"))
            continue
        out[f.stem.replace("__", ":", 1)] = arr
    return out, skipped


def index_preds_by_hash(P: dict, runs, seed_hash: str):
    """{хэш текста: (имя, предсказания)}. Имя -> хэш: seed, alt:<хэш> и прогоны по их best_prompt.txt."""
    run_hash = {r.name: r.final_hash for r in runs if r.usable}
    by_hash, conflicts, unknown = {}, [], []
    order = (["seed"] + [r.name for r in runs if r.usable] + sorted(k for k in P if k.startswith("alt:")))
    seen = set()
    for name in order + sorted(k for k in P if k not in set(order)):
        if name in seen or name not in P:
            continue
        seen.add(name)
        if name == "seed":
            h = seed_hash
        elif name.startswith("alt:"):
            h = name[4:]
        elif name in run_hash:
            h = run_hash[name]
        else:
            unknown.append(name)
            continue
        if h in by_hash:
            if not np.array_equal(by_hash[h][1], P[name]):
                conflicts.append((by_hash[h][0], name))
            continue
        by_hash[h] = (name, P[name])
    return by_hash, conflicts, unknown


# ---------------------------------------------------------------------------------------------
# блок 1: инвентарь
# ---------------------------------------------------------------------------------------------
def block_inventory(runs, pending, plan_cells, by_hash, seed_hash, seed_found):
    header("1. ИНВЕНТАРЬ")
    inplan = [r for r in runs if r.in_plan]
    extra = [r for r in runs if not r.in_plan]
    usable = [r for r in inplan if r.usable]
    total = len(plan_cells)
    print(f"готово {len(inplan)} из {total} прогонов плана; seed-промпт sha256[:12] = {seed_hash}"
          + ("" if seed_found else "  (!) ни в одном архиве не найден"))
    for r in runs:
        r.pred_src = by_hash[r.final_hash][0] if (r.usable and r.final_hash in by_hash) else ""
    print(f"\n{'сид':>4s} {'метод':13s} {'протокол':9s} {'|арх|':>5s} {'финал=seed':>10s} "
          f"{'дубликат текста: чей':28s} {'слов':>5s} {'тест':>5s}  примечания")
    for r in sorted(inplan, key=lambda r: (r.seed, r.method, r.protocol)):
        dup = r.dup_of if r.dup_of else "-"
        tst = "да" if r.pred_src else "нет"
        print(f"{r.seed:4d} {r.method:13s} {r.protocol:9s} {r.n_eval:5d} "
              f"{('да' if r.dup_of == 'seed' else 'нет'):>10s} {dup:28s} {r.words:5d} {tst:>5s}  "
              + "; ".join(r.notes))
    for r in extra:
        print(f"  вне плана (в анализ не входит): {r.name}")
    if pending:
        cells = ", ".join(f"{s}/{m}__{p}" for (s, m, p) in [x.cell for x in pending][:8])
        print(f"\nне готово {len(pending)}: {cells}" + (" ..." if len(pending) > 8 else ""))
    n_seed = sum(r.dup_of == "seed" for r in usable)
    n_dup_other = sum(bool(r.dup_of) and r.dup_of != "seed" for r in usable)
    n_test = sum(bool(r.pred_src) for r in usable)
    n_own = sum(bool(r.pred_src) and r.pred_src == r.name for r in usable)
    print(f"\nоптимизатор вернул seed: {n_seed} из {len(usable)}; совпало с финалом другого прогона: {n_dup_other}; "
          f"уникальных текстов финалов: {len({r.final_hash for r in usable})}")
    print(f"финалов с оценкой на тесте: {n_test} из {len(usable)} (по собственному имени {n_own}, "
          f"через текст-представитель {n_test - n_own}); слов в финале: "
          + (f"медиана {np.median([r.words for r in usable]):.0f}, "
             f"{min(r.words for r in usable)}..{max(r.words for r in usable)}" if usable else "-"))
    print(f"проверка разбора: seed есть в архиве у {sum(r.seed_in_archive for r in usable)}/{len(usable)}, "
          f"последним кандидатом у {sum(r.seed_last for r in usable)}/{len(usable)}; "
          f"финал найден в архиве у {sum(r.final_in_archive for r in usable)}/{len(usable)}")

    # Прогон, вернувший seed, — это признание «улучшения не нашлось». Чем шумнее статистика
    # отбора, тем реже такое признание: минимум по нескольким шумным группам почти всегда
    # находит мутанта, случайно оказавшегося выше. Разбивка проверяет именно это.
    by_proto = collections.defaultdict(lambda: [0, 0])
    by_method = collections.defaultdict(lambda: [0, 0])
    for r in usable:
        for d, k in ((by_proto, r.protocol), (by_method, r.method)):
            d[k][0] += 1
            d[k][1] += int(r.dup_of == "seed")
    print("вернули seed без изменений — по протоколу отбора: "
          + ", ".join(f"{k} {v[1]}/{v[0]}" for k, v in sorted(by_proto.items()))
          + "; по методу: " + ", ".join(f"{k} {v[1]}/{v[0]}" for k, v in sorted(by_method.items())))
    hm = by_proto.get("hard_min", [0, 0])
    rest = [sum(v[0] for k, v in by_proto.items() if k != "hard_min"),
            sum(v[1] for k, v in by_proto.items() if k != "hard_min")]
    if hm[0] and rest[0]:
        from scipy.stats import fisher_exact
        p = fisher_exact([[hm[1], hm[0] - hm[1]], [rest[1], rest[0] - rest[1]]])[1]
        print(f"  hard_min {hm[1]}/{hm[0]} против прочих протоколов {rest[1]}/{rest[0]}: "
              f"точный критерий Фишера p = {p:.3f}"
              + ("" if p < 0.05 else " — различие НЕ значимо, отчитывать как наблюдение"))

    return {"planned": total, "done": len(inplan), "returned_seed": n_seed, "dup_of_other_final": n_dup_other,
            "with_test": n_test, "seed_hash": seed_hash,
            "runs": [{"name": r.name, "seed": r.seed, "method": r.method, "protocol": r.protocol,
                      "n_evaluated": r.n_eval, "final_hash": r.final_hash, "returned_seed": r.dup_of == "seed",
                      "dup_of": r.dup_of, "words": r.words, "has_test": bool(r.pred_src),
                      "pred_src": r.pred_src, "seed_in_archive": r.seed_in_archive, "seed_last": r.seed_last,
                      "final_in_archive": r.final_in_archive, "notes": r.notes} for r in inplan],
            "pending": [f"{r.seed}/{r.method}__{r.protocol}" for r in pending]}


# ---------------------------------------------------------------------------------------------
# блок 2: контрасты против seed
# ---------------------------------------------------------------------------------------------
def compute_contrasts(test_runs, ob, seed_hash):
    """По каждой метрике: список контрастов «финал минус seed» с бутстреп-интервалом и поправкой Холма.

    Финал, чей текст равен seed, даёт нулевой контраст по построению (те же предсказания): он не
    проверяет никакой гипотезы, в семейство Холма не входит и в победы не засчитывается.
    """
    so, sb = ob[seed_hash]
    out = {}
    for m in METRICS:
        recs = []
        for r in test_runs:
            o, b = ob[r.final_hash]
            zero = r.final_hash == seed_hash
            dobs = float(o[m] - so[m])
            if zero:
                lo = hi = 0.0
                p = 1.0
            else:
                d = b[m] - sb[m]
                lo, hi = (float(v) for v in np.percentile(d, [2.5, 97.5]))
                p = min(1.0, boot_p(b, sb, m))
            recs.append({"name": r.name, "protocol": r.protocol, "method": r.method, "seed": r.seed,
                         "zero": zero, "delta": dobs, "lo": lo, "hi": hi, "p": p,
                         "value": float(o[m]), "seed_value": float(so[m])})
        nz = [i for i, x in enumerate(recs) if not x["zero"]]
        rej = holm(np.array([recs[i]["p"] for i in nz])) if nz else np.zeros(0, bool)
        rej_zero_counted = (holm(np.array([recs[i]["p"] for i in nz] + [1.0] * (len(recs) - len(nz))))[:len(nz)]
                            if nz else np.zeros(0, bool))
        for x in recs:
            x["holm_reject"] = False
        for i, rj in zip(nz, rej):
            recs[i]["holm_reject"] = bool(rj)
        for x in recs:
            tie = abs(x["delta"]) <= TIE_TOL
            x["point_win"] = bool(x["delta"] > TIE_TOL)
            x["tie"] = bool(tie)
            x["resolved"] = bool(not x["zero"] and (x["lo"] > 0 or x["hi"] < 0))
            x["sig_win"] = bool(not x["zero"] and x["lo"] > 0)
            x["holm_win"] = bool(x["holm_reject"] and x["lo"] > 0)
        d_all = np.array([x["delta"] for x in recs]) if recs else np.zeros(0)
        widths = [x["hi"] - x["lo"] for x in recs if not x["zero"]]
        out[m] = {
            "n": len(recs), "n_zero": len(recs) - len(nz), "family_m": len(nz),
            "point_win": sum(x["point_win"] for x in recs), "ties": sum(x["tie"] for x in recs),
            "resolved": sum(x["resolved"] for x in recs), "resolved_holm": int(rej.sum()),
            "resolved_holm_zero_counted_in_m": int(rej_zero_counted.sum()),
            "sig_win": sum(x["sig_win"] for x in recs), "holm_win": sum(x["holm_win"] for x in recs),
            "median_delta": float(np.median(d_all)) if len(d_all) else float("nan"),
            "p05_delta": float(np.percentile(d_all, 5)) if len(d_all) else float("nan"),
            "p95_delta": float(np.percentile(d_all, 95)) if len(d_all) else float("nan"),
            "mean_ci_width": float(np.mean(widths)) if widths else float("nan"),
            "runs": {x["name"]: x for x in recs}}
    return out


def print_contrasts(con, metric_name):
    header("2. ЗНАЧИМОСТЬ ПРОТИВ SEED (парный бутстреп, общие индексы; финал минус seed)")
    first = con[METRICS[0]]
    print(f"контрастов {first['n']}, из них нулевых по построению (финал = текст seed) {first['n_zero']}; "
          f"семейство Холма m = {first['family_m']} на метрику (нулевые в семейство не входят).")
    print("«победа» = Δ>0 по точечной оценке; ничья (Δ=0, в т.ч. финал = seed) победой не считается.\n"
          "«различимы» = 95%-й интервал не содержит 0 (любой знак); «значимая победа» = нижняя граница > 0.")
    print(f"\n{'метрика':12s} {'Δ>0':>5s} {'ничьи':>5s} {'различ.95%':>10s} {'различ.Холм':>11s} "
          f"{'знач.побед':>10s} {'побед Холм':>10s} {'медиана Δ':>10s} {'5..95% Δ':>19s} {'ср.ширина CI':>12s}")
    for m in METRICS:
        c = con[m]
        print(f"{m:12s} {c['point_win']:5d} {c['ties']:5d} {c['resolved']:10d} {c['resolved_holm']:11d} "
              f"{c['sig_win']:10d} {c['holm_win']:10d} {fmt(c['median_delta'], 4, True, 10)} "
              f"[{fmt(c['p05_delta'], 4, True)}, {fmt(c['p95_delta'], 4, True)}] {fmt(c['mean_ci_width'], 4, False, 12)}")
    z = con[metric_name]["resolved_holm_zero_counted_in_m"]
    if first["n_zero"]:
        print(f"(если нулевые контрасты считать в m Холма: различимых по {metric_name} было бы {z}, "
              f"а не {con[metric_name]['resolved_holm']})")
    print(f"\nпо финалам, метрика {metric_name}:")
    print(f"  {'прогон':34s} {'значение':>8s} {'Δ vs seed':>9s} {'95% CI':>19s} {'p':>7s}  флаги")
    for x in sorted(con[metric_name]["runs"].values(), key=lambda x: (x["seed"], x["method"], x["protocol"])):
        fl = []
        if x["zero"]:
            fl.append("НУЛЕВОЙ: финал = seed")
        elif x["sig_win"]:
            fl.append("значимая победа" + (" (Холм)" if x["holm_win"] else ""))
        elif x["resolved"]:
            fl.append("значимо ХУЖЕ" + (" (Холм)" if x["holm_reject"] else ""))
        elif x["point_win"]:
            fl.append("точечная победа")
        print(f"  {x['name']:34s} {fmt(x['value'], 4, False, 8)} {fmt(x['delta'], 4, True, 9)} "
              f"[{fmt(x['lo'], 4, True)}, {fmt(x['hi'], 4, True)}] {x['p']:7.4f}  {'; '.join(fl)}")


# ---------------------------------------------------------------------------------------------
# блок 3: разложение дисперсии, перестановочные критерии, отношение разброса
# ---------------------------------------------------------------------------------------------
def ss_components(Yb):
    """Суммы квадратов для пачки таблиц Yb формы (B, S, M, P): сид, метод, протокол, метод x протокол.

    Одно наблюдение в ячейке, поэтому остаток = итого - остальное: он вбирает все взаимодействия
    с сидом (сид x метод, сид x протокол, тройное).
    """
    B, S, M, P = Yb.shape
    mu = Yb.mean(axis=(1, 2, 3), keepdims=True)
    m_s = Yb.mean(axis=(2, 3), keepdims=True)
    m_m = Yb.mean(axis=(1, 3), keepdims=True)
    m_p = Yb.mean(axis=(1, 2), keepdims=True)
    m_mp = Yb.mean(axis=1, keepdims=True)

    def sq(a):
        return (a ** 2).sum(axis=(1, 2, 3))

    out = {"seed": M * P * sq(m_s - mu), "method": S * P * sq(m_m - mu), "protocol": S * M * sq(m_p - mu),
           "method_x_protocol": S * sq(m_mp - m_m - m_p + mu), "total": sq(Yb - mu)}
    res = out["total"] - out["seed"] - out["method"] - out["protocol"] - out["method_x_protocol"]
    out["residual"] = np.where(res < 0, np.maximum(res, 0.0) if False else np.where(res > -1e-12, 0.0, res), res)
    return out


def perm_pvalues(Y, n_perm, rng, chunk=2000):
    """Перестановочные критерии главных эффектов; статистика -- сумма квадратов эффекта.

    Протокол: метки протокола переставляются внутри каждой пары (сид, метод).
    Метод: метки метода переставляются внутри каждой пары (сид, протокол).
    p = (1 + #{SS_перест >= SS_набл}) / (1 + n_perm).
    """
    S, M, P = Y.shape
    obs = ss_components(Y[None])
    res = {}
    for kind, axis in (("protocol", 3), ("method", 2)):
        o = float(obs[kind][0])
        tol = 1e-12 * max(1.0, abs(o))
        cnt, done = 0, 0
        while done < n_perm:
            b = min(chunk, n_perm - done)
            idx = np.argsort(rng.random((b, S, M, P)), axis=axis)
            Yp = np.take_along_axis(np.broadcast_to(Y, (b, S, M, P)), idx, axis=axis)
            cnt += int((ss_components(Yp)[kind] >= o - tol).sum())
            done += b
        res[kind] = (1 + cnt) / (1 + n_perm)
    return res


def spread_ratio(Y, n_boot, rng):
    """sd средних по протоколам / sd средних по методам (ddof=0, как mv.std() в S11); бутстреп по сидам."""
    S = Y.shape[0]

    def sds(Yc):
        ym = Yc.mean(axis=-3)                         # (..., M, P): среднее по сидам
        return ym.mean(axis=-1).std(axis=-1), ym.mean(axis=-2).std(axis=-1)   # sd по методам, sd по протоколам

    sd_m, sd_p = (float(v) for v in sds(Y))
    sel = rng.integers(0, S, size=(n_boot, S))
    bm, bp = sds(Y[sel])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_b = np.where(bm > 0, bp / bm, np.nan)
    ok = ratio_b[~np.isnan(ratio_b)]
    lo, hi = (float(v) for v in np.percentile(ok, [2.5, 97.5])) if len(ok) else (float("nan"),) * 2
    return {"sd_method": sd_m, "sd_protocol": sd_p, "ratio": (sd_p / sd_m) if sd_m > 0 else float("nan"),
            "ci95": [lo, hi], "n_nan": int(np.isnan(ratio_b).sum()), "n_seeds": S,
            "max_distinct_resamples": math.comb(2 * S - 1, S)}


def largest_complete_box(present, seeds, methods, protocols, min_levels=(2, 2, 2)):
    """Наибольший полный подкуб (сиды x методы x протоколы), где заполнены все ячейки."""
    best, best_key = None, None
    for ns in range(min_levels[0], len(seeds) + 1):
        for S in itertools.combinations(seeds, ns):
            for nm in range(min_levels[1], len(methods) + 1):
                for M in itertools.combinations(methods, nm):
                    for np_ in range(min_levels[2], len(protocols) + 1):
                        for P in itertools.combinations(protocols, np_):
                            if all((s, m, p) in present for s in S for m in M for p in P):
                                key = (ns * nm * np_, ns, nm, np_)
                                if best_key is None or key > best_key:
                                    best, best_key = (S, M, P), key
    return best


def marginals(vals, methods, protocols):
    by_m = {m: [v for (s, mm, p), v in vals.items() if mm == m] for m in methods}
    by_p = {p: [v for (s, m, pp), v in vals.items() if pp == p] for p in protocols}
    return by_m, by_p


def block_design(vals, seeds, methods, protocols, n_perm, n_boot, metric_name):
    header(f"3. РАЗЛОЖЕНИЕ ДИСПЕРСИИ истинной метрики {metric_name}: сид x метод x протокол, одно наблюдение в ячейке")
    total_cells = len(seeds) * len(methods) * len(protocols)
    print(f"наблюдений {len(vals)} из {total_cells} ячеек")
    box = largest_complete_box(set(vals), seeds, methods, protocols)
    if box is None:
        print("Полного подблока (не меньше 2 сидов x 2 методов x 2 протоколов) нет: разложение при неполном\n"
              "дизайне не определено; вместо него только средние по методу и по протоколу с числом наблюдений.")
        by_m, by_p = marginals(vals, methods, protocols)
        for title, d in (("по методу", by_m), ("по протоколу", by_p)):
            print(f"  {title}: " + ", ".join(f"{k} {fmt(float(np.mean(v)) if v else float('nan'))} (n={len(v)})"
                                             for k, v in d.items()))
        return {"defined": False, "n_obs": len(vals)}
    S, M, P = box
    Y = np.array([[[vals[(s, m, p)] for p in P] for m in M] for s in S], dtype=float)
    complete = (len(S), len(M), len(P)) == (len(seeds), len(methods), len(protocols))
    print(f"дизайн {len(S)}x{len(M)}x{len(P)}: сиды {list(S)}, методы {list(M)}, протоколы {list(P)}"
          + ("" if complete else f"  <- ПОДБЛОК ({Y.size} из {total_cells} ячеек); остальные наблюдения не использованы"))
    ss = {k: float(v[0]) for k, v in ss_components(Y[None]).items()}
    s_, m_, p_ = Y.shape
    df = {"seed": s_ - 1, "method": m_ - 1, "protocol": p_ - 1, "method_x_protocol": (m_ - 1) * (p_ - 1),
          "total": Y.size - 1}
    df["residual"] = df["total"] - sum(df[k] for k in ("seed", "method", "protocol", "method_x_protocol"))
    pv = perm_pvalues(Y, n_perm, np.random.default_rng(PERM_SEED))
    print(f"\n{'источник':22s} {'df':>3s} {'SS':>11s} {'доля от итога':>13s}  перест. p (n_perm={n_perm})")
    labels = (("seed", "сид (блок)"), ("method", "метод"), ("protocol", "протокол"),
              ("method_x_protocol", "метод x протокол"), ("residual", "остаток"), ("total", "итого"))
    for k, lab in labels:
        share = ss[k] / ss["total"] if ss["total"] > 0 else float("nan")
        p_txt = f"{pv[k]:.4f}" if k in pv else "-"
        print(f"{lab:22s} {df[k]:3d} {ss[k]:11.6f} {fmt(share, 3, False, 13)}  {p_txt}")
    print("остаток включает взаимодействия с сидом (сид x метод, сид x протокол, тройное); "
          "взаимодействие и сид не тестируются.")
    rr = spread_ratio(Y, n_boot, np.random.default_rng(RATIO_SEED))
    print(f"\nотношение разброса «протокол / метод» = sd(средние по протоколам) / sd(средние по методам) = "
          f"{fmt(rr['sd_protocol'])} / {fmt(rr['sd_method'])} = {fmt(rr['ratio'], 3)}")
    print(f"  бутстреп по сидам, 95%: [{fmt(rr['ci95'][0], 3)}, {fmt(rr['ci95'][1], 3)}] "
          f"(нед. значений {rr['n_nan']}/{n_boot}); при {rr['n_seeds']} сидах различных ресэмплов не более "
          f"{rr['max_distinct_resamples']}, интервал грубый")
    print("  S11 на 37 промптах: протокол/метод = 0.75 (CVaR@25%), 0.82 (hard-min); 1.26 на старом кэше.")
    return {"defined": True, "complete_design": complete, "box": {"seeds": list(S), "methods": list(M),
            "protocols": list(P)}, "ss": ss, "df": df, "share": {k: ss[k] / ss["total"] for k in ss if ss["total"] > 0},
            "perm_p": pv, "n_perm": n_perm, "spread_ratio": rr}


# ---------------------------------------------------------------------------------------------
# блок 4: средние по протоколам и методам
# ---------------------------------------------------------------------------------------------
def block_means(vals, con, seeds, methods, protocols, metric_name):
    header(f"4. СРЕДНИЕ ИСТИННОЙ МЕТРИКИ {metric_name} ПО ПРОТОКОЛАМ И МЕТОДАМ")
    w = 16
    print(f"{'метод \\ протокол':18s} " + " ".join(f"{p:>{w}s}" for p in protocols) + f" {'среднее по методу':>{w + 4}s}")
    cell = {}
    for m in methods:
        row = []
        for p in protocols:
            v = [x for (s, mm, pp), x in vals.items() if mm == m and pp == p]
            cell[(m, p)] = (float(np.mean(v)) if v else float("nan"), len(v))
            row.append(f"{fmt(cell[(m, p)][0]):>7s} (n={len(v)})".rjust(w))
        allm = [x for (s, mm, pp), x in vals.items() if mm == m]
        print(f"{m:18s} " + " ".join(row) + f" {fmt(float(np.mean(allm)) if allm else float('nan')):>7s} (n={len(allm)})".rjust(w + 4))
    allp = [[x for (s, mm, pp), x in vals.items() if pp == p] for p in protocols]
    print(f"{'среднее по протоколу':18s} " + " ".join(
        f"{fmt(float(np.mean(v)) if v else float('nan')):>7s} (n={len(v)})".rjust(w) for v in allp))
    recs = con[metric_name]["runs"]

    def share(keyfun, levels, label):
        print(f"\nдоля финалов, «выигравших у seed» ({metric_name}), {label}: "
              f"точечная Δ>0 / значимая 95% / значимая после Холма")
        res = {}
        for lv in levels:
            rs = [x for x in recs.values() if keyfun(x) == lv]
            n = len(rs)
            pw, sg, hw = (sum(x[k] for x in rs) for k in ("point_win", "sig_win", "holm_win"))
            res[lv] = {"n": n, "point_win": pw, "sig_win": sg, "holm_win": hw}
            pct = f"{100 * pw / n:3.0f}%" if n else "  - "
            print(f"  {lv:14s} {pw:2d}/{n:<2d} ({pct})   {sg:2d}/{n:<2d}   {hw:2d}/{n:<2d}")
        return res

    by_p = share(lambda x: x["protocol"], protocols, "по протоколам")
    by_m = share(lambda x: x["method"], methods, "по методам")
    ties = [x["name"] for x in recs.values() if x["tie"]]
    if ties:
        print(f"ничьи (Δ=0; в точечные победы не входят): {len(ties)}")
    return {"cells": {f"{m}|{p}": {"mean": v[0], "n": v[1]} for (m, p), v in cell.items()},
            "point_win_by_protocol": by_p, "point_win_by_method": by_m}


# ---------------------------------------------------------------------------------------------
# блок 5: только архивы
# ---------------------------------------------------------------------------------------------
def archive_picks(run: Run):
    """Хэш промпта, который выбрала бы каждая вычислимая статистика на архиве прогона.

    Оценка промпта = ПЕРВАЯ запись в evals (та, по которой петля его отбирала: у EvoPrompt и GEPA
    вторая запись -- подтверждение лучшего на полном dev, в отборе не участвует). Отбракованные
    (rejected) не выбираются. Ничья -- первый в списке evaluated (строгое `>`, как в петле).
    """
    rows = []
    for rec in run.evaluated:
        evs = rec.get("evals") or []
        if evs and not evs[0].get("rejected"):
            rows.append((rec["hash"], evs[0]))
    keys = dict(STAT_KEY)
    if run.protocol == "soft_min":
        keys["soft_min"] = "R_soft_min_gba"      # настоящий soft-min только у прогонов soft_min
    picks = {}
    for st, key in keys.items():
        best_h, best_v = None, None
        for h, e in rows:
            v = e.get(key)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            if best_v is None or v > best_v:
                best_h, best_v = h, v
        picks[st] = best_h
    own = None
    if run.protocol in PROTOCOL_KEY:
        best_v = None
        for h, e in rows:
            v = e.get(PROTOCOL_KEY[run.protocol])
            if v is not None and (best_v is None or v > best_v):
                own, best_v = h, v
    return picks, own


def block_archives(runs, by_hash, seed_hash, y, c, keep_main, args, alt_out: Path | None, price=UNIT_PRICE_PER_1K):
    header("5. ТОЛЬКО ПО АРХИВАМ (тест не нужен): что выбрала бы другая статистика на архиве того же прогона")
    usable = [r for r in runs if r.usable and r.in_plan]
    if not usable:
        print("готовых прогонов нет")
        return {"runs": 0}
    final_hashes = {r.final_hash for r in usable}
    text_of = {}
    for r in usable:
        for rec in r.evaluated:
            text_of.setdefault(rec["hash"], rec["prompt"])
        text_of.setdefault(r.final_hash, r.final_text)
    per_run, alt_pairs = {}, []
    stat_tot, stat_diff = {}, {}
    consistent = 0
    n_any, n_any_diff = 0, 0
    for r in usable:
        picks, own = archive_picks(r)
        ok_own = own == r.final_hash
        consistent += ok_own
        alts = {s: h for s, h in picks.items() if s != r.protocol}
        diffs = {s: (h is not None and h != r.final_hash) for s, h in alts.items()}
        for s, d in diffs.items():
            stat_tot[(r.protocol, s)] = stat_tot.get((r.protocol, s), 0) + 1
            stat_diff[(r.protocol, s)] = stat_diff.get((r.protocol, s), 0) + int(d)
            if d:
                alt_pairs.append((r, s, alts[s]))
        n_any += bool(alts)
        n_any_diff += any(diffs.values())
        per_run[r.name] = {"final": r.final_hash, "own_argmax": own, "own_argmax_equals_final": ok_own,
                           "picks": picks, "differs": diffs}
    print(f"готовых прогонов {len(usable)}. Проверка разбора: argmax по числу СВОЕГО протокола по архиву совпал с "
          f"реальным финалом у {consistent} из {len(usable)}"
          + ("" if consistent == len(usable) else
             " (несовпадения: округление до 5 знаков в архиве даёт ничьи; у GEPA финал выбирает ещё Парето-отсев)"))
    print("выбор считается ПОСТФАКТУМ на уже оценённом архиве: это не повтор петли с другим протоколом "
          "(траектория поиска при другом протоколе была бы иной).")
    by_stat = {}
    for (proto, s), t in stat_tot.items():
        d = stat_diff[(proto, s)]
        a = by_stat.setdefault(s, [0, 0])
        a[0] += t
        a[1] += d
    print(f"\nдоля прогонов, где выбор ДРУГОЙ вычислимой статистики отличается от реального финала:")
    print(f"  {'статистика':12s} {'всего':>12s} {'отличается':>11s}")
    for s, (t, d) in sorted(by_stat.items()):
        print(f"  {s:12s} {t:12d} {d:5d} ({100 * d / t:.0f}%)")
    print(f"  хотя бы одна из альтернатив отличается: {n_any_diff} из {n_any} прогонов "
          f"({100 * n_any_diff / max(1, n_any):.0f}%)")
    print("  по паре (протокол прогона -> альтернативная статистика): "
          + "; ".join(f"{p}->{s} {stat_diff[(p, s)]}/{stat_tot[(p, s)]}" for (p, s) in sorted(stat_tot)))

    # промпты-альтернативы и их оценка на тесте
    alt_hashes = sorted({h for _, _, h in alt_pairs})
    seed_h = seed_hash
    have = [h for h in alt_hashes if h in by_hash]
    covered = [h for h in alt_hashes if h not in by_hash and (h == seed_h or h in final_hashes)]
    need = [h for h in alt_hashes if h not in by_hash and h != seed_h and h not in final_hashes]
    rows_cost = int(args.cost_rows or len(y))
    per_prompt = rows_cost * price / 1000.0
    print(f"\nразличных альтернативных промптов (по хэшу, не совпавших с финалом своего прогона): {len(alt_hashes)}")
    print(f"  уже есть оценка на тесте: {len(have)}; совпали с seed или финалом другого прогона, "
          f"оценка придёт вместе с финалами: {len(covered)}; НЕТ оценки и нужна отдельная: {len(need)}")
    print(f"  оценка отдельных на {SET} ({rows_cost} строк x ${price}/1000 вызовов = ${per_prompt:.4f} на промпт): "
          f"{len(need)} x ${per_prompt:.4f} = ${len(need) * per_prompt:.2f}"
          + f"; верхняя оценка, если оценивать и «covered» отдельно: ${(len(need) + len(covered)) * per_prompt:.2f}")
    alt_file = None
    if alt_out is not None:
        alt_out.parent.mkdir(parents=True, exist_ok=True)
        alt_out.write_text(json.dumps({f"alt:{h}": text_of[h] for h in need}, ensure_ascii=False, indent=1),
                           encoding="utf-8")
        alt_file = str(alt_out)
        print(f"  тексты {len(need)} промптов записаны в {alt_out} ({{'alt:<хэш>': текст}}) для скорера")

    # регрет смены статистики: только там, где обе оценки на тесте есть
    regret = {"pairs": 0}
    pairs_ok = [(r, s, h) for r, s, h in alt_pairs if h in by_hash and r.final_hash in by_hash]
    print(f"\nрегрет смены статистики ({args.metric}; Δ = метрика(выбор другой статистики) - метрика(реальный финал);"
          f" Δ>0: смена улучшила бы истинное качество):")
    if not pairs_ok:
        print("  нет ни одной пары, где обе оценки на тесте есть (предсказания финалов и альтернатив ещё не посчитаны)")
    else:
        pool = {by_hash[h][0]: by_hash[h][1] for h in pool_hashes(pairs_ok, by_hash)}
        arrs = list(pool.values()) + ([by_hash[seed_hash][1]] if seed_hash in by_hash else [])
        bad = np.zeros(len(y), bool)
        for arr in arrs:
            bad |= arr < 0
        keep = ~bad          # то же правило, что valid_rows: строки с -1 выпадают у всех сразу
        tv = {h: metric(args.metric, by_hash[h][1], y, c, rows=keep) for h in pool_hashes(pairs_ok, by_hash)}
        agg = {}
        for r, s, h in pairs_ok:
            d = tv[h] - tv[r.final_hash]
            agg.setdefault((r.protocol, s), []).append(d)
        allv = [d for v in agg.values() for d in v]
        print(f"  пар с оценками {len(pairs_ok)} из {len(alt_pairs)} различающихся; строк в оценке {int(keep.sum())}")
        print(f"  {'протокол->статистика':24s} {'n':>3s} {'средн.':>9s} {'медиана':>9s} {'Δ>0':>5s}")
        for k in sorted(agg):
            v = np.array(agg[k])
            print(f"  {k[0] + '->' + k[1]:24s} {len(v):3d} {v.mean():+9.4f} {np.median(v):+9.4f} {int((v > TIE_TOL).sum()):5d}")
        av = np.array(allv)
        # Неопределённость среднего регрета. Пары не независимы: один прогон даёт до трёх пар
        # (по числу альтернативных статистик), поэтому ресэмплим ПРОГОНЫ, а не пары.
        by_run = {}
        for r, st, h in pairs_ok:
            by_run.setdefault(r.name, []).append(tv[h] - tv[r.final_hash])
        runs_list = list(by_run.values())
        rng_r = np.random.default_rng(RATIO_SEED)
        boot = np.array([float(np.mean([d for i in rng_r.integers(0, len(runs_list), len(runs_list))
                                        for d in runs_list[i]])) for _ in range(4000)])
        lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
        print(f"  все: среднее {av.mean():+.4f}, медиана {np.median(av):+.4f}, "
              f"Δ>0 у {int((av > TIE_TOL).sum())} из {len(av)}")
        print(f"  разброс по парам: sd {av.std(ddof=1):.4f}, размах [{av.min():+.4f}, {av.max():+.4f}]; "
              f"кластерный бутстреп по {len(runs_list)} прогонам, 95% для среднего: [{lo:+.4f}, {hi:+.4f}] "
              f"-> среднее {'ОТЛИЧИМО' if lo > 0 or hi < 0 else 'НЕ отличимо'} от нуля")
        regret = {"pairs": len(pairs_ok), "mean": float(av.mean()), "median": float(np.median(av)),
                  "sd": float(av.std(ddof=1)), "ci95_cluster_boot": [lo, hi], "n_runs": len(runs_list),
                  "deltas": [float(x) for x in av],
                  "by_pair": {f"{k[0]}->{k[1]}": {"n": len(v), "mean": float(np.mean(v))} for k, v in agg.items()},
                  "keep_rows": int(keep.sum())}
    return {"runs": len(usable), "own_argmax_consistent": consistent, "per_run": per_run,
            "differs_by_stat": {s: {"total": t, "differs": d} for s, (t, d) in by_stat.items()},
            "any_differs": n_any_diff, "any_computable": n_any,
            "alt_distinct": len(alt_hashes), "alt_have_test": len(have), "alt_covered_by_finals": len(covered),
            "alt_need_scoring": len(need), "cost_per_prompt": per_prompt, "cost_total": len(need) * per_prompt,
            "cost_rows": rows_cost, "alt_file": alt_file, "regret": regret}


def pool_hashes(pairs_ok, by_hash):
    hs = {h for _, _, h in pairs_ok} | {r.final_hash for r, _, _ in pairs_ok}
    return sorted(h for h in hs if h in by_hash)


# ---------------------------------------------------------------------------------------------
# основной проход
# ---------------------------------------------------------------------------------------------
def run_analysis(args) -> dict:
    if STAND is None:
        raise SystemExit(f"для стенда S11_DATASET={cfg.key!r} не заданы пути живой петли; "
                         "добавьте его в STAND или укажите --runs, --preds и --seed-prompt явно")
    runs_root, preds_dir = rp(args.runs), rp(args.preds)
    out_json = rp(args.out)
    methods, protocols, seeds = args.methods, args.protocols, args.seeds
    plan_cells = [(s, m, p) for s in seeds for m in methods for p in protocols]
    y, c, rec = load_set(SET)
    n = len(y)
    seed_text = rp(args.seed_prompt).read_text(encoding="utf-8")
    seed_hash = text_hash(seed_text)
    print(f"S12: прогоны {runs_root}\n     предсказания {preds_dir}\n     множество {SET}: n={n}, "
          f"fingerprint {rec.get('fingerprint')}; метрика для блоков 3-4: {args.metric}; n_boot={args.n_boot}, "
          f"n_perm={args.n_perm}")
    runs, pending = load_runs(runs_root, plan_cells, seed_hash)
    seed_found = any(r.seed_in_archive for r in runs)

    P, skipped, by_hash, conflicts, unknown = {}, [], {}, [], []
    fallback_used = False
    have_dir = preds_dir.is_dir()
    if have_dir:
        P, skipped = load_pred_files(preds_dir, n)
        # Альтернативные выборы блока 5 держим ОТДЕЛЬНО от матрицы: если положить их файлы
        # рядом с матрицей, analyze_s11.py примет их за полноправные промпты пула.
        for extra in (args.extra_preds or []):
            ed = rp(extra)
            if not ed.is_dir():
                skipped.append((str(ed), "каталога нет"))
                continue
            more, more_skipped = load_pred_files(ed, n)
            skipped += more_skipped
            for k, v in more.items():
                if k not in P:          # основной каталог главнее
                    P[k] = v
        if "seed" not in P and args.seed_preds_fallback:
            fb = rp(args.seed_preds_fallback)
            if fb.is_file():
                arr = np.load(fb)
                if arr.ndim == 1 and len(arr) == n:
                    P["seed"] = arr
                    fallback_used = True
        if P:
            lens = {len(v) for v in P.values()}
            if len(lens) > 1 or min(lens) < n:
                m = min(lens)
                print(f"неровная матрица предсказаний: длины {sorted(lens)} -> срез до общего префикса "
                      f"в {m} строк ({sum(len(v) > m for v in P.values())} из {len(P)} файлов длиннее). "
                      f"Так бюджетная остановка оставляет прямоугольный, пригодный для анализа кусок.")
                P = {k: v[:m] for k, v in P.items()}
                y, c, n = y[:m], c[:m], m
        by_hash, conflicts, unknown = index_preds_by_hash(P, runs, seed_hash)
    inv = block_inventory(runs, pending, plan_cells, by_hash, seed_hash, seed_found)
    summary = {"params": {"runs": runs_root, "preds": preds_dir, "metric": args.metric, "n_boot": args.n_boot,
                          "n_perm": args.n_perm, "methods": methods, "protocols": protocols, "seeds": seeds,
                          "set": SET, "n_rows": n, "fingerprint": rec.get("fingerprint")},
               "inventory": inv}
    if have_dir:
        print(f"\nпредсказания: {len(P)} файлов в каталоге принято"
              + (f"; seed взят из резерва {rp(args.seed_preds_fallback)}" if fallback_used else "")
              + (f"; пропущено {len(skipped)}: " + "; ".join(f"{a} ({b})" for a, b in skipped[:4]) if skipped else "")
              + (f"; неизвестных имён {len(unknown)}: {unknown[:3]}" if unknown else ""))
        if conflicts:
            print(f"  (!) предсказания одного текста различаются в файлах: {conflicts[:3]}")
    keep_main = None
    usable = [r for r in runs if r.usable and r.in_plan]
    test_runs = [r for r in usable if r.final_hash in by_hash]
    if not have_dir:
        header("2-4. ПРЕДСКАЗАНИЯ ФИНАЛОВ НА ТЕСТЕ")
        print(f"каталога предсказаний нет: {preds_dir}\nоценки финалов на {SET} ещё не посчитаны "
              "(scripts/score_s11_scorer2.py, см. заголовок); блоки 2-4 пропущены, блок 5 работает по архивам.")
    elif seed_hash not in by_hash:
        header("2-4. ПРЕДСКАЗАНИЯ ФИНАЛОВ НА ТЕСТЕ")
        print("в каталоге нет предсказаний seed (файл seed.npy), парные контрасты невозможны; блоки 2-4 пропущены.")
    elif not test_runs:
        header("2-4. ПРЕДСКАЗАНИЯ ФИНАЛОВ НА ТЕСТЕ")
        print("ни у одного готового финала пока нет оценки на тесте; блоки 2-4 пропущены.")
    else:
        need_h = {seed_hash} | {r.final_hash for r in test_runs}
        pool = {by_hash[h][0]: by_hash[h][1] for h in need_h}
        print()
        keep_main = valid_rows(pool, n)
        bs = Bootstrap(y, c, args.n_boot, np.random.default_rng(BOOT_SEED), cells=cell_index(y, c, keep_main))
        ob = {h: bs.metrics(by_hash[h][1]) for h in need_h}
        print(f"строк в анализе: {int(keep_main.sum())} из {n}; уникальных предсказаний: {len(need_h)} "
              f"(seed + {len(need_h) - 1} различных текстов финалов) на {len(test_runs)} прогонов")
        con = compute_contrasts(test_runs, ob, seed_hash)
        print_contrasts(con, args.metric)
        vals = {r.cell: float(ob[r.final_hash][0][args.metric]) for r in test_runs}
        summary["contrasts"] = con
        summary["design"] = block_design(vals, seeds, methods, protocols, args.n_perm, args.n_boot, args.metric)
        summary["means"] = block_means(vals, con, seeds, methods, protocols, args.metric)
        summary["rows_used"] = int(keep_main.sum())
    alt_out = out_json.parent / "alt_picks_prompts.json"
    summary["archive_only"] = block_archives(runs, by_hash, seed_hash, y, c, keep_main, args, alt_out)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nзаписано: {out_json}")
    return summary


# ---------------------------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------------------------
class Checks:
    def __init__(self):
        self.rows = []

    def add(self, ok, text):
        self.rows.append((bool(ok), text))
        print(("PASS  " if ok else "FAIL  ") + text)

    @property
    def ok(self):
        return all(o for o, _ in self.rows)


class SynthWorld:
    """Синтетический «тест»: ячейки группа x метка из truth_large.json.

    У строки есть общая трудность z (одинаковая для всех промптов), у промпта -- независимый шум:
    строка решена, если rho*z + sqrt(1-rho^2)*e < Phi^-1(p_ячейки + сдвиг). Так предсказания разных
    промптов коррелированы, как настоящие (тот же тест, те же трудные строки). Сдвиг delta двигает
    вероятность решить строку во всех ячейках, то есть и метрики худшей группы, на delta.
    Предсказание = метка, если строка решена, иначе 1-метка; редкие -1 имитируют неразобранные ответы.
    """

    def __init__(self, y, c, rng, rho=0.7, bad_rate=1.5e-4):
        self.y, self.c, self.rho, self.bad_rate = y, c, rho, bad_rate
        cells = sorted(set(zip(c.tolist(), y.tolist())))
        idx = {k: i for i, k in enumerate(cells)}
        self.cell_of = np.array([idx[(int(g), int(l))] for g, l in zip(c, y)])
        self.p0 = rng.uniform(0.60, 0.85, size=len(cells))
        self.z = rng.standard_normal(len(y))
        self.nd = NormalDist()

    def predict(self, delta, rng):
        thr = np.array([self.nd.inv_cdf(min(max(float(p + delta), 0.01), 0.99)) for p in self.p0])[self.cell_of]
        e = rng.standard_normal(len(self.y))
        correct = self.rho * self.z + math.sqrt(1 - self.rho ** 2) * e < thr
        pred = np.where(correct, self.y, 1 - self.y).astype(np.int16)
        pred[rng.random(len(self.y)) < self.bad_rate] = -1
        return pred


def synth_cells(world, rng, eff_m, eff_p, dims, seed_sd, n_dup_seed=3, n_dup_pair=2):
    """Предсказания 27 прогонов (i,j,k) + seed, с подложенными эффектами и дубликатами текста.

    Возвращает {ячейка: индекс уникального предсказания}, список уникальных предсказаний и метки дубликатов.
    Дубликаты ставятся в равномерно случайные ячейки, чтобы не нарушать обмениваемость меток под нулём.
    """
    S, M, P = dims
    cells = list(itertools.product(range(S), range(M), range(P)))
    seed_shift = rng.normal(0.0, seed_sd, S)
    uniq = [world.predict(0.0, rng)]                       # 0 = seed
    which = {}
    for (i, j, k) in cells:
        uniq.append(world.predict(float(seed_shift[i] + eff_m[j] + eff_p[k]), rng))
        which[(i, j, k)] = len(uniq) - 1
    order = rng.permutation(len(cells))
    for t in order[:n_dup_seed]:
        which[cells[t]] = 0                                   # финал вернул seed
    if n_dup_pair >= 2:
        a = cells[order[n_dup_seed]]
        for t in order[n_dup_seed + 1:n_dup_seed + n_dup_pair]:
            which[cells[t]] = which[a]                       # два финала совпали друг с другом
    return which, uniq, cells


def fast_pvalues(world, y, c, rng, eff_m, eff_p, n_perm, dims, seed_sd, metric_name="cvar25"):
    which, uniq, cells = synth_cells(world, rng, eff_m, eff_p, dims, seed_sd)
    used = sorted(set(which.values()) | {0})
    bad = np.zeros(len(y), bool)
    for u in used:
        bad |= uniq[u] < 0
    keep = ~bad
    tv = {u: metric(metric_name, uniq[u], y, c, rows=keep) for u in used}
    S, M, P = dims
    Y = np.zeros((S, M, P))
    for (i, j, k), u in which.items():
        Y[i, j, k] = tv[u]
    pv = perm_pvalues(Y, n_perm, rng)
    sm = Y.mean(axis=(0, 1)).std()
    mm = Y.mean(axis=(0, 2)).std()
    return pv["protocol"], pv["method"], (sm / mm if mm > 0 else float("nan"))


# ---- синтетические архивы для сквозного прохода ----
WORDS = ("токсичность комментарий группа метка ответ классифицируй инструкция пример строго нейтрально "
         "hate slur insult identity threat context sarcasm quote reply label output json").split()


def synth_text(rng, tag, n_words):
    return f"[{tag}] " + " ".join(rng.choice(WORDS, n_words))


def synth_archive(rng, run, seed_text, final_text, borrowed_text, flags):
    """Архив с известными по построению победителями каждой статистики.

    Ловушки: отбракованный промпт с рекордными числами; у финала EvoPrompt/GEPA вторая оценка (полный
    dev) с завышенными числами; близнец победителя с теми же числами, стоящий позже (ничья -> первый).
    """
    protocol = run["protocol"]
    stats = ["hard_min", "global", "mean_gba"] + (["soft_min"] if protocol == "soft_min" else [])
    alt_stats = [s for s in stats if s != protocol]
    texts = [final_text]
    if seed_text != final_text:
        texts.append(seed_text)
    if borrowed_text and borrowed_text not in texts:
        texts.append(borrowed_text)
    fresh = [synth_text(rng, f"cand {run['name']} #{k}", 30) for k in range(int(rng.integers(3, 6)))]
    texts += fresh
    base = {t: {"R_worst_gba": rng.uniform(0.50, 0.60), "R_global": rng.uniform(0.66, 0.70),
                "R_gba_mean": rng.uniform(0.66, 0.70), "soft": rng.uniform(0.58, 0.66)} for t in texts}
    planted = {}
    key_of = {"hard_min": "R_worst_gba", "global": "R_global", "mean_gba": "R_gba_mean", "soft_min": "soft"}
    others = [t for t in texts if t != final_text]
    for s in alt_stats:
        w = final_text if rng.random() < 0.25 else others[int(rng.integers(0, len(others)))]
        key = key_of[s]
        base[w][key] = max(v[key] for v in base.values()) + 0.08   # гарантированный, не вероятный, максимум
        planted[s] = w
    own_key = {"soft_min": "soft", "hard_min": "R_worst_gba", "global": "R_global"}[protocol]
    base[final_text][own_key] = max(v[own_key] for v in base.values()) + 0.1
    # число протокола в поле R_soft_min_gba (ArchiveTap): soft_min -> сам soft-min, иначе число протокола

    def evals_for(t, rows=900):
        b = base[t]
        used = b["soft"] if protocol == "soft_min" else (b["R_worst_gba"] if protocol == "hard_min" else b["R_global"])
        return {"rows": rows, "selected_on": round(used, 5), "rejected": None, "R_soft_min_gba": round(used, 5),
                "R_worst_gba": round(b["R_worst_gba"], 5), "R_gba_mean": round(b["R_gba_mean"], 5),
                "R_global": round(b["R_global"], 5), "R_worst_group": 0.5, "toxic_recall": 0.7,
                "invalid_rate": 0.0, "pred_pos_rate": 0.55}
    order = [texts[i] for i in rng.permutation(len(texts))]
    recs = []
    sub_rows = 400 if run["method"] in ("gepa", "evoprompt_de") else 900
    for t in order:
        ev = [evals_for(t, sub_rows)]
        if flags.get("second_eval") and t == final_text and sub_rows != 900:
            e2 = evals_for(t, 900)
            for k in ("R_soft_min_gba", "R_worst_gba", "R_gba_mean", "R_global"):
                e2[k] = 0.95
            ev.append(e2)                                    # полный dev: числа завышены нарочно
        recs.append({"hash": text_hash(t), "prompt": t, "evals": ev})
    if flags.get("reject"):
        t = synth_text(rng, f"rejected {run['name']}", 20)
        ev = evals_for(final_text)
        ev.update({"selected_on": -1e9, "rejected": "invalid_rate", "R_worst_gba": 0.9, "R_global": 0.9,
                   "R_gba_mean": 0.9, "invalid_rate": 0.2})
        del ev["R_soft_min_gba"]
        recs.insert(int(rng.integers(0, len(recs) + 1)), {"hash": text_hash(t), "prompt": t, "evals": [ev]})
    if flags.get("twin") and alt_stats:
        w = planted[alt_stats[-1]]
        twin = synth_text(rng, f"twin {run['name']}", 20)
        src = next(r_ for r_ in recs if r_["prompt"] == w)
        recs.append({"hash": text_hash(twin), "prompt": twin, "evals": [dict(e) for e in src["evals"]]})
    cands = [r_["prompt"] for r_ in recs if r_["prompt"] != seed_text] + [seed_text]
    arch = {"protocol": protocol, "method": run["method"], "seed": run["seed"], "all_candidates": cands,
            "evaluated": recs}
    return arch, {s: text_hash(w) for s, w in planted.items()}, fresh


def write_synth_runs(root, preds_dir, world, y, c, rng, seed_text, eff_m, eff_p, methods, protocols, seeds):
    """27 готовых прогонов + предсказания, разложенные так, как это делает collect_prompts_json."""
    root.mkdir(parents=True, exist_ok=True)
    (root.parent / "seed_prompt.txt").write_text(seed_text, encoding="utf-8")
    cells = list(itertools.product(seeds, methods, protocols))
    names = {cell: f"s12:{cell[0]}_{cell[1]}__{cell[2]}" for cell in cells}
    order = sorted(cells, key=lambda t: (f"seed{t[0]}", f"{t[1]}__{t[2]}"))   # порядок sorted(glob)
    pick = rng.permutation(len(cells))
    seed_dups = [cells[i] for i in pick[:3]]
    pair = [cells[pick[3]], cells[pick[4]]]
    seed_shift = {s: float(rng.normal(0, 0.008)) for s in seeds}
    text_of, delta_of = {}, {}
    shared = synth_text(rng, "общий финал двух прогонов", 60)
    for cell in cells:
        s, m, p = cell
        if cell in seed_dups:
            text_of[cell] = seed_text
        elif cell in pair:
            text_of[cell] = shared
        else:
            text_of[cell] = synth_text(rng, f"final {names[cell]}", int(rng.integers(150, 260)))
        delta_of[cell] = seed_shift[s] + eff_m[methods.index(m)] + eff_p[protocols.index(p)]
    pred_of_text = {seed_text: world.predict(0.0, rng)}
    for cell in cells:
        t = text_of[cell]
        if t not in pred_of_text:
            pred_of_text[t] = world.predict(float(delta_of[cell]), rng)
    borrowed = text_of[cells[pick[5]]]                        # чужой финал как альтернатива в архивах
    expected = {"picks": {}, "fresh": {}, "dup_of": {}, "final_hash": {}}
    all_fresh = {}
    first_of = {text_hash(seed_text): "seed"}
    for cell in order:
        s, m, p = cell
        rdir = root / f"seed{s}" / f"{m}__{p}"
        rdir.mkdir(parents=True)
        run = {"name": names[cell], "method": m, "protocol": p, "seed": s}
        flags = {"reject": bool(rng.random() < 0.4), "second_eval": True, "twin": bool(rng.random() < 0.3)}
        arch, planted, fresh = synth_archive(rng, run, seed_text, text_of[cell], borrowed, flags)
        (rdir / "best_prompt.txt").write_text(text_of[cell], encoding="utf-8")     # CRLF на Windows, как в петле
        (rdir / "archive.json").write_text(json.dumps(arch, ensure_ascii=False), encoding="utf-8")
        (rdir / "optimizer_result.json").write_text("{}", encoding="utf-8")
        (rdir / "done.json").write_text(json.dumps({"seconds": 1.0, "unique_prompts_evaluated": len(arch["evaluated"]),
                                                    "scorer_calls": 0, "optimizer_calls": 0,
                                                    "finished": "2000-01-01 00:00:00"}), encoding="utf-8")
        h = text_hash(text_of[cell])
        expected["final_hash"][names[cell]] = h
        expected["dup_of"][names[cell]] = first_of.get(h, "")
        first_of.setdefault(h, names[cell])
        expected["picks"][names[cell]] = planted
        for t in fresh:
            all_fresh[text_hash(t)] = t
    # предсказания только для имён из finals_prompts.json, собранного НАСТОЯЩИМ collect_prompts_json.py
    fp = root.parent / "finals_prompts.json"
    import os as _os
    env = dict(_os.environ, PYTHONIOENCODING="utf-8")   # stdout сабпроцесса на Windows иначе падает на кириллице
    subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "collect_prompts_json.py"),
                    "--seed-prompt", str(root.parent / "seed_prompt.txt"), "--finals-root", str(root),
                    "--prefix", "s12", "--out", str(fp)], check=True, capture_output=True, env=env)
    uniq_prompts = json.loads(fp.read_text(encoding="utf-8"))
    preds_dir.mkdir(parents=True, exist_ok=True)
    for name, text in uniq_prompts.items():
        np.save(preds_dir / (name.replace(":", "__", 1) + ".npy"), pred_of_text[text])
    expected["n_pred_files"] = len(uniq_prompts)
    expected["fresh"] = all_fresh
    expected["pred_of_text"] = pred_of_text
    expected["text_of"] = {names[c_]: text_of[c_] for c_ in cells}
    expected["seed_dups"] = [names[c_] for c_ in seed_dups]
    expected["pair"] = [names[c_] for c_ in sorted(pair, key=lambda t: order.index(t))]
    return expected


def selftest(args) -> int:
    ck = Checks()
    root = rp(args.runs) / "analysis"
    tmp = root / "selftest_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        y, c, _ = load_set(SET)
        rng = np.random.default_rng(12_012)
        world = SynthWorld(y, c, rng)
        methods, protocols, seeds = args.methods, args.protocols, args.seeds
        dims = (len(seeds), len(methods), len(protocols))
        header("SELFTEST 0. Численные тождества")
        # 0.1: разложение суммы квадратов против прямого счёта по определению
        Yt = np.random.default_rng(1).normal(size=dims)
        ssv = {k: float(v[0]) for k, v in ss_components(Yt[None]).items()}
        mu = Yt.mean()
        S, M, P = dims
        brute = {"seed": sum(M * P * (Yt[i].mean() - mu) ** 2 for i in range(S)),
                 "method": sum(S * P * (Yt[:, j].mean() - mu) ** 2 for j in range(M)),
                 "protocol": sum(S * M * (Yt[:, :, k].mean() - mu) ** 2 for k in range(P)),
                 "method_x_protocol": sum(S * (Yt[:, j, k].mean() - Yt[:, j].mean() - Yt[:, :, k].mean() + mu) ** 2
                                          for j in range(M) for k in range(P)),
                 "total": float(((Yt - mu) ** 2).sum())}
        ck.add(all(abs(ssv[k] - brute[k]) < 1e-10 for k in brute)
               and abs(ssv["total"] - sum(ssv[k] for k in ("seed", "method", "protocol", "method_x_protocol", "residual"))) < 1e-10,
               "суммы квадратов совпали с прямым счётом по определению, части складываются в итог")
        # 0.2: аддитивная таблица без остатка и без взаимодействий
        a, b, g = np.array([0.0, .01, -.01]), np.array([.02, -.02, 0.0]), np.array([.03, 0.0, -.03])
        Ya = a[:, None, None] + b[None, :, None] + g[None, None, :]
        sa = {k: float(v[0]) for k, v in ss_components(Ya[None]).items()}
        ck.add(abs(sa["residual"]) < 1e-12 and abs(sa["method_x_protocol"]) < 1e-12
               and abs(sa["method"] - 9 * float((b ** 2).sum())) < 1e-12
               and abs(sa["protocol"] - 9 * float((g ** 2).sum())) < 1e-12,
               "аддитивная таблица: остаток и метод x протокол равны 0, SS метода/протокола = 9*сумма квадратов эффектов")
        # 0.3: наибольший полный подблок
        present = {(s, m, p) for s in (42, 43) for m in methods for p in protocols} | {(44, "ape", "soft_min")}
        bx = largest_complete_box(present, seeds, methods, protocols)
        ck.add(bx is not None and set(bx[0]) == {42, 43} and len(bx[1]) == 3 and len(bx[2]) == 3,
               "неполный дизайн: найден полный подблок 2 сида x 3 x 3")
        only_ape = {(s, "ape", p) for s in seeds for p in protocols}
        ck.add(largest_complete_box(only_ape, seeds, methods, protocols) is None,
               "только один метод: подблока нет, разложение не определено")
        # 0.4: Bootstrap.metrics совпадает с metric() на синтетике
        pr = world.predict(0.0, rng)
        keep = pr >= 0
        bs0 = Bootstrap(y, c, 50, np.random.default_rng(3), cells=cell_index(y, c, keep))
        o0, _ = bs0.metrics(pr)
        ck.add(all(abs(o0[m] - metric(m, pr, y, c, rows=keep)) < 1e-9 for m in METRICS),
               "наблюдаемые метрики Bootstrap.metrics совпали с metric() по всем пяти метрикам")
        gb = gba_by_group(pr, y, c, keep)
        print(f"  (синтетический seed: GBA групп 1..8 от {min(gb.values()):.3f} до {max(gb.values()):.3f}, "
              f"строк -1: {int((pr < 0).sum())})")

        # ---- сквозной проход: файлы -> collect_prompts_json.py -> analyze -> проверки ----
        header("SELFTEST 1. Сквозной проход на синтетических файлах (27 прогонов, дубликаты текста)")
        seed_text = rp(args.seed_prompt).read_text(encoding="utf-8")
        eff_m, eff_p = [-0.02, 0.0, 0.02], [0.0, 0.0, 0.0]
        ex = write_synth_runs(tmp / "runs", tmp / "runs" / "finals_scorer_gemma" / "preds" / "truth_large",
                              world, y, c, np.random.default_rng(77), seed_text, eff_m, eff_p,
                              list(methods), list(protocols), list(seeds))
        ck.add(ex["n_pred_files"] == 1 + 27 - 3 - 1, f"collect_prompts_json оставил {ex['n_pred_files']} промптов "
               "(seed + 27 финалов - 3 совпавших с seed - 1 совпавший с другим финалом = 24)")
        ns = argparse.Namespace(**{**vars(args), "runs": tmp / "runs",
                                   "preds": tmp / "runs" / "finals_scorer_gemma" / "preds" / "truth_large",
                                   "out": tmp / "runs" / "analysis" / "s12_summary.json",
                                   "n_boot": 300, "n_perm": 2000, "seed_preds_fallback": None})
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            sm = run_analysis(ns)
        inv = {r["name"]: r for r in sm["inventory"]["runs"]}
        ck.add(sm["inventory"]["done"] == 27 and len(inv) == 27, "инвентарь: готово 27 из 27")
        ck.add(all(inv[k]["final_hash"] == v for k, v in ex["final_hash"].items()),
               "хэш финала из best_prompt.txt (CRLF на диске) совпал с ожидаемым sha256[:12] текста")
        ck.add(all(inv[k]["dup_of"] == v for k, v in ex["dup_of"].items()),
               "«дубликат текста: чей»: seed-дубликаты -> seed, второй из пары -> первый в порядке sorted(glob)")
        ck.add(sm["inventory"]["returned_seed"] == 3 and sm["inventory"]["dup_of_other_final"] == 1,
               "вернули seed: 3; совпал с чужим финалом: 1")
        ck.add(all(r["has_test"] for r in inv.values()), "у всех 27 финалов есть оценка на тесте (в т.ч. через представителя)")
        ck.add(all(r["seed_in_archive"] and r["seed_last"] and r["final_in_archive"] for r in inv.values()),
               "разбор архива: seed найден (последним кандидатом), финал найден в архиве у всех 27")
        con = sm["contrasts"]["cvar25"]["runs"]
        ck.add(all(con[k]["zero"] and con[k]["lo"] == 0.0 and con[k]["hi"] == 0.0 and not con[k]["point_win"]
                   and con[k]["tie"] and not con[k]["sig_win"] for k in ex["seed_dups"]),
               "контраст финала-дубликата seed нулевой (интервал [0,0]), это ничья, не победа")
        ck.add(sm["contrasts"]["cvar25"]["n_zero"] == 3 and sm["contrasts"]["cvar25"]["family_m"] == 24,
               "семейство Холма = 24 ненулевых контраста, 3 нулевых в него не входят")
        a_, b_ = ex["pair"]
        ck.add(abs(con[a_]["delta"] - con[b_]["delta"]) < 1e-15 and abs(con[a_]["lo"] - con[b_]["lo"]) < 1e-15,
               "два совпавших между собой финала имеют одинаковые контрасты (предсказания представителя)")
        # блок 5: победители статистик известны по построению
        ar = sm["archive_only"]
        ok_pick = True
        for name, planted in ex["picks"].items():
            got = ar["per_run"][name]["picks"]
            ok_pick &= all(got.get(s) == h for s, h in planted.items())
            ok_pick &= ("soft_min" in got) == name.endswith("__soft_min")
            ok_pick &= ar["per_run"][name]["own_argmax_equals_final"]
        ck.add(ok_pick, "блок 5: выбор каждой статистики совпал с подложенным победителем (ловушки: отбракованный "
                        "промпт, вторая оценка на полном dev, близнец с той же ценой, ничья -> первый); "
                        "soft_min только у прогонов soft_min; argmax по своему числу = финал у всех 27")
        exp_need = set()
        alt_pairs = {(n_, s, h) for n_, pl in ex["picks"].items() for s, h in pl.items() if h != ex["final_hash"][n_]}
        exp_need = {h for _, _, h in alt_pairs if h in {text_hash(t) for t in ex["fresh"].values()}}
        exp_need = {h for h in exp_need}
        # ожидаемые: альтернативы, не равные seed и не равные финалам
        seed_h_, finals_ = text_hash(seed_text), set(ex["final_hash"].values())
        exp_need = {h for _, _, h in alt_pairs if h != seed_h_ and h not in finals_}
        alt_json = json.loads((tmp / "runs" / "analysis" / "alt_picks_prompts.json").read_text(encoding="utf-8"))
        ck.add(set(alt_json) == {f"alt:{h}" for h in exp_need}
               and all(text_hash(t) == k[4:] for k, t in alt_json.items()) and ar["alt_need_scoring"] == len(exp_need),
               f"alt_picks_prompts.json: ровно {len(exp_need)} альтернатив без оценки (не seed, не финалы), "
               "ключи alt:<хэш>, тексты совпали с хэшем")
        ck.add(abs(ar["cost_total"] - len(exp_need) * 3600 * 0.029 / 1000) < 1e-9,
               f"стоимость = {len(exp_need)} x 3600 x $0.029/1000 = ${ar['cost_total']:.2f}")
        ck.add(ar["regret"]["pairs"] > 0, f"регрет: есть пары с оценками на тесте (альтернатива = финал другого прогона или seed): "
                                          f"{ar['regret']['pairs']}")
        # докладываем оценки для одной «свежей» альтернативы и сверяем регрет независимо
        target_alt = sorted(exp_need)[0]
        pred_alt = world.predict(0.0, np.random.default_rng(5))
        np.save(tmp / "runs" / "finals_scorer_gemma" / "preds" / "truth_large" / f"alt__{target_alt}.npy", pred_alt)
        with contextlib.redirect_stdout(io.StringIO()):
            sm2 = run_analysis(ns)
        ck.add(sm2["archive_only"]["alt_need_scoring"] == len(exp_need) - 1
               and sm2["archive_only"]["regret"]["pairs"] > ar["regret"]["pairs"],
               "после добавления предсказаний alt:<хэш> он выпал из «нужна оценка», а число пар регрета выросло")
        # независимый пересчёт регрета: тот же промпт против финала прогона
        fr = {}
        for name, pl in ex["picks"].items():
            for s, h in pl.items():
                if h == target_alt and h != ex["final_hash"][name]:
                    fr[(name, s)] = ex["final_hash"][name]
        pool_arrs = [pred_alt] + [ex["pred_of_text"][ex["text_of"][n_]] for (n_, _) in fr]
        keep_r = np.ones(len(y), bool)
        for arr in pool_arrs + [ex["pred_of_text"][seed_text]]:
            keep_r &= arr >= 0
        d_exp = []
        for (name, s), fh in fr.items():
            d_exp.append(metric(args.metric, pred_alt, y, c, rows=keep_r)
                         - metric(args.metric, ex["pred_of_text"][ex["text_of"][name]], y, c, rows=keep_r))
        got_pairs = sm2["archive_only"]["regret"]["pairs"] - ar["regret"]["pairs"]
        ck.add(got_pairs == len(fr), f"регрет: число добавленных пар {got_pairs} = ожидаемому {len(fr)}")
        del d_exp  # сравнение самих Δ ниже, на уровне суммарного среднего, было бы зависимым от других пар

        # ---- перестановочные критерии на синтетике ----
        header(f"SELFTEST 2. Перестановочные критерии: {args.selftest_reps} повторов на сценарий, "
               f"дизайн {dims[0]}x{dims[1]}x{dims[2]}, 3 финала = seed, 2 финала совпали (случайные ячейки), "
               f"n_perm={args.selftest_perm}")
        scenarios = [("нуль: эффектов метода и протокола нет", [0, 0, 0], [0, 0, 0]),
                     ("эффект метода ±0.02, протокола нет", [-0.02, 0, 0.02], [0, 0, 0]),
                     ("эффект протокола ±0.02, метода нет", [0, 0, 0], [-0.02, 0, 0.02]),
                     ("слабый эффект метода ±0.0135 (sd 0.011, как в S11)", [-0.0135, 0, 0.0135], [0, 0, 0]),
                     ("слабый эффект протокола ±0.0135 (sd 0.011)", [0, 0, 0], [-0.0135, 0, 0.0135])]
        res = {}
        for si, (title, em, ep) in enumerate(scenarios):
            r_ = np.random.default_rng(9_000 + si)
            pp, pm, rt = [], [], []
            for _ in range(args.selftest_reps):
                a1, a2, a3 = fast_pvalues(world, y, c, r_, em, ep, args.selftest_perm, dims, 0.008)
                pp.append(a1)
                pm.append(a2)
                rt.append(a3)
            pp, pm, rt = np.array(pp), np.array(pm), np.array(rt)
            res[si] = (pp, pm, rt)
            print(f"  {title}\n     протокол: p<0.05 в {100 * (pp < .05).mean():5.1f}% повторов (p<0.10: {100 * (pp < .10).mean():4.1f}%, "
                  f"среднее p {pp.mean():.3f});  метод: p<0.05 в {100 * (pm < .05).mean():5.1f}% "
                  f"(p<0.10: {100 * (pm < .10).mean():4.1f}%, среднее p {pm.mean():.3f});  "
                  f"медиана sd_прот/sd_метод {np.nanmedian(rt):.2f}")
        lo, hi = 0.01, 0.09      # 5% +- 2.6 SE при 200 повторах
        pp, pm, _ = res[0]
        ck.add(lo <= (pp < .05).mean() <= hi and lo <= (pm < .05).mean() <= hi,
               f"(а) под нулём доля p<0.05: протокол {100 * (pp < .05).mean():.1f}%, метод {100 * (pm < .05).mean():.1f}% "
               f"(норма 5%, допуск {int(lo * 100)}-{int(hi * 100)}%); средние p {pp.mean():.2f}/{pm.mean():.2f}")
        pp, pm, _ = res[1]
        ck.add((pm < .05).mean() >= 0.95 and lo <= (pp < .05).mean() <= hi,
               f"(б) эффект метода: критерий метода находит его в {100 * (pm < .05).mean():.0f}% повторов, "
               f"критерий протокола ложно срабатывает в {100 * (pp < .05).mean():.1f}%")
        pp, pm, _ = res[2]
        ck.add((pp < .05).mean() >= 0.95 and lo <= (pm < .05).mean() <= hi,
               f"(в) эффект протокола: критерий протокола находит его в {100 * (pp < .05).mean():.0f}% повторов, "
               f"критерий метода ложно срабатывает в {100 * (pm < .05).mean():.1f}%")
        print(f"  справочно (без проверок): при эффекте размера S11 (sd 0.011) находится методный "
              f"{100 * (res[3][1] < .05).mean():.0f}%, протокольный {100 * (res[4][0] < .05).mean():.0f}% повторов на 27 прогонах")
        # полный проход на неполном дизайне: только 5 прогонов (как сейчас на диске) -> разложение не определено
        header("SELFTEST 3. Неполный дизайн")
        vals_part = {(42, "ape", p): 0.6 + 0.01 * i for i, p in enumerate(protocols)}
        vals_part.update({(43, "ape", "soft_min"): 0.61, (43, "ape", "hard_min"): 0.62})
        with contextlib.redirect_stdout(io.StringIO()) as bf:
            dz = block_design(vals_part, seeds, methods, protocols, 200, 200, "cvar25")
        ck.add(dz["defined"] is False and "не определено" in bf.getvalue() and "n=" in bf.getvalue(),
               "5 прогонов одного метода: разложение не определено, напечатаны средние с числом наблюдений")
        vals_box = {(s, m, p): 0.6 for s in (42, 43) for m in methods for p in protocols}
        vals_box[(44, "ape", "soft_min")] = 0.7
        with contextlib.redirect_stdout(io.StringIO()):
            dz2 = block_design(vals_box, seeds, methods, protocols, 200, 200, "cvar25")
        ck.add(dz2["defined"] and dz2["box"]["seeds"] == [42, 43] and dz2["complete_design"] is False,
               "неполный, но с полным подблоком: разложение считается на подблоке 2x3x3 и помечено как подблок")
    finally:
        if tmp.exists() and tmp.name == "selftest_tmp" and tmp.parent.name == "analysis":
            shutil.rmtree(tmp)
    header("SELFTEST: итог")
    n_ok = sum(o for o, _ in ck.rows)
    print(f"{n_ok} из {len(ck.rows)} проверок пройдено" + ("" if ck.ok else "; ЕСТЬ ОШИБКИ"))
    print(f"временный каталог {tmp} " + ("удалён" if not tmp.exists() else "НЕ удалён"))
    return 0 if ck.ok else 1


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path(STAND["runs"]) if STAND else None,
                    help="каталог прогонов seed*/<метод>__<протокол>/ (относительный путь -- от корня проекта)")
    ap.add_argument("--preds", type=Path, default=None,
                    help=f"предсказания финалов на {SET}; по умолчанию каталог стенда "
                         f"({STAND['preds'] if STAND else '--'})")
    ap.add_argument("--out", type=Path, default=None,
                    help="JSON-сводка; по умолчанию <runs>/analysis/s12_summary.json; рядом кладётся alt_picks_prompts.json")
    ap.add_argument("--seed-prompt", type=Path,
                    default=Path(STAND["seed_prompt"]) if STAND else None)
    ap.add_argument("--seed-preds-fallback", type=Path,
                    default=Path(STAND["seed_fallback"]) if STAND else None,
                    help="предсказания seed из S11 (тот же скорер, то же множество), если seed нет среди финалов")
    ap.add_argument("--extra-preds", type=Path, action="append", default=None,
                    help="дополнительный каталог предсказаний (можно повторять); нужен, когда альтернативные "
                         "выборы блока 5 считаются отдельно от матрицы, чтобы не подмешивать их в её пул")
    ap.add_argument("--metric", default="cvar25", choices=METRICS, help="истинная метрика для блоков 3-5")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    ap.add_argument("--protocols", default=",".join(DEFAULT_PROTOCOLS))
    ap.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    ap.add_argument("--price-per-1k", type=float, default=UNIT_PRICE_PER_1K, help="$ за 1000 вызовов скорера")
    ap.add_argument("--cost-rows", type=int, default=None,
                    help="строк на промпт при оценке стоимости; по умолчанию все строки множества")
    ap.add_argument("--selftest", action="store_true", help="проверка на синтетике во временном каталоге")
    ap.add_argument("--selftest-reps", type=int, default=200)
    ap.add_argument("--selftest-perm", type=int, default=2000)
    args = ap.parse_args()
    args.methods = tuple(x.strip() for x in args.methods.split(",") if x.strip())
    args.protocols = tuple(x.strip() for x in args.protocols.split(",") if x.strip())
    args.seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    root = rp(args.runs)
    if args.preds is None:
        args.preds = Path(STAND["preds"])
    if args.out is None:
        args.out = root / "analysis" / f"{RUN_PREFIX}_summary.json"
    if args.selftest:
        return selftest(args)
    run_analysis(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

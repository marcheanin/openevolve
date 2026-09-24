#!/usr/bin/env python
"""S15: третий стенд — многоязычная токсичность, группы — языки, а не демографические категории.

Задача НЕ меняется относительно CivilComments (S11): бинарная детекция токсичности, тот же
стартовый промпт (`prompts/initial_prompt_civilcomments.txt`) и тот же пул из 11 однострочных
правок строгости (`scripts/build_r15_strictness_pool.py`). Меняется только источник текста (не
английские комментарии, а размеченная многоязычная токсичность) и ось группировки (язык вместо
демографической категории). Это осознанно контролируемое сравнение с S11/S13: если выводы о
протоколах отбора держатся и здесь, они не были артефактом одной конкретной оси групп.

Источник: `textdetox/multilingual_toxicity_dataset` (Hugging Face), 15 языковых сплитов, колонки
`text` и `toxic`; ровно 2500 нетоксичных и 2500 токсичных строк на язык. Метка в файле может
храниться строкой, поэтому она всегда приводится через `int()`.

Выбор языков (id группы = порядок ниже, id 0 не используется, как в S13):
    1=en, 2=de, 3=ru, 4=ar, 5=hi, 6=am
Языки подобраны ДО какого-либо скоринга, по двум формальным осям — письменность и ресурсный
уровень, — а не по тому, где ошибается модель:
    en, de — латиница,          высокий ресурс
    ru     — кириллица,         высокий ресурс
    ar     — арабское письмо,   средний ресурс
    hi     — деванагари,        средний ресурс
    am     — эфиопское письмо,  низкий ресурс
Важно: выбор НЕ зависит от того, на каких языках ошибается стартовый промпт. Если бы группы
подбирались по ошибкам, это были бы группы, выбранные постфактум под удобный результат, — ровно
то, что запрещено методикой этой серии экспериментов (сначала фиксируется ось групп, потом
считается метрика, а не наоборот).

Резерв (см. RESERVE) — упорядоченный список языков на случай, если один из шести придётся
заменить (например, если у модели-скорера не найдётся токенизации/качества для него). Замена
сейчас НЕ делается, список только документирует порядок будущей замены.

Как и в S13 (в отличие от S11), тексты хранятся прямо в JSON, а не как индексы во внешнем
загрузчике: у этого стенда нет общего загрузчика с фиксированным разбиением, поэтому хранить
нужно сами строки.

Два непересекающихся множества, как в S11/S13:
    truth_tox         — «истинное значение промпта»;
    dev_universe_tox  — из него нарезаются выборки для отбора.
Ячейка — «язык × метка» (6 × 2 = 12 ячеек). Для каждой ячейки берётся `cap` строк в truth и
следующие `cap` строк (из того же перемешанного пула, без пересечения) в dev. Порядок строк
внутри каждого готового множества дополнительно перемешан, поэтому любой префикс — стратифицированная
подвыборка по язык×метка (этим пользуется скорер с остановкой, который скорит по префиксу и может
быть прерван в любой момент).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/S15_toxlang_matrix"

DATASET = "textdetox/multilingual_toxicity_dataset"

# id группы -> код языка на HF. Порядок и id фиксированы, менять нельзя (см. docstring).
LANGS: dict[int, str] = {1: "en", 2: "de", 3: "ru", 4: "ar", 5: "hi", 6: "am"}

# Резерв на случай замены одного из шести языков; упорядочен, замена сейчас не делается.
RESERVE: tuple[str, ...] = ("it", "es", "uk", "zh", "ja")

SEED = 20260921


def load_rows() -> list[dict]:
    """Читает все шесть языковых сплитов и складывает их в один список строк с общим индексом.

    В сырых сплитах встречаются буквально повторяющиеся строки текста (внутри одного языка, с
    одной и той же меткой — проверено отдельно, конфликтов метки на дубликатах нет; между
    языками дубликатов текста нет вовсе). Без дедупликации один и тот же текст мог попасть и в
    truth, и в dev по разным исходным позициям, что нарушило бы непересечение множеств по текстам.
    Поэтому внутри каждого языка дубликаты по тексту схлопываются (остаётся первое вхождение) ещё
    до нарезки на ячейки; после этого в каждой ячейке язык×метка остаётся не меньше ~2000 строк —
    с большим запасом над требуемыми 2*cap=500.
    """
    from datasets import load_dataset

    rows = []
    for gid, lang in LANGS.items():
        seen_text: set[str] = set()
        n_dup = 0
        for r in load_dataset(DATASET, split=lang):
            t = r["text"]
            if t in seen_text:
                n_dup += 1
                continue
            seen_text.add(t)
            rows.append({"lang": lang, "gid": gid, "text": t, "label": int(r["toxic"])})
        if n_dup:
            print(f"[load_rows] {lang}: отброшено {n_dup} дублирующихся по тексту строк")
    return rows


def render(r: dict) -> str:
    return r["text"].strip()


def build(rows: list[dict], cap: int) -> dict:
    rng = np.random.RandomState(SEED)
    picked = {"truth_tox": [], "dev_universe_tox": []}
    counts = {}
    for gid, lang in LANGS.items():
        for y in (0, 1):
            pool = [i for i, r in enumerate(rows) if r["gid"] == gid and r["label"] == y]
            pool = list(rng.permutation(pool))
            if len(pool) < 2 * cap:
                raise SystemExit(f"ячейка {lang}/{y}: {len(pool)} строк, нужно {2 * cap}")
            picked["truth_tox"] += pool[:cap]
            picked["dev_universe_tox"] += pool[cap:2 * cap]
            counts[f"{lang}_y{y}"] = {"available": len(pool), "per_set": cap}
    sets = {}
    for name, idx in picked.items():
        idx = [int(i) for i in idx]
        rng.shuffle(idx)  # перемешано: любой префикс — стратифицированная подвыборка
        recs = [rows[i] for i in idx]
        texts = [render(r) for r in recs]
        labels = [int(r["label"]) for r in recs]
        fp = hashlib.sha256(json.dumps([texts, labels], ensure_ascii=False).encode("utf-8")).hexdigest()[:12]
        sets[name] = {
            "name": name, "dataset": "toxlang", "source_split": "+".join(LANGS.values()),
            "cap_per_cell": cap, "n": len(idx), "fingerprint": fp, "cell_counts": counts,
            "group_names": {str(gid): lang for gid, lang in LANGS.items()},
            "group_ids": sorted(LANGS.keys()),
            "indices": idx, "texts": texts, "labels": labels,
            "cluster_ids": [r["gid"] for r in recs],
            "lang": [r["lang"] for r in recs],
        }
    assert not set(sets["truth_tox"]["indices"]) & set(sets["dev_universe_tox"]["indices"]), \
        "множества пересеклись"
    return sets


def cmd_build(cap: int) -> int:
    rows = load_rows()
    sets = build(rows, cap)
    (OUT / "fixed_sets").mkdir(parents=True, exist_ok=True)
    for name, rec in sets.items():
        (OUT / "fixed_sets" / f"{name}.json").write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        y = np.asarray(rec["labels"])
        print(f"{name}: n={rec['n']} fp={rec['fingerprint']}  доля метки 1 {y.mean():.3f}")
    return 0


def cmd_check() -> int:
    """Читает уже собранные fixed_sets и печатает контрольные сводки, без обращения к сети/ключу."""
    sets = {}
    for name in ("truth_tox", "dev_universe_tox"):
        p = OUT / "fixed_sets" / f"{name}.json"
        if not p.is_file():
            raise SystemExit(f"не найден {p}; сначала запустите сборку без --check")
        sets[name] = json.loads(p.read_text(encoding="utf-8"))

    for name, rec in sets.items():
        print(f"\n=== {name} ===")
        print(f"n={rec['n']}  fingerprint={rec['fingerprint']}  cap_per_cell={rec['cap_per_cell']}")
        lang = np.asarray(rec["lang"])
        labels = np.asarray(rec["labels"])
        texts = rec["texts"]
        print("размеры всех 12 ячеек (язык x метка):")
        for gid in rec["group_ids"]:
            code = rec["group_names"][str(gid)]
            for y in (0, 1):
                n_cell = int(((lang == code) & (labels == y)).sum())
                print(f"    {code} y={y}: {n_cell}")
        print("баланс метки (доля токсичных) по языку:")
        for gid in rec["group_ids"]:
            code = rec["group_names"][str(gid)]
            mask = lang == code
            print(f"    {code}: {labels[mask].mean():.3f}  (n={int(mask.sum())})")
        print("средняя длина текста в символах по языку:")
        for gid in rec["group_ids"]:
            code = rec["group_names"][str(gid)]
            lens = [len(t) for t, c in zip(texts, rec["lang"]) if c == code]
            print(f"    {code}: {sum(lens) / len(lens):.1f}")

    inter = set(sets["truth_tox"]["texts"]) & set(sets["dev_universe_tox"]["texts"])
    print(f"\nпересечение текстов truth_tox и dev_universe_tox: {len(inter)} (должно быть 0)")
    assert not inter, "множества пересекаются по текстам!"
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cap", type=int, default=250, help="строк на ячейку (язык x метка) в каждом множестве")
    ap.add_argument("--check", action="store_true",
                    help="только проверить уже собранные fixed_sets (без обращения к HF)")
    args = ap.parse_args()

    if args.check:
        return cmd_check()
    return cmd_build(args.cap)


if __name__ == "__main__":
    raise SystemExit(main())

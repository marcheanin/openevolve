#!/usr/bin/env python
"""S13: второй датасет — MultiNLI, построенный так же, как S11 строился для CivilComments.

Что решается. Бинарная задача «противоречит ли гипотеза посылке» (метка 1 = contradiction,
0 = entailment или neutral). Группа — жанр (10 жанров: 5 из validation_matched и 5 из
validation_mismatched). Строк из train нет: обучающие примеры для методов берутся отдельно, а
проверочные множества строятся только из двух официальных проверочных частей.

Два непересекающихся множества, как в S11:
  truth_mnli         — «истинное значение промпта»;
  dev_universe_mnli  — из него нарезаются выборки для отбора.
Для каждой ячейки (жанр × метка) берётся `cap` строк в truth и следующие `cap` в dev; порядок
строк внутри множества перемешан, поэтому любой префикс — стратифицированная подвыборка
(тем же свойством пользуется скорер с остановкой).

Почему жанры, а не отрицания. Ячейки «жанр × метка» большие (650–1300 строк), потолок бенчмарка
по данным не давит — в отличие от CivilComments, где у группы other_religions всего 240
токсичных. Это и нужно второму датасету: проверить, что происходит с выводами S11, когда
скудных ячеек нет. Отрицание записано в строки как дополнительный признак (`neg_broad`,
`neg_sagawa`), и при анализе группы можно резать тоньше — «жанр × отрицание», ячейки около 55 —
на тех же самых предсказаниях. Так размер ячейки меняется внутри одного датасета и одной матрицы.

Тексты хранятся прямо в JSON (в отличие от S11, где хранились индексы в загрузчике).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/S13_mnli_matrix"
EXP = ROOT / "experiments/S13_mnli"

NEG_BROAD = re.compile(r"\b(no|never|nobody|nothing|nowhere|none|neither|nor|not|cannot)\b|n't\b", re.I)
NEG_SAGAWA = re.compile(r"\b(nobody|no|never|nothing)\b", re.I)  # список из Sagawa et al. 2020
SEED = 20260919


def load_rows():
    from datasets import load_dataset

    rows = []
    for split in ("validation_matched", "validation_mismatched"):
        for r in load_dataset("nyu-mll/multi_nli", split=split):
            if int(r["label"]) not in (0, 1, 2):
                continue
            rows.append({"genre": r["genre"], "premise": r["premise"], "hypothesis": r["hypothesis"],
                         "label3": int(r["label"]), "split": split})
    return rows


def render(r: dict) -> str:
    return f"Premise: {r['premise'].strip()}\nHypothesis: {r['hypothesis'].strip()}"


def build(rows, cap: int):
    genres = sorted({r["genre"] for r in rows})
    gid = {g: i + 1 for i, g in enumerate(genres)}  # группа 0 не используется
    rng = np.random.RandomState(SEED)
    picked = {"truth_mnli": [], "dev_universe_mnli": []}
    counts = {}
    for g in genres:
        for y in (0, 1):
            pool = [i for i, r in enumerate(rows) if r["genre"] == g and int(r["label3"] == 2) == y]
            pool = list(rng.permutation(pool))
            if len(pool) < 2 * cap:
                raise SystemExit(f"ячейка {g}/{y}: {len(pool)} строк, нужно {2 * cap}")
            picked["truth_mnli"] += pool[:cap]
            picked["dev_universe_mnli"] += pool[cap:2 * cap]
            counts[f"{g}_y{y}"] = {"available": len(pool), "per_set": cap}
    sets = {}
    for name, idx in picked.items():
        idx = [int(i) for i in idx]
        rng.shuffle(idx)  # перемешано: любой префикс — стратифицированная подвыборка
        recs = [rows[i] for i in idx]
        texts = [render(r) for r in recs]
        labels = [int(r["label3"] == 2) for r in recs]
        fp = hashlib.sha256(json.dumps([texts, labels], ensure_ascii=False).encode("utf-8")).hexdigest()[:12]
        sets[name] = {
            "name": name, "dataset": "mnli", "source_split": "validation_matched+validation_mismatched",
            "cap_per_cell": cap, "n": len(idx), "fingerprint": fp, "cell_counts": counts,
            "group_names": {str(v): k for k, v in gid.items()}, "group_ids": sorted(gid.values()),
            "indices": idx, "texts": texts, "labels": labels,
            "cluster_ids": [gid[r["genre"]] for r in recs],
            "genre": [r["genre"] for r in recs],
            "label3": [r["label3"] for r in recs],
            "neg_broad": [bool(NEG_BROAD.search(r["hypothesis"])) for r in recs],
            "neg_sagawa": [bool(NEG_SAGAWA.search(r["hypothesis"])) for r in recs],
        }
    assert not set(sets["truth_mnli"]["indices"]) & set(sets["dev_universe_mnli"]["indices"]), "множества пересеклись"
    return sets


SEED_PROMPT = """<System>
    <Role>You are a natural language inference classifier. Given a premise and a hypothesis, decide whether the hypothesis contradicts the premise.</Role>

    <BaseGuidelines>
        - Output a single binary label only: 0 (does not contradict) or 1 (contradicts)
        - Format: "Label: 0" or "Label: 1" (or just "0" / "1")
        - Contradiction = the hypothesis cannot be true if the premise is true
        - Not a contradiction = the hypothesis follows from the premise, or is neutral (could be true or false given the premise)
        - Judge only from the premise; do not invent outside assumptions to create a conflict
    </BaseGuidelines>

    <DynamicRules>
        Decision process:
        1. Do the premise and hypothesis make incompatible claims about the same thing? → lean contradiction
        2. Does the hypothesis only add new information that the premise neither supports nor rules out? → not a contradiction
        3. Does the hypothesis restate or follow from the premise? → not a contradiction
        4. When unsure, prefer 0 (not a contradiction) unless the incompatibility is clear

        Core rules:
        - A negation word alone does not imply contradiction
        - A different level of detail is not a contradiction
        - A hypothesis about something the premise does not mention is neutral, not a contradiction
    </DynamicRules>
</System>

<FewShotExamples>
Example 1:
Premise: The meeting took place in the morning.
Hypothesis: The meeting took place in the evening.
Label: 1

Example 2:
Premise: Nobody was in the office after six o'clock.
Hypothesis: The office was empty in the evening.
Label: 0

Example 3:
Premise: The committee met on Tuesday to review the budget.
Hypothesis: The committee approved the new budget unanimously.
Label: 0
</FewShotExamples>

<Task>
    {review}
</Task>
"""

ANCHOR = "        4. When unsure, prefer 0 (not a contradiction) unless the incompatibility is clear"
EDITS = {
    **{f"thr_p{p}": f"        4. Label 1 only if the probability that the hypothesis contradicts the premise "
                    f"is at least {p}%; otherwise label 0" for p in (20, 30, 40, 50, 60, 70, 80, 90)},
    "nl_lenient_max": "        4. When unsure, label 0. Reserve label 1 for hypotheses that are unambiguously "
                      "incompatible with the premise",
    "nl_strict": "        4. When unsure, prefer 1 (contradiction); err on the side of flagging",
    "nl_strict_max": "        4. When unsure, label 1. Flag any hypothesis that could reasonably be read as "
                     "conflicting with the premise",
}


def write_prompts():
    (EXP / "prompts").mkdir(parents=True, exist_ok=True)
    (EXP / "prompts/seed.txt").write_text(SEED_PROMPT, encoding="utf-8", newline="\n")
    pool = EXP / "pools/strictness_sweep"
    pool.mkdir(parents=True, exist_ok=True)
    assert SEED_PROMPT.count(ANCHOR) == 1, "строка-якорь должна встречаться ровно один раз"
    cands = []
    for rank, (name, line) in enumerate(EDITS.items()):
        fname = f"{rank:02d}_{name}.txt"
        (pool / fname).write_text(SEED_PROMPT.replace(ANCHOR, line), encoding="utf-8", newline="\n")
        cands.append({"rank": rank, "name": name, "file": fname})
    (pool / "manifest.json").write_text(json.dumps(
        {"pool": "strictness_sweep", "dataset": "mnli", "edit": "single-line substitution of the tie-break rule",
         "base_prompt": "experiments/S13_mnli/prompts/seed.txt", "anchor_line": ANCHOR, "candidates": cands},
        indent=2, ensure_ascii=False), encoding="utf-8")
    return len(cands)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cap", type=int, default=250, help="строк на ячейку (жанр × метка) в каждом множестве")
    args = ap.parse_args()

    n_edits = write_prompts()
    rows = load_rows()
    sets = build(rows, args.cap)
    (OUT / "fixed_sets").mkdir(parents=True, exist_ok=True)
    for name, rec in sets.items():
        (OUT / "fixed_sets" / f"{name}.json").write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        y = np.asarray(rec["labels"])
        neg = np.asarray(rec["neg_broad"])
        print(f"{name}: n={rec['n']} fp={rec['fingerprint']}  доля метки 1 {y.mean():.3f}  "
              f"доля с отрицанием (широкий список) {neg.mean():.3f}  (Sagawa) {np.mean(rec['neg_sagawa']):.3f}")
        # какие ячейки получаются при тонкой нарезке «жанр × отрицание × метка»
        c = np.asarray(rec["cluster_ids"])
        fine = {}
        for g in rec["group_ids"]:
            for n_ in (False, True):
                for lab in (0, 1):
                    fine[(g, n_, lab)] = int(((c == g) & (neg == n_) & (y == lab)).sum())
        print(f"    тонкая нарезка жанр×отрицание×метка: минимальная ячейка {min(fine.values())}, "
              f"медиана {int(np.median(list(fine.values())))}")
    print(f"seed-промпт и {n_edits} правок записаны в {EXP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

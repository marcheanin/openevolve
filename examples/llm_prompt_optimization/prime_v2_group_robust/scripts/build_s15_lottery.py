#!/usr/bin/env python3
"""Пункт 7: «лотерея валидации» на toxlang.

Собирает все уникальные тексты промптов из архивов 12 живых прогонов
(results/S15_toxlang_loop/seed*/*/archive.json) и готовит их к оценке
на тестовом наборе truth_tox (первые 1000 строк).

Тексты, совпадающие с сидом или с финалами s15:*, повторно не оцениваются:
их предсказания на 3000 строк уже лежат в scorer_gemma/preds/truth_tox,
а первые 1000 строк — те же самые строки.

Выход:
  results/S15_toxlang_matrix/prompts_archive.json — {имя: текст} только новых
  results/S15_toxlang_matrix/archive_map.json     — по каждому прогону:
      список кандидатов (имя для оценки, оценки на валидации) и выбранный финал
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOOP = ROOT / "results" / "S15_toxlang_loop"
MATRIX = ROOT / "results" / "S15_toxlang_matrix"


def key(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def main() -> None:
    known = json.loads((MATRIX / "prompts.json").read_text(encoding="utf-8"))
    # уже оценённые тексты: сид и финалы живых прогонов
    scored = {key(t): n for n, t in known.items() if n == "seed" or n.startswith("s15:")}

    new: dict[str, str] = {}
    runs = []
    for f in sorted(glob.glob(str(LOOP / "seed*" / "*" / "archive.json"))):
        d = Path(f).parent
        a = json.loads(Path(f).read_text(encoding="utf-8"))
        best = (d / "best_prompt.txt").read_text(encoding="utf-8")
        run = f"{a['seed']}_{d.name}"
        cands = []
        for e in a["evaluated"]:
            k = key(e["prompt"])
            name = scored.get(k)
            if name is None:
                name = f"arc:{k[:10]}"
                new.setdefault(name, e["prompt"])
            cands.append({"name": name, "hash": e["hash"], "evals": e["evals"]})
        final = scored.get(key(best)) or f"arc:{key(best)[:10]}"
        runs.append({"run": run, "method": a["method"], "seed": a["seed"],
                     "final": final, "candidates": cands})

    uniq = {c["name"] for r in runs for c in r["candidates"]}
    (MATRIX / "prompts_archive.json").write_text(
        json.dumps(new, ensure_ascii=False, indent=1), encoding="utf-8")
    (MATRIX / "archive_map.json").write_text(
        json.dumps({"n_runs": len(runs), "n_unique": len(uniq), "n_new": len(new),
                    "set": "truth_tox", "rows": 1000, "runs": runs},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"прогонов {len(runs)}, оценок {sum(len(r['candidates']) for r in runs)}, "
          f"уникальных текстов {len(uniq)}, новых к оценке {len(new)}")
    for r in runs:
        print(f"  {r['run']:<28} кандидатов {len(r['candidates']):>2}  финал {r['final']}")


if __name__ == "__main__":
    main()

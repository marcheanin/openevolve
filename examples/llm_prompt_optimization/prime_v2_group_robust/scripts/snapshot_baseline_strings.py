"""Слепок текстов, которые базовые методы вставляют в инструкции мутатору и в промпты.

Нужен, чтобы вынос зашитых строк в профиль задачи (`baselines/task_profile.py`) для второго
датасета не изменил ни одного символа для CivilComments: слепок снимается до правки и после, и
два файла должны совпасть побайтово. Это единственная защита от тихой порчи базовых методов,
на которых держатся все прежние результаты.

Запуск: python scripts/snapshot_baseline_strings.py --out <файл.json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class Stub:
    """Вместо LLM: запоминает то, что у него просили, и отвечает фиксированной инструкцией."""

    def __init__(self, reply: str = "Classify carefully and output Label: 0 or Label: 1."):
        self.reply, self.calls = reply, []

    def complete(self, prompt, *, max_tokens=None, temperature=None):
        self.calls.append(prompt)
        return self.reply


def snapshot() -> dict:
    from baselines import evoprompt, gepa_baseline, optimizer_llm

    seed = (ROOT / "prompts/initial_prompt_civilcomments.txt").read_text(encoding="utf-8")
    out: dict = {}

    # чистые функции склейки промптов
    out["inject_seed"] = optimizer_llm.inject_instruction_into_seed("Judge the comment carefully.", seed)
    out["inject_no_tags"] = optimizer_llm.inject_instruction_into_seed("Do the task.", "plain {review}")
    out["contract_freeform"] = optimizer_llm.ensure_prompt_contract("Decide label 0 or 1 for the text.")
    out["contract_with_seed"] = optimizer_llm.ensure_prompt_contract("Decide label 0 or 1.", seed_fallback=seed)
    out["contract_last_resort"] = optimizer_llm.ensure_prompt_contract("nothing useful {review}")
    out["contract_ordinal"] = optimizer_llm.ensure_prompt_contract("Rate it.", label_space="ordinal5")

    # инструкции мутатору EvoPrompt
    for name, fn, args in (
        ("evo_paraphrase", evoprompt._paraphrase, (seed, seed, None)),
        ("evo_mutate", evoprompt._mutate, (seed, seed, None)),
        ("evo_crossover", evoprompt._crossover, (seed, "PARENT B TEXT {review} 0 1 label", seed, None)),
    ):
        stub = Stub()
        fn(stub, *args, label_space="binary")
        out[name] = stub.calls[0]
    stub = Stub()
    evoprompt._mutate(stub, seed, seed, None, label_space="ordinal5")
    out["evo_mutate_ordinal"] = stub.calls[0]

    # рефлексия GEPA
    stub = Stub()
    gepa_baseline._reflect(stub, seed, [("some comment text", 1, 0), ("another one", 0, 1)], seed, None)
    out["gepa_reflect"] = stub.calls[0]

    # строки, которые в коде записаны литералами и достаются только при сбое; берём из исходников
    src = {p: (ROOT / p).read_text(encoding="utf-8") for p in
           ("baselines/ape.py", "baselines/evoprompt.py", "baselines/gepa_baseline.py", "baselines/optimizer_llm.py")}
    out["literal_ape_fallback"] = (
        "classify the comment as toxic (1) or non-toxic (0); "
        "mentioning identity alone is not toxic; output Label: 0 or Label: 1")
    present = out["literal_ape_fallback"] in src["baselines/ape.py"].replace('"\n                    "', "")
    try:  # после выноса строк литерал живёт в профиле CIVIL
        from baselines import task_profile

        present = present or task_profile.CIVIL.ape_fallback_instruction == out["literal_ape_fallback"]
    except ImportError:
        pass
    out["literal_ape_fallback_present"] = present
    out["literal_default_seed"] = "Classify toxicity. Output Label: 0 or Label: 1.\nComment: {review}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    snap = snapshot()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(snap, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"записано {len(snap)} текстов в {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

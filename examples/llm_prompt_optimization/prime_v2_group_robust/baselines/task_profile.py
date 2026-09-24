"""Профиль задачи: слова, которые базовые методы вставляют в инструкции мутатору.

Раньше «токсичность» и «комментарий» были зашиты в APE, EvoPrompt, GEPA и в склейку промптов.
Для второго датасета (S13, MultiNLI) эти тексты вынесены сюда. Профиль CIVIL содержит ДОСЛОВНЫЕ
копии прежних строк и включён по умолчанию, так что поведение на CivilComments не меняется;
это проверяется побайтовым сравнением слепков (`scripts/snapshot_baseline_strings.py`).

Профиль выбирается один раз на процесс: `set_active(MNLI)`.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskProfile:
    name: str
    ape_fallback_instruction: str  # APE: инструкция, если мутатор не ответил
    out_line: str                  # строка вывода внутри блока <DynamicRules>
    answer_tail: str               # хвост промпта, собранного вокруг свободной инструкции
    last_resort: str               # промпт последней надежды, если контракт не выполнен
    mutate_head: str               # EvoPrompt: инструкция мутатору для мутации (кончается \n\n)
    crossover_head: str            # EvoPrompt: инструкция для скрещивания (кончается \n\n)
    default_seed: str              # запасной стартовый промпт, если task.seed_prompt не задан
    reflect_head: str              # GEPA: вводная часть рефлексии (кончается \n\n)
    reflect_item: str              # GEPA: одна ошибка, поля {text} {gold} {pred}
    reflect_max_chars: int         # GEPA: сколько символов текста показывать в ошибке


CIVIL = TaskProfile(
    name="civilcomments",
    ape_fallback_instruction=(
        "classify the comment as toxic (1) or non-toxic (0); "
        "mentioning identity alone is not toxic; output Label: 0 or Label: 1"
    ),
    out_line="        Output exactly Label: 0 (non-toxic) or Label: 1 (toxic).",
    answer_tail=(
        "Output exactly 'Label: 0' (non-toxic) or 'Label: 1' (toxic).\n"
        "Comment: {review}"
    ),
    last_resort=(
        "Classify the comment as toxic (1) or not (0). "
        "Identity mentions alone are not toxic. "
        "Output Label: 0 or Label: 1.\nComment: {review}"
    ),
    mutate_head=(
        "Mutate the prompt to improve group-robust toxicity classification. "
        "Preserve {review} and Label: 0/1. Return only the full prompt.\n\n"
    ),
    crossover_head=(
        "Combine two parent prompts into one stronger toxicity classifier prompt. "
        "Preserve {review} and Label: 0/1. Return only the full child prompt.\n\n"
    ),
    default_seed="Classify toxicity. Output Label: 0 or Label: 1.\nComment: {review}",
    reflect_head=(
        "You are reflecting on toxicity-classification failures. "
        "Rewrite the prompt to fix these errors without breaking correct cases. "
        "Preserve {review} and Label: 0/1. Return only the full prompt.\n\n"
    ),
    reflect_item="Comment: {text}\nGold: {gold} Pred: {pred}",
    reflect_max_chars=240,
)

# MultiNLI, бинарная постановка: 1 = гипотеза противоречит посылке, 0 = следует или нейтральна.
# В {review} подставляется готовая пара «Premise: ...\nHypothesis: ...».
MNLI = TaskProfile(
    name="mnli",
    ape_fallback_instruction=(
        "decide whether the hypothesis contradicts the premise (1) or not (0); "
        "a negation word alone does not imply contradiction; output Label: 0 or Label: 1"
    ),
    out_line="        Output exactly Label: 0 (does not contradict) or Label: 1 (contradicts).",
    answer_tail=(
        "Output exactly 'Label: 0' (does not contradict) or 'Label: 1' (contradicts).\n"
        "{review}"
    ),
    last_resort=(
        "Decide whether the hypothesis contradicts the premise (1) or not (0). "
        "A negation word alone does not imply contradiction. "
        "Output Label: 0 or Label: 1.\n{review}"
    ),
    mutate_head=(
        "Mutate the prompt to improve group-robust natural language inference "
        "(does the hypothesis contradict the premise). "
        "Preserve {review} and Label: 0/1. Return only the full prompt.\n\n"
    ),
    crossover_head=(
        "Combine two parent prompts into one stronger contradiction-detection prompt "
        "for premise/hypothesis pairs. "
        "Preserve {review} and Label: 0/1. Return only the full child prompt.\n\n"
    ),
    default_seed="Decide whether the hypothesis contradicts the premise. Output Label: 0 or Label: 1.\n{review}",
    reflect_head=(
        "You are reflecting on failures of a contradiction classifier for premise/hypothesis pairs. "
        "Rewrite the prompt to fix these errors without breaking correct cases. "
        "Preserve {review} and Label: 0/1. Return only the full prompt.\n\n"
    ),
    reflect_item="Pair: {text}\nGold: {gold} Pred: {pred}",
    reflect_max_chars=480,
)

_ACTIVE = CIVIL


def active() -> TaskProfile:
    return _ACTIVE


def set_active(profile: TaskProfile) -> None:
    global _ACTIVE
    _ACTIVE = profile

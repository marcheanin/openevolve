"""APE-ut: APE meta-prompt augmented with unlabeled target texts (GPO paper §4.1)."""

from __future__ import annotations

import random
from typing import List, Optional, Sequence

from baselines.ape import (
    GPO_PROMPT_GEN_TEMPLATE,
    _extract_instruction,
    select_best_on_dev,
)
from baselines.api import LabeledSet, OptimizerResult, Task
from baselines.optimizer_llm import ensure_prompt_contract


def _format_demo_with_unlabeled(
    texts: Sequence[str],
    labels: Sequence[int],
    unlabeled_texts: Sequence[str],
) -> str:
    parts: List[str] = []
    for t, y in zip(texts, labels):
        parts.append(
            f"Input: {t[:400].replace(chr(10), ' ')}\nOutput: {int(y)}"
        )
    if unlabeled_texts:
        parts.append(
            "Here are additional unlabeled inputs from the target distribution "
            "(no outputs). Use them to infer a more robust instruction:"
        )
        for t in unlabeled_texts:
            parts.append(f"Input: {t[:400].replace(chr(10), ' ')}\nOutput: ?")
    return "\n\n".join(parts)


def generate_ape_ut_candidates(
    train: LabeledSet,
    unlabeled_texts: Sequence[str],
    optimizer_llm,
    *,
    k_prompts: int = 6,
    n_shots: int = 36,
    n_unlabeled: int = 12,
    max_tokens: int = 100,
    temperature: float = 0.0,
    seed: int = 42,
    seed_prompt: Optional[str] = None,
    budget=None,
) -> List[str]:
    rng = random.Random(int(seed))
    n = min(int(n_shots), len(train.texts))
    order = list(range(len(train.texts)))
    rng.shuffle(order)
    order = order[:n]
    texts = [train.texts[i] for i in order]
    labels = [train.labels[i] for i in order]
    u = list(unlabeled_texts)
    rng.shuffle(u)
    u = u[: max(0, int(n_unlabeled))]

    demos_per = max(1, n // k_prompts)
    u_per = max(0, len(u) // max(1, k_prompts))
    candidates: List[str] = []
    for k in range(k_prompts):
        chunk_t = texts[k * demos_per : (k + 1) * demos_per]
        chunk_y = labels[k * demos_per : (k + 1) * demos_per]
        if not chunk_t:
            chunk_t, chunk_y = texts[:demos_per], labels[:demos_per]
        chunk_u = u[k * u_per : (k + 1) * u_per] if u_per else u[:2]
        full_demo = _format_demo_with_unlabeled(chunk_t, chunk_y, chunk_u)
        meta = GPO_PROMPT_GEN_TEMPLATE.replace("[full_DEMO]", full_demo)
        if meta.endswith("[APE]"):
            meta = meta[: -len("[APE]")]
        try:
            if optimizer_llm is None:
                raise RuntimeError("no optimizer_llm")
            print(f"[ape_ut] generate candidate {k + 1}/{k_prompts}", flush=True)
            raw = optimizer_llm.complete(meta, max_tokens=max_tokens, temperature=temperature)
            if budget is not None:
                budget.charge("mutator", n_calls=1, kind="optimizer", note="ape_ut_gen")
            instr = _extract_instruction(raw)
        except Exception:
            instr = (
                "classify the comment as toxic (1) or non-toxic (0); "
                "mentioning identity alone is not toxic; output Label: 0 or Label: 1"
            )
        candidates.append(ensure_prompt_contract(instr, seed_fallback=seed_prompt))
    if seed_prompt:
        candidates.append(seed_prompt)
    seen = set()
    uniq: List[str] = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


class APEUTOptimizer:
    """APE with unlabeled target texts in the meta-prompt."""

    def __init__(
        self,
        *,
        k_prompts: int = 6,
        n_shots: int = 36,
        n_unlabeled: int = 12,
        max_tokens: int = 100,
        temperature: float = 0.0,
    ) -> None:
        self.k_prompts = k_prompts
        self.n_shots = n_shots
        self.n_unlabeled = n_unlabeled
        self.max_tokens = max_tokens
        self.temperature = temperature

    def run(self, task: Task) -> OptimizerResult:
        seed = int(task.meta.get("seed", 42))
        candidates = generate_ape_ut_candidates(
            task.train,
            list(task.unlabeled.texts),
            task.optimizer_llm,
            k_prompts=self.k_prompts,
            n_shots=self.n_shots,
            n_unlabeled=self.n_unlabeled,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            seed=seed,
            seed_prompt=task.seed_prompt,
            budget=task.budget,
        )
        best_prompt, best_score, scored = select_best_on_dev(candidates, task, note="ape_ut_dev")
        return OptimizerResult(
            best_prompt=best_prompt,
            all_candidates=list(candidates),
            trace={
                "method": "ape_ut",
                "k": self.k_prompts,
                "n_shots": self.n_shots,
                "n_unlabeled": self.n_unlabeled,
                "scored": scored,
                "best_softmin": best_score,
            },
            scorer_calls=task.budget.scorer_calls,
            optimizer_calls=task.budget.optimizer_calls,
        )

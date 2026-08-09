"""Minimal APE (Zhou+ 2023) via shared baselines API — native K=6, N=36."""

from __future__ import annotations

import json
import random
from typing import List

from baselines.api import OptimizerResult, Task, score_prompt_on_set


def _format_demos(texts: List[str], labels: List[int], k: int) -> str:
    lines = []
    for t, y in zip(texts[:k], labels[:k]):
        lines.append(f"Comment: {t[:240]}\nLabel: {int(y)}")
    return "\n\n".join(lines)


class APEOptimizer:
    """Generate K instruction candidates from shot splits; pick best on D_dev."""

    def __init__(self, *, k_prompts: int = 6, n_shots: int = 36, max_tokens: int = 100) -> None:
        self.k_prompts = k_prompts
        self.n_shots = n_shots
        self.max_tokens = max_tokens

    def run(self, task: Task) -> OptimizerResult:
        rng = random.Random(int(task.meta.get("seed", 42)))
        n = min(self.n_shots, len(task.train.texts))
        order = list(range(len(task.train.texts)))
        rng.shuffle(order)
        order = order[:n]
        texts = [task.train.texts[i] for i in order]
        labels = [task.train.labels[i] for i in order]

        split = max(1, n // self.k_prompts)
        candidates: List[str] = []
        for k in range(self.k_prompts):
            chunk_t = texts[k * split : (k + 1) * split] or texts[: max(1, n // self.k_prompts)]
            chunk_y = labels[k * split : (k + 1) * split] or labels[: len(chunk_t)]
            demos = _format_demos(chunk_t, chunk_y, len(chunk_t))
            meta = (
                "Write a short instruction for a binary toxicity classifier. "
                "The model must output exactly 'Label: 0' or 'Label: 1'. "
                "Return ONLY the instruction text, including a '{review}' placeholder "
                "for the comment.\n\nExamples:\n"
                f"{demos}\n\nInstruction:"
            )
            try:
                # optimizer_llm is an LLMWorker-like object with .complete or .predict
                llm = task.optimizer_llm
                if hasattr(llm, "complete"):
                    instr = llm.complete(meta, max_tokens=self.max_tokens)
                elif hasattr(llm, "chat"):
                    instr = llm.chat(meta, max_tokens=self.max_tokens)
                else:
                    # Fallback: wrap demos into a usable prompt without a live mutator.
                    instr = (
                        "Classify the comment as toxic (1) or not (0). "
                        "Identity mentions alone are not toxic. "
                        "Output Label: 0 or Label: 1.\nComment: {review}"
                    )
                task.budget.charge("mutator", n_calls=1, kind="optimizer", note="ape_gen")
            except Exception:
                instr = (
                    "Classify the comment as toxic (1) or not (0). "
                    "Output Label: 0 or Label: 1.\nComment: {review}"
                )
            instr = instr.replace("{text}", "{review}")
            if "{review}" not in instr:
                instr = instr.rstrip() + "\nComment: {review}"
            candidates.append(instr.strip())

        if task.seed_prompt:
            candidates.append(task.seed_prompt)

        best_prompt = candidates[0]
        best_score = -1e9
        scored = []
        for p in candidates:
            m = score_prompt_on_set(task.scorer, p, task.dev)
            task.budget.charge("val", n_calls=len(task.dev.texts), kind="scorer", note="ape_dev")
            s = float(m.get("R_soft_min_gba", m.get("fitness", 0.0)))
            scored.append({"softmin": s, "prompt_len": len(p)})
            if s > best_score:
                best_score = s
                best_prompt = p

        return OptimizerResult(
            best_prompt=best_prompt,
            all_candidates=candidates,
            trace={"k": self.k_prompts, "scored": scored, "best_softmin": best_score},
            scorer_calls=task.budget.scorer_calls,
            optimizer_calls=task.budget.optimizer_calls,
        )


def save_result(result: OptimizerResult, path) -> None:
    path.write_text(
        json.dumps(
            {
                "best_prompt": result.best_prompt,
                "n_candidates": len(result.all_candidates),
                "trace": result.trace,
                "scorer_calls": result.scorer_calls,
                "optimizer_calls": result.optimizer_calls,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

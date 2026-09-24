"""APE (Zhou+ 2023) via GPO App. A.2 / upstream meta-prompt — native K=6, N=36.

Faithful to ``baselines/_upstream/GPO``:
- meta prompt with [full_DEMO] / [APE] continuation
- K=6 subsamples × 6 demos (N=36), temperature=0, max_tokens=100
- Monte-Carlo search disabled; pick best on D_dev with our Scorer
"""

from __future__ import annotations

import json
import random
import re
from typing import List, Optional, Sequence, Tuple

from baselines.api import LabeledSet, OptimizerResult, Task, score_prompt_on_set
from baselines.optimizer_llm import ensure_prompt_contract

# From GPO experiments/run_gpo.py (meta prompt).
GPO_PROMPT_GEN_TEMPLATE = (
    "I provide my friend with an instruction. Based on the instruction, I gave him "
    "several inputs, and he generated the corresponding outputs. Here are the "
    "input-output examples:\n\n[full_DEMO]\n\nPlease briefly illustrate the "
    "instruction and describe the output format. The instruction is to [APE]"
)

DEMOS_TEMPLATE = "Input: [INPUT]\nOutput: [OUTPUT]"


def _format_full_demo(texts: Sequence[str], labels: Sequence[int]) -> str:
    parts = []
    for t, y in zip(texts, labels):
        # CivilComments adaptation of Input/Output demo format.
        parts.append(
            DEMOS_TEMPLATE.replace("[INPUT]", t[:400].replace("\n", " ")).replace(
                "[OUTPUT]", str(int(y))
            )
        )
    return "\n\n".join(parts)


def _extract_instruction(raw: str) -> str:
    text = (raw or "").strip()
    # Model may continue after "[APE]" or echo the template.
    if "[APE]" in text:
        text = text.split("[APE]", 1)[-1].strip()
    # Stop at common trailing chatter.
    for stop in ("\n\nInput:", "\nInput:", "\n\nComment:", "\n\n# "):
        if stop in text:
            text = text.split(stop, 1)[0].strip()
    # Drop wrapping quotes.
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    # Collapse to a short instruction (GPO max_tokens=100).
    text = re.sub(r"\s+", " ", text).strip()
    return text


def generate_ape_candidates(
    train: LabeledSet,
    optimizer_llm,
    *,
    k_prompts: int = 6,
    n_shots: int = 36,
    max_tokens: int = 100,
    temperature: float = 0.0,
    seed: int = 42,
    seed_prompt: Optional[str] = None,
    budget=None,
) -> List[str]:
    """Generate K instruction prompts from N-shot splits (no MC search)."""
    rng = random.Random(int(seed))
    # Budget-matched APE (K=48) needs ~6 demos/subsample ⇒ up to 288 shots;
    # sample with replacement if |train| is smaller.
    demos_per = 6 if k_prompts >= 12 else max(1, min(int(n_shots), len(train.texts)) // max(1, k_prompts))
    need = max(int(n_shots), k_prompts * demos_per)
    pool_idx = list(range(len(train.texts)))
    rng.shuffle(pool_idx)
    if len(pool_idx) < need:
        while len(pool_idx) < need:
            pool_idx.append(rng.choice(range(len(train.texts))))
    else:
        pool_idx = pool_idx[:need]
    texts = [train.texts[i] for i in pool_idx]
    labels = [train.labels[i] for i in pool_idx]

    candidates: List[str] = []
    for k in range(k_prompts):
        chunk_t = texts[k * demos_per : (k + 1) * demos_per]
        chunk_y = labels[k * demos_per : (k + 1) * demos_per]
        if not chunk_t:
            chunk_t, chunk_y = texts[:demos_per], labels[:demos_per]
        full_demo = _format_full_demo(chunk_t, chunk_y)
        meta = GPO_PROMPT_GEN_TEMPLATE.replace("[full_DEMO]", full_demo)
        # Upstream stops at [APE] for completion; we ask the model to continue.
        # Provide the prefix ending at "The instruction is to " so completion is the instruction.
        if meta.endswith("[APE]"):
            meta = meta[: -len("[APE]")]
        ls = "binary"
        if optimizer_llm is not None:
            w = getattr(optimizer_llm, "worker", None)
            ls = str(
                getattr(w, "label_space", None)
                or getattr(optimizer_llm, "label_space", None)
                or "binary"
            )
        try:
            if optimizer_llm is None:
                raise RuntimeError("no optimizer_llm")
            print(f"[ape] generate candidate {k + 1}/{k_prompts}", flush=True)
            raw = optimizer_llm.complete(meta, max_tokens=max_tokens, temperature=temperature)
            if budget is not None:
                budget.charge("mutator", n_calls=1, kind="optimizer", note="ape_gen")
            instr = _extract_instruction(raw)
        except Exception:
            if ls == "ordinal5":
                instr = (
                    "rate the product review from 1 (awful) to 5 (excellent); "
                    "output a single integer rating"
                )
            else:
                from baselines.task_profile import active as _profile

                instr = _profile().ape_fallback_instruction
        prompt = ensure_prompt_contract(
            instr,
            seed_fallback=seed_prompt,
            label_space=ls,
        )
        candidates.append(prompt)

    if seed_prompt:
        candidates.append(seed_prompt)
    # Dedup while preserving order.
    seen = set()
    uniq: List[str] = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def select_best_on_dev(
    candidates: Sequence[str],
    task: Task,
    *,
    note: str = "ape_dev",
) -> Tuple[str, float, List[dict]]:
    best_prompt = candidates[0]
    best_score = -1e9
    scored: List[dict] = []
    for i, p in enumerate(candidates):
        print(
            f"[ape] score candidate {i + 1}/{len(candidates)} on |dev|={len(task.dev.texts)}",
            flush=True,
        )
        m = score_prompt_on_set(task.scorer, p, task.dev)
        task.budget.charge("val", n_calls=len(task.dev.texts), kind="scorer", note=note)
        key = str(task.meta.get("dev_score_key", ""))
        if key and key in m:
            s = float(m[key])
        else:
            s = float(m.get("R_soft_min_gba", m.get("R_global", m.get("fitness", 0.0))))
        scored.append(
            {
                "score": s,
                "softmin": m.get("R_soft_min_gba"),
                "R_global": m.get("R_global"),
                "worst_gba": m.get("R_worst_gba"),
                "prompt_len": len(p),
            }
        )
        if s > best_score:
            best_score = s
            best_prompt = p
    return best_prompt, best_score, scored


class APEOptimizer:
    """Generate K instruction candidates from shot splits; pick best on D_dev."""

    def __init__(
        self,
        *,
        k_prompts: int = 6,
        n_shots: int = 36,
        max_tokens: int = 100,
        temperature: float = 0.0,
    ) -> None:
        self.k_prompts = k_prompts
        self.n_shots = n_shots
        self.max_tokens = max_tokens
        self.temperature = temperature

    def run(self, task: Task) -> OptimizerResult:
        seed = int(task.meta.get("seed", 42))
        candidates = generate_ape_candidates(
            task.train,
            task.optimizer_llm,
            k_prompts=self.k_prompts,
            n_shots=self.n_shots,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            seed=seed,
            seed_prompt=task.seed_prompt,
            budget=task.budget,
        )
        best_prompt, best_score, scored = select_best_on_dev(candidates, task)
        return OptimizerResult(
            best_prompt=best_prompt,
            all_candidates=list(candidates),
            trace={
                "method": "ape",
                "k": self.k_prompts,
                "n_shots": self.n_shots,
                "temperature": self.temperature,
                "scored": scored,
                "best_softmin": best_score,
                "meta_prompt": "gpo_run_gpo.py",
            },
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

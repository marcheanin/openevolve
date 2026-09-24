"""EvoPrompt GA / DE (Guo+ ICLR'24) — ported onto shared baselines API.

Native-ish defaults from ROADMAP §6.2: population 10, 10 generations,
dev subsample 500 for fitness. Initial population = seed + paraphrases.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence

from baselines.api import LabeledSet, OptimizerResult, Task, score_prompt_on_set
from baselines.optimizer_llm import ensure_prompt_contract


@dataclass
class _Cand:
    prompt: str
    score: float


def _subsample_dev(dev: LabeledSet, n: int, seed: int) -> LabeledSet:
    if len(dev.texts) <= n:
        return dev
    rng = random.Random(seed)
    order = list(range(len(dev.texts)))
    rng.shuffle(order)
    order = order[:n]
    return LabeledSet(
        texts=[dev.texts[i] for i in order],
        labels=[dev.labels[i] for i in order],
        group_ids=[dev.group_ids[i] for i in order] if dev.group_ids else None,
        example_ids=[dev.example_ids[i] for i in order] if dev.example_ids else None,
    )


def _score(task: Task, prompt: str, dev: LabeledSet, note: str) -> float:
    m = score_prompt_on_set(task.scorer, prompt, dev)
    task.budget.charge("val", n_calls=len(dev.texts), kind="scorer", note=note)
    key = str(task.meta.get("dev_score_key", ""))
    if key and key in m:
        return float(m[key])
    return float(m.get("R_soft_min_gba", m.get("R_global", m.get("fitness", 0.0))))


def _ls(task: Task) -> str:
    return str(task.meta.get("label_space", getattr(task.scorer, "label_space", "binary")))


def _paraphrase(llm, prompt: str, seed_prompt: Optional[str], budget, *, label_space: str) -> str:
    if label_space == "ordinal5":
        meta = (
            "Paraphrase the following Amazon review rating prompt while preserving the "
            "{review} placeholder and Rating 1-5 output format. "
            "Return only the full prompt.\n\n"
            f"{prompt}"
        )
    else:
        meta = (
            "Paraphrase the following classification prompt while preserving the "
            "{review} placeholder and binary Label: 0/1 output format. "
            "Return only the full prompt.\n\n"
            f"{prompt}"
        )
    try:
        raw = llm.complete(meta, max_tokens=800, temperature=0.8)
        if budget is not None:
            budget.charge("mutator", n_calls=1, kind="optimizer", note="evoprompt_para")
        return ensure_prompt_contract(raw, seed_fallback=seed_prompt, label_space=label_space)
    except Exception:
        return prompt


def _mutate(llm, prompt: str, seed_prompt: Optional[str], budget, *, label_space: str) -> str:
    if label_space == "ordinal5":
        meta = (
            "Mutate the prompt to improve robust 1-5 star rating of Amazon reviews "
            "under category shift. Preserve {review} and Rating 1-5. "
            "Return only the full prompt.\n\n"
            f"{prompt}"
        )
    else:
        from baselines.task_profile import active as _profile

        meta = _profile().mutate_head + f"{prompt}"
    try:
        raw = llm.complete(meta, max_tokens=800, temperature=0.8)
        if budget is not None:
            budget.charge("mutator", n_calls=1, kind="optimizer", note="evoprompt_mut")
        return ensure_prompt_contract(raw, seed_fallback=seed_prompt, label_space=label_space)
    except Exception:
        return prompt


def _crossover(llm, a: str, b: str, seed_prompt: Optional[str], budget, *, label_space: str) -> str:
    if label_space == "ordinal5":
        meta = (
            "Combine two parent prompts into one stronger Amazon star-rating prompt. "
            "Preserve {review} and Rating 1-5. Return only the full child prompt.\n\n"
            f"Parent A:\n{a}\n\nParent B:\n{b}"
        )
    else:
        from baselines.task_profile import active as _profile

        meta = _profile().crossover_head + f"Parent A:\n{a}\n\nParent B:\n{b}"
    try:
        raw = llm.complete(meta, max_tokens=800, temperature=0.8)
        if budget is not None:
            budget.charge("mutator", n_calls=1, kind="optimizer", note="evoprompt_xover")
        return ensure_prompt_contract(raw, seed_fallback=seed_prompt, label_space=label_space)
    except Exception:
        return a


def _roulette(pop: Sequence[_Cand], rng: random.Random) -> _Cand:
    scores = [max(0.0, c.score) + 1e-6 for c in pop]
    total = sum(scores)
    r = rng.random() * total
    acc = 0.0
    for c, s in zip(pop, scores):
        acc += s
        if acc >= r:
            return c
    return pop[-1]


class EvoPromptOptimizer:
    """EvoPrompt with mode='ga' (crossover+mutation) or mode='de' (differential-style)."""

    def __init__(
        self,
        *,
        mode: str = "ga",
        population_size: int = 10,
        generations: int = 10,
        mutation_rate: float = 0.2,
        dev_subsample: int = 500,
    ) -> None:
        self.mode = mode.lower().strip()
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.dev_subsample = dev_subsample

    def run(self, task: Task) -> OptimizerResult:
        seed = int(task.meta.get("seed", 42))
        rng = random.Random(seed)
        llm = task.optimizer_llm
        ls = _ls(task)
        from baselines.task_profile import active as _profile

        seed_prompt = task.seed_prompt or (
            "Classify Amazon reviews. Output Rating: 1-5.\nReview: {review}"
            if ls == "ordinal5"
            else _profile().default_seed
        )
        dev = _subsample_dev(task.dev, self.dev_subsample, seed)

        prompts: List[str] = [seed_prompt]
        while len(prompts) < self.population_size:
            prompts.append(
                _paraphrase(llm, seed_prompt, seed_prompt, task.budget, label_space=ls)
            )

        pop: List[_Cand] = []
        for i, p in enumerate(prompts):
            print(f"[evoprompt-{self.mode}] init {i + 1}/{len(prompts)}", flush=True)
            pop.append(_Cand(prompt=p, score=_score(task, p, dev, f"evo_{self.mode}_init")))
        pop.sort(key=lambda c: c.score, reverse=True)
        history = [{"gen": 0, "best": pop[0].score}]

        for gen in range(1, self.generations + 1):
            children: List[_Cand] = []
            for j in range(self.population_size):
                if self.mode == "de":
                    # DE-style: mutate toward difference of two parents.
                    a, b, c = rng.sample(pop, k=min(3, len(pop)))
                    base = _crossover(
                        llm, a.prompt, b.prompt, seed_prompt, task.budget, label_space=ls
                    )
                    child_p = _mutate(llm, base, seed_prompt, task.budget, label_space=ls)
                    if rng.random() < 0.5:
                        child_p = _crossover(
                            llm, child_p, c.prompt, seed_prompt, task.budget, label_space=ls
                        )
                else:
                    # GA: roulette crossover + occasional mutation.
                    if len(pop) > 1 and rng.random() > self.mutation_rate:
                        pa = _roulette(pop, rng)
                        pb = _roulette(pop, rng)
                        child_p = _crossover(
                            llm, pa.prompt, pb.prompt, seed_prompt, task.budget, label_space=ls
                        )
                    else:
                        child_p = _mutate(
                            llm,
                            _roulette(pop, rng).prompt,
                            seed_prompt,
                            task.budget,
                            label_space=ls,
                        )
                print(
                    f"[evoprompt-{self.mode}] gen {gen} child {j + 1}/{self.population_size}",
                    flush=True,
                )
                children.append(
                    _Cand(prompt=child_p, score=_score(task, child_p, dev, f"evo_{self.mode}_g{gen}"))
                )
            pop = sorted(pop + children, key=lambda c: c.score, reverse=True)[: self.population_size]
            history.append({"gen": gen, "best": pop[0].score})
            print(f"[evoprompt-{self.mode}] gen {gen} best={pop[0].score:.4f}", flush=True)

        # Final pick on full D_dev.
        best = pop[0]
        full = _score(task, best.prompt, task.dev, f"evo_{self.mode}_full_dev")
        return OptimizerResult(
            best_prompt=best.prompt,
            all_candidates=[c.prompt for c in pop],
            trace={
                "method": f"evoprompt_{self.mode}",
                "population_size": self.population_size,
                "generations": self.generations,
                "dev_subsample": self.dev_subsample,
                "history": history,
                "best_softmin_sub": best.score,
                "best_softmin_full_dev": full,
            },
            scorer_calls=task.budget.scorer_calls,
            optimizer_calls=task.budget.optimizer_calls,
        )

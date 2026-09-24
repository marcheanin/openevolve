"""GEPA-style reflective prompt optimization (Agrawal+ 2025) — lightweight adapter.

Full ``gepa`` PyPI package is not required. We implement the core loop used in
ROADMAP §6.2: reflective mutations from error feedback + Pareto retention over
instance-level correctness, capped by ``max_metric_calls``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from baselines.api import LabeledSet, OptimizerResult, Task
from baselines.optimizer_llm import ensure_prompt_contract


@dataclass
class _Cand:
    prompt: str
    score: float
    correct: Set[int] = field(default_factory=set)


def _subsample(dev: LabeledSet, n: int, seed: int) -> LabeledSet:
    if len(dev.texts) <= n:
        return dev
    rng = random.Random(seed)
    idxs = list(range(len(dev.texts)))
    rng.shuffle(idxs)
    idxs = idxs[:n]
    return LabeledSet(
        texts=[dev.texts[i] for i in idxs],
        labels=[dev.labels[i] for i in idxs],
        group_ids=[dev.group_ids[i] for i in idxs] if dev.group_ids else None,
        example_ids=list(idxs),
    )


def _eval_cand(task: Task, prompt: str, dev: LabeledSet, note: str) -> _Cand:
    from prime.config import FitnessCfg
    from prime.fitness.objective import compute_fitness
    import numpy as np

    result = task.scorer.predict_batch(list(dev.texts), prompt, labels_for_mock=dev.labels)
    task.budget.charge("val", n_calls=len(dev.texts), kind="scorer", note=note)
    preds = [int(x) for x in result.preds]
    correct = {i for i, (p, y) in enumerate(zip(preds, dev.labels)) if p == int(y)}
    cfg = FitnessCfg(
        mode="soft_min_lex",
        group_acc="balanced_within",
        fail_closed=True,
        class_balanced=True,
        shrink_prior_weight=40.0,
        soft_min_tau=0.10,
        len_penalty_start=10_000,
    )
    user_ids = np.arange(len(dev.labels))
    cluster_ids = np.asarray(dev.group_ids) if dev.group_ids else None
    fit = compute_fitness(
        np.asarray(preds),
        np.asarray(dev.labels),
        user_ids,
        prompt,
        cfg,
        cluster_ids=cluster_ids,
    )
    s = float(fit.get("R_soft_min_gba", fit.get("fitness", 0.0)))
    return _Cand(prompt=prompt, score=s, correct=correct)


def _pareto_keep(cands: Sequence[_Cand], max_keep: int) -> List[_Cand]:
    """Keep non-dominated candidates by instance-correct set inclusion + score."""
    kept: List[_Cand] = []
    for c in sorted(cands, key=lambda x: x.score, reverse=True):
        dominated = False
        for o in kept:
            if c.correct < o.correct or (
                c.correct <= o.correct and c.score < o.score - 1e-9
            ):
                dominated = True
                break
        if not dominated:
            # Drop anyone we now dominate.
            kept = [
                o
                for o in kept
                if not (
                    o.correct < c.correct
                    or (o.correct <= c.correct and o.score < c.score - 1e-9)
                )
            ]
            kept.append(c)
        if len(kept) >= max_keep:
            break
    return kept[:max_keep]


def _reflect(llm, prompt: str, errors: Sequence[Tuple[str, int, int]], seed_prompt, budget) -> str:
    from baselines.task_profile import active as _profile

    err_lines = []
    for text, gold, pred in errors[:6]:
        err_lines.append(
            _profile().reflect_item.format(
                text=text[: _profile().reflect_max_chars], gold=gold, pred=pred
            )
        )
    meta = (
        _profile().reflect_head
        + f"Current prompt:\n{prompt}\n\nFailures:\n" + "\n\n".join(err_lines)
    )
    try:
        raw = llm.complete(meta, max_tokens=900, temperature=0.7)
        if budget is not None:
            budget.charge("mutator", n_calls=1, kind="optimizer", note="gepa_reflect")
        return ensure_prompt_contract(raw, seed_fallback=seed_prompt)
    except Exception:
        return prompt


class GEPAOptimizer:
    """Reflective mutations + Pareto retention under a metric-call budget."""

    def __init__(
        self,
        *,
        max_metric_calls: int = 60_000,
        population: int = 8,
        reflections: int = 12,
        dev_subsample: int = 400,
    ) -> None:
        self.max_metric_calls = max_metric_calls
        self.population = population
        self.reflections = reflections
        self.dev_subsample = dev_subsample

    def run(self, task: Task) -> OptimizerResult:
        seed = int(task.meta.get("seed", 42))
        rng = random.Random(seed)
        from baselines.task_profile import active as _profile

        seed_prompt = task.seed_prompt or _profile().default_seed
        dev = _subsample(task.dev, self.dev_subsample, seed)
        calls0 = int(getattr(task.budget, "scorer_calls", 0) or 0)

        print("[gepa] evaluate seed", flush=True)
        front = [_eval_cand(task, seed_prompt, dev, "gepa_init")]
        history: List[Dict] = [{"step": 0, "best": front[0].score, "front": 1}]

        for step in range(1, self.reflections + 1):
            spent = int(getattr(task.budget, "scorer_calls", 0) or 0) - calls0
            if spent >= self.max_metric_calls:
                print(f"[gepa] budget exhausted at step {step}", flush=True)
                break
            parent = rng.choice(front)
            # Collect errors for reflection.
            result = task.scorer.predict_batch(
                list(dev.texts), parent.prompt, labels_for_mock=dev.labels
            )
            task.budget.charge("val", n_calls=len(dev.texts), kind="scorer", note="gepa_err")
            errors = [
                (dev.texts[i], int(dev.labels[i]), int(result.preds[i]))
                for i in range(len(dev.texts))
                if int(result.preds[i]) != int(dev.labels[i])
            ]
            rng.shuffle(errors)
            child_p = _reflect(task.optimizer_llm, parent.prompt, errors, seed_prompt, task.budget)
            print(f"[gepa] reflect step {step}/{self.reflections}", flush=True)
            child = _eval_cand(task, child_p, dev, f"gepa_s{step}")
            front = _pareto_keep(front + [child], self.population)
            best = max(front, key=lambda c: c.score)
            history.append({"step": step, "best": best.score, "front": len(front)})
            print(f"[gepa] step {step} best={best.score:.4f} front={len(front)}", flush=True)

        best = max(front, key=lambda c: c.score)
        # Full D_dev confirm.
        full = _eval_cand(task, best.prompt, task.dev, "gepa_full_dev")
        return OptimizerResult(
            best_prompt=best.prompt,
            all_candidates=[c.prompt for c in front],
            trace={
                "method": "gepa",
                "max_metric_calls": self.max_metric_calls,
                "reflections": self.reflections,
                "dev_subsample": self.dev_subsample,
                "history": history,
                "best_softmin_sub": best.score,
                "best_softmin_full_dev": full.score,
                "note": "lightweight reflective+Pareto adapter (gepa PyPI not installed)",
            },
            scorer_calls=task.budget.scorer_calls,
            optimizer_calls=task.budget.optimizer_calls,
        )

"""Shared baseline optimizer API (ROADMAP_PHASE3 F11).

All foreign methods (APE, APO, GPO, …) must go through the same Scorer,
parser, prompt contract, and budget guard as PRIME.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

from prime.workers.scorer import Scorer


@dataclass
class LabeledSet:
    texts: List[str]
    labels: List[int]
    group_ids: Optional[List[int]] = None
    example_ids: Optional[List[int]] = None


@dataclass
class UnlabeledSet:
    texts: List[str]
    group_ids: Optional[List[int]] = None
    example_ids: Optional[List[int]] = None


@dataclass
class Task:
    train: LabeledSet
    dev: LabeledSet
    unlabeled: UnlabeledSet
    label_budget: int
    scorer: Scorer
    optimizer_llm: Any  # LLMWorker or compatible
    budget: Any  # TokenTracker / BudgetGuard
    seed_prompt: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizerResult:
    best_prompt: str
    all_candidates: List[str] = field(default_factory=list)
    trace: Dict[str, Any] = field(default_factory=dict)
    scorer_calls: int = 0
    optimizer_calls: int = 0


class Optimizer(Protocol):
    def run(self, task: Task) -> OptimizerResult: ...


def score_prompt_on_set(
    scorer: Scorer,
    prompt: str,
    labeled: LabeledSet,
    *,
    fitness_mode: str | None = None,
) -> Dict[str, Any]:
    """Convenience: score prompt, return fitness-ready pred arrays.

    Defaults: binary CivilComments soft_min_lex + balanced_within.
    For Amazon ordinal5, pass fitness_mode=\"global\" (or set scorer.label_space).
    """
    from prime.fitness.objective import compute_fitness
    from prime.config import FitnessCfg
    import numpy as np

    result = scorer.predict_batch(labeled.texts, prompt, labels_for_mock=labeled.labels)
    user_ids = np.arange(len(labeled.labels))
    cluster_ids = np.asarray(labeled.group_ids) if labeled.group_ids else None
    ls = getattr(scorer, "label_space", "binary") or "binary"
    mode = fitness_mode or ("global" if ls == "ordinal5" else "soft_min_lex")
    if mode == "global":
        cfg = FitnessCfg(
            mode="global",
            fail_closed=True,
            class_balanced=False,
            shrink_prior_weight=50.0,
            len_penalty_start=10_000,
        )
    else:
        cfg = FitnessCfg(
            mode="soft_min_lex",
            group_acc="balanced_within",
            fail_closed=True,
            class_balanced=True,
            shrink_prior_weight=40.0,
            soft_min_tau=0.10,
            len_penalty_start=10_000,
        )
    return compute_fitness(
        result.preds,
        np.asarray(labeled.labels),
        user_ids,
        prompt,
        cfg,
        cluster_ids=cluster_ids,
    )

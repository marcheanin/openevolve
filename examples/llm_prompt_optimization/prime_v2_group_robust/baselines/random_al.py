"""Random-AL and Oracle upper-bound controls (ROADMAP §6.1 R11/R12).

Random-AL: APO on S_source ∪ 240 random labeled target examples.
Oracle: APO on S_source ∪ fully labeled U_target (seed 42 only in matrix).
"""

from __future__ import annotations

import random
from typing import Optional, Sequence

from baselines.apo import APOOptimizer
from baselines.api import LabeledSet, OptimizerResult, Task, UnlabeledSet
from baselines.regime_sets import materialize_gold_for_indices


def _merge(a: LabeledSet, b: LabeledSet, *, seed: int) -> LabeledSet:
    texts = list(a.texts) + list(b.texts)
    labels = list(a.labels) + list(b.labels)
    gids = None
    if a.group_ids is not None or b.group_ids is not None:
        ag = a.group_ids or [0] * len(a.texts)
        bg = b.group_ids or [0] * len(b.texts)
        gids = list(ag) + list(bg)
    eids = None
    if a.example_ids is not None or b.example_ids is not None:
        ae = a.example_ids or list(range(len(a.texts)))
        be = b.example_ids or list(range(len(b.texts)))
        eids = list(ae) + list(be)
    order = list(range(len(texts)))
    random.Random(seed).shuffle(order)
    return LabeledSet(
        texts=[texts[i] for i in order],
        labels=[labels[i] for i in order],
        group_ids=[gids[i] for i in order] if gids is not None else None,
        example_ids=[eids[i] for i in order] if eids is not None else None,
    )


def sample_random_target_labels(
    train_split,
    u_example_ids: Sequence[int],
    *,
    n: int = 240,
    seed: int = 42,
) -> LabeledSet:
    """Reveal gold labels for a random subset of U_target indices."""
    rng = random.Random(seed)
    ids = list(u_example_ids)
    if len(ids) < n:
        raise ValueError(f"U_target has {len(ids)} ids, need {n}")
    pick = sorted(rng.sample(ids, k=n))
    return materialize_gold_for_indices(train_split, pick)


class RandomALOptimizer:
    """APO on S_source ∪ random L labeled target examples."""

    def __init__(
        self,
        *,
        label_budget: int = 240,
        apo_rounds: int = 6,
        beam_size: int = 4,
        train_split=None,
    ) -> None:
        self.label_budget = label_budget
        self.apo_rounds = apo_rounds
        self.beam_size = beam_size
        self.train_split = train_split

    def run(self, task: Task) -> OptimizerResult:
        seed = int(task.meta.get("seed", 42))
        if self.train_split is None:
            raise ValueError("RandomALOptimizer requires train_split for gold labels")
        u_ids = task.unlabeled.example_ids
        if not u_ids:
            raise ValueError("Random-AL needs unlabeled.example_ids")
        extra = sample_random_target_labels(
            self.train_split,
            u_ids,
            n=min(self.label_budget, len(u_ids)),
            seed=seed + 99,
        )
        mixed = _merge(task.train, extra, seed=seed)
        t2 = Task(
            train=mixed,
            dev=task.dev,
            unlabeled=UnlabeledSet(texts=[]),
            label_budget=self.label_budget,
            scorer=task.scorer,
            optimizer_llm=task.optimizer_llm,
            budget=task.budget,
            seed_prompt=task.seed_prompt,
            meta=dict(task.meta),
        )
        result = APOOptimizer(rounds=self.apo_rounds, beam_size=self.beam_size).run(t2)
        result.trace = {
            **result.trace,
            "method": "random_al",
            "label_budget": self.label_budget,
            "extra_n": len(extra.texts),
            "mixed_n": len(mixed.texts),
            "base": "apo_protegi",
        }
        return result


class OracleALOptimizer:
    """APO on S_source ∪ fully labeled U_target (upper bound)."""

    def __init__(
        self,
        *,
        apo_rounds: int = 6,
        beam_size: int = 4,
        train_split=None,
        max_target_labels: Optional[int] = None,
    ) -> None:
        self.apo_rounds = apo_rounds
        self.beam_size = beam_size
        self.train_split = train_split
        self.max_target_labels = max_target_labels

    def run(self, task: Task) -> OptimizerResult:
        seed = int(task.meta.get("seed", 42))
        if self.train_split is None:
            raise ValueError("OracleALOptimizer requires train_split")
        u_ids = list(task.unlabeled.example_ids or [])
        if self.max_target_labels is not None and len(u_ids) > self.max_target_labels:
            u_ids = sorted(random.Random(seed).sample(u_ids, k=self.max_target_labels))
        gold = materialize_gold_for_indices(self.train_split, u_ids)
        mixed = _merge(task.train, gold, seed=seed)
        t2 = Task(
            train=mixed,
            dev=task.dev,
            unlabeled=UnlabeledSet(texts=[]),
            label_budget=len(u_ids),
            scorer=task.scorer,
            optimizer_llm=task.optimizer_llm,
            budget=task.budget,
            seed_prompt=task.seed_prompt,
            meta=dict(task.meta),
        )
        result = APOOptimizer(rounds=self.apo_rounds, beam_size=self.beam_size).run(t2)
        result.trace = {
            **result.trace,
            "method": "oracle",
            "target_labeled_n": len(u_ids),
            "mixed_n": len(mixed.texts),
            "base": "apo_protegi",
        }
        return result

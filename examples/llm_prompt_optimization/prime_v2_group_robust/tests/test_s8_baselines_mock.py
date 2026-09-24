"""Mock-path smoke for S8 baselines (no API)."""

from __future__ import annotations

from baselines.ape import APEOptimizer, generate_ape_candidates
from baselines.apo import seed_to_protegi, parse_sectioned_prompt
from baselines.api import LabeledSet, Task, UnlabeledSet
from baselines.gpo import ensemble_label_unlabeled
from baselines.optimizer_llm import ensure_prompt_contract
from prime.experiment.budget import TokenTracker
from prime.config import BudgetCfg


class _FakeLLM:
    def __init__(self) -> None:
        self._i = 0

    def complete(self, prompt, *, max_tokens=None, temperature=None):
        self._i += 1
        return f"variant {self._i}: classify toxicity with Label 0 or 1 for the comment"

    def complete_n(self, prompt, n=1, *, max_tokens=None, temperature=None):
        return [self.complete(prompt) for _ in range(n)]


class _FakeScorer:
    def predict_batch(self, texts, prompt, labels_for_mock=None):
        from types import SimpleNamespace
        import numpy as np

        n = len(texts)
        preds = np.zeros(n, dtype=int)
        if labels_for_mock is not None:
            preds = np.asarray(labels_for_mock, dtype=int)
        return SimpleNamespace(preds=preds)


def test_ensure_prompt_contract_adds_review():
    p = ensure_prompt_contract("say if toxic using 0 and 1")
    assert "{review}" in p
    assert "0" in p and "1" in p


def test_protegi_seed_sections():
    seed = (
        "<System><DynamicRules>Rule A</DynamicRules></System>\n"
        "<Task>Comment: {review}</Task>"
    )
    sect = seed_to_protegi(seed)
    parsed = parse_sectioned_prompt(sect)
    assert "task" in parsed
    assert "{review}" in sect


def test_ape_generate_mock():
    train = LabeledSet(
        texts=[f"t{i}" for i in range(36)],
        labels=[i % 2 for i in range(36)],
        group_ids=[1] * 36,
    )
    bud = TokenTracker.from_cfg(BudgetCfg(total_calls=10000, on_exhausted="warn"))
    cands = generate_ape_candidates(
        train, _FakeLLM(), k_prompts=6, n_shots=36, seed=0, budget=bud
    )
    assert len(cands) >= 6
    assert all("{review}" in c for c in cands)


def test_gpo_ensemble_label_mock():
    scorer = _FakeScorer()
    bud = TokenTracker.from_cfg(BudgetCfg(total_calls=10000, on_exhausted="warn"))
    prompts = [
        "Output Label: 0 or Label: 1\nComment: {review}",
        "Output Label: 0 or Label: 1\nComment: {review}",
    ]
    labeled, trace = ensemble_label_unlabeled(
        prompts, ["a", "b", "c", "d"], scorer, bud, conf_threshold=0.5
    )
    assert trace["total"] == 4
    assert len(labeled.texts) in (0, 4)  # upsample to |U| or empty

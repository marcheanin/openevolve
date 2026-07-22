from prime.consolidation.base_consolidator import (
    consolidate_base_guidelines_mock,
    run_base_consolidation,
)
from prime.consolidation.pool_carryover import PromptRecord, gate_consolidation
from prime.evolution.prompt_blocks import extract_block


PROMPT = """<System>
    <BaseGuidelines>
        - Star rating scale: 1-5
    </BaseGuidelines>
    <DynamicRules>
        If short hyperbolic review, check sarcasm.
    </DynamicRules>
</System>
"""


def test_mock_consolidation_only_touches_base():
    carry = [PromptRecord(prompt=PROMPT, fitness=0.5)]
    out = consolidate_base_guidelines_mock(PROMPT, carry)
    assert "sarcasm" in (extract_block(out, "BaseGuidelines") or "") or "consolidated" in (
        extract_block(out, "BaseGuidelines") or ""
    )
    assert extract_block(out, "DynamicRules").strip().startswith("If short")


def test_gate_reject():
    best = PromptRecord(prompt=PROMPT, fitness=0.80)
    bad = PromptRecord(prompt=PROMPT, fitness=0.50)
    assert gate_consolidation(bad, best, delta=0.01) is False
    assert gate_consolidation(bad, best, delta=0.5) is True


def test_run_base_consolidation_mock_gate():
    best = PromptRecord(prompt=PROMPT, fitness=0.5)
    carry = [best]
    prompt, rec, passed = run_base_consolidation(
        PROMPT, carry, best, use_mock=True, gate_delta=0.01, consolidated_fitness=0.5
    )
    assert passed is True
    assert "BaseGuidelines" in prompt
    assert rec.metrics.get("source") == "base_guidelines_consolidation"

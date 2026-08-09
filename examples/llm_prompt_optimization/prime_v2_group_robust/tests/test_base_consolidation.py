from prime.consolidation.base_consolidator import (
    _parse_two_sections,
    build_consolidated_prompt,
    check_not_lossy,
    consolidate_mock,
)
from prime.consolidation.pool_carryover import PromptRecord
from prime.evolution.prompt_blocks import extract_block


PROMPT = """<System>
    <BaseGuidelines>
        - Star rating scale: 1-5
    </BaseGuidelines>
    <DynamicRules>
        If short hyperbolic review, check sarcasm.
        Ratings below 3 need an explicit complaint.
    </DynamicRules>
</System>
"""


def test_mock_consolidation_promotes_and_prunes():
    """OBSERVATIONS O9: whatever moves into the base must leave DynamicRules."""
    carry = [PromptRecord(prompt=PROMPT, fitness=0.5)]
    out = consolidate_mock(PROMPT, carry)
    base = extract_block(out, "BaseGuidelines") or ""
    dynamic = extract_block(out, "DynamicRules") or ""
    assert "sarcasm" in base
    assert "sarcasm" not in dynamic
    assert "explicit complaint" in dynamic


def test_mock_consolidation_is_idempotent():
    carry = [PromptRecord(prompt=PROMPT, fitness=0.5)]
    once = consolidate_mock(PROMPT, carry)
    twice = consolidate_mock(once, carry)
    assert twice == once


def test_build_consolidated_prompt_mock_path():
    out = build_consolidated_prompt(PROMPT, [PromptRecord(prompt=PROMPT, fitness=0.5)], use_mock=True)
    assert "BaseGuidelines" in out and "DynamicRules" in out


def test_lossy_consolidation_is_rejected():
    """A rewrite may move rules between blocks; it may not delete them (O21)."""
    moved = """<System>
    <BaseGuidelines>
        - Star rating scale: 1-5
        If short hyperbolic review, check sarcasm.
    </BaseGuidelines>
    <DynamicRules>
        Ratings below 3 need an explicit complaint.
    </DynamicRules>
</System>
"""
    dropped = """<System>
    <BaseGuidelines>
        - Star rating scale: 1-5
    </BaseGuidelines>
    <DynamicRules>
    </DynamicRules>
</System>
"""
    ok, reason = check_not_lossy(PROMPT, moved)
    assert ok, reason
    ok, reason = check_not_lossy(PROMPT, dropped)
    assert not ok and "lossy" in reason


def test_parse_two_sections():
    reply = (
        "```text\n"
        "=== BASE_GUIDELINES ===\n"
        "- Scale 1-5\n"
        "=== DYNAMIC_RULES ===\n"
        "- Hedged praise caps at 4\n"
        "```"
    )
    base, dyn = _parse_two_sections(reply)
    assert base == "- Scale 1-5"
    assert dyn == "- Hedged praise caps at 4"


def test_parse_two_sections_rejects_garbage():
    assert _parse_two_sections("just some prose") == (None, None)


def test_parse_two_sections_tolerates_decorated_headers():
    """Models decorate separators; a strict match silently degrades a live cycle."""
    reply = (
        "## BASE GUIDELINES\n"
        "- Scale 1-5\n"
        "**DYNAMIC_RULES:**\n"
        "- Hedged praise caps at 4\n"
    )
    base, dyn = _parse_two_sections(reply)
    assert base == "- Scale 1-5"
    assert dyn == "- Hedged praise caps at 4"

"""Phase3 F7 prompt contract."""

from __future__ import annotations

from prime.evolution.prompt_contract import check_prompt_contract


SEED = """<System>
Output Label: 0 or Label: 1 for toxic vs non-toxic.
</System>
<Task>Comment: {review}</Task>
"""


def test_seed_ok():
    r = check_prompt_contract(SEED, label_space="binary")
    assert r.ok


def test_unbalanced_brace_rejected():
    bad = "Label 0 or 1. Comment: {review} extra {unclosed"
    r = check_prompt_contract(bad, label_space="binary")
    assert not r.ok
    assert r.reason and "format_error" in r.reason


def test_missing_placeholder_rejected():
    r = check_prompt_contract("Label 0 or 1. No slot.", label_space="binary")
    assert not r.ok
    assert r.reason == "missing_placeholder"


def test_missing_binary_labels_rejected():
    r = check_prompt_contract("Classify. Comment: {review}", label_space="binary")
    assert not r.ok
    assert r.reason == "missing_binary_labels"

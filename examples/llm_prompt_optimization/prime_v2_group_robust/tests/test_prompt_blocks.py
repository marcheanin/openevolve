from prime.evolution.prompt_blocks import extract_block, has_block, list_blocks, replace_block


SAMPLE = """<System>
    <BaseGuidelines>
        - rule A
    </BaseGuidelines>
    <DynamicRules>
        Step 1
    </DynamicRules>
</System>
"""


def test_extract_and_list_blocks():
    assert "BaseGuidelines" in list_blocks(SAMPLE)
    assert extract_block(SAMPLE, "DynamicRules").strip() == "Step 1"
    assert has_block(SAMPLE, "FewShotExamples") is False


def test_replace_roundtrip():
    updated = replace_block(SAMPLE, "BaseGuidelines", "\n        - rule B\n    ")
    assert "rule B" in extract_block(updated, "BaseGuidelines")
    assert "Step 1" in extract_block(updated, "DynamicRules")
    # DynamicRules unchanged
    assert extract_block(updated, "DynamicRules").strip() == "Step 1"


def test_replace_missing_appends():
    out = replace_block(SAMPLE, "FewShotExamples", "ex1")
    assert has_block(out, "FewShotExamples")
    assert "ex1" in extract_block(out, "FewShotExamples")

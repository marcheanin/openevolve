"""Unit tests for Phase 2a fixes: M17 rotate, O22 fewshot inject, O13/contrastive artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from prime.acquisition.eval_sets import (  # noqa: E402
    EvalSets,
    build_d_select,
    rotate_d_select,
)
from prime.data.wilds_loader import ReviewSplit  # noqa: E402
from prime.evolution.artifacts import (  # noqa: E402
    extract_systematic_fewshot_examples,
    format_error_artifacts,
    lint_dynamic_rules_for_triggers,
)
from prime.evolution.prompt_blocks import extract_block, inject_verbatim_fewshot  # noqa: E402


def _toy_heldout(n: int = 40) -> ReviewSplit:
    texts, labels, users, groups = [], [], [], []
    for i in range(n):
        g = i % 4
        y = (i // 4) % 2  # both labels per group
        texts.append(f"comment {i} about group{g} label{y}")
        labels.append(y)
        users.append(i)
        groups.append(g)
    return ReviewSplit(
        name="heldout",
        texts=texts,
        labels=labels,
        user_ids=users,
        example_cluster_ids=groups,
    )


def test_build_d_select_group_label_stratified():
    heldout = _toy_heldout(40)
    idx = build_d_select(heldout, size=16, seed=0, stratify_by_label=True)
    assert len(idx) == 16
    cells = {(heldout.example_cluster_ids[i], heldout.labels[i]) for i in idx}
    # 4 groups × 2 labels = 8 cells; with size 16 all cells should appear.
    assert len(cells) >= 6


def test_rotate_d_select_keeps_anchor_changes_select():
    heldout = _toy_heldout(60)
    d_select = build_d_select(heldout, 20, seed=1, stratify_by_label=True)
    d_anchor = [i for i in range(60) if i not in set(d_select)][:10]
    sets = EvalSets(d_select=d_select, d_anchor=d_anchor, d_audit=[], meta={})
    sets.content_hash = sets.compute_hash()
    rotated = rotate_d_select(
        heldout,
        sets,
        d_select_size=20,
        d_audit_size=8,
        seed=1,
        cycle=2,
        stratify_by_label=True,
    )
    assert rotated.d_anchor == d_anchor
    assert set(rotated.d_select) & set(rotated.d_anchor) == set()
    assert set(rotated.d_select) & set(rotated.d_audit) == set()
    # Different cycle seed should usually change the set (not guaranteed but likely).
    assert rotated.content_hash != sets.content_hash or rotated.d_select != d_select


def test_inject_verbatim_fewshot_binary():
    prompt = (
        "<System>x</System>\n"
        "<FewShotExamples>\nOld junk\n</FewShotExamples>\n"
        "<Task>Comment: {review}</Task>\n"
    )
    out = inject_verbatim_fewshot(
        prompt,
        [("Those people are animals.", 1), ("Equal protection under the law.", 0)],
        label_space="binary",
    )
    block = extract_block(out, "FewShotExamples") or ""
    assert "Those people are animals." in block
    assert "Equal protection under the law." in block
    assert "Label: 1" in block
    assert "Label: 0" in block
    assert "Old junk" not in block


def test_contrastive_pairs_and_o13_lint():
    texts = [
        "muslims are all terrorists and must die",
        "muslims celebrate eid with family",
        "random other text",
    ]
    gold = [1, 0, 0]
    preds = [0, 0, 0]  # first is systematic error if workers agree on 0
    wp = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    report = format_error_artifacts(
        preds,
        gold,
        wp,
        texts,
        cluster_ids=[1, 1, 0],
        label_space="binary",
        contrastive_pairs=True,
        contrastive_pair_limit=2,
    )
    assert "CONTRASTIVE PAIRS" in report
    assert "FAIL:" in report and "OK:" in report
    assert "CANDIDATE FEW-SHOT LINES" in report
    assert "Comment:" in report  # binary tag

    bad_prompt = (
        "<DynamicRules>\n"
        "        - CLUSTER 3: always rate lower\n"
        "        - If the text contains a slur, Label: 1\n"
        "</DynamicRules>\n"
    )
    viol = lint_dynamic_rules_for_triggers(bad_prompt)
    assert len(viol) == 1
    assert "CLUSTER 3" in viol[0]

    from_art = extract_systematic_fewshot_examples(report, limit=2)
    assert from_art
    assert from_art[0][1] in (0, 1)

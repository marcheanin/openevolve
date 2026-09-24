"""E5 mutator-feedback fixes: polarity, freeze compose, GBA dash, QD exclude none."""

from __future__ import annotations

import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from prime.evolution.artifacts import (  # noqa: E402
    compose_mutator_artifacts,
    format_dselect_error_sample,
    format_error_artifacts,
    format_gba_dashboard,
)
from prime.evolution.qd_features import (  # noqa: E402
    build_qd_metrics,
    qd_feature_dimension_names,
)


def test_binary_single_scorer_polarity_and_named_groups():
    texts = [
        "muslims are all terrorists and must die",
        "muslims celebrate eid with family",
        " Progressive Psychosis is loud and clear!",
    ]
    gold = [1, 0, 0]
    preds = [0, 0, 1]  # FN + FP
    wp = [[0, 0, 1]]  # single scorer
    names = {0: "none", 1: "muslim", 2: "white"}
    report = format_error_artifacts(
        preds,
        gold,
        wp,
        texts,
        cluster_ids=[1, 1, 2],
        label_space="binary",
        contrastive_pairs=True,
        contrastive_pair_limit=2,
        group_names=names,
        gba_dashboard=format_gba_dashboard(
            cluster_gba={1: 0.60, 2: 0.70},
            softmin=0.61,
            worst_gba=0.60,
            toxic_recall=0.5,
            specificity=0.5,
            group_names=names,
        ),
    )
    assert "POLARITY:" in report
    assert "FP 0->1" in report and "FN 1->0" in report
    assert "single scorer" in report
    assert "never rate this above 4" not in report
    assert "group=muslim" in report or "muslim:" in report
    assert "GBA DASHBOARD" in report
    assert "softmin bottleneck: muslim" in report
    assert "mutation_log> MUST name which polarity" in report or "MUST name which polarity" in report


def test_compose_uses_frozen_when_live_empty():
    frozen = "POLARITY: FP 0->1=3 | FN 1->0=1\nERRORS (fix these):\n  1. gold=0 pred=1"
    out = compose_mutator_artifacts(frozen_text=frozen, live_text="")
    assert out is not None
    assert "PRIMARY TARGETS" in out
    assert "0 remaining errors" in out
    assert "Progressive" not in out or "PRIMARY" in out


def test_compose_keeps_live_and_frozen():
    frozen = "POLARITY: FP=2\nfrozen-error"
    live = "POLARITY: FP=1\nlive-error"
    out = compose_mutator_artifacts(frozen_text=frozen, live_text=live)
    assert "PRIMARY TARGETS" in out
    assert "THIS CANDIDATE" in out
    assert "frozen-error" in out and "live-error" in out


def test_dselect_error_sample():
    sample = format_dselect_error_sample(
        [1, 0, 1],
        [0, 0, 1],
        ["false toxic", "ok", "caught"],
        limit=2,
    )
    assert "FP=" in sample and "FN=" in sample
    assert "false toxic" in sample


def test_qd_excludes_none_cluster():
    names = qd_feature_dimension_names(9, exclude_clusters=[0])
    assert "cluster_acc_0" not in names
    assert "cluster_acc_1" in names
    assert names[-1] == "prompt_length"
    m = build_qd_metrics({1: 0.7, 8: 0.5}, "hi there", n_clusters=9, exclude_clusters=[0])
    assert "cluster_acc_0" not in m
    assert m["cluster_acc_1"] == 0.7
    assert m["cluster_acc_8"] == 0.5

"""Fixed CivilComments evaluation sets (ROADMAP_PHASE3 §4.3 / F9).

Designs (seed-deterministic, fingerprinted):
  - test_fixed: test split, groups 0..8 × labels {0,1} × 100 = 1800
  - D_dev:      validation, groups 0..8 × {0,1} × 50 = 900
  - D_select:   train heldout, identities 1..8 × {0,1} × 45 = 720 (controller)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from prime.data.balanced_cells import (
    cell_counts,
    fingerprint_indices,
    sample_balanced_from_split,
)
from prime.data.civilcomments_loader import IDENTITY_VARS
from prime.data.wilds_loader import ReviewSplit


@dataclass
class FixedSet:
    name: str
    source_split: str
    indices: List[int]
    fingerprint: str
    design: Dict[str, Any]
    cell_counts: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "source_split": self.source_split,
            "indices": list(self.indices),
            "fingerprint": self.fingerprint,
            "design": dict(self.design),
            "cell_counts": dict(self.cell_counts),
            "n": len(self.indices),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "FixedSet":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            name=str(d["name"]),
            source_split=str(d["source_split"]),
            indices=[int(i) for i in d["indices"]],
            fingerprint=str(d["fingerprint"]),
            design=dict(d.get("design") or {}),
            cell_counts=dict(d.get("cell_counts") or {}),
        )


def _identity_groups(*, include_none: bool) -> List[int]:
    # CIVILCOMMENTS: 0=none, 1..8 = IDENTITY_VARS order
    return list(range(0 if include_none else 1, 1 + len(IDENTITY_VARS)))


def build_balanced_fixed_set(
    split: ReviewSplit,
    *,
    name: str,
    source_split: str,
    per_cell: int,
    seed: int,
    include_none: bool = True,
    exclude: Optional[Set[int]] = None,
) -> FixedSet:
    groups = _identity_groups(include_none=include_none)
    indices = sample_balanced_from_split(
        split,
        groups=groups,
        per_cell=per_cell,
        seed=seed,
        exclude=exclude,
    )
    labels = split.labels
    gids = split.example_cluster_ids or []
    return FixedSet(
        name=name,
        source_split=source_split,
        indices=indices,
        fingerprint=fingerprint_indices(indices),
        design={
            "groups": groups,
            "per_cell": per_cell,
            "include_none": include_none,
            "seed": seed,
            "n_expected": len(groups) * 2 * per_cell,
        },
        cell_counts=cell_counts(labels, gids, indices),
    )


def build_test_fixed(test: ReviewSplit, *, seed: int = 42) -> FixedSet:
    """8 identity + none × 2 × 100 = 1800 from WILDS test."""
    return build_balanced_fixed_set(
        test,
        name="test_fixed",
        source_split="test",
        per_cell=100,
        seed=seed,
        include_none=True,
    )


def build_d_dev(validation: ReviewSplit, *, seed: int = 42) -> FixedSet:
    """8 identity + none × 2 × 50 = 900 from WILDS validation."""
    return build_balanced_fixed_set(
        validation,
        name="d_dev",
        source_split="validation",
        per_cell=50,
        seed=seed + 17,
        include_none=True,
    )


def materialize_split(split: ReviewSplit, indices: Sequence[int]) -> Dict[str, Any]:
    """Arrays for scoring / persistence."""
    idxs = [int(i) for i in indices]
    cluster_ids = None
    if split.example_cluster_ids:
        cluster_ids = [int(split.example_cluster_ids[i]) for i in idxs]
    return {
        "texts": [split.texts[i] for i in idxs],
        "labels": [int(split.labels[i]) for i in idxs],
        "user_ids": [split.user_ids[i] for i in idxs],
        "cluster_ids": cluster_ids,
        "indices": idxs,
        "source_split": split.name,
    }


def assert_disjoint(*index_lists: Sequence[int]) -> None:
    seen: Set[int] = set()
    for lst in index_lists:
        s = set(int(i) for i in lst)
        if seen & s:
            raise AssertionError(f"index overlap: {sorted(seen & s)[:20]}")
        seen |= s

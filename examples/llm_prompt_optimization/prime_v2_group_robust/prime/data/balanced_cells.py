"""Balanced group×label cell sampling (ROADMAP_PHASE3 F9)."""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from prime.data.wilds_loader import ReviewSplit


def sample_balanced_cells(
    labels: Sequence[int],
    group_ids: Sequence[int],
    *,
    groups: Sequence[int],
    per_cell: int,
    seed: int,
    exclude: Optional[Set[int]] = None,
    label_values: Sequence[int] = (0, 1),
) -> List[int]:
    """
    Sample exactly ``per_cell`` indices for each (group, label) cell.
    Raises if a cell lacks enough examples.
    """
    exclude = exclude or set()
    rng = np.random.RandomState(seed)
    by_cell: Dict[Tuple[int, int], List[int]] = {}
    for i, (lab, gid) in enumerate(zip(labels, group_ids)):
        if i in exclude:
            continue
        key = (int(gid), int(lab))
        by_cell.setdefault(key, []).append(i)

    chosen: List[int] = []
    for g in groups:
        for lab in label_values:
            pool = by_cell.get((int(g), int(lab)), [])
            if len(pool) < per_cell:
                raise ValueError(
                    f"Cell (group={g}, label={lab}) has {len(pool)} examples, need {per_cell}"
                )
            pick = rng.choice(pool, size=per_cell, replace=False)
            chosen.extend(int(i) for i in pick)
    return sorted(chosen)


def sample_balanced_from_split(
    split: ReviewSplit,
    *,
    groups: Sequence[int],
    per_cell: int,
    seed: int,
    exclude: Optional[Set[int]] = None,
    label_values: Sequence[int] = (0, 1),
) -> List[int]:
    if not split.example_cluster_ids:
        raise ValueError("split.example_cluster_ids required for balanced cell sampling")
    return sample_balanced_cells(
        split.labels,
        split.example_cluster_ids,
        groups=groups,
        per_cell=per_cell,
        seed=seed,
        exclude=exclude,
        label_values=label_values,
    )


def fingerprint_indices(indices: Sequence[int]) -> str:
    payload = json.dumps([int(i) for i in indices], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def cell_counts(
    labels: Sequence[int],
    group_ids: Sequence[int],
    indices: Sequence[int],
) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for i in indices:
        key = f"g{int(group_ids[i])}_y{int(labels[i])}"
        out[key] = out.get(key, 0) + 1
    return out

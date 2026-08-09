"""Fixed evaluation sets D_select / D_anchor / D_audit (SPEC v3 §4.3, Р9 / Р15).

- D_select: heldout examples, group×class-stratified; sole basis for fitness /
  CVaR / QD / carryover. With ``d_select_rotate`` (OBSERVATIONS M17) it is
  re-sampled each AL cycle; D_anchor stays fixed as the regression gate.
- D_anchor: confidently solved group×class cells; CI/δ regression gate.
- D_audit: heldout slice disjoint from both; never used for selection —
  measures post-selection bias via D_select↔D_audit gap (Р15).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from prime.data.wilds_loader import ReviewSplit

CellKey = Union[int, Tuple[int, int]]


@dataclass
class EvalSets:
    """Indices are positions inside the heldout split arrays."""

    d_select: List[int]
    d_anchor: List[int]
    d_audit: List[int] = field(default_factory=list)
    meta: Dict[str, object] = field(default_factory=dict)
    content_hash: str = ""

    def validate(self) -> None:
        if set(self.d_select) & set(self.d_anchor):
            raise ValueError("D_select and D_anchor must be disjoint (Р9)")
        if set(self.d_select) & set(self.d_audit):
            raise ValueError("D_select and D_audit must be disjoint (Р15)")
        if set(self.d_anchor) & set(self.d_audit):
            raise ValueError("D_anchor and D_audit must be disjoint (Р15)")

    def compute_hash(self) -> str:
        payload = json.dumps(
            {
                "d_select": self.d_select,
                "d_anchor": self.d_anchor,
                "d_audit": self.d_audit,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def save(self, path: Path) -> None:
        self.validate()
        self.content_hash = self.compute_hash()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "d_select": self.d_select,
                    "d_anchor": self.d_anchor,
                    "d_audit": self.d_audit,
                    "meta": self.meta,
                    "content_hash": self.content_hash,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path, *, verify_hash: bool = True) -> "EvalSets":
        data = json.loads(path.read_text(encoding="utf-8"))
        obj = cls(
            d_select=[int(i) for i in data["d_select"]],
            d_anchor=[int(i) for i in data["d_anchor"]],
            d_audit=[int(i) for i in data.get("d_audit", [])],
            meta=data.get("meta", {}),
            content_hash=str(data.get("content_hash", "")),
        )
        obj.validate()
        if verify_hash and obj.content_hash:
            actual = obj.compute_hash()
            if actual != obj.content_hash:
                raise ValueError(
                    f"EvalSets hash mismatch: stored={obj.content_hash} actual={actual}"
                )
        return obj


def _allocate_quotas(
    pools: Dict[CellKey, List[int]],
    size: int,
) -> Dict[CellKey, int]:
    """Proportional quotas with ≥1 per non-empty cell when size allows."""
    keys = sorted(pools.keys(), key=lambda k: (k if isinstance(k, tuple) else (k,)))
    n = sum(len(pools[k]) for k in keys)
    if n == 0 or size <= 0:
        return {}
    quotas: Dict[CellKey, int] = {}
    for k in keys:
        quotas[k] = max(1, int(round(size * len(pools[k]) / n)))
    # Cap at pool size.
    for k in keys:
        quotas[k] = min(quotas[k], len(pools[k]))
    diff = size - sum(quotas.values())
    order = sorted(keys, key=lambda k: -len(pools[k]))
    guard = 0
    while diff != 0 and guard < 10_000:
        guard += 1
        progressed = False
        for k in order:
            if diff == 0:
                break
            adj = 1 if diff > 0 else -1
            nxt = quotas[k] + adj
            if 0 <= nxt <= len(pools[k]) and (nxt >= 1 or len(keys) == 1):
                quotas[k] = nxt
                diff -= adj
                progressed = True
        if not progressed:
            break
    return quotas


def build_d_select(
    heldout: ReviewSplit,
    size: int,
    seed: int = 42,
    *,
    excluded: Optional[Sequence[int]] = None,
    stratify_by_label: bool = True,
) -> List[int]:
    """
    Stratified sample from heldout.

    Default (Phase 2 / M17): quotas over (group, label) cells so rare identity×
    toxicity combinations survive caps. Pass ``stratify_by_label=False`` for the
    legacy group-only behaviour. ``excluded`` keeps D_select disjoint from a
    frozen D_anchor under rotation.
    """
    n = len(heldout)
    if n == 0:
        return []
    blocked = {int(i) for i in (excluded or [])}
    available = [i for i in range(n) if i not in blocked]
    if not available:
        return []
    size = min(size, len(available))
    cluster_ids = heldout.example_cluster_ids or [0] * n
    rng = np.random.RandomState(seed)

    pools: Dict[CellKey, List[int]] = {}
    for i in available:
        if stratify_by_label:
            key: CellKey = (int(cluster_ids[i]), int(heldout.labels[i]))
        else:
            key = int(cluster_ids[i])
        pools.setdefault(key, []).append(i)

    quotas = _allocate_quotas(pools, size)
    picked: List[int] = []
    for key, take in quotas.items():
        pool = pools[key]
        if take <= 0:
            continue
        chosen = rng.choice(pool, size=min(take, len(pool)), replace=False)
        picked.extend(int(i) for i in chosen)
    if len(picked) < size:
        rest = [i for i in available if i not in set(picked)]
        if rest:
            extra = rng.choice(
                rest, size=min(size - len(picked), len(rest)), replace=False
            )
            picked.extend(int(i) for i in extra)
    return sorted(picked[:size])


def build_d_anchor(
    heldout: ReviewSplit,
    d_select: Sequence[int],
    predictions: Sequence[int],
    disagreements: Sequence[float],
    size: int,
    seed: int = 42,
    max_disagreement: float = 0.2,
) -> List[int]:
    """Confidently solved examples, round-robin over group×class, excl. D_select."""
    n = len(heldout)
    if n == 0 or size <= 0:
        return []
    cluster_ids = heldout.example_cluster_ids or [0] * n
    excluded = set(int(i) for i in d_select)
    rng = np.random.RandomState(seed + 1)

    cells: Dict[tuple, List[int]] = {}
    for i in range(n):
        if i in excluded:
            continue
        if int(predictions[i]) != int(heldout.labels[i]):
            continue
        if float(disagreements[i]) > max_disagreement:
            continue
        key = (int(cluster_ids[i]), int(heldout.labels[i]))
        cells.setdefault(key, []).append(i)

    for key in cells:
        rng.shuffle(cells[key])

    picked: List[int] = []
    keys = sorted(cells.keys())
    while len(picked) < size and any(cells[k] for k in keys):
        for k in keys:
            if cells[k] and len(picked) < size:
                picked.append(cells[k].pop())
    return sorted(picked)


def build_d_audit(
    heldout: ReviewSplit,
    excluded: Sequence[int],
    size: int,
    seed: int = 42,
    *,
    stratify_by_label: bool = True,
) -> List[int]:
    """Heldout slice disjoint from D_select/D_anchor; stratified if possible."""
    n = len(heldout)
    if n == 0 or size <= 0:
        return []
    blocked = set(int(i) for i in excluded)
    available = [i for i in range(n) if i not in blocked]
    if not available:
        return []
    size = min(size, len(available))
    cluster_ids = heldout.example_cluster_ids or [0] * n
    tmp = ReviewSplit(
        name="audit_pool",
        texts=[heldout.texts[i] for i in available],
        labels=[heldout.labels[i] for i in available],
        user_ids=[heldout.user_ids[i] for i in available],
        example_cluster_ids=[cluster_ids[i] for i in available],
    )
    local = build_d_select(
        tmp, size=size, seed=seed + 2, stratify_by_label=stratify_by_label
    )
    return sorted(available[i] for i in local)


def rotate_d_select(
    heldout: ReviewSplit,
    eval_sets: EvalSets,
    *,
    d_select_size: int,
    d_audit_size: int,
    seed: int,
    stratify_by_label: bool = True,
    cycle: int = 1,
) -> EvalSets:
    """
    Re-sample D_select for a new AL cycle (OBSERVATIONS M17).

    D_anchor is kept byte-identical (regression gate). D_audit is rebuilt to stay
    disjoint from the new D_select + frozen D_anchor.
    """
    new_select = build_d_select(
        heldout,
        size=d_select_size,
        seed=seed + int(cycle),
        excluded=eval_sets.d_anchor,
        stratify_by_label=stratify_by_label,
    )
    new_audit = build_d_audit(
        heldout,
        excluded=list(new_select) + list(eval_sets.d_anchor),
        size=d_audit_size,
        seed=seed + int(cycle),
        stratify_by_label=stratify_by_label,
    )
    meta = dict(eval_sets.meta)
    meta.update(
        {
            "d_select_size": len(new_select),
            "d_audit_size": len(new_audit),
            "rotated_cycle": int(cycle),
            "rotate_seed": int(seed + int(cycle)),
            "stratify_by_label": bool(stratify_by_label),
            "prev_d_select_hash": eval_sets.content_hash,
        }
    )
    out = EvalSets(
        d_select=new_select,
        d_anchor=list(eval_sets.d_anchor),
        d_audit=new_audit,
        meta=meta,
    )
    out.validate()
    out.content_hash = out.compute_hash()
    return out


def build_eval_sets(
    heldout: ReviewSplit,
    *,
    d_select_size: int,
    d_anchor_size: int,
    d_audit_size: int = 0,
    predictions: Optional[Sequence[int]] = None,
    disagreements: Optional[Sequence[float]] = None,
    seed: int = 42,
    max_disagreement: float = 0.2,
    stratify_by_label: bool = True,
) -> EvalSets:
    """Build and validate the three fixed evaluation sets (INV-5)."""
    d_select = build_d_select(
        heldout, size=d_select_size, seed=seed, stratify_by_label=stratify_by_label
    )
    if predictions is None:
        predictions = heldout.labels  # fallback: treat gold as "correct" (smoke)
    if disagreements is None:
        disagreements = [0.0] * len(heldout)
    d_anchor = build_d_anchor(
        heldout,
        d_select,
        predictions,
        disagreements,
        size=d_anchor_size,
        seed=seed,
        max_disagreement=max_disagreement,
    )
    d_audit = build_d_audit(
        heldout,
        excluded=list(d_select) + list(d_anchor),
        size=d_audit_size,
        seed=seed,
        stratify_by_label=stratify_by_label,
    )
    sets = EvalSets(
        d_select=d_select,
        d_anchor=d_anchor,
        d_audit=d_audit,
        meta={
            "heldout_n": len(heldout),
            "d_select_size": len(d_select),
            "d_anchor_size": len(d_anchor),
            "d_audit_size": len(d_audit),
            "seed": seed,
            "stratify_by_label": bool(stratify_by_label),
        },
    )
    sets.validate()
    sets.content_hash = sets.compute_hash()
    return sets


def power_rule_max_k(d_select_size: int, n_min_per_group: int) -> int:
    """K <= |D_select| / n_min (SPEC v3 §4.1, Р9); at least 2 groups."""
    return max(2, d_select_size // max(1, n_min_per_group))


def split_arrays(split: ReviewSplit, indices: Sequence[int]) -> Dict[str, list]:
    return {
        "texts": [split.texts[i] for i in indices],
        "labels": [split.labels[i] for i in indices],
        "user_ids": [split.user_ids[i] for i in indices],
        "cluster_ids": (
            [split.example_cluster_ids[i] for i in indices]
            if split.example_cluster_ids
            else [0] * len(indices)
        ),
        "example_ids": [int(i) for i in indices],
    }

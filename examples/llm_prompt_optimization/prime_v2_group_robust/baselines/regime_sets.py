"""Regime-Shift pools from ROADMAP_PHASE3 §4.3 / §6.1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from baselines.api import LabeledSet, UnlabeledSet
from prime.data.balanced_cells import fingerprint_indices, sample_balanced_from_split
from prime.data.civilcomments_loader import ORACLE_GROUP_NAMES
from prime.data.wilds_loader import ReviewSplit

# Source / target identity split (ROADMAP §4.3).
SOURCE_GROUP_NAMES = ("none", "male", "female", "christian")
TARGET_GROUP_NAMES = ("LGBTQ", "muslim", "other_religions", "black", "white")


def _name_to_id(name: str) -> int:
    return ORACLE_GROUP_NAMES.index(name)


SOURCE_GROUP_IDS = tuple(_name_to_id(n) for n in SOURCE_GROUP_NAMES)
TARGET_GROUP_IDS = tuple(_name_to_id(n) for n in TARGET_GROUP_NAMES)


def build_s_source(
    train: ReviewSplit,
    *,
    per_cell: int = 45,
    seed: int = 42,
    exclude: Optional[Set[int]] = None,
) -> Tuple[LabeledSet, List[int]]:
    """S_source: {none,male,female,christian} × {0,1} × per_cell (=360 at 45)."""
    idxs = sample_balanced_from_split(
        train,
        groups=SOURCE_GROUP_IDS,
        per_cell=per_cell,
        seed=seed,
        exclude=exclude,
    )
    texts = [train.texts[i] for i in idxs]
    labels = [int(train.labels[i]) for i in idxs]
    gids = [int(train.example_cluster_ids[i]) for i in idxs]
    return (
        LabeledSet(texts=texts, labels=labels, group_ids=gids, example_ids=list(idxs)),
        idxs,
    )


def build_u_target(
    train: ReviewSplit,
    *,
    n: int = 4000,
    seed: int = 42,
    exclude: Optional[Set[int]] = None,
) -> Tuple[UnlabeledSet, List[int]]:
    """U_target: unlabeled pool from hard identity groups (labels hidden)."""
    exclude = exclude or set()
    rng = np.random.RandomState(seed + 17)
    pool: List[int] = []
    gids = train.example_cluster_ids or []
    target = set(TARGET_GROUP_IDS)
    for i, gid in enumerate(gids):
        if i in exclude:
            continue
        if int(gid) in target:
            pool.append(i)
    if len(pool) < n:
        raise ValueError(f"U_target pool has {len(pool)} examples, need {n}")
    pick = rng.choice(pool, size=n, replace=False)
    idxs = sorted(int(i) for i in pick)
    texts = [train.texts[i] for i in idxs]
    ugids = [int(gids[i]) for i in idxs]
    return UnlabeledSet(texts=texts, group_ids=ugids, example_ids=list(idxs)), idxs


def materialize_gold_for_indices(train: ReviewSplit, idxs: Sequence[int]) -> LabeledSet:
    """Reveal labels for AL / Random-AL control (not used by GPO pseudo-label path)."""
    return LabeledSet(
        texts=[train.texts[i] for i in idxs],
        labels=[int(train.labels[i]) for i in idxs],
        group_ids=[int(train.example_cluster_ids[i]) for i in idxs],
        example_ids=list(idxs),
    )


def save_regime_manifest(
    path: Path,
    *,
    s_source_idxs: Sequence[int],
    u_target_idxs: Sequence[int],
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "seed": seed,
                "s_source": {
                    "n": len(s_source_idxs),
                    "fingerprint": fingerprint_indices(s_source_idxs),
                    "groups": list(SOURCE_GROUP_NAMES),
                    "indices": list(s_source_idxs),
                },
                "u_target": {
                    "n": len(u_target_idxs),
                    "fingerprint": fingerprint_indices(u_target_idxs),
                    "groups": list(TARGET_GROUP_NAMES),
                    "indices": list(u_target_idxs),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_or_build_regime_sets(
    train: ReviewSplit,
    fixed_dir: Path,
    *,
    seed: int = 42,
    s_per_cell: int = 45,
    u_n: int = 4000,
) -> Dict[str, object]:
    """Load cached S_source / U_target indices if present; else build and save.

    Seed 42 uses ``regime_shift_manifest.json`` (S8 fingerprint). Other seeds use
    ``regime_shift_manifest_seed{seed}.json`` so S/U are re-sampled while D_dev /
    test_fixed stay fixed (ROADMAP §7.1).
    """
    if int(seed) == 42:
        man_path = fixed_dir / "regime_shift_manifest.json"
    else:
        man_path = fixed_dir / f"regime_shift_manifest_seed{int(seed)}.json"
    if man_path.is_file():
        man = json.loads(man_path.read_text(encoding="utf-8"))
        s_idxs = [int(i) for i in man["s_source"]["indices"]]
        u_idxs = [int(i) for i in man["u_target"]["indices"]]
    else:
        s_set, s_idxs = build_s_source(train, per_cell=s_per_cell, seed=seed)
        u_set, u_idxs = build_u_target(train, n=u_n, seed=seed, exclude=set(s_idxs))
        save_regime_manifest(man_path, s_source_idxs=s_idxs, u_target_idxs=u_idxs, seed=seed)
        return {"s_source": s_set, "u_target": u_set, "s_idxs": s_idxs, "u_idxs": u_idxs, "manifest": man_path}

    s_set = LabeledSet(
        texts=[train.texts[i] for i in s_idxs],
        labels=[int(train.labels[i]) for i in s_idxs],
        group_ids=[int(train.example_cluster_ids[i]) for i in s_idxs],
        example_ids=list(s_idxs),
    )
    u_set = UnlabeledSet(
        texts=[train.texts[i] for i in u_idxs],
        group_ids=[int(train.example_cluster_ids[i]) for i in u_idxs],
        example_ids=list(u_idxs),
    )
    return {"s_source": s_set, "u_target": u_set, "s_idxs": s_idxs, "u_idxs": u_idxs, "manifest": man_path}

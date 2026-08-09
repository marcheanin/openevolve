"""Source-level fit/heldout split of train (SPEC v3 §4.3, Р7 / Р15).

Train users are split 70/30 (configurable) by user_id:
- fit_sources: cluster fitting + mutator minibatches (D_mut role);
- heldout_sources: candidate selection (D_select, D_anchor, D_audit).

All examples of one user go entirely to one side (user-disjoint).
When stratify=True (Р15), users are binned by length quantile and (optional)
PCA-1 of the mean embedding so halves do not acquire a systematic shift.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from prime.data.wilds_loader import ReviewSplit


@dataclass
class SourceSplit:
    fit: ReviewSplit
    heldout: ReviewSplit
    fit_users: Set[int]
    heldout_users: Set[int]
    strata_balance: Dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if self.fit_users & self.heldout_users:
            raise ValueError("fit_sources and heldout_sources overlap by user_id")
        if not self.fit.user_disjoint_check(self.heldout):
            raise ValueError("fit/heldout example-level user overlap detected")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "fit_users": sorted(self.fit_users),
                    "heldout_users": sorted(self.heldout_users),
                    "strata_balance": self.strata_balance,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def _subset_by_users(split: ReviewSplit, users: Set[int], name: str) -> ReviewSplit:
    idxs = [i for i, u in enumerate(split.user_ids) if u in users]
    return ReviewSplit(
        name=name,
        texts=[split.texts[i] for i in idxs],
        labels=[split.labels[i] for i in idxs],
        user_ids=[split.user_ids[i] for i in idxs],
        example_cluster_ids=(
            [split.example_cluster_ids[i] for i in idxs]
            if split.example_cluster_ids
            else None
        ),
    )


def _user_mean_lengths(split: ReviewSplit) -> Dict[int, float]:
    by_user: Dict[int, List[int]] = {}
    for i, u in enumerate(split.user_ids):
        by_user.setdefault(int(u), []).append(i)
    return {
        u: float(np.mean([len(split.texts[i]) for i in idxs]))
        for u, idxs in by_user.items()
    }


def _quantile_bins(values: Dict[int, float], n_bins: int = 4) -> Dict[int, int]:
    if not values:
        return {}
    users = sorted(values.keys())
    arr = np.array([values[u] for u in users], dtype=np.float64)
    # Digitize into n_bins roughly equal-count bins.
    edges = np.unique(np.quantile(arr, np.linspace(0, 1, n_bins + 1)[1:-1]))
    if len(edges) == 0:
        return {u: 0 for u in users}
    bins = np.digitize(arr, edges, right=True)
    return {u: int(b) for u, b in zip(users, bins)}


def _stratify_assign(
    users: List[int],
    length_bins: Dict[int, int],
    pca_bins: Optional[Dict[int, int]],
    fit_fraction: float,
    rng: np.random.RandomState,
) -> Tuple[Set[int], Set[int], Dict[str, object]]:
    """Within each stratum, assign ~fit_fraction of users to fit."""
    strata: Dict[Tuple[int, int], List[int]] = {}
    for u in users:
        key = (length_bins.get(u, 0), (pca_bins or {}).get(u, 0))
        strata.setdefault(key, []).append(u)

    fit_users: Set[int] = set()
    heldout_users: Set[int] = set()
    balance: Dict[str, object] = {"strata": {}}
    for key, members in sorted(strata.items()):
        members = list(members)
        rng.shuffle(members)
        n_fit = max(1, min(len(members) - 1, int(round(len(members) * fit_fraction)))) if len(members) > 1 else 1
        if len(members) == 1:
            # Singleton stratum: alternate by hash of key to avoid emptying a half.
            if (key[0] + key[1]) % 2 == 0:
                fit_users.add(members[0])
                n_fit = 1
            else:
                heldout_users.add(members[0])
                n_fit = 0
        else:
            fit_users.update(members[:n_fit])
            heldout_users.update(members[n_fit:])
        balance["strata"][f"L{key[0]}_P{key[1]}"] = {
            "n": len(members),
            "n_fit": n_fit,
            "n_heldout": len(members) - n_fit,
        }
    # Guarantee both sides non-empty.
    if not fit_users and heldout_users:
        moved = next(iter(heldout_users))
        heldout_users.remove(moved)
        fit_users.add(moved)
    if not heldout_users and fit_users:
        moved = next(iter(fit_users))
        fit_users.remove(moved)
        heldout_users.add(moved)
    balance["n_fit_users"] = len(fit_users)
    balance["n_heldout_users"] = len(heldout_users)
    return fit_users, heldout_users, balance


def split_fit_heldout(
    train: ReviewSplit,
    fit_fraction: float = 0.7,
    seed: int = 42,
    stratify: bool = True,
    user_embeddings: Optional[Dict[int, np.ndarray]] = None,
    n_length_bins: int = 4,
    n_pca_bins: int = 4,
) -> SourceSplit:
    """
    User-level split of train into fit/heldout sources.

    If ``stratify`` (Р15): bin by mean-text-length quantiles and optional PCA-1
    of per-user embeddings; sample within strata. Otherwise: random permutation.
    """
    if not (0.0 < fit_fraction < 1.0):
        raise ValueError("fit_fraction must be in (0, 1)")
    users: List[int] = sorted(set(int(u) for u in train.user_ids))
    if len(users) < 2:
        raise ValueError("Need at least 2 train users for a fit/heldout source split")
    rng = np.random.RandomState(seed)

    if stratify:
        lengths = _user_mean_lengths(train)
        length_bins = _quantile_bins(lengths, n_bins=n_length_bins)
        pca_bins: Optional[Dict[int, int]] = None
        if user_embeddings:
            pca1 = {
                u: float(np.asarray(user_embeddings[u]).ravel()[0])
                for u in users
                if u in user_embeddings
            }
            if pca1:
                pca_bins = _quantile_bins(pca1, n_bins=n_pca_bins)
        fit_users, heldout_users, balance = _stratify_assign(
            users, length_bins, pca_bins, fit_fraction, rng
        )
        balance["stratify"] = True
    else:
        perm = rng.permutation(len(users))
        n_fit = max(1, min(len(users) - 1, int(round(len(users) * fit_fraction))))
        fit_users = {users[i] for i in perm[:n_fit]}
        heldout_users = {users[i] for i in perm[n_fit:]}
        balance = {
            "stratify": False,
            "n_fit_users": len(fit_users),
            "n_heldout_users": len(heldout_users),
        }

    result = SourceSplit(
        fit=_subset_by_users(train, fit_users, "train_fit"),
        heldout=_subset_by_users(train, heldout_users, "train_heldout"),
        fit_users=fit_users,
        heldout_users=heldout_users,
        strata_balance=balance,
    )
    result.validate()
    return result


# Alias matching IMPLEMENTATION.md §4 API naming.
def split_sources(
    train: ReviewSplit,
    fit_fraction: float = 0.7,
    seed: int = 42,
    stratify: bool = True,
    user_embeddings: Optional[Dict[int, np.ndarray]] = None,
) -> SourceSplit:
    return split_fit_heldout(
        train,
        fit_fraction=fit_fraction,
        seed=seed,
        stratify=stratify,
        user_embeddings=user_embeddings,
    )

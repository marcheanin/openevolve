"""Stratified downsample by discrete category id (e.g. WILDS Amazon product category)."""
from __future__ import annotations

from typing import Sequence
import numpy as np


def stratified_downsample_pick(
    categories: Sequence[int],
    n_pick: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    Choose exactly n_pick row indices in [0, n) with per-category counts
    sampled multinomially from category prevalence, then clipped to available
    counts and repaired to sum to n_pick.
    """
    n = len(categories)
    if n_pick <= 0:
        return np.array([], dtype=np.int64)
    if n_pick >= n:
        return np.arange(n, dtype=np.int64)

    cats = np.asarray(categories, dtype=np.int64)
    uniq, inv = np.unique(cats, return_inverse=True)
    counts = np.bincount(inv, minlength=len(uniq))
    if len(uniq) == 1:
        return np.sort(rng.choice(n, size=n_pick, replace=False))

    p = counts.astype(np.float64) / float(counts.sum())
    alloc = rng.multinomial(n_pick, p)
    alloc = np.minimum(alloc, counts)
    need = n_pick - int(alloc.sum())
    while need > 0:
        slack = counts - alloc
        viable = np.where(slack > 0)[0]
        if viable.size == 0:
            break
        alloc[int(rng.choice(viable))] += 1
        need -= 1

    while int(alloc.sum()) > n_pick:
        viable = np.where(alloc > 0)[0]
        if viable.size == 0:
            break
        excess = alloc.astype(np.float64) - n_pick * p
        j = int(viable[np.argmax(excess[viable])])
        alloc[j] -= 1

    parts: list[np.ndarray] = []
    for j in range(len(uniq)):
        idx = np.where(inv == j)[0]
        k = int(alloc[j])
        if k <= 0:
            continue
        k = min(k, len(idx))
        parts.append(rng.choice(idx, size=k, replace=False))

    if not parts:
        return np.sort(rng.choice(n, size=n_pick, replace=False))

    picked = np.concatenate(parts)
    if picked.size < n_pick:
        used = set(picked.tolist())
        pool = [i for i in range(n) if i not in used]
        pad = min(n_pick - picked.size, len(pool))
        if pad > 0:
            picked = np.concatenate([picked, rng.choice(pool, size=pad, replace=False)])
    if picked.size > n_pick:
        picked = rng.choice(picked, size=n_pick, replace=False)
    return np.sort(picked.astype(np.int64, copy=False))


def select_stratified_users_and_cap(
    user_ids: np.ndarray,
    categories: np.ndarray,
    max_users: int,
    max_reviews_per_user: int,
    rng: np.random.RandomState
) -> np.ndarray:
    """
    Selects `max_users` evenly across categories (based on user's primary category),
    then caps each user to `max_reviews_per_user`.
    Returns the selected indices.
    """
    from collections import Counter
    unique_users = sorted(set(user_ids.tolist()))
    
    # 1. Find primary category for each user
    user_primary_cats = []
    for u in unique_users:
        u_cats = categories[user_ids == u]
        if len(u_cats) > 0:
            primary = Counter(u_cats).most_common(1)[0][0]
        else:
            primary = 0
        user_primary_cats.append(primary)
        
    # 2. Even selection of users
    unique_cats = sorted(set(categories.tolist()))
    n_cats = len(unique_cats)
    
    user_primary_cats = np.array(user_primary_cats)
    even_pick = []
    
    target_per_cat = max(1, max_users // max(1, n_cats))
    
    for c in unique_cats:
        cat_user_indices = np.where(user_primary_cats == c)[0]
        if len(cat_user_indices) > 0:
            k = min(len(cat_user_indices), target_per_cat)
            even_pick.extend(rng.choice(cat_user_indices, size=k, replace=False))
            
    # Pad to max_users if needed
    if len(even_pick) < max_users:
        remain = list(set(range(len(unique_users))) - set(even_pick))
        pad = min(max_users - len(even_pick), len(remain))
        if pad > 0:
            even_pick.extend(rng.choice(remain, size=pad, replace=False))
            
    selected_users = set([unique_users[i] for i in even_pick])
    
    # 3. Filter reviews and apply per-user cap
    final_indices = []
    for u in selected_users:
        u_idx = np.where(user_ids == u)[0]
        if max_reviews_per_user and max_reviews_per_user > 0 and len(u_idx) > max_reviews_per_user:
            u_idx = rng.choice(u_idx, size=max_reviews_per_user, replace=False)
        final_indices.extend(u_idx)
        
    final_indices = np.array(final_indices)
    final_indices.sort()
    return final_indices

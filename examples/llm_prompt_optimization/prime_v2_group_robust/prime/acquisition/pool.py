"""Seen/unseen pool management and cluster-margin expansion."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from prime.config import AcquisitionCfg, DatasetCfg
from prime.data.wilds_loader import ReviewSplit


class AcquisitionPool:
    """
    Manages train pool indices: seen/unseen, hard/anchor classification,
    and semantic expansion from unseen when anchor grows.
    """

    def __init__(
        self,
        split: ReviewSplit,
        cfg: DatasetCfg,
        acq_cfg: AcquisitionCfg,
        seed: int = 42,
    ) -> None:
        self.texts = split.texts
        self.labels = split.labels
        self.user_ids = split.user_ids
        self.cluster_ids = split.example_cluster_ids or [0] * len(split)
        self.cfg = cfg
        self.acq_cfg = acq_cfg
        self.seed = seed

        n = len(split)
        cap = min(n, cfg.al_candidate_pool_size) if cfg.al_candidate_pool_size else n
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)[:cap]
        self.pool_indices = sorted(int(i) for i in perm)

        self.seen: Set[int] = set()
        self.unseen: Set[int] = set(self.pool_indices)
        self.hard: Set[int] = set()
        self.anchor: Set[int] = set()
        self._embeddings: Optional[np.ndarray] = None

    def initialize_seen(self, batch_size: int) -> List[int]:
        rng = np.random.RandomState(self.seed)
        pick = rng.choice(self.pool_indices, size=min(batch_size, len(self.pool_indices)), replace=False)
        for idx in pick:
            self.mark_seen(int(idx))
        return sorted(int(i) for i in pick)

    def mark_seen(self, pool_index: int) -> None:
        self.seen.add(pool_index)
        self.unseen.discard(pool_index)

    def update_hard_anchor(
        self,
        predictions: Sequence[int],
        worker_preds: Sequence[Sequence[int]],
    ) -> None:
        from prime.workers.ensemble import disagreement_score

        for i, idx in enumerate(sorted(self.seen)):
            pred = int(predictions[i])
            gold = int(self.labels[idx])
            wp = [int(worker_preds[w][i]) for w in range(len(worker_preds))]
            d = disagreement_score(wp)
            is_hard = pred != gold or d > 0
            if is_hard:
                self.hard.add(idx)
                self.anchor.discard(idx)
            else:
                self.anchor.add(idx)
                self.hard.discard(idx)

    def maybe_expand(self) -> List[int]:
        """Pull cluster-margin examples from unseen when anchor count exceeds trigger."""
        if len(self.anchor) < self.acq_cfg.expansion_trigger or not self.unseen:
            return []
        n_add = min(self.acq_cfg.expansion_batch, len(self.unseen))
        if self._embeddings is None:
            self._embeddings = self._compute_embeddings()
        anchor_emb = self._embeddings[list(self.anchor)] if self.anchor else self._embeddings[:1]
        centroid = anchor_emb.mean(axis=0)
        candidates = sorted(self.unseen)
        dists = []
        for idx in candidates:
            dists.append((idx, float(np.linalg.norm(self._embeddings[idx] - centroid))))
        dists.sort(key=lambda x: x[1], reverse=True)
        added = [idx for idx, _ in dists[:n_add]]
        for idx in added:
            self.mark_seen(idx)
        return added

    def _compute_embeddings(self) -> np.ndarray:
        from prime.data.cache import encode_texts_cached, resolve_cache_dir

        texts = [self.texts[i] for i in self.pool_indices]
        tag = f"pool_expansion_{len(self.pool_indices)}"
        if self.cfg.use_cache:
            cache_root = resolve_cache_dir(self.cfg.data_root, self.cfg.cache_dir)
            emb = encode_texts_cached(
                texts, "all-MiniLM-L6-v2", cache_root, tag, use_cache=True
            )
        else:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("all-MiniLM-L6-v2")
            emb = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        full = np.zeros((len(self.texts), emb.shape[1]), dtype=np.float32)
        for j, pool_idx in enumerate(self.pool_indices):
            full[pool_idx] = emb[j]
        return full

    def seen_arrays(self) -> Tuple[List[int], List[str], List[int], List[int], List[int]]:
        idxs = sorted(self.seen)
        return (
            idxs,
            [self.texts[i] for i in idxs],
            [self.labels[i] for i in idxs],
            [self.user_ids[i] for i in idxs],
            [self.cluster_ids[i] for i in idxs],
        )

    def to_dict(self) -> Dict[str, List[int]]:
        return {
            "pool_indices": list(self.pool_indices),
            "seen": sorted(self.seen),
            "unseen": sorted(self.unseen),
            "hard": sorted(self.hard),
            "anchor": sorted(self.anchor),
        }

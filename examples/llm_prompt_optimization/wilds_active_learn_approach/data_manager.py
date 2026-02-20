"""
DataManager for Active Learning with Seen/Unseen pool management.

Core idea: work with a fixed-size active batch (~80 examples). As the prompt
improves and Hard examples are solved (move to Anchor), expand the pool by
pulling in new examples from Unseen that are semantically distant from Anchor.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
WILDS_EXPERIMENT = SCRIPT_DIR.parent / "wilds_experiment"
EXPERIMENTS_ROOT = WILDS_EXPERIMENT / "experiments"
if str(WILDS_EXPERIMENT) not in sys.path:
    sys.path.insert(0, str(WILDS_EXPERIMENT))
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))


def _load_pool_data(config: dict, split_name: str = "train"):
    """Load pool data (texts, labels, user_ids, indices) for a split."""
    import importlib.util
    import yaml

    base_eval_path = WILDS_EXPERIMENT / "evaluator.py"
    spec = importlib.util.spec_from_file_location("wilds_base_evaluator", base_eval_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    load_preprocessed_data = mod.load_preprocessed_data
    load_wilds_dataset = mod.load_wilds_dataset
    preprocess_category_data = mod.preprocess_category_data
    create_splits_from_preprocessed = mod.create_splits_from_preprocessed
    save_preprocessed_data = mod.save_preprocessed_data

    dataset_cfg_path = SCRIPT_DIR / "dataset.yaml"
    with open(dataset_cfg_path, "r", encoding="utf-8") as f:
        dataset_cfg = yaml.safe_load(f)
    dataset_cfg.setdefault("data_root", "./data")

    preprocessed = load_preprocessed_data(dataset_cfg)
    if preprocessed is None:
        dataset, _ = load_wilds_dataset(dataset_cfg)
        preprocessed = preprocess_category_data(dataset, dataset_cfg["category_id"])
        save_preprocessed_data(dataset_cfg, preprocessed)

    splits = create_splits_from_preprocessed(
        preprocessed,
        train_ratio=dataset_cfg.get("train_ratio", 0.7),
        val_ratio=dataset_cfg.get("validation_ratio", 0.15),
        test_ratio=dataset_cfg.get("test_ratio", 0.15),
        seed=dataset_cfg.get("split_seed", 42),
    )

    split = splits[split_name]
    indices = list(split["indices"])
    texts = [preprocessed["texts"][i] for i in indices]
    labels = [int(preprocessed["labels"][i]) for i in indices]
    user_ids = [int(preprocessed["user_ids"][i]) for i in indices]

    ds_cfg = config.get("dataset", {})
    max_users = ds_cfg.get("max_train_users" if split_name == "train" else "max_val_users")
    if max_users and max_users > 0:
        unique_users = sorted(set(user_ids))
        selected_users = set(unique_users[: int(max_users)])
        mask = [u in selected_users for u in user_ids]
        texts = [t for t, m in zip(texts, mask) if m]
        labels = [l for l, m in zip(labels, mask) if m]
        user_ids = [u for u, m in zip(user_ids, mask) if m]
        indices = [i for i, m in zip(indices, mask) if m]

    return texts, labels, user_ids, indices


def _compute_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("sentence-transformers required for embeddings")
    model = SentenceTransformer(model_name)
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)


def disagreement_score(
    worker_predictions: List[int],
    rating_min: int = 1,
    rating_max: int = 5,
) -> float:
    """
    Ordinal-aware disagreement in [0, 1].

    mean(|pred_i - pred_j|) over all pairs, normalized by (rating_max - rating_min).
    """
    if len(worker_predictions) < 2:
        return 0.0
    scale = rating_max - rating_min
    n = len(worker_predictions)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            total += abs(worker_predictions[i] - worker_predictions[j])
    num_pairs = n * (n - 1) / 2
    return float(total / num_pairs / scale)


class DataManager:
    """
    Manages Seen/Unseen pools, Hard/Anchor classification, and pool expansion.

    All data arrays (texts, labels, user_ids) are indexed by "pool index" (0..N-1
    where N is the full train set). Seen/Unseen/Hard/Anchor are sets of these
    indices.
    """

    def __init__(self, config: dict, split_name: str = "train"):
        self.config = config
        self.split_name = split_name

        self.texts: List[str] = []
        self.labels: List[int] = []
        self.user_ids: List[int] = []
        self.pool_indices: List[int] = []

        self.seen_indices: Set[int] = set()
        self.unseen_indices: Set[int] = set()
        self.hard_indices: List[int] = []
        self.anchor_indices: List[int] = []

        self._embeddings_cache: Optional[np.ndarray] = None

        self._load_all_data()

    def _load_all_data(self) -> None:
        self.texts, self.labels, self.user_ids, self.pool_indices = _load_pool_data(
            self.config, self.split_name
        )
        self.n_total = len(self.texts)
        self.unseen_indices = set(range(self.n_total))
        self.seen_indices = set()

    @property
    def n_hard(self) -> int:
        return len(self.hard_indices)

    @property
    def n_anchor(self) -> int:
        return len(self.anchor_indices)

    @property
    def n_seen(self) -> int:
        return len(self.seen_indices)

    @property
    def n_unseen(self) -> int:
        return len(self.unseen_indices)

    def _get_embeddings(self) -> np.ndarray:
        """Compute and cache embeddings for the full pool."""
        if self._embeddings_cache is None:
            self._embeddings_cache = _compute_embeddings(self.texts)
        return self._embeddings_cache

    # ------------------------------------------------------------------
    # Initialization: build the first active batch from a full-pool eval
    # ------------------------------------------------------------------

    def initialize_from_evaluation(
        self,
        predictions: List[int],
        gold_labels: List[int],
        worker_predictions: List[List[int]],
        batch_size: int = 80,
        hard_ratio: float = 0.7,
        seed: int = 42,
    ) -> List[int]:
        """
        One-time initialization after the first full-pool evaluation.

        1. Classify every example as Hard or Anchor.
        2. Select an initial active batch of `batch_size`.
        3. Mark selected examples as Seen; the rest stay Unseen.
        4. Return the list of selected indices.
        """
        all_hard, all_anchor = self._classify(predictions, gold_labels, worker_predictions)

        n_hard = int(batch_size * hard_ratio)
        n_anchor = batch_size - n_hard

        hard_batch = self._select_diverse(all_hard, n_hard, seed)
        anchor_batch = self._select_stratified(all_anchor, n_anchor, seed)
        batch = list(dict.fromkeys(hard_batch + anchor_batch))[:batch_size]

        self.seen_indices = set(batch)
        self.unseen_indices = set(range(self.n_total)) - self.seen_indices

        batch_hard_set = set(all_hard)
        self.hard_indices = [i for i in batch if i in batch_hard_set]
        self.anchor_indices = [i for i in batch if i not in batch_hard_set]

        return batch

    # ------------------------------------------------------------------
    # Reclassify only the current batch (not the full pool)
    # ------------------------------------------------------------------

    def reclassify_batch(
        self,
        batch_indices: List[int],
        predictions: List[int],
        gold_labels: List[int],
        worker_predictions: List[List[int]],
    ) -> Tuple[List[int], List[int]]:
        """
        Reclassify examples within the current batch after evolution.

        Returns updated (hard_indices, anchor_indices) within the batch.
        """
        uncertainty_threshold = self.config.get("active_learning", {}).get(
            "uncertainty_threshold", 0.0
        )
        hard = []
        anchor = []
        for j, idx in enumerate(batch_indices):
            pred = predictions[j]
            gold = gold_labels[j]
            wp = [int(w[j]) for w in worker_predictions]
            correct = pred == gold
            d_score = disagreement_score(wp, rating_min=1, rating_max=5)
            if (not correct) or (d_score > uncertainty_threshold):
                hard.append(idx)
            else:
                anchor.append(idx)

        self.hard_indices = hard
        self.anchor_indices = anchor
        return hard, anchor

    # ------------------------------------------------------------------
    # Pool expansion: add new Hard from Unseen, far from Anchor
    # ------------------------------------------------------------------

    def needs_expansion(self, threshold: int = 5) -> bool:
        return self.n_hard <= threshold and self.n_unseen > 0

    def expand_pool(
        self,
        n_new: int,
        seed: int = 42,
    ) -> List[int]:
        """
        Select `n_new` examples from Unseen that are semantically farthest
        from the current Anchor set. They are added to Seen and returned
        (to be treated as new Hard).

        Distance: for each unseen example, compute min cosine distance to
        any Anchor example. Pick top-n_new by descending distance.
        """
        if self.n_unseen == 0:
            return []

        n_new = min(n_new, self.n_unseen)
        embeddings = self._get_embeddings()

        anchor_set = set(self.anchor_indices)
        if not anchor_set:
            rng = np.random.default_rng(seed)
            unseen_list = sorted(self.unseen_indices)
            selected = rng.choice(unseen_list, size=n_new, replace=False).tolist()
            self._move_to_seen(selected)
            return selected

        anchor_list = sorted(anchor_set)
        anchor_embs = embeddings[anchor_list]

        unseen_list = sorted(self.unseen_indices)
        unseen_embs = embeddings[unseen_list]

        # Cosine similarity → distance = 1 - similarity
        anchor_norms = anchor_embs / (np.linalg.norm(anchor_embs, axis=1, keepdims=True) + 1e-9)
        unseen_norms = unseen_embs / (np.linalg.norm(unseen_embs, axis=1, keepdims=True) + 1e-9)
        # sim_matrix[i, j] = cosine_sim(unseen[i], anchor[j])
        sim_matrix = unseen_norms @ anchor_norms.T
        # For each unseen: min distance to any anchor = 1 - max similarity
        min_distance = 1.0 - np.max(sim_matrix, axis=1)

        top_indices = np.argsort(min_distance)[::-1][:n_new]
        selected = [unseen_list[i] for i in top_indices]

        self._move_to_seen(selected)
        return selected

    # ------------------------------------------------------------------
    # Build active batch from current Hard + Anchor within Seen
    # ------------------------------------------------------------------

    def build_active_batch(
        self,
        batch_size: int = 80,
        hard_ratio: float = 0.7,
        seed: int = 42,
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Assemble an active batch from current Hard and Anchor in Seen.

        Returns (batch_indices, hard_in_batch, anchor_in_batch).
        """
        n_hard = int(batch_size * hard_ratio)
        n_anchor = batch_size - n_hard

        hard_batch = self._select_diverse(self.hard_indices, n_hard, seed)
        anchor_batch = self._select_stratified(self.anchor_indices, n_anchor, seed)
        batch = list(dict.fromkeys(hard_batch + anchor_batch))[:batch_size]

        hard_set = set(self.hard_indices)
        hard_in = [i for i in batch if i in hard_set]
        anchor_in = [i for i in batch if i not in hard_set]
        return batch, hard_in, anchor_in

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify(
        self,
        predictions: List[int],
        gold_labels: List[int],
        worker_predictions: List[List[int]],
    ) -> Tuple[List[int], List[int]]:
        """Classify all examples (full pool). Returns (hard, anchor) index lists."""
        uncertainty_threshold = self.config.get("active_learning", {}).get(
            "uncertainty_threshold", 0.0
        )
        hard = []
        anchor = []
        for i in range(len(predictions)):
            pred = predictions[i]
            gold = gold_labels[i]
            wp = [int(w[i]) for w in worker_predictions]
            correct = pred == gold
            d_score = disagreement_score(wp, rating_min=1, rating_max=5)
            if (not correct) or (d_score > uncertainty_threshold):
                hard.append(i)
            else:
                anchor.append(i)
        return hard, anchor

    def _select_diverse(self, indices: List[int], size: int, seed: int) -> List[int]:
        """K-Means diversity selection from a set of indices."""
        if not indices:
            return []
        if len(indices) <= size:
            return list(indices)

        rng = np.random.default_rng(seed)
        embeddings = self._get_embeddings()
        embs = embeddings[indices]
        n_clusters = min(8, len(indices) // 2, size)
        if n_clusters < 2:
            return rng.choice(indices, size=size, replace=False).tolist()

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        kmeans.fit(embs)

        selected = []
        per_cluster = max(1, size // n_clusters)
        for c in range(n_clusters):
            mask = kmeans.labels_ == c
            cluster_idx = [indices[j] for j in range(len(indices)) if mask[j]]
            cluster_emb = embs[mask]
            centroid = kmeans.cluster_centers_[c]
            dists = np.linalg.norm(cluster_emb - centroid, axis=1)
            order = np.argsort(dists)
            take = min(per_cluster, len(order))
            for j in order[:take]:
                selected.append(cluster_idx[j])
            if len(selected) >= size:
                break

        if len(selected) < size:
            remaining = [i for i in indices if i not in set(selected)]
            extra = rng.choice(remaining, min(size - len(selected), len(remaining)), replace=False)
            selected.extend(extra.tolist())
        return selected[:size]

    def _select_stratified(self, indices: List[int], size: int, seed: int) -> List[int]:
        """Stratified random sample by label."""
        if not indices:
            return []
        if len(indices) <= size:
            return list(indices)

        rng = np.random.default_rng(seed)
        by_label: Dict[int, List[int]] = {}
        for i in indices:
            lb = self.labels[i]
            by_label.setdefault(lb, []).append(i)
        selected = []
        per_class = max(1, size // max(len(by_label), 1))
        for lb in sorted(by_label.keys()):
            pool = by_label[lb]
            n_take = min(per_class, len(pool))
            selected.extend(rng.choice(pool, n_take, replace=False).tolist())
        if len(selected) < size:
            remaining = [i for i in indices if i not in set(selected)]
            if remaining:
                extra = rng.choice(remaining, min(size - len(selected), len(remaining)), replace=False)
                selected.extend(extra.tolist())
        return selected[:size]

    def _move_to_seen(self, indices: List[int]) -> None:
        for idx in indices:
            self.unseen_indices.discard(idx)
            self.seen_indices.add(idx)

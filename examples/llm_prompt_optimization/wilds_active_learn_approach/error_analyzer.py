"""
ErrorAnalyzer: cluster hard examples and format for evolution prompt.

Identifies error patterns and formats them for the LLM mutator.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent


def _compute_embeddings(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("sentence-transformers required")
    model = SentenceTransformer(model_name)
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)


class ErrorAnalyzer:
    """
    Analyzes hard examples (errors) via clustering and formats for evolution.
    """

    def __init__(self, data_manager: "DataManager"):
        self.dm = data_manager

    def analyze_errors(
        self,
        hard_indices: List[int],
        k_clusters: int = 6,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Cluster hard examples and return analysis.

        Returns:
            {
                "clusters": [
                    {
                        "id": int,
                        "indices": list of pool indices,
                        "texts": list of review texts,
                        "labels": list of gold labels,
                        "centroid_idx": index of representative example,
                    }
                ],
                "n_total": int,
            }
        """
        if not hard_indices:
            return {"clusters": [], "n_total": 0}

        texts = [self.dm.texts[i] for i in hard_indices]
        labels = [self.dm.labels[i] for i in hard_indices]
        embeddings = _compute_embeddings(texts)

        n_clusters = min(k_clusters, len(hard_indices) // 2, len(hard_indices) - 1)
        if n_clusters < 2:
            return {
                "clusters": [
                    {
                        "id": 0,
                        "indices": hard_indices,
                        "texts": texts,
                        "labels": labels,
                        "centroid_idx": 0,
                    }
                ],
                "n_total": len(hard_indices),
            }

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        kmeans.fit(embeddings)

        clusters = []
        for c in range(n_clusters):
            mask = kmeans.labels_ == c
            cluster_indices = [hard_indices[i] for i in range(len(hard_indices)) if mask[i]]
            cluster_texts = [texts[i] for i in range(len(texts)) if mask[i]]
            cluster_labels = [labels[i] for i in range(len(labels)) if mask[i]]
            cluster_emb = embeddings[mask]
            centroid = kmeans.cluster_centers_[c]
            dists = np.linalg.norm(cluster_emb - centroid, axis=1)
            centroid_local_idx = int(np.argmin(dists))
            centroid_pool_idx = cluster_indices[centroid_local_idx]

            clusters.append({
                "id": c,
                "indices": cluster_indices,
                "texts": cluster_texts,
                "labels": cluster_labels,
                "centroid_idx": centroid_pool_idx,
            })

        return {"clusters": clusters, "n_total": len(hard_indices)}

    def format_for_evolution(
        self,
        analysis: Dict[str, Any],
        max_examples: int = 15,
        max_text_len: int = 300,
        *,
        predictions: Optional[List[int]] = None,
        worker_predictions: Optional[List[List[int]]] = None,
        batch_indices: Optional[List[int]] = None,
        batch_stats: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Format error analysis as text for the LLM mutation prompt.
        """
        clusters = analysis.get("clusters", [])
        lines: List[str] = []

        if batch_stats:
            bsz = int(batch_stats.get("batch_size", 0))
            b_hard = int(batch_stats.get("batch_hard", 0))
            b_anchor = int(batch_stats.get("batch_anchor", 0))
            seen = int(batch_stats.get("n_seen", 0))
            unseen = int(batch_stats.get("n_unseen", 0))
            h_total = int(batch_stats.get("n_hard_total", 0))
            a_total = int(batch_stats.get("n_anchor_total", 0))
            hard_frac = (b_hard / bsz) if bsz > 0 else 0.0
            lines.append(
                "BATCH COMPOSITION: "
                f"{b_hard} Hard / {b_anchor} Anchor (batch={bsz}, hard_frac={hard_frac:.1%}); "
                f"Seen={seen}, Unseen={unseen}; Pool Hard={h_total}, Anchor={a_total}."
            )

        if (
            predictions is not None
            and worker_predictions is not None
            and batch_indices is not None
            and len(predictions) == len(batch_indices)
        ):
            # Rich mode: include confusion matrix and representative examples from previous cycle.
            confusion: Dict[tuple, int] = {}
            grouped_examples: Dict[tuple, List[str]] = {}

            n_workers = len(worker_predictions)
            for j, idx in enumerate(batch_indices):
                gold = int(self.dm.labels[idx])
                pred = int(predictions[j])
                pair = (gold, pred)
                confusion[pair] = confusion.get(pair, 0) + 1
                if len(grouped_examples.get(pair, [])) >= 2:
                    continue
                review = self.dm.texts[idx]
                short_review = review[:max_text_len] + ("..." if len(review) > max_text_len else "")
                workers_here = []
                for w in range(n_workers):
                    if j < len(worker_predictions[w]):
                        workers_here.append(int(worker_predictions[w][j]))
                grouped_examples.setdefault(pair, []).append(
                    f'  - Review: "{short_review}"\n'
                    f"    Gold={gold}, Pred={pred}, Workers={workers_here}"
                )

            if confusion:
                sorted_pairs = sorted(confusion.items(), key=lambda kv: kv[1], reverse=True)
                lines.append("CONFUSION MATRIX (from previous cycle batch):")
                for (gold, pred), count in sorted_pairs[:5]:
                    lines.append(f"  - gold={gold} -> pred={pred}: {count}")
                lines.append("")
                lines.append("REPRESENTATIVE EXAMPLES BY CONFUSION TYPE:")
                for (gold, pred), _count in sorted_pairs[:4]:
                    lines.append(f"* Pair gold={gold} -> pred={pred}:")
                    for ex in grouped_examples.get((gold, pred), []):
                        lines.append(ex)
                lines.append("")

        if not clusters:
            if lines:
                return "\n".join(lines)
            return "No error examples available."

        examples_added = 0
        lines.append("CLUSTERED HARD EXAMPLES (review + gold label):")
        for cluster in clusters:
            if examples_added >= max_examples:
                break
            texts = cluster.get("texts", [])
            labels = cluster.get("labels", [])
            if not texts:
                continue
            # Add 2–3 examples per cluster
            n_per = min(3, max(1, (max_examples - examples_added) // len(clusters)))
            for i in range(min(n_per, len(texts))):
                t = texts[i][:max_text_len] + ("..." if len(texts[i]) > max_text_len else "")
                lines.append(f"  - Review: {t}")
                lines.append(f"    Gold: {labels[i]}")
                examples_added += 1
                if examples_added >= max_examples:
                    break
            if examples_added >= max_examples:
                break

        return "\n".join(lines)

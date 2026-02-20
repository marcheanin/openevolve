"""
ErrorAnalyzer: cluster hard examples and format for evolution prompt.

Identifies error patterns and formats them for the LLM mutator.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Any

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
    ) -> str:
        """
        Format error analysis as text for the LLM mutation prompt.
        """
        clusters = analysis.get("clusters", [])
        if not clusters:
            return "No error examples available."

        lines = []
        examples_added = 0
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

        header = "Examples where the model made errors (review + gold label):\n"
        return header + "\n".join(lines)

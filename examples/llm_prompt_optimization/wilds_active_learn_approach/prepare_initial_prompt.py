"""
APE (Automatic Prompt Engineer) Initial Prompt Preparation.

Performs Stratified Diversity Sampling on WILDS train data and generates
a meta-prompt for the user to paste into ChatGPT for APE-style prompt generation.

Usage:
    python prepare_initial_prompt.py

Output:
    ape_meta_prompt.txt - Paste this into ChatGPT web to generate prompt variants.
"""

import os
import sys
from pathlib import Path

# Load .env before any imports that need OPENAI_API_KEY (e.g. evaluator)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

import numpy as np
import yaml

# Add parent paths for imports
SCRIPT_DIR = Path(__file__).resolve().parent
WILDS_EXPERIMENT = SCRIPT_DIR.parent / "wilds_experiment"
EXPERIMENTS_ROOT = WILDS_EXPERIMENT / "experiments"
if str(WILDS_EXPERIMENT) not in sys.path:
    sys.path.insert(0, str(WILDS_EXPERIMENT))
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))


def load_dataset_config() -> dict:
    """Load dataset config from dataset.yaml."""
    config_path = SCRIPT_DIR / "dataset.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_train_data():
    """
    Load train split texts and labels.
    Reuses wilds_experiment data loading.
    """
    from evaluator import (
        load_wilds_dataset,
        preprocess_category_data,
        create_splits_from_preprocessed,
        load_preprocessed_data,
        save_preprocessed_data,
    )

    dataset_cfg = load_dataset_config()
    category_id = dataset_cfg.get("category_id", 24)
    # data_root relative to wilds_experiment dir
    dataset_cfg.setdefault("data_root", "./data")

    preprocessed = load_preprocessed_data(dataset_cfg)
    if preprocessed is None:
        print("Cache miss - loading WILDS dataset (first run may take a few minutes)...")
        dataset, _ = load_wilds_dataset(dataset_cfg)
        preprocessed = preprocess_category_data(dataset, category_id)
        save_preprocessed_data(dataset_cfg, preprocessed)

    splits = create_splits_from_preprocessed(
        preprocessed,
        train_ratio=dataset_cfg.get("train_ratio", 0.7),
        val_ratio=dataset_cfg.get("validation_ratio", 0.15),
        test_ratio=dataset_cfg.get("test_ratio", 0.15),
        seed=dataset_cfg.get("split_seed", 42),
    )

    train_split = splits["train"]
    indices = train_split["indices"]
    texts = [preprocessed["texts"][i] for i in indices]
    labels = [int(preprocessed["labels"][i]) for i in indices]
    return texts, labels, preprocessed


def compute_embeddings(texts, model_name: str = "all-MiniLM-L6-v2"):
    """Compute embeddings using sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers required. Install: pip install sentence-transformers"
        )
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings


def stratified_diversity_sampling(
    texts: list,
    labels: list,
    n_per_class: int = 3,
    k_clusters: int = 3,
    seed: int = 42,
):
    """
    Stratified Diversity Sampling:
    - Group by class (1-5)
    - Within each class: K-Means on embeddings
    - Select centroid (typical) + farthest-from-centroid (outlier) per cluster
    - Total: ~n_per_class * k_clusters * 2 examples (capped)
    """
    from sklearn.cluster import KMeans

    np.random.seed(seed)
    embeddings = compute_embeddings(texts)

    selected_indices = []
    for class_label in range(1, 6):
        class_mask = np.array(labels) == class_label
        class_indices = np.where(class_mask)[0]
        if len(class_indices) == 0:
            continue

        class_emb = embeddings[class_indices]
        n_clusters = min(k_clusters, len(class_indices))
        if n_clusters < 2:
            selected_indices.extend(class_indices.tolist())
            continue

        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        kmeans.fit(class_emb)
        labels_cluster = kmeans.labels_

        for c in range(n_clusters):
            cluster_mask = labels_cluster == c
            cluster_idx = class_indices[cluster_mask]
            cluster_emb = class_emb[cluster_mask]
            centroid = kmeans.cluster_centers_[c]

            # Centroid: closest to cluster center
            dists = np.linalg.norm(cluster_emb - centroid, axis=1)
            centroid_idx = cluster_idx[np.argmin(dists)]
            selected_indices.append(int(centroid_idx))

            # Outlier: farthest from centroid (if cluster has >1 point)
            if len(cluster_idx) > 1:
                outlier_idx = cluster_idx[np.argmax(dists)]
                selected_indices.append(int(outlier_idx))

    # Cap total and deduplicate
    selected_indices = list(dict.fromkeys(selected_indices))
    max_examples = 25
    if len(selected_indices) > max_examples:
        np.random.shuffle(selected_indices)
        selected_indices = selected_indices[:max_examples]
    return selected_indices


def format_ape_meta_prompt(examples: list[tuple[str, int]]) -> str:
    """Format examples for APE meta-prompt."""
    lines = []
    for i, (text, label) in enumerate(examples, 1):
        text_preview = text[:500] + "..." if len(text) > 500 else text
        lines.append(f"Example {i}:")
        lines.append(f"Review: {text_preview}")
        lines.append(f"Gold label: {label}")
        lines.append("")
    return "\n".join(lines)


def main():
    print("Loading train data...")
    texts, labels, _ = load_train_data()
    print(f"Loaded {len(texts)} train examples")

    print("Stratified diversity sampling (K-Means)...")
    selected = stratified_diversity_sampling(texts, labels)
    examples = [(texts[i], labels[i]) for i in selected]
    print(f"Selected {len(examples)} golden examples")

    examples_block = format_ape_meta_prompt(examples)

    meta_prompt = f"""You are an expert at designing prompts for LLM-based sentiment classification.

Here are {len(examples)} representative examples of Amazon Home & Kitchen product reviews with gold labels (1-5 stars):

{examples_block}

Write a detailed System Prompt that teaches another LLM to classify such reviews into 1-5 stars.

IMPORTANT: The prompt MUST use the following XML structure with three sections:

<System>
    <Role>You are a sentiment classification model for Amazon Home & Kitchen product reviews.</Role>
    
    <BaseGuidelines>
        (Static rules that should NOT change during evolution:
         - Star rating scale: 1-5
         - Output format: ONLY a number (1, 2, 3, 4, or 5), no explanation
         - Task: classify sentiment based on review text)
    </BaseGuidelines>

    <DynamicRules>
        (Rules that WILL be evolved during active learning:
         - Criteria for each star level (1=very negative, 5=very positive)
         - Edge case handling: sarcasm, mixed sentiment, nuanced reviews
         - Specific patterns and examples)
    </DynamicRules>
</System>

<FewShotExamples>
    (Include 5-8 representative examples from the list above, formatted as:
     Example 1:
     Review: [text]
     Rating: [label]
     
     Example 2:
     ...)
</FewShotExamples>

<Task>
    Review: {{review}}
    
    Rating:
</Task>

The prompt MUST:
- Use the exact XML structure above with <System>, <BaseGuidelines>, <DynamicRules>, <FewShotExamples>, and <Task> sections
- Put static, unchanging rules in <BaseGuidelines>
- Put evolvable rules and criteria in <DynamicRules>
- Include 5-8 few-shot examples in <FewShotExamples>
- Use {{review}} as the placeholder in <Task>
- Request ONLY a number (1, 2, 3, 4, or 5) as output—no explanation

Generate 5 different prompt variants following this structure. For each variant, briefly explain its strengths.
"""

    output_path = SCRIPT_DIR / "ape_meta_prompt.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(meta_prompt)
    print(f"Wrote {output_path}")
    print("Next: Paste the content into ChatGPT web, then save the best variant to initial_prompt.txt")


if __name__ == "__main__":
    main()

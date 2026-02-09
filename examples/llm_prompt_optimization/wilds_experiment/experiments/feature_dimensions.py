"""
Feature dimensions for MAP-Elites in prompt evolution.

These metrics are designed to create meaningful diversity in the prompt space,
enabling effective exploration of different prompt strategies.

Key principles:
1. Metrics should have wide range of values (fill the grid evenly)
2. Metrics should correlate with different approaches to the task
3. Metrics should be interpretable
"""

import re
from typing import Dict, Any, Optional


def calculate_criteria_explicitness(prompt: str) -> float:
    """
    Calculate how explicitly the prompt describes criteria for each star rating (1-5).
    
    This metric measures whether the prompt clearly defines what each rating means,
    which is crucial for consistent classification.
    
    Returns:
        float: 0.0 to 1.0, where 1.0 means all 5 star ratings are explicitly described
    
    Examples:
        - "1 star: broken product" -> counts as explicit for 1 star
        - "- 5: excellent" -> counts as explicit for 5 stars
        - Generic prompt without star descriptions -> 0.0
    """
    prompt_lower = prompt.lower()
    explicit_criteria = 0
    
    for star in range(1, 6):
        # Various patterns for explicit star rating criteria
        patterns = [
            # "1 star:" or "1 stars:"
            rf'{star}\s*stars?\s*:',
            # "- 1:" at start of line or after newline
            rf'[-•]\s*{star}\s*:',
            # "1:" at start of line (numbered list)
            rf'^\s*{star}\s*:', 
            # "rating of 1" or "1 rating"
            rf'rating\s+of\s+{star}|{star}\s+rating',
            # "1-star" or "1 star"
            rf'{star}[-\s]stars?\s*[-:–]',
            # Explicit mentions like "For 1 star"
            rf'for\s+{star}\s+stars?',
        ]
        
        for pattern in patterns:
            if re.search(pattern, prompt_lower, re.MULTILINE | re.IGNORECASE):
                explicit_criteria += 1
                break
    
    return explicit_criteria / 5.0


def calculate_domain_focus(prompt: str) -> float:
    """
    Calculate how specific the prompt is to the Home & Kitchen product domain.
    
    Domain-specific prompts may generalize better within the domain but worse
    outside of it. This creates an interesting exploration axis.
    
    Returns:
        float: 0.0 to 1.0, where higher values indicate more domain specificity
    """
    # Domain-specific keywords for Home & Kitchen products
    domain_keywords = [
        # Product categories
        'kitchen', 'home', 'appliance', 'cookware', 'utensil',
        'gadget', 'tool', 'furniture', 'decor', 'storage',
        # Quality attributes
        'durable', 'sturdy', 'quality', 'material', 'build',
        'design', 'ergonomic', 'compact', 'portable',
        # Usage context
        'daily use', 'everyday', 'cooking', 'cleaning', 'organizing',
        # Value indicators
        'value', 'price', 'worth', 'investment', 'affordable',
        # Recommendation patterns
        'recommend', 'buy', 'purchase', 'gift',
    ]
    
    prompt_lower = prompt.lower()
    
    # Count unique domain keywords found
    found_keywords = sum(1 for kw in domain_keywords if kw in prompt_lower)
    
    # Normalize: saturate at ~10 keywords (half the list)
    return min(1.0, found_keywords / 10.0)


def calculate_instruction_specificity(prompt: str) -> float:
    """
    Calculate how specific and actionable the instructions in the prompt are.
    
    Specific instructions (e.g., "focus on emotional words") tend to produce
    more consistent results than vague ones (e.g., "analyze the review").
    
    Returns:
        float: 0.0 to 1.0, where higher values indicate more specific instructions
    """
    prompt_lower = prompt.lower()
    
    # Specific instruction patterns (actionable, concrete)
    specific_patterns = [
        r'focus on',
        r'look for',
        r'pay attention to',
        r'consider\s+the',
        r'evaluate\s+the',
        r'identify',
        r'check\s+(for|if|whether)',
        r'note\s+(the|that|whether)',
        r'prioritize',
        r'weigh\s+the',
        r'compare',
        r'distinguish\s+between',
    ]
    
    # Vague instruction patterns (abstract, general)
    vague_patterns = [
        r'analyze\s+the\s+review',
        r'determine\s+the\s+rating',
        r'predict\s+the\s+star',
        r'assign\s+a\s+rating',
        r'evaluate\s+the\s+review',
    ]
    
    specific_count = sum(1 for p in specific_patterns if re.search(p, prompt_lower))
    vague_count = sum(1 for p in vague_patterns if re.search(p, prompt_lower))
    
    # Ratio of specific to total instructions
    total = specific_count + vague_count
    if total == 0:
        return 0.5  # Neutral if no instructions detected
    
    return min(1.0, specific_count / max(total, 1))


def calculate_example_richness(prompt: str) -> float:
    """
    Calculate how many examples or illustrative phrases the prompt contains.
    
    Examples help models understand the task better, but too many can be
    overwhelming or cause overfitting to specific patterns.
    
    Returns:
        float: 0.0 to 1.0, where higher values indicate more examples
    """
    prompt_lower = prompt.lower()
    
    # Patterns that indicate examples
    example_patterns = [
        r'for example',
        r'e\.g\.',
        r'such as',
        r'like\s+"[^"]+"',  # like "word"
        r'"[^"]+"\s*,\s*"[^"]+"',  # quoted lists
        r'words?\s+like\s+"',
        r'phrases?\s+like\s+"',
        r'including\s+"',
    ]
    
    # Count example indicators
    example_count = sum(1 for p in example_patterns if re.search(p, prompt_lower))
    
    # Also count quoted strings as potential examples
    quoted_strings = re.findall(r'"[^"]{2,30}"', prompt)  # 2-30 char quoted strings
    example_count += len(quoted_strings) // 2  # Divide by 2 to not overcount
    
    # Normalize: saturate at 8 examples
    return min(1.0, example_count / 8.0)


def calculate_sentiment_vocabulary(prompt: str) -> float:
    """
    Calculate the richness of sentiment vocabulary in the prompt.
    
    Prompts with rich sentiment vocabulary help models distinguish between
    nuanced sentiment levels.
    
    Returns:
        float: 0.0 to 1.0, where higher values indicate richer vocabulary
    """
    # Positive sentiment words
    positive_words = [
        'love', 'excellent', 'perfect', 'amazing', 'great', 'fantastic',
        'wonderful', 'outstanding', 'superb', 'exceptional', 'satisfied',
        'happy', 'pleased', 'recommend', 'best', 'favorite', 'impressed',
        'delighted', 'thrilled', 'exceeded',
    ]
    
    # Negative sentiment words
    negative_words = [
        'hate', 'terrible', 'awful', 'horrible', 'worst', 'broken',
        'defective', 'useless', 'disappointed', 'frustrated', 'angry',
        'regret', 'waste', 'failed', 'poor', 'bad', 'cheap', 'flimsy',
        'warning', 'avoid', 'return', 'refund',
    ]
    
    # Neutral/mixed sentiment words
    neutral_words = [
        'okay', 'fine', 'average', 'decent', 'acceptable', 'mixed',
        'neutral', 'lukewarm', 'indifferent', 'mediocre',
    ]
    
    prompt_lower = prompt.lower()
    
    positive_found = sum(1 for w in positive_words if w in prompt_lower)
    negative_found = sum(1 for w in negative_words if w in prompt_lower)
    neutral_found = sum(1 for w in neutral_words if w in prompt_lower)
    
    total_found = positive_found + negative_found + neutral_found
    
    # Normalize: saturate at 15 sentiment words
    return min(1.0, total_found / 15.0)


def calculate_structural_complexity(prompt: str) -> float:
    """
    Calculate the structural complexity of the prompt.
    
    More structured prompts (with sections, bullet points, etc.) may be
    easier for models to follow, but overly complex structures can confuse.
    
    Returns:
        float: 0.0 to 1.0, where higher values indicate more structure
    """
    # Count structural elements
    bullet_points = len(re.findall(r'^[\s]*[-•*]\s', prompt, re.MULTILINE))
    numbered_items = len(re.findall(r'^[\s]*\d+[.)]\s', prompt, re.MULTILINE))
    sections = len(re.findall(r'^#+\s|^[A-Z][^.!?]*:\s*$', prompt, re.MULTILINE))
    newlines = prompt.count('\n')
    
    # Combine metrics
    structure_score = (
        min(5, bullet_points) * 0.15 +
        min(5, numbered_items) * 0.15 +
        min(3, sections) * 0.2 +
        min(10, newlines) * 0.05
    )
    
    return min(1.0, structure_score)


def calculate_sentiment_vocabulary_richness(prompt: str) -> float:
    """
    Calculate a combined metric that measures both sentiment vocabulary richness
    and example richness in the prompt.
    
    This metric combines:
    - sentiment_vocabulary: Richness of sentiment words (positive, negative, neutral)
    - example_richness: Presence of examples and illustrative phrases
    
    The combination creates a meaningful dimension that captures how well the prompt
    guides the model through both vocabulary and concrete examples.
    
    Formula: weighted average of sentiment_vocabulary (60%) and example_richness (40%)
    
    Returns:
        float: 0.0 to 1.0, where higher values indicate richer vocabulary and more examples
    """
    sentiment_vocab = calculate_sentiment_vocabulary(prompt)
    example_rich = calculate_example_richness(prompt)
    
    # Weighted combination: vocabulary is slightly more important
    combined = 0.6 * sentiment_vocab + 0.4 * example_rich
    
    return min(1.0, combined)


def calculate_all_features(
    prompt: str,
    metrics: Optional[Dict[str, Any]] = None,
    is_ensemble: bool = False,
) -> Dict[str, float]:
    """
    Calculate all feature dimensions for a prompt.
    
    Args:
        prompt: The prompt text
        metrics: Optional metrics dict (for ensemble-specific features like mean_kappa)
        is_ensemble: Whether this is an ensemble experiment
    
    Returns:
        Dict with all calculated feature dimensions
    """
    features = {
        'criteria_explicitness': calculate_criteria_explicitness(prompt),
        'domain_focus': calculate_domain_focus(prompt),
        'instruction_specificity': calculate_instruction_specificity(prompt),
        'example_richness': calculate_example_richness(prompt),
        'sentiment_vocabulary': calculate_sentiment_vocabulary(prompt),
        'sentiment_vocabulary_richness': calculate_sentiment_vocabulary_richness(prompt),
        'structural_complexity': calculate_structural_complexity(prompt),
        # Keep legacy features for backward compatibility
        'prompt_length': min(1.0, len(prompt) / 2000.0),
    }
    
    # For ensemble experiments, add mean_kappa (Cohen's Kappa) as feature; normalize to [0, 1]
    if is_ensemble and metrics:
        mean_kappa = metrics.get('mean_kappa', 0.0)
        if isinstance(mean_kappa, (int, float)):
            features['mean_kappa'] = max(0.0, float(mean_kappa))
            features['ensemble_agreement'] = features['mean_kappa']
    
    return features


# Recommended feature dimension combinations for different experiments
RECOMMENDED_FEATURES = {
    'exp2_single': ['sentiment_vocabulary_richness', 'domain_focus'],
    'exp3_ensemble': ['sentiment_vocabulary_richness', 'mean_kappa'],
    'exp4_llm_aggregator': ['sentiment_vocabulary_richness', 'ensemble_agreement'],
}

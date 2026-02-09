#!/usr/bin/env python3
"""
Final validation script for the best evolved prompt.
Runs comprehensive evaluation on ALL examples from test split (default) or validation split.

IMPORTANT: Test split is NOT used during evolution - this is a fair evaluation of generalization.

Usage:
    python validate_best_prompt.py                    # Default: ALL test examples
    python validate_best_prompt.py --all              # Explicitly use ALL examples
    python validate_best_prompt.py --split test      # Use test split (default)
    python validate_best_prompt.py --split validation # Use validation split
    python validate_best_prompt.py --samples 200      # Use sample instead (not recommended)
    python validate_best_prompt.py --prompt custom.txt # Use custom prompt file
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from evaluator import (
    get_dataset_splits,
    stratified_sample,
    evaluate_on_samples,
    calculate_features,
    load_prompt_config
)


def validate_prompt(
    prompt: str,
    splits_data: Dict,
    split_name: str = 'validation',
    n_samples: Optional[int] = None,
    use_all: bool = False
) -> Dict[str, Any]:
    """
    Validate a prompt on a sample or all examples from the specified split.
    
    Args:
        prompt: The prompt text
        splits_data: Dataset splits
        split_name: 'validation' or 'test'
        n_samples: Number of samples to evaluate (if None and use_all=False, uses all)
        use_all: If True, use ALL examples from the split (ignores n_samples)
    
    Returns:
        Dictionary with detailed metrics
    """
    split_data = splits_data['splits'][split_name]
    
    if use_all:
        # Используем ВСЕ примеры из сплита
        sample_indices = split_data['indices'].tolist() if hasattr(split_data['indices'], 'tolist') else list(split_data['indices'])
        print(f"Evaluating on ALL {len(sample_indices)} examples from {split_name} split...")
    elif n_samples is not None:
        # Используем стратифицированную выборку
        sample_indices = stratified_sample(split_data, n_samples, seed=42)
        print(f"Evaluating on {len(sample_indices)} samples from {split_name} split...")
    else:
        # По умолчанию используем все примеры
        sample_indices = split_data['indices'].tolist() if hasattr(split_data['indices'], 'tolist') else list(split_data['indices'])
        print(f"Evaluating on ALL {len(sample_indices)} examples from {split_name} split (default)...")
    
    results = evaluate_on_samples(
        prompt,
        sample_indices,
        splits_data['preprocessed'],
        verbose=False
    )
    
    return results


def format_results(results: Dict, split_name: str) -> str:
    """Format results as a readable string."""
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"Validation Results ({split_name} split)")
    lines.append(f"{'='*60}")
    
    lines.append(f"\nSamples evaluated: {results.get('total', 0)}")
    lines.append(f"Correct predictions: {results.get('correct', 0)}")
    lines.append(f"\nAccuracy: {results.get('accuracy', 0):.2%}")
    lines.append(f"Mean Absolute Error: {results.get('mae', 0):.4f}")
    
    lines.append(f"\nPer-class Accuracy:")
    for i in range(1, 6):
        key = f"class_{i}_accuracy"
        if key in results:
            lines.append(f"  {i} star: {results[key]:.2%}")
    
    lines.append(f"\n{'='*60}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Validate the best evolved prompt")
    parser.add_argument(
        "--prompt", 
        type=str, 
        default=None,
        help="Path to prompt file. Default: openevolve_output/best/best_program.txt"
    )
    parser.add_argument(
        "--samples", 
        type=int, 
        default=None,
        help="Number of samples to evaluate (default: None = use all examples)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["validation", "test"],
        help="Which split to use for validation (default: test)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Use ALL examples from the split (overrides --samples)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (default: openevolve_output/final_validation.json)"
    )
    
    args = parser.parse_args()
    
    # Determine prompt path
    if args.prompt:
        prompt_path = args.prompt
    else:
        prompt_path = "openevolve_output/best/best_program.txt"
    
    if not os.path.exists(prompt_path):
        print(f"Error: Prompt file not found: {prompt_path}")
        print("Run the evolution first: .\\run_evolution.ps1")
        sys.exit(1)
    
    # Load prompt
    print(f"Loading prompt from: {prompt_path}")
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt = f.read()
    
    print(f"Prompt length: {len(prompt)} characters")
    print(f"\nPrompt preview:\n{'-'*40}")
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
    print(f"{'-'*40}\n")
    
    # Load dataset
    print("Loading dataset...")
    splits_data = get_dataset_splits()
    
    # Run validation
    if args.all:
        print(f"\nRunning validation on ALL examples from {args.split} split...")
    elif args.samples:
        print(f"\nRunning validation on {args.samples} samples from {args.split} split...")
    else:
        print(f"\nRunning validation on ALL examples from {args.split} split (default)...")
    
    results = validate_prompt(
        prompt,
        splits_data,
        split_name=args.split,
        n_samples=args.samples,
        use_all=args.all or args.samples is None
    )
    
    # Calculate combined score
    accuracy = results.get('accuracy', 0.0)
    mae = results.get('mae', 0.0)
    mae_penalty = mae / 4.0
    combined_score = accuracy * (1 - 0.2 * mae_penalty)
    
    # Calculate features
    features = calculate_features(prompt, results)
    
    # Print results
    print(format_results(results, args.split))
    print(f"Combined Score: {combined_score:.4f}")
    print(f"Prompt Length (normalized): {features['prompt_length']:.4f}")
    print(f"Reasoning Strategy: {features['reasoning_strategy']:.4f}")
    
    # Prepare output data
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "prompt_path": prompt_path,
        "prompt_length": len(prompt),
        "split": args.split,
        "samples": results.get('total', 0) if (args.all or args.samples is None) else args.samples,
        "used_all": args.all or args.samples is None,
        "metrics": {
            "accuracy": results.get('accuracy', 0.0),
            "mae": results.get('mae', 0.0),
            "combined_score": combined_score,
            "correct": results.get('correct', 0),
            "total": results.get('total', 0),
            "errors": results.get('errors', 0),
        },
        "per_class_accuracy": {
            f"{i}_star": results.get(f'class_{i}_accuracy', 0.0)
            for i in range(1, 6)
        },
        "features": features
    }
    
    # Save results
    output_path = args.output or "openevolve_output/final_validation.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")
    
    # Also save as markdown report
    md_path = output_path.replace('.json', '.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Final Validation Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Configuration\n\n")
        f.write(f"- **Prompt file**: `{prompt_path}`\n")
        f.write(f"- **Split**: {args.split}\n")
        if args.all or args.samples is None:
            f.write(f"- **Samples**: ALL examples from {args.split} split\n")
        else:
            f.write(f"- **Samples**: {args.samples}\n")
        f.write(f"\n")
        f.write(f"## Results\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| **Accuracy** | {accuracy:.2%} |\n")
        f.write(f"| **MAE** | {mae:.4f} |\n")
        f.write(f"| **Combined Score** | {combined_score:.4f} |\n")
        f.write(f"| Correct | {results.get('correct', 0)} |\n")
        f.write(f"| Total | {results.get('total', 0)} |\n")
        f.write(f"| Errors | {results.get('errors', 0)} |\n\n")
        f.write(f"## Per-class Accuracy\n\n")
        f.write(f"| Star Rating | Accuracy |\n")
        f.write(f"|-------------|----------|\n")
        for i in range(1, 6):
            acc = results.get(f'class_{i}_accuracy', 0.0)
            f.write(f"| {i} ⭐ | {acc:.2%} |\n")
        f.write(f"\n## Prompt\n\n")
        f.write(f"```\n{prompt}\n```\n")
    
    print(f"Markdown report saved to: {md_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


#!/usr/bin/env python3
"""
Script to visualize evolution results from the IFEval experiment.
Generates learning curves, best metrics, and evolution trace analysis.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

# Try to import jsonlines, but fall back to manual JSONL reading if not available
try:
    import jsonlines
except ImportError:
    jsonlines = None

# Part 1: Imports and helper functions

def load_evolution_trace(trace_path: str) -> List[Dict[str, Any]]:
    """Load evolution trace from JSONL file."""
    traces = []
    if not os.path.exists(trace_path):
        print(f"Warning: Evolution trace not found at {trace_path}")
        return traces
    
    try:
        # Try to use jsonlines if available
        try:
            import jsonlines
            with jsonlines.open(trace_path) as reader:
                for trace in reader:
                    traces.append(trace)
        except ImportError:
            # Fallback: read JSONL file manually
            print("jsonlines not available, reading JSONL file manually...")
            with open(trace_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            trace = json.loads(line)
                            traces.append(trace)
                        except json.JSONDecodeError as e:
                            print(f"Warning: Could not parse line: {e}")
                            continue
        
        # Sort traces by iteration to ensure chronological order
        traces.sort(key=lambda x: (x.get("iteration", 0), x.get("timestamp", 0)))
        print(f"Loaded {len(traces)} evolution traces (sorted by iteration)")
    except Exception as e:
        print(f"Error loading evolution trace: {e}")
        import traceback
        traceback.print_exc()
    
    return traces


def load_checkpoint_data(checkpoint_dir: str) -> Dict[str, Any]:
    """Load checkpoint data including best program and metrics."""
    checkpoint_path = Path(checkpoint_dir)
    data = {}
    
    # Load best program info
    best_info_path = checkpoint_path / "best" / "best_program_info.json"
    if best_info_path.exists():
        with open(best_info_path, "r") as f:
            data["best_program"] = json.load(f)
    
    # Load database state from latest checkpoint
    checkpoint_dirs = sorted([d for d in checkpoint_path.glob("checkpoint_*") if d.is_dir()])
    if checkpoint_dirs:
        latest_checkpoint = checkpoint_dirs[-1]
        db_path = latest_checkpoint / "database.json"
        if db_path.exists():
            with open(db_path, "r") as f:
                data["database"] = json.load(f)
    
    return data


def extract_learning_curve(traces: List[Dict[str, Any]]) -> Dict[str, List]:
    """Extract learning curve data from traces."""
    # Ensure traces are sorted by iteration
    sorted_traces = sorted(traces, key=lambda x: (x.get("iteration", 0), x.get("timestamp", 0)))
    
    # Build a dictionary to track scores at each iteration
    iteration_scores = {}
    
    # First pass: collect all scores per iteration
    for trace in sorted_traces:
        iteration = trace.get("iteration", 0)
        child_metrics = trace.get("child_metrics", {})
        score = child_metrics.get("combined_score", 0.0)
        
        if iteration not in iteration_scores:
            iteration_scores[iteration] = []
        iteration_scores[iteration].append(score)
    
    # Second pass: calculate cumulative best score
    all_iterations = sorted(iteration_scores.keys())
    current_best = 0.0
    improvements = []
    
    iterations = []
    best_scores = []  # Cumulative maximum (never decreases)
    best_per_gen = []  # Best score in each generation (can go up and down)
    avg_scores = []
    
    for iteration in all_iterations:
        scores = iteration_scores[iteration]
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        
        # Update cumulative best
        if max_score > current_best:
            current_best = max_score
            improvements.append((iteration, max_score))
        
        # Store data for this iteration
        iterations.append(iteration)
        best_scores.append(current_best)
        best_per_gen.append(max_score)  # Best score of this generation (not cumulative)
        avg_scores.append(avg_score)
    
    return {
        "iterations": iterations,
        "best_scores": best_scores,
        "best_per_gen": best_per_gen,
        "avg_scores": avg_scores,
        "improvements": improvements
    }


def analyze_inspirations(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze which inspirations led to improvements."""
    inspiration_stats = defaultdict(int)
    successful_inspirations = []
    
    for trace in traces:
        improvement = trace.get("improvement_delta", {})
        if improvement.get("combined_score", 0) > 0:
            # This was a successful evolution
            metadata = trace.get("metadata", {})
            inspirations = metadata.get("inspirations", [])
            for insp_id in inspirations:
                inspiration_stats[insp_id] += 1
            successful_inspirations.append({
                "iteration": trace.get("iteration"),
                "improvement": improvement.get("combined_score", 0),
                "inspirations": inspirations
            })
    
    return {
        "stats": dict(inspiration_stats),
        "successful": successful_inspirations
    }


def plot_learning_curve(curve_data: Dict[str, List], output_path: str):
    """Plot learning curve showing best and average scores over iterations."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    iterations = curve_data["iterations"]
    best_scores = curve_data["best_scores"]
    best_per_gen = curve_data["best_per_gen"]
    avg_scores = curve_data["avg_scores"]
    
    # Plot best score as a solid line (cumulative maximum, never decreases)
    ax.plot(iterations, best_scores, label="Best Score (Cumulative Max)", 
            linewidth=2.5, color="green", alpha=0.9, marker='o', markersize=5)
    
    # Plot best score per generation (can go up and down)
    ax.plot(iterations, best_per_gen, label="Best Score per Generation", 
            linewidth=2.0, color="orange", alpha=0.8, marker='^', markersize=4, linestyle='-')
    
    # Average scores as a dashed line (can go up and down)
    ax.plot(iterations, avg_scores, label="Average Score", linewidth=1.5, 
            alpha=0.6, color="blue", linestyle='--', marker='s', markersize=3)
    
    # Mark improvement points
    improvements = curve_data["improvements"]
    if improvements:
        imp_iterations, imp_scores = zip(*improvements)
        ax.scatter(imp_iterations, imp_scores, color="red", s=80, zorder=5, 
                  label="New Best", marker='*', edgecolors='darkred', linewidths=1)
    
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Score (Accuracy)", fontsize=12)
    ax.set_title("Learning Curve: IFEval Prompt Evolution", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    # Set x-axis to show all iterations
    if iterations:
        ax.set_xlim(min(iterations) - 1, max(iterations) + 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved learning curve to {output_path}")
    plt.close()


def plot_metrics_over_time(traces: List[Dict[str, Any]], output_path: str):
    """Plot various metrics over time."""
    # Ensure traces are sorted by iteration
    sorted_traces = sorted(traces, key=lambda x: (x.get("iteration", 0), x.get("timestamp", 0)))
    
    # Group data by iteration (in case multiple traces per iteration)
    iteration_data = {}
    for trace in sorted_traces:
        iteration = trace.get("iteration", 0)
        child_metrics = trace.get("child_metrics", {})
        
        if iteration not in iteration_data:
            iteration_data[iteration] = {
                "scores": [],
                "prompt_lengths": [],
                "reasoning_scores": []
            }
        
        iteration_data[iteration]["scores"].append(child_metrics.get("combined_score", 0.0))
        iteration_data[iteration]["prompt_lengths"].append(child_metrics.get("prompt_length", 0))
        iteration_data[iteration]["reasoning_scores"].append(child_metrics.get("reasoning_strategy", 0.0))
    
    # Extract data sorted by iteration
    iterations = sorted(iteration_data.keys())
    scores = [sum(iteration_data[it]["scores"]) / len(iteration_data[it]["scores"]) for it in iterations]
    prompt_lengths = [sum(iteration_data[it]["prompt_lengths"]) / len(iteration_data[it]["prompt_lengths"]) for it in iterations]
    reasoning_scores = [sum(iteration_data[it]["reasoning_scores"]) / len(iteration_data[it]["reasoning_scores"]) for it in iterations]
    
    # Calculate moving average for smoother lines (window=3)
    def moving_average(data, window=3):
        """Calculate moving average with given window size."""
        if len(data) < window:
            return data
        result = []
        for i in range(len(data)):
            start = max(0, i - window // 2)
            end = min(len(data), i + window // 2 + 1)
            result.append(sum(data[start:end]) / (end - start))
        return result
    
    scores_smooth = moving_average(scores, window=3)
    prompt_lengths_smooth = moving_average(prompt_lengths, window=3)
    reasoning_scores_smooth = moving_average(reasoning_scores, window=3)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Score over time
    axes[0].plot(iterations, scores, 'o-', markersize=3, linewidth=1, alpha=0.4, color="blue", label="Raw")
    axes[0].plot(iterations, scores_smooth, linewidth=2, alpha=0.8, color="blue", label="Smoothed")
    axes[0].set_ylabel("Combined Score", fontsize=10)
    axes[0].set_title("Metrics Evolution Over Time", fontsize=12, fontweight="bold")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # Prompt length over time
    axes[1].plot(iterations, prompt_lengths, 'o-', markersize=3, linewidth=1, alpha=0.4, color="green", label="Raw")
    axes[1].plot(iterations, prompt_lengths_smooth, linewidth=2, alpha=0.8, color="green", label="Smoothed")
    axes[1].set_ylabel("Prompt Length", fontsize=10)
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    # Reasoning strategy over time
    axes[2].plot(iterations, reasoning_scores, 'o-', markersize=3, linewidth=1, alpha=0.4, color="orange", label="Raw")
    axes[2].plot(iterations, reasoning_scores_smooth, linewidth=2, alpha=0.8, color="orange", label="Smoothed")
    axes[2].set_xlabel("Iteration", fontsize=10)
    axes[2].set_ylabel("Reasoning Strategy", fontsize=10)
    axes[2].legend(loc="upper left", fontsize=8)
    axes[2].grid(True, alpha=0.3)
    
    # Set x-axis limits
    if iterations:
        for ax in axes:
            ax.set_xlim(min(iterations) - 1, max(iterations) + 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved metrics plot to {output_path}")
    plt.close()


def generate_report(traces: List[Dict[str, Any]], checkpoint_data: Dict[str, Any], output_path: str):
    """Generate a text report with key statistics."""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("IFEVAL PROMPT EVOLUTION REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Basic statistics
    report_lines.append(f"Total Evolution Steps: {len(traces)}")
    
    if traces:
        final_trace = traces[-1]
        final_score = final_trace.get("child_metrics", {}).get("combined_score", 0.0)
        initial_score = traces[0].get("parent_metrics", {}).get("combined_score", 0.0)
        improvement = final_score - initial_score
        
        report_lines.append(f"Initial Score: {initial_score:.4f}")
        report_lines.append(f"Final Score: {final_score:.4f}")
        report_lines.append(f"Total Improvement: {improvement:+.4f} ({improvement/initial_score*100:+.2f}%)")
        report_lines.append("")
    
    # Best program info
    if "best_program" in checkpoint_data:
        best = checkpoint_data["best_program"]
        report_lines.append("BEST PROGRAM:")
        report_lines.append("-" * 80)
        metrics = best.get("metrics", {})
        for key, value in metrics.items():
            if isinstance(value, float):
                report_lines.append(f"  {key}: {value:.4f}")
            else:
                report_lines.append(f"  {key}: {value}")
        report_lines.append("")
    
    # Inspiration analysis
    inspiration_analysis = analyze_inspirations(traces)
    report_lines.append("TOP INSPIRATIONS (by success count):")
    report_lines.append("-" * 80)
    if inspiration_analysis["stats"]:
        sorted_insp = sorted(inspiration_analysis["stats"].items(), key=lambda x: x[1], reverse=True)
        for insp_id, count in sorted_insp[:10]:
            report_lines.append(f"  {insp_id[:8]}...: {count} successful evolutions")
    else:
        report_lines.append("  (No inspiration metadata recorded in evolution trace)")
    report_lines.append("")
    
    # Improvement timeline
    improvements = [t for t in traces if t.get("improvement_delta", {}).get("combined_score", 0) > 0]
    if improvements:
        report_lines.append(f"Total Improvements: {len(improvements)}")
        report_lines.append("Major Improvements:")
        report_lines.append("-" * 80)
        for imp in sorted(improvements, key=lambda x: x.get("improvement_delta", {}).get("combined_score", 0), reverse=True)[:5]:
            iter_num = imp.get("iteration", 0)
            delta = imp.get("improvement_delta", {}).get("combined_score", 0)
            report_lines.append(f"  Iteration {iter_num}: +{delta:.4f}")
        report_lines.append("")
    
    report_lines.append("=" * 80)
    
    with open(output_path, "w") as f:
        f.write("\n".join(report_lines))
    
    print(f"Saved report to {output_path}")


def main():
    """Main function to generate all visualizations and reports."""
    # Paths
    output_dir = Path("openevolve_output")
    trace_path = output_dir / "evolution_trace.jsonl"
    
    if not trace_path.exists():
        print(f"Error: Evolution trace not found at {trace_path}")
        print("Please run the evolution first using run_evolution.ps1")
        sys.exit(1)
    
    print("Loading evolution data...")
    traces = load_evolution_trace(str(trace_path))
    checkpoint_data = load_checkpoint_data(str(output_dir))
    
    if not traces:
        print("No traces found. Cannot generate visualizations.")
        sys.exit(1)
    
    # Create output directory for visualizations
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    print("\nGenerating visualizations...")
    
    # Learning curve
    curve_data = extract_learning_curve(traces)
    plot_learning_curve(curve_data, str(viz_dir / "learning_curve.png"))
    
    # Metrics over time
    plot_metrics_over_time(traces, str(viz_dir / "metrics_evolution.png"))
    
    # Report
    generate_report(traces, checkpoint_data, str(viz_dir / "evolution_report.txt"))
    
    print("\nVisualization complete!")
    print(f"Results saved to: {viz_dir}")


if __name__ == "__main__":
    main()


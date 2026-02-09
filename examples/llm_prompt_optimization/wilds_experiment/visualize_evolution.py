#!/usr/bin/env python3
"""
Script to visualize evolution results from the WILDS Amazon experiment.
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
        accuracy = child_metrics.get("accuracy", 0.0)
        val_accuracy = child_metrics.get("val_accuracy", None)  # May be None
        
        if iteration not in iteration_scores:
            iteration_scores[iteration] = {
                'scores': [],
                'accuracies': [],
                'val_accuracies': [],
            }
        iteration_scores[iteration]['scores'].append(score)
        iteration_scores[iteration]['accuracies'].append(accuracy)
        if val_accuracy is not None:
            iteration_scores[iteration]['val_accuracies'].append(val_accuracy)
    
    # Second pass: calculate cumulative best score
    all_iterations = sorted(iteration_scores.keys())
    current_best = 0.0
    current_best_accuracy = 0.0
    current_best_val_accuracy = None
    improvements = []
    
    iterations = []
    best_scores = []  # Cumulative maximum (never decreases)
    best_per_gen = []  # Best score in each generation (can go up and down)
    avg_scores = []
    best_accuracies = []
    avg_accuracies = []
    val_accuracy_points = []  # (iteration, val_accuracy) tuples - only when available
    best_val_accuracies = []  # Cumulative best val_accuracy
    
    for iteration in all_iterations:
        scores = iteration_scores[iteration]['scores']
        accuracies = iteration_scores[iteration]['accuracies']
        val_accs = iteration_scores[iteration]['val_accuracies']
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        max_accuracy = max(accuracies)
        avg_accuracy = sum(accuracies) / len(accuracies)
        
        # Update cumulative best
        if max_score > current_best:
            current_best = max_score
            current_best_accuracy = max_accuracy
            improvements.append((iteration, max_score, max_accuracy))
        
        # Track val_accuracy when available
        # NOTE: По новой архитектуре val_accuracy должна быть только раз в 10 итераций
        if val_accs:
            max_val_acc = max(val_accs)
            validation_interval = 10  # Периодичность валидации
            
            # Всегда фильтруем по периодичности: показываем только iteration % 10 == 0
            # Исключение: первая точка (iteration 0 или 1), если она не на границе
            if len(val_accuracy_points) == 0:
                # Первая точка - добавляем всегда
                val_accuracy_points.append((iteration, max_val_acc))
            elif iteration % validation_interval == 0:
                # Периодическая валидация - добавляем
                val_accuracy_points.append((iteration, max_val_acc))
            # Иначе пропускаем (не периодическая валидация)
            
            if current_best_val_accuracy is None or max_val_acc > current_best_val_accuracy:
                current_best_val_accuracy = max_val_acc
        
        # Store data for this iteration
        iterations.append(iteration)
        best_scores.append(current_best)
        best_per_gen.append(max_score)  # Best score of this generation (not cumulative)
        avg_scores.append(avg_score)
        best_accuracies.append(current_best_accuracy)
        avg_accuracies.append(avg_accuracy)
        best_val_accuracies.append(current_best_val_accuracy)  # May be None
    
    return {
        "iterations": iterations,
        "best_scores": best_scores,
        "best_per_gen": best_per_gen,
        "avg_scores": avg_scores,
        "best_accuracies": best_accuracies,
        "avg_accuracies": avg_accuracies,
        "improvements": improvements,
        "val_accuracy_points": val_accuracy_points,  # Points where val was computed
        "best_val_accuracies": best_val_accuracies,  # Cumulative best (with Nones)
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
    # Check if we have validation accuracy data
    val_points = curve_data.get("val_accuracy_points", [])
    has_val_data = len(val_points) > 0
    
    if has_val_data:
        # Two subplots: scores and validation accuracy
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])
    else:
        fig, ax1 = plt.subplots(figsize=(12, 6))
    
    iterations = curve_data["iterations"]
    best_scores = curve_data["best_scores"]
    best_per_gen = curve_data["best_per_gen"]
    avg_scores = curve_data["avg_scores"]
    
    # Plot 1: Best score as a solid line (cumulative maximum, never decreases)
    ax1.plot(iterations, best_scores, label="Best Score (Cumulative Max)", 
            linewidth=2.5, color="green", alpha=0.9, marker='o', markersize=5)
    
    # Plot best score per generation (can go up and down)
    ax1.plot(iterations, best_per_gen, label="Best Score per Generation", 
            linewidth=2.0, color="orange", alpha=0.8, marker='^', markersize=4, linestyle='-')
    
    # Average scores as a dashed line (can go up and down)
    ax1.plot(iterations, avg_scores, label="Average Score", linewidth=1.5, 
            alpha=0.6, color="blue", linestyle='--', marker='s', markersize=3)
    
    # Mark improvement points
    improvements = curve_data["improvements"]
    if improvements:
        imp_iterations = [i[0] for i in improvements]
        imp_scores = [i[1] for i in improvements]
        ax1.scatter(imp_iterations, imp_scores, color="red", s=80, zorder=5, 
                  label="New Best", marker='*', edgecolors='darkred', linewidths=1)
    
    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel("Score", fontsize=12)
    ax1.set_title("Learning Curve: WILDS Amazon (Office Products) Prompt Evolution", fontsize=14, fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # Set x-axis to show all iterations
    if iterations:
        ax1.set_xlim(min(iterations) - 1, max(iterations) + 1)
    
    # Plot 2: Validation Accuracy (only when available)
    if has_val_data:
        val_iters = [p[0] for p in val_points]
        val_accs = [p[1] for p in val_points]
        
        # Plot validation accuracy points
        ax2.scatter(val_iters, val_accs, color="purple", s=100, zorder=5,
                   label="Validation Accuracy", marker='D', edgecolors='darkviolet', linewidths=1)
        
        # Connect with line
        ax2.plot(val_iters, val_accs, color="purple", linewidth=1.5, alpha=0.5, linestyle='--')
        
        # Add cumulative best validation line
        best_val = curve_data.get("best_val_accuracies", [])
        # Filter out None values for plotting
        valid_best_val = [(it, bv) for it, bv in zip(iterations, best_val) if bv is not None]
        if valid_best_val:
            bv_iters, bv_vals = zip(*valid_best_val)
            ax2.plot(bv_iters, bv_vals, color="magenta", linewidth=2, alpha=0.8,
                    label="Best Val Accuracy (Cumulative)", linestyle='-')
        
        ax2.set_xlabel("Iteration", fontsize=12)
        ax2.set_ylabel("Validation Accuracy", fontsize=12)
        # Определяем интервал валидации по распределению точек
        if len(val_iters) > 1:
            intervals = [val_iters[i+1] - val_iters[i] for i in range(len(val_iters)-1)]
            avg_interval = sum(intervals) / len(intervals) if intervals else 10
            if 9 <= avg_interval <= 11:
                ax2.set_title(f"Validation Accuracy (computed every ~{int(avg_interval)} iterations)", fontsize=12)
            else:
                ax2.set_title("Validation Accuracy (periodic validation)", fontsize=12)
        else:
            ax2.set_title("Validation Accuracy (periodic validation)", fontsize=12)
        ax2.legend(loc="lower right")
        ax2.grid(True, alpha=0.3)
        
        if iterations:
            ax2.set_xlim(min(iterations) - 1, max(iterations) + 1)
        ax2.set_ylim(0, 1)
    
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
    val_accuracy_points = []  # (iteration, val_accuracy) for sparse plotting
    
    for trace in sorted_traces:
        iteration = trace.get("iteration", 0)
        child_metrics = trace.get("child_metrics", {})
        
        if iteration not in iteration_data:
            iteration_data[iteration] = {
                "scores": [],
                "accuracies": [],
                "prompt_lengths": [],
                "sentiment_vocabulary_richness": [],
                "domain_focus": [],
                "mean_kappa": [],
                "maes": [],
                "r_global": [],
                "r_worst": []
            }
        
        iteration_data[iteration]["scores"].append(child_metrics.get("combined_score", 0.0))
        iteration_data[iteration]["accuracies"].append(child_metrics.get("accuracy", 0.0))
        iteration_data[iteration]["prompt_lengths"].append(child_metrics.get("prompt_length", 0))
        # New feature dimensions (with fallback to old metrics for backward compatibility)
        sentiment_richness = child_metrics.get("sentiment_vocabulary_richness")
        if sentiment_richness is None:
            # Fallback to criteria_explicitness for old experiments
            sentiment_richness = child_metrics.get("criteria_explicitness", 0.0)
        iteration_data[iteration]["sentiment_vocabulary_richness"].append(sentiment_richness)
        iteration_data[iteration]["domain_focus"].append(child_metrics.get("domain_focus", 0.0))
        # Ensemble: Cohen's Kappa (top-level or metrics); fallback to (1 - disagreement_rate) for old traces
        kappa_val = child_metrics.get("mean_kappa")
        if kappa_val is None:
            kappa_val = child_metrics.get("metrics", {}).get("mean_kappa")
        if kappa_val is None and child_metrics.get("disagreement_rate") is not None:
            kappa_val = 1.0 - child_metrics.get("disagreement_rate")  # approximate for old traces
        iteration_data[iteration]["mean_kappa"].append(float(kappa_val) if kappa_val is not None else 0.0)
        iteration_data[iteration]["maes"].append(child_metrics.get("mae", 0.0))
        
        # Extract R_global and R_worst from metrics dict
        metrics_dict = child_metrics.get("metrics", {})
        iteration_data[iteration]["r_global"].append(metrics_dict.get("R_global", 0.0))
        iteration_data[iteration]["r_worst"].append(metrics_dict.get("R_worst", 0.0))
        
        # Collect val_accuracy if present
        # Фильтруем: показываем только периодические валидации (раз в 10 итераций)
        val_acc = child_metrics.get("val_accuracy")
        if val_acc is not None:
            validation_interval = 10
            
            # Всегда фильтруем по периодичности: показываем только iteration % 10 == 0
            # Исключение: первая точка, если она не на границе
            if len(val_accuracy_points) == 0:
                # Первая точка - добавляем всегда
                val_accuracy_points.append((iteration, val_acc))
            elif iteration % validation_interval == 0:
                # Периодическая валидация - добавляем
                val_accuracy_points.append((iteration, val_acc))
            # Иначе пропускаем (не периодическая валидация)
    
    # Extract data sorted by iteration
    iterations = sorted(iteration_data.keys())
    scores = [sum(iteration_data[it]["scores"]) / len(iteration_data[it]["scores"]) for it in iterations]
    accuracies = [sum(iteration_data[it]["accuracies"]) / len(iteration_data[it]["accuracies"]) for it in iterations]
    prompt_lengths = [sum(iteration_data[it]["prompt_lengths"]) / len(iteration_data[it]["prompt_lengths"]) for it in iterations]
    sentiment_vocabulary_richness = [sum(iteration_data[it]["sentiment_vocabulary_richness"]) / len(iteration_data[it]["sentiment_vocabulary_richness"]) for it in iterations]
    domain_focus = [sum(iteration_data[it]["domain_focus"]) / len(iteration_data[it]["domain_focus"]) for it in iterations]
    mean_kappa = [sum(iteration_data[it]["mean_kappa"]) / len(iteration_data[it]["mean_kappa"]) for it in iterations]
    maes = [sum(iteration_data[it]["maes"]) / len(iteration_data[it]["maes"]) for it in iterations]
    r_global = [sum(iteration_data[it]["r_global"]) / len(iteration_data[it]["r_global"]) for it in iterations]
    r_worst = [sum(iteration_data[it]["r_worst"]) / len(iteration_data[it]["r_worst"]) for it in iterations]
    
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
    accuracies_smooth = moving_average(accuracies, window=3)
    prompt_lengths_smooth = moving_average(prompt_lengths, window=3)
    sentiment_vocabulary_richness_smooth = moving_average(sentiment_vocabulary_richness, window=3)
    domain_focus_smooth = moving_average(domain_focus, window=3)
    mean_kappa_smooth = moving_average(mean_kappa, window=3)
    r_global_smooth = moving_average(r_global, window=3)
    r_worst_smooth = moving_average(r_worst, window=3)
    
    # Determine if this is an ensemble experiment (has mean_kappa data)
    has_kappa = any(mk != 0.0 for mk in mean_kappa)
    
    # Cumulative best (best-so-far) for combined_score, r_global, kappa/domain_focus
    def cumulative_max(values: List[float]) -> List[float]:
        out = []
        best = float("-inf") if values else 0.0
        for v in values:
            best = max(best, v)
            out.append(best)
        return out
    best_combined_score = cumulative_max(scores)
    best_r_global = cumulative_max(r_global)
    best_kappa_or_focus = cumulative_max(mean_kappa if has_kappa else domain_focus)
    
    # Metrics of the solution that is best by combined_score (at each iteration: all components of that best solution)
    kappa_values = mean_kappa if has_kappa else domain_focus
    kappa_of_best_by_score: List[float] = []
    r_global_of_best: List[float] = []
    r_worst_of_best: List[float] = []
    mae_component_of_best: List[float] = []  # 1 - MAE/4 (higher = better, as in combined_score)
    best_score_so_far = float("-inf")
    idx_of_best = 0
    for i, s in enumerate(scores):
        if s > best_score_so_far:
            best_score_so_far = s
            idx_of_best = i
        kappa_of_best_by_score.append(kappa_values[idx_of_best])
        r_global_of_best.append(r_global[idx_of_best])
        r_worst_of_best.append(r_worst[idx_of_best])
        mae_component_of_best.append(1.0 - maes[idx_of_best] / 4.0)
    
    # Determine number of subplots: combined_score, R_global, R_worst, feature dimensions (4-5 base plots)
    # Plus validation accuracy if available
    has_val_data = len(val_accuracy_points) > 0
    # Base: combined_score, R_global, R_worst, sentiment_vocabulary_richness, domain_focus (or mean_kappa for ensemble)
    n_base_plots = 5  # combined_score, R_global, R_worst, sentiment_vocabulary_richness, domain_focus/mean_kappa
    n_plots = n_base_plots + (1 if has_val_data else 0)  # + optional validation
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots))
    fig.suptitle('WILDS Amazon (Home and Kitchen) - Metrics Evolution', fontsize=14, fontweight='bold')
    
    plot_idx = 0
    
    # Combined Score over time
    axes[plot_idx].plot(iterations, scores, 'o-', markersize=3, linewidth=1, alpha=0.4, color="blue", label="Raw")
    axes[plot_idx].plot(iterations, scores_smooth, linewidth=2, alpha=0.8, color="blue", label="Smoothed")
    axes[plot_idx].plot(iterations, best_combined_score, linewidth=2, alpha=0.9, color="darkblue", linestyle="--", label="Best (cumulative)")
    axes[plot_idx].set_ylabel("Combined Score", fontsize=10)
    axes[plot_idx].legend(loc="upper left", fontsize=8)
    axes[plot_idx].grid(True, alpha=0.3)
    plot_idx += 1
    
    # R_global over time
    axes[plot_idx].plot(iterations, r_global, 'o-', markersize=3, linewidth=1, alpha=0.4, color="green", label="Raw")
    axes[plot_idx].plot(iterations, r_global_smooth, linewidth=2, alpha=0.8, color="green", label="Smoothed")
    axes[plot_idx].plot(iterations, best_r_global, linewidth=2, alpha=0.9, color="darkgreen", linestyle="--", label="Best (cumulative)")
    axes[plot_idx].set_ylabel("R_global", fontsize=10)
    axes[plot_idx].legend(loc="upper left", fontsize=8)
    axes[plot_idx].grid(True, alpha=0.3)
    axes[plot_idx].set_ylim(0, 1)
    plot_idx += 1
    
    # R_worst over time
    axes[plot_idx].plot(iterations, r_worst, 'o-', markersize=3, linewidth=1, alpha=0.4, color="red", label="Raw")
    axes[plot_idx].plot(iterations, r_worst_smooth, linewidth=2, alpha=0.8, color="red", label="Smoothed")
    axes[plot_idx].set_ylabel("R_worst", fontsize=10)
    axes[plot_idx].legend(loc="upper left", fontsize=8)
    axes[plot_idx].grid(True, alpha=0.3)
    axes[plot_idx].set_ylim(0, 1)
    plot_idx += 1
    
    # Sentiment Vocabulary Richness over time
    axes[plot_idx].plot(iterations, sentiment_vocabulary_richness, 'o-', markersize=3, linewidth=1, alpha=0.4, color="orange", label="Raw")
    axes[plot_idx].plot(iterations, sentiment_vocabulary_richness_smooth, linewidth=2, alpha=0.8, color="orange", label="Smoothed")
    axes[plot_idx].set_ylabel("Sentiment Vocabulary Richness", fontsize=10)
    axes[plot_idx].legend(loc="upper left", fontsize=8)
    axes[plot_idx].grid(True, alpha=0.3)
    axes[plot_idx].set_ylim(0, 1)
    plot_idx += 1
    
    # Domain Focus or Cohen's Kappa (depending on experiment type)
    if has_kappa:
        # Ensemble experiment: show mean_kappa (Cohen's Kappa)
        axes[plot_idx].plot(iterations, mean_kappa, 'o-', markersize=3, linewidth=1, alpha=0.4, color="teal", label="Raw")
        axes[plot_idx].plot(iterations, mean_kappa_smooth, linewidth=2, alpha=0.8, color="teal", label="Smoothed")
        axes[plot_idx].plot(iterations, best_kappa_or_focus, linewidth=2, alpha=0.9, color="darkcyan", linestyle="--", label="Best (cumulative)")
        axes[plot_idx].set_ylabel("Cohen's Kappa (κ)", fontsize=10)
    else:
        # Single model experiment: show domain_focus
        axes[plot_idx].plot(iterations, domain_focus, 'o-', markersize=3, linewidth=1, alpha=0.4, color="purple", label="Raw")
        axes[plot_idx].plot(iterations, domain_focus_smooth, linewidth=2, alpha=0.8, color="purple", label="Smoothed")
        axes[plot_idx].plot(iterations, best_kappa_or_focus, linewidth=2, alpha=0.9, color="indigo", linestyle="--", label="Best (cumulative)")
        axes[plot_idx].set_ylabel("Domain Focus", fontsize=10)
    axes[plot_idx].legend(loc="upper left", fontsize=8)
    axes[plot_idx].grid(True, alpha=0.3)
    axes[plot_idx].set_ylim(0, 1)
    plot_idx += 1
    
    # Validation Accuracy (separate subplot if data exists) - placed at the end
    if has_val_data:
        val_iters = [p[0] for p in val_accuracy_points]
        val_accs = [p[1] for p in val_accuracy_points]
        
        axes[plot_idx].scatter(val_iters, val_accs, color="purple", s=80, zorder=5,
                   label="Val Accuracy", marker='D', edgecolors='darkviolet', linewidths=1)
        axes[plot_idx].plot(val_iters, val_accs, color="purple", linewidth=1.5, alpha=0.5, linestyle='--')
        
        # Cumulative best val accuracy
        best_val = 0.0
        best_val_line = []
        for va in val_accs:
            if va > best_val:
                best_val = va
            best_val_line.append(best_val)
        axes[plot_idx].plot(val_iters, best_val_line, color="magenta", linewidth=2, alpha=0.8,
                label="Best Val (Cumulative)", linestyle='-')
        
        # Определяем интервал для заголовка
        if len(val_iters) > 1:
            intervals = [val_iters[i+1] - val_iters[i] for i in range(len(val_iters)-1)]
            avg_interval = sum(intervals) / len(intervals) if intervals else 10
            if 9 <= avg_interval <= 11:
                title = f"Validation Accuracy (every ~{int(avg_interval)} iterations)"
            else:
                title = "Validation Accuracy (periodic)"
        else:
            title = "Validation Accuracy (periodic)"
        
        axes[plot_idx].set_xlabel("Iteration", fontsize=10)
        axes[plot_idx].set_ylabel("Validation Accuracy", fontsize=10)
        axes[plot_idx].set_title(title, fontsize=10)
        axes[plot_idx].set_ylim(0, 1)
        axes[plot_idx].legend(loc="upper left", fontsize=8)
        axes[plot_idx].grid(True, alpha=0.3)
    else:
        # If no validation data, set xlabel on the last feature dimension plot
        axes[plot_idx - 1].set_xlabel("Iteration", fontsize=10)
    
    # Set x-axis limits
    if iterations:
        for ax in axes:
            ax.set_xlim(min(iterations) - 1, max(iterations) + 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved metrics plot to {output_path}")
    plt.close()
    
    # Second figure: Best solutions over time (combined_score, r_global, kappa/domain_focus)
    best_output_path = str(Path(output_path).parent / "best_solutions_over_time.png")
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))
    fig2.suptitle("WILDS Amazon (Home and Kitchen) - Best Solutions Over Time", fontsize=12, fontweight="bold")
    
    axes2[0].plot(iterations, best_combined_score, linewidth=2, color="darkblue")
    axes2[0].set_ylabel("Best Combined Score", fontsize=10)
    axes2[0].set_xlabel("Iteration", fontsize=10)
    axes2[0].grid(True, alpha=0.3)
    axes2[0].set_title("Best by combined_score", fontsize=10)
    
    axes2[1].plot(iterations, best_r_global, linewidth=2, color="darkgreen")
    axes2[1].set_ylabel("Best R_global", fontsize=10)
    axes2[1].set_xlabel("Iteration", fontsize=10)
    axes2[1].grid(True, alpha=0.3)
    axes2[1].set_ylim(0, 1)
    axes2[1].set_title("Best by R_global", fontsize=10)
    
    ylabel_best = "Best Cohen's Kappa (κ)" if has_kappa else "Best Domain Focus"
    title_best = "Best by κ" if has_kappa else "Best by domain_focus"
    axes2[2].plot(iterations, best_kappa_or_focus, linewidth=2, color="darkcyan" if has_kappa else "indigo")
    axes2[2].set_ylabel(ylabel_best, fontsize=10)
    axes2[2].set_xlabel("Iteration", fontsize=10)
    axes2[2].grid(True, alpha=0.3)
    axes2[2].set_ylim(0, 1)
    axes2[2].set_title(title_best, fontsize=10)
    
    if iterations:
        for ax in axes2:
            ax.set_xlim(min(iterations) - 1, max(iterations) + 1)
    plt.tight_layout()
    plt.savefig(best_output_path, dpi=300, bbox_inches="tight")
    print(f"Saved best-solutions plot to {best_output_path}")
    plt.close()
    
    # Third figure: one graph — all components of combined_score for the solution best by combined_score
    score_kappa_path = str(Path(output_path).parent / "best_score_and_components.png")
    fig3, ax3 = plt.subplots(1, 1, figsize=(11, 5))
    ax3.set_title(
        "WILDS Amazon (Home and Kitchen) — Components of combined_score for the best solution (by score)",
        fontsize=11,
        fontweight="bold",
    )
    ax3.set_xlabel("Iteration", fontsize=10)
    ax3.set_ylabel("Value (0–1, higher is better)", fontsize=10)
    ax3.plot(iterations, best_combined_score, linewidth=2.5, color="darkblue", label="combined_score (best cumulative)")
    ax3.plot(iterations, r_global_of_best, linewidth=1.8, color="green", linestyle="-", alpha=0.9, label="R_global (of best)")
    ax3.plot(iterations, r_worst_of_best, linewidth=1.8, color="red", linestyle="-", alpha=0.9, label="R_worst (of best)")
    ax3.plot(iterations, mae_component_of_best, linewidth=1.8, color="orange", linestyle="-", alpha=0.9, label="1 − MAE/4 (of best)")
    if has_kappa:
        ax3.plot(iterations, kappa_of_best_by_score, linewidth=1.8, color="darkcyan", linestyle="--", alpha=0.9, label="κ (of best)")
    else:
        ax3.plot(iterations, kappa_of_best_by_score, linewidth=1.8, color="indigo", linestyle="--", alpha=0.9, label="Domain focus (of best)")
    ax3.set_ylim(0.2, 0.8)
    ax3.legend(loc="lower right", fontsize=8)
    ax3.grid(True, alpha=0.3)
    if iterations:
        ax3.set_xlim(min(iterations) - 1, max(iterations) + 1)
    plt.tight_layout()
    plt.savefig(score_kappa_path, dpi=300, bbox_inches="tight")
    print(f"Saved best-score-and-components plot to {score_kappa_path}")
    plt.close()


def generate_report(traces: List[Dict[str, Any]], checkpoint_data: Dict[str, Any], output_path: str):
    """Generate a text report with key statistics."""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("WILDS AMAZON (HOME AND KITCHEN) - EVOLUTION REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Extract model name from traces (if available)
    model_name = "unknown"
    if traces:
        first_trace = traces[0]
        # Try parent_metrics.train.metrics.model_name
        train_metrics = first_trace.get("parent_metrics", {}).get("train", {}).get("metrics", {})
        if "model_name" in train_metrics:
            model_name = train_metrics["model_name"]
        else:
            # Fallback: try child_metrics.train.metrics.model_name
            child_train_metrics = first_trace.get("child_metrics", {}).get("train", {}).get("metrics", {})
            if "model_name" in child_train_metrics:
                model_name = child_train_metrics["model_name"]
            else:
                # Fallback: try child_metrics.metrics.model_name (top-level)
                child_metrics = first_trace.get("child_metrics", {}).get("metrics", {})
                if "model_name" in child_metrics:
                    model_name = child_metrics["model_name"]
    
    # Basic statistics
    report_lines.append(f"Model: {model_name}")
    report_lines.append(f"Total Evolution Steps: {len(traces)}")
    
    # Collect validation accuracy data
    val_accuracy_points = []
    for trace in traces:
        val_acc = trace.get("child_metrics", {}).get("val_accuracy")
        if val_acc is not None:
            val_accuracy_points.append((trace.get("iteration", 0), val_acc))
    
    if traces:
        # Find initial and final scores
        sorted_traces = sorted(traces, key=lambda x: x.get("iteration", 0))
        
        # Get best score overall
        best_score = 0.0
        best_accuracy = 0.0
        best_val_accuracy = None
        best_iteration = 0
        for trace in traces:
            score = trace.get("child_metrics", {}).get("combined_score", 0.0)
            if score > best_score:
                best_score = score
                best_accuracy = trace.get("child_metrics", {}).get("accuracy", 0.0)
                best_val_accuracy = trace.get("child_metrics", {}).get("val_accuracy")
                best_iteration = trace.get("iteration", 0)
        
        initial_score = sorted_traces[0].get("child_metrics", {}).get("combined_score", 0.0)
        improvement = best_score - initial_score
        
        report_lines.append(f"Initial Score: {initial_score:.4f}")
        report_lines.append(f"Best Score: {best_score:.4f} (Iteration {best_iteration})")
        report_lines.append(f"Best Train Accuracy: {best_accuracy:.2%}")
        
        # Best validation accuracy overall
        if val_accuracy_points:
            overall_best_val = max(va for _, va in val_accuracy_points)
            report_lines.append(f"Best Validation Accuracy: {overall_best_val:.2%}")
            report_lines.append(f"Validation runs: {len(val_accuracy_points)}")
        
        if initial_score > 0:
            report_lines.append(f"Total Improvement: {improvement:+.4f} ({improvement/initial_score*100:+.2f}%)")
        else:
            report_lines.append(f"Total Improvement: {improvement:+.4f}")
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
    
    # Validation accuracy history
    if val_accuracy_points:
        report_lines.append("VALIDATION ACCURACY HISTORY:")
        report_lines.append("-" * 80)
        for iteration, val_acc in sorted(val_accuracy_points, key=lambda x: x[0]):
            report_lines.append(f"  Iteration {iteration}: {val_acc:.2%}")
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
            accuracy = imp.get("child_metrics", {}).get("accuracy", 0)
            val_acc = imp.get("child_metrics", {}).get("val_accuracy")
            val_str = f", val: {val_acc:.2%}" if val_acc is not None else ""
            report_lines.append(f"  Iteration {iter_num}: +{delta:.4f} (train: {accuracy:.2%}{val_str})")
        report_lines.append("")
    
    report_lines.append("=" * 80)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    print(f"Saved report to {output_path}")


def generate_summary(traces: List[Dict], curve_data: Dict, output_dir: str):
    """Generate a summary file for the evolution (legacy format)."""
    report_path = os.path.join(output_dir, "evolution_summary.txt")
    
    # Collect validation accuracy data
    val_accuracy_points = curve_data.get("val_accuracy_points", [])
    
    # Extract model name from traces (if available)
    model_name = "unknown"
    if traces:
        first_trace = traces[0]
        # Try parent_metrics.train.metrics.model_name
        train_metrics = first_trace.get("parent_metrics", {}).get("train", {}).get("metrics", {})
        if "model_name" in train_metrics:
            model_name = train_metrics["model_name"]
        else:
            # Fallback: try child_metrics.train.metrics.model_name
            child_train_metrics = first_trace.get("child_metrics", {}).get("train", {}).get("metrics", {})
            if "model_name" in child_train_metrics:
                model_name = child_train_metrics["model_name"]
            else:
                # Fallback: try child_metrics.metrics.model_name (top-level)
                child_metrics = first_trace.get("child_metrics", {}).get("metrics", {})
                if "model_name" in child_metrics:
                    model_name = child_metrics["model_name"]
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("WILDS Amazon (Home and Kitchen) - Evolution Summary\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Model: {model_name}\n")
        f.write(f"Total iterations: {len(curve_data['iterations'])}\n")
        f.write(f"Total evaluations: {len(traces)}\n")
        f.write(f"Validation runs: {len(val_accuracy_points)}\n\n")
        
        if curve_data["best_scores"]:
            f.write("Performance Metrics:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Initial best score: {curve_data['best_scores'][0]:.4f}\n")
            f.write(f"  Final best score: {curve_data['best_scores'][-1]:.4f}\n")
            f.write(f"  Improvement: {curve_data['best_scores'][-1] - curve_data['best_scores'][0]:.4f}\n")
            f.write(f"  Final best train accuracy: {curve_data['best_accuracies'][-1]:.2%}\n")
            
            # Validation accuracy
            if val_accuracy_points:
                best_val = max(va for _, va in val_accuracy_points)
                f.write(f"  Best validation accuracy: {best_val:.2%}\n")
            f.write("\n")
        
        # Validation history
        if val_accuracy_points:
            f.write("Validation Accuracy History:\n")
            f.write("-" * 40 + "\n")
            for iteration, val_acc in sorted(val_accuracy_points, key=lambda x: x[0]):
                f.write(f"  Iteration {iteration}: {val_acc:.2%}\n")
            f.write("\n")
        
        # Find best prompt
        best_trace = None
        best_score = 0
        for trace in traces:
            score = trace.get("child_metrics", {}).get("combined_score", 0)
            if score > best_score:
                best_score = score
                best_trace = trace
        
        if best_trace:
            f.write("Best Prompt Details:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Iteration: {best_trace.get('iteration', 'N/A')}\n")
            f.write(f"  Score: {best_score:.4f}\n")
            metrics = best_trace.get("child_metrics", {})
            f.write(f"  Train Accuracy: {metrics.get('accuracy', 0):.2%}\n")
            val_acc = metrics.get('val_accuracy')
            if val_acc is not None:
                f.write(f"  Val Accuracy: {val_acc:.2%}\n")
            f.write(f"  MAE: {metrics.get('mae', 0):.4f}\n")
            
            # Per-class accuracy
            f.write("\n  Per-class Accuracy:\n")
            for i in range(1, 6):
                key = f"class_{i}_accuracy"
                if key in metrics:
                    f.write(f"    {i} star: {metrics[key]:.2%}\n")
    
    print(f"Saved summary to {report_path}")


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
    plot_learning_curve(curve_data, str(output_dir / "learning_curves.png"))
    
    # Metrics over time
    plot_metrics_over_time(traces, str(viz_dir / "metrics_evolution.png"))
    
    # Reports
    generate_report(traces, checkpoint_data, str(viz_dir / "evolution_report.txt"))
    generate_summary(traces, curve_data, str(output_dir))
    
    print("\nVisualization complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

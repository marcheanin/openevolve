#!/usr/bin/env python3
"""
Script to analyze prompt improvements from evolution trace.
Shows which prompts gave the biggest improvements and compares them with parent prompts.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

try:
    import jsonlines
except ImportError:
    jsonlines = None


def load_evolution_trace(trace_path: str) -> List[Dict[str, Any]]:
    """Load evolution trace from JSONL file."""
    traces = []
    if not os.path.exists(trace_path):
        print(f"Error: Evolution trace not found at {trace_path}")
        return traces
    
    try:
        if jsonlines:
            with jsonlines.open(trace_path) as reader:
                for trace in reader:
                    traces.append(trace)
        else:
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
        
        print(f"Loaded {len(traces)} evolution traces")
    except Exception as e:
        print(f"Error loading evolution trace: {e}")
        import traceback
        traceback.print_exc()
    
    return traces


def has_substantial_change(parent_code: str, child_code: str, parent_metrics: Dict, child_metrics: Dict) -> bool:
    """Check if the change is substantial, not just whitespace."""
    parent_reasoning = parent_metrics.get("reasoning_strategy", 0.0)
    child_reasoning = child_metrics.get("reasoning_strategy", 0.0)
    reasoning_delta = abs(child_reasoning - parent_reasoning)
    
    parent_score = parent_metrics.get("combined_score", 0.0)
    child_score = child_metrics.get("combined_score", 0.0)
    score_delta = child_score - parent_score
    
    if not parent_code or not child_code:
        return reasoning_delta > 0.01 or score_delta > 0.02
    
    parent_normalized = " ".join(parent_code.split()).lower()
    child_normalized = " ".join(child_code.split()).lower()
    
    if parent_normalized == child_normalized:
        return False
    
    parent_words = parent_normalized.split()
    child_words = child_normalized.split()
    
    if parent_words == child_words:
        return False
    
    parent_set = set(parent_words)
    child_set = set(child_words)
    
    if parent_set == child_set:
        return score_delta > 0.02 or reasoning_delta > 0.1
    
    intersection = parent_set & child_set
    union = parent_set | child_set
    overlap = len(intersection) / len(union) if union else 0
    
    if overlap > 0.98:
        return score_delta > 0.02 or reasoning_delta > 0.1
    
    return True


def find_improvements(traces: List[Dict[str, Any]], min_improvement: float = 0.001, 
                      check_substantial: bool = True) -> List[Dict[str, Any]]:
    """Find all evolution steps that resulted in improvement."""
    improvements = []
    
    for trace in traces:
        improvement_delta = trace.get("improvement_delta", {})
        combined_score_delta = improvement_delta.get("combined_score", 0)
        
        if combined_score_delta > min_improvement:
            parent_metrics = trace.get("parent_metrics", {})
            child_metrics = trace.get("child_metrics", {})
            parent_code = trace.get("parent_code", "")
            child_code = trace.get("child_code", "")
            
            if check_substantial:
                if not has_substantial_change(parent_code, child_code, parent_metrics, child_metrics):
                    continue
            
            improvements.append({
                "iteration": trace.get("iteration", 0),
                "timestamp": trace.get("timestamp", 0),
                "parent_score": parent_metrics.get("combined_score", 0),
                "child_score": child_metrics.get("combined_score", 0),
                "improvement": combined_score_delta,
                "parent_code": parent_code,
                "child_code": child_code,
                "parent_metrics": parent_metrics,
                "child_metrics": child_metrics,
                "metadata": trace.get("metadata", {})
            })
    
    return sorted(improvements, key=lambda x: x["improvement"], reverse=True)


def format_prompt_diff(parent: str, child: str, max_length: int = 500) -> str:
    """Create a simple diff-like comparison of two prompts."""
    parent_lines = parent.split('\n') if parent else []
    child_lines = child.split('\n') if child else []
    
    diff_lines = []
    diff_lines.append("=" * 60)
    diff_lines.append("PARENT PROMPT:")
    diff_lines.append("-" * 60)
    parent_preview = parent[:max_length] + "..." if len(parent) > max_length else parent
    diff_lines.append(parent_preview)
    diff_lines.append("")
    diff_lines.append("=" * 60)
    diff_lines.append("CHILD PROMPT (IMPROVED):")
    diff_lines.append("-" * 60)
    child_preview = child[:max_length] + "..." if len(child) > max_length else child
    diff_lines.append(child_preview)
    diff_lines.append("=" * 60)
    
    return "\n".join(diff_lines)


def generate_improvements_report(improvements: List[Dict[str, Any]], output_dir: Path, top_n: int = 10):
    """Generate detailed improvement reports."""
    output_dir.mkdir(exist_ok=True)
    
    # Summary JSON
    summary = {
        "total_improvements": len(improvements),
        "top_improvements": []
    }
    
    for i, imp in enumerate(improvements[:top_n]):
        summary["top_improvements"].append({
            "rank": i + 1,
            "iteration": imp["iteration"],
            "improvement": imp["improvement"],
            "parent_score": imp["parent_score"],
            "child_score": imp["child_score"]
        })
    
    with open(output_dir / "improvements_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Detailed markdown reports
    for i, imp in enumerate(improvements[:top_n]):
        report_path = output_dir / f"improvement_{i+1:02d}_iter_{imp['iteration']:03d}.md"
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# Improvement #{i+1}\n\n")
            f.write(f"**Iteration:** {imp['iteration']}\n\n")
            f.write(f"**Score Change:** {imp['parent_score']:.4f} → {imp['child_score']:.4f} (+{imp['improvement']:.4f})\n\n")
            
            parent_metrics = imp['parent_metrics']
            child_metrics = imp['child_metrics']
            
            f.write("## Metrics Comparison\n\n")
            f.write("| Metric | Parent | Child | Delta |\n")
            f.write("|--------|--------|-------|-------|\n")
            
            all_keys = set(parent_metrics.keys()) | set(child_metrics.keys())
            for key in sorted(all_keys):
                pval = parent_metrics.get(key, 0)
                cval = child_metrics.get(key, 0)
                if isinstance(pval, float) and isinstance(cval, float):
                    delta = cval - pval
                    f.write(f"| {key} | {pval:.4f} | {cval:.4f} | {delta:+.4f} |\n")
                else:
                    f.write(f"| {key} | {pval} | {cval} | - |\n")
            
            f.write("\n## Parent Prompt\n\n```\n")
            f.write(imp['parent_code'] or "(not available)")
            f.write("\n```\n\n")
            
            f.write("## Child Prompt (Improved)\n\n```\n")
            f.write(imp['child_code'] or "(not available)")
            f.write("\n```\n")
    
    print(f"Generated {min(top_n, len(improvements))} improvement reports in {output_dir}")


def print_summary(improvements: List[Dict[str, Any]], top_n: int = 5):
    """Print a summary of top improvements to console."""
    print("\n" + "=" * 80)
    print("TOP PROMPT IMPROVEMENTS (LiveBench IF)")
    print("=" * 80)
    
    if not improvements:
        print("\nNo substantial improvements found in evolution trace.")
        return
    
    print(f"\nTotal substantial improvements: {len(improvements)}")
    print(f"\nTop {min(top_n, len(improvements))} improvements:\n")
    
    for i, imp in enumerate(improvements[:top_n]):
        print(f"#{i+1}: Iteration {imp['iteration']}")
        print(f"    Score: {imp['parent_score']:.4f} → {imp['child_score']:.4f} (+{imp['improvement']:.4f})")
        
        parent_len = len(imp['parent_code']) if imp['parent_code'] else 0
        child_len = len(imp['child_code']) if imp['child_code'] else 0
        print(f"    Length: {parent_len} → {child_len} chars")
        print()


def main():
    """Main function."""
    output_dir = Path("openevolve_output")
    trace_path = output_dir / "evolution_trace.jsonl"
    
    if not trace_path.exists():
        print(f"Error: Evolution trace not found at {trace_path}")
        print("Please run the evolution first using run_evolution.ps1")
        sys.exit(1)
    
    print("Loading evolution trace...")
    traces = load_evolution_trace(str(trace_path))
    
    if not traces:
        print("No traces found.")
        sys.exit(1)
    
    print("\nFinding improvements...")
    improvements = find_improvements(traces, min_improvement=0.001, check_substantial=True)
    
    print_summary(improvements, top_n=10)
    
    # Generate detailed reports
    improvements_dir = output_dir / "visualizations" / "improvements"
    generate_improvements_report(improvements, improvements_dir, top_n=20)
    
    print(f"\nDetailed reports saved to: {improvements_dir}")


if __name__ == "__main__":
    main()


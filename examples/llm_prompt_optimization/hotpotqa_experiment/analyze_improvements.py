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

# Try to import jsonlines, but fall back to manual JSONL reading if not available
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
        # Try to use jsonlines if available
        if jsonlines:
            with jsonlines.open(trace_path) as reader:
                for trace in reader:
                    traces.append(trace)
        else:
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
        
        print(f"Loaded {len(traces)} evolution traces")
    except Exception as e:
        print(f"Error loading evolution trace: {e}")
        import traceback
        traceback.print_exc()
    
    return traces


def has_substantial_change(parent_code: str, child_code: str, parent_metrics: Dict, child_metrics: Dict) -> bool:
    """
    Check if the change is substantial, not just prompt length.
    
    Returns False if:
    - Only whitespace/formatting changed
    - Only prompt length changed (same words, same order)
    - Only word order changed (same words, different order)
    - Very high overlap (>98%) with no significant metric changes
    
    Returns True if:
    - Different words were added/removed
    - Reasoning strategy changed significantly
    - Score improved significantly
    - Substantial content changes
    """
    # Get metrics
    parent_reasoning = parent_metrics.get("reasoning_strategy", 0.0)
    child_reasoning = child_metrics.get("reasoning_strategy", 0.0)
    reasoning_delta = abs(child_reasoning - parent_reasoning)
    
    parent_score = parent_metrics.get("combined_score", 0.0)
    child_score = child_metrics.get("combined_score", 0.0)
    score_delta = child_score - parent_score
    
    # If we don't have code, check if metrics changed substantially
    if not parent_code or not child_code:
        # At least reasoning strategy changed or significant score improvement
        return reasoning_delta > 0.01 or score_delta > 0.02
    
    # Normalize prompts by removing extra whitespace and converting to lowercase
    parent_normalized = " ".join(parent_code.split()).lower()
    child_normalized = " ".join(child_code.split()).lower()
    
    # Check if prompts are identical after normalization
    if parent_normalized == child_normalized:
        return False
    
    # Get word lists (lowercase for comparison)
    parent_words = parent_normalized.split()
    child_words = child_normalized.split()
    
    # Check if word sequences are identical (only whitespace/formatting changed)
    if parent_words == child_words:
        return False
    
    # Compare word sets (ignoring order)
    parent_word_set = set(parent_words)
    child_word_set = set(child_words)
    
    # If word sets are identical, only order/formatting changed
    if parent_word_set == child_word_set:
        # Check if reasoning strategy or score changed significantly
        # If only order/formatting changed and metrics didn't change, filter it out
        if reasoning_delta < 0.01 and score_delta < 0.02:
            return False
        # If metrics changed, it's still substantial (even if words are the same)
        return True
    
    # Calculate word differences
    new_words = child_word_set - parent_word_set
    removed_words = parent_word_set - child_word_set
    word_changes = len(new_words) + len(removed_words)
    
    # Calculate word overlap ratio
    common_words = len(parent_word_set & child_word_set)
    total_unique_words = len(parent_word_set | child_word_set)
    
    if total_unique_words > 0:
        overlap_ratio = common_words / total_unique_words
        
        # If overlap is very high (>98%), check if it's just minor changes
        if overlap_ratio > 0.98:
            # If very few words changed and metrics didn't change significantly, filter it out
            if word_changes <= 2 and reasoning_delta < 0.01 and score_delta < 0.02:
                return False
    
    # If reasoning strategy changed significantly, it's substantial
    if reasoning_delta > 0.05:
        return True
    
    # If score improved significantly (>2%), it's substantial
    if score_delta > 0.02:
        return True
    
    # If words actually changed (different content), it's substantial
    # But only if significant word changes (more than just 1-2 words)
    if word_changes > 2:
        return True
    
    # If overlap is low (<95%), it means substantial content change
    if total_unique_words > 0:
        if overlap_ratio < 0.95:
            return True
    
    # Default: if we got here, it's likely a minor change
    # (very high overlap, few word changes, no significant metric changes)
    return False


def find_major_improvements(traces: List[Dict[str, Any]], top_n: int = 10) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Find traces with the biggest improvements in combined_score.
    
    Returns:
        (improvements, filtered_count, total_improvements)
    """
    improvements = []
    filtered_count = 0
    total_improvements = 0
    
    for trace in traces:
        improvement_delta = trace.get("improvement_delta", {})
        score_delta = improvement_delta.get("combined_score", 0.0)
        
        if score_delta > 0:
            total_improvements += 1
            parent_code = trace.get("parent_code", "")
            child_code = trace.get("child_code", "")
            parent_metrics = trace.get("parent_metrics", {})
            child_metrics = trace.get("child_metrics", {})
            
            # Check if change is substantial (not just length)
            if has_substantial_change(parent_code, child_code, parent_metrics, child_metrics):
                improvements.append({
                    "trace": trace,
                    "iteration": trace.get("iteration", 0),
                    "improvement": score_delta,
                    "parent_score": parent_metrics.get("combined_score", 0.0),
                    "child_score": child_metrics.get("combined_score", 0.0),
                    "parent_code": parent_code,
                    "child_code": child_code,
                    "parent_metrics": parent_metrics,
                    "child_metrics": child_metrics,
                    "prompt": trace.get("prompt", {}),
                    "llm_response": trace.get("llm_response", ""),
                    "island_id": trace.get("island_id"),
                    "generation": trace.get("generation", 0),
                })
            else:
                filtered_count += 1
    
    # Sort by improvement (descending)
    improvements.sort(key=lambda x: x["improvement"], reverse=True)
    
    return improvements[:top_n], filtered_count, total_improvements


def format_prompt_comparison(improvement: Dict[str, Any]) -> str:
    """Format a comparison between parent and child prompts in Markdown format."""
    lines = []
    
    iteration = improvement["iteration"]
    rank = improvement.get('rank', 0)
    improvement_val = improvement["improvement"]
    parent_score = improvement["parent_score"]
    child_score = improvement["child_score"]
    improvement_percent = improvement_val/parent_score*100 if parent_score > 0 else 0
    
    # Header
    lines.append(f"## Improvement #{rank} - Iteration {iteration}")
    lines.append("")
    
    # Score improvement
    lines.append("### Score Improvement")
    lines.append("")
    lines.append(f"- **Improvement**: `+{improvement_val:.4f}` ({improvement_percent:+.2f}%)")
    lines.append(f"- **Parent Score**: `{parent_score:.4f}`")
    lines.append(f"- **Child Score**: `{child_score:.4f}`")
    lines.append("")
    
    # Metrics comparison table
    parent_metrics = improvement.get("parent_metrics", {})
    child_metrics = improvement.get("child_metrics", {})
    
    lines.append("### Metrics Comparison")
    lines.append("")
    lines.append("| Metric | Parent | Child | Delta |")
    lines.append("|--------|--------|-------|-------|")
    for key in sorted(set(list(parent_metrics.keys()) + list(child_metrics.keys()))):
        if key == "combined_score":
            continue
        parent_val = parent_metrics.get(key, 0)
        child_val = child_metrics.get(key, 0)
        if isinstance(parent_val, (int, float)) and isinstance(child_val, (int, float)):
            delta = child_val - parent_val
            delta_str = f"{delta:+.4f}" if delta != 0 else "0.0000"
            lines.append(f"| `{key}` | {parent_val:.4f} | {child_val:.4f} | {delta_str} |")
    lines.append("")
    
    # Parent prompt
    lines.append("### Parent Prompt (Before)")
    lines.append("")
    parent_code = improvement.get("parent_code", "")
    if parent_code:
        lines.append("```")
        lines.append(parent_code)
        lines.append("```")
    else:
        lines.append("*(Parent code not available in trace)*")
    lines.append("")
    
    # Child prompt
    lines.append("### Child Prompt (After - with improvement)")
    lines.append("")
    child_code = improvement.get("child_code", "")
    if child_code:
        lines.append("```")
        lines.append(child_code)
        lines.append("```")
    else:
        lines.append("*(Child code not available in trace)*")
    lines.append("")
    
    # Key changes
    if parent_code and child_code:
        lines.append("### Key Changes")
        lines.append("")
        parent_len = len(parent_code)
        child_len = len(child_code)
        length_delta = child_len - parent_len
        lines.append(f"- **Length**: {parent_len} → {child_len} chars ({length_delta:+d})")
        lines.append("")
        
        # Check for common patterns
        changes_list = []
        if "step" in child_code.lower() and "step" not in parent_code.lower():
            changes_list.append("Added step-by-step reasoning")
        if "example" in child_code.lower() and "example" not in parent_code.lower():
            changes_list.append("Added examples")
        if "multiple" in child_code.lower() and "multiple" not in parent_code.lower():
            changes_list.append("Emphasized multiple paragraphs")
        if "paragraph" in child_code.lower() and "paragraph" not in parent_code.lower():
            changes_list.append("Added paragraph-specific instructions")
        
        if changes_list:
            for change in changes_list:
                lines.append(f"- {change}")
            lines.append("")
    
    # LLM prompt used (if available) - FULL PROMPT
    prompt = improvement.get("prompt", {})
    if prompt:
        lines.append("---")
        lines.append("")
        lines.append("## Full LLM Prompt Used for Evolution")
        lines.append("")
        lines.append("This is the complete prompt that was sent to the LLM to generate")
        lines.append("the improved prompt. It includes all filled placeholders from the")
        lines.append("template (`full_rewrite_user.txt`).")
        lines.append("")
        
        system_msg = prompt.get("system", "")
        user_msg = prompt.get("user", "")
        
        if system_msg:
            lines.append("### System Message")
            lines.append("")
            lines.append("```")
            lines.append(system_msg)
            lines.append("```")
            lines.append("")
        else:
            lines.append("*(System message not available in trace)*")
            lines.append("")
        
        if user_msg:
            lines.append("### User Message (Full filled template from `full_rewrite_user.txt`)")
            lines.append("")
            lines.append(f"*Length: {len(user_msg)} characters*")
            lines.append("")
            lines.append("```")
            lines.append(user_msg)
            lines.append("```")
            lines.append("")
        else:
            lines.append("*(User message not available in trace)*")
            lines.append("")
        
        # Also show LLM response if available
        llm_response = improvement.get("llm_response", "")
        if llm_response:
            lines.append("### LLM Response (Generated new prompt)")
            lines.append("")
            lines.append(f"*Length: {len(llm_response)} characters*")
            lines.append("")
            lines.append("```")
            lines.append(llm_response)
            lines.append("```")
            lines.append("")
        else:
            lines.append("*(LLM response not available in trace)*")
            lines.append("")
    
    # Island and generation info
    island_id = improvement.get("island_id")
    generation = improvement.get("generation", 0)
    if island_id is not None:
        lines.append("---")
        lines.append("")
        lines.append(f"**Island ID**: {island_id}, **Generation**: {generation}")
        lines.append("")
    
    return "\n".join(lines)


def generate_improvements_report(traces: List[Dict[str, Any]], output_path: str, top_n: int = 10):
    """Generate a detailed report of major improvements."""
    print(f"\nAnalyzing improvements from {len(traces)} traces...")
    print("Filtering out improvements where only prompt length/formatting changed...")
    
    improvements, filtered_count, total_improvements = find_major_improvements(traces, top_n=top_n)
    
    # Print filtering statistics
    if filtered_count > 0:
        print(f"  Filtered out {filtered_count} improvements (only length/formatting/order changed)")
        print(f"  Kept {len(improvements)} substantial improvements out of {total_improvements} total")
    
    if not improvements:
        print("No substantial improvements found in traces.")
        print("(All improvements were filtered out as only length/formatting changes)")
        return
    
    print(f"\nFound {len(improvements)} substantial improvements (after filtering)")
    print(f"\nTop {min(top_n, len(improvements))} improvements:")
    for i, imp in enumerate(improvements[:top_n], 1):
        print(f"  {i}. Iteration {imp['iteration']}: +{imp['improvement']:.4f} "
              f"({imp['parent_score']:.4f} → {imp['child_score']:.4f})")
    
    # Generate report in Markdown format
    report_lines = []
    report_lines.append("# HotpotQA Prompt Evolution - Major Improvements Analysis")
    report_lines.append("")
    report_lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append("## Summary Statistics")
    report_lines.append("")
    report_lines.append(f"- **Total improvements found**: {total_improvements}")
    report_lines.append(f"- **Filtered out** (only length/formatting): {filtered_count}")
    report_lines.append(f"- **Substantial improvements**: {len(improvements)}")
    report_lines.append("")
    report_lines.append("> **Note**: Improvements where only prompt length, formatting, or word order")
    report_lines.append("> changed (without content changes) have been filtered out.")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Add ranking
    for i, improvement in enumerate(improvements, 1):
        improvement['rank'] = i
    
    # Generate comparison for each improvement
    for improvement in improvements:
        report_lines.append(format_prompt_comparison(improvement))
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
    
    # Final summary
    report_lines.append("## Final Summary")
    report_lines.append("")
    report_lines.append("| Metric | Value |")
    report_lines.append("|--------|-------|")
    report_lines.append(f"| Total improvements found | {total_improvements} |")
    report_lines.append(f"| Filtered out (only length/formatting) | {filtered_count} |")
    report_lines.append(f"| Substantial improvements | {len(improvements)} |")
    if improvements:
        avg_improvement = sum(imp['improvement'] for imp in improvements) / len(improvements)
        report_lines.append(f"| Average improvement | {avg_improvement:.4f} |")
        report_lines.append(f"| Largest improvement | {improvements[0]['improvement']:.4f} (Iteration {improvements[0]['iteration']}) |")
        report_lines.append(f"| Smallest major improvement | {improvements[-1]['improvement']:.4f} (Iteration {improvements[-1]['iteration']}) |")
    report_lines.append("")
    
    # Save report
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    print(f"\nReport saved to: {output_path}")
    
    # Also save individual improvements as separate files
    output_dir = Path(output_path).parent
    improvements_dir = output_dir / "improvements"
    improvements_dir.mkdir(exist_ok=True)
    
    for improvement in improvements:
        rank = improvement['rank']
        iteration = improvement['iteration']
        filename = f"improvement_{rank:02d}_iter_{iteration:03d}.md"
        filepath = improvements_dir / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(format_prompt_comparison(improvement))
        
        print(f"  Saved improvement #{rank} to: {filepath}")
    
    # Save JSON summary
    json_summary = []
    for improvement in improvements:
        json_summary.append({
            "rank": improvement['rank'],
            "iteration": improvement['iteration'],
            "improvement": improvement['improvement'],
            "parent_score": improvement['parent_score'],
            "child_score": improvement['child_score'],
            "parent_code": improvement.get('parent_code', ''),
            "child_code": improvement.get('child_code', ''),
            "parent_metrics": improvement.get('parent_metrics', {}),
            "child_metrics": improvement.get('child_metrics', {}),
            "island_id": improvement.get('island_id'),
            "generation": improvement.get('generation', 0),
        })
    
    json_path = output_dir / "improvements_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nJSON summary saved to: {json_path}")


def main():
    """Main function to analyze improvements."""
    # Paths
    output_dir = Path("openevolve_output")
    trace_path = output_dir / "evolution_trace.jsonl"
    
    if not trace_path.exists():
        print(f"Error: Evolution trace not found at {trace_path}")
        print("Please run the evolution first using run_evolution.ps1")
        sys.exit(1)
    
    print("Loading evolution trace...")
    traces = load_evolution_trace(str(trace_path))
    
    if not traces:
        print("No traces found. Cannot analyze improvements.")
        sys.exit(1)
    
    # Create output directory
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Generate improvements report (Markdown format)
    output_path = viz_dir / "improvements_analysis.md"
    generate_improvements_report(traces, str(output_path), top_n=10)
    
    print("\nAnalysis complete!")
    print(f"Results saved to: {viz_dir}")


if __name__ == "__main__":
    main()


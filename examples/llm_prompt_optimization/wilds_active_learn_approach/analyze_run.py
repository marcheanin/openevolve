"""
Comprehensive analysis of an Active Learning evolution run.

Reads all output artifacts and produces a diagnostic report:
  - Pipeline health checks (artifacts, trace, DB)
  - Per-cycle and per-iteration metrics summary
  - Consolidation decisions
  - Token usage breakdown
  - Final delta vs baseline
"""

import json
import glob
import os
import sys
from pathlib import Path
from collections import Counter

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"  # overridden by --results-dir when given


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path):
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def section(title):
    w = 70
    print(f"\n{'=' * w}")
    print(f"  {title}")
    print(f"{'=' * w}")


def check_pipeline_health():
    """Verify all pipeline components worked correctly."""
    section("1. Pipeline Health Check")
    issues = []

    # Check evolution trace
    traces = glob.glob(str(RESULTS_DIR / "al_iter_*" / "openevolve_output" / "evolution_trace.jsonl"))
    if traces:
        total_entries = 0
        with_artifacts = 0
        for tp in sorted(traces):
            entries = _load_jsonl(tp)
            total_entries += len(entries)
            for e in entries:
                if e.get("artifacts"):
                    with_artifacts += 1
        print(f"  [OK] Evolution traces: {len(traces)} files, {total_entries} total iterations")
        if with_artifacts == total_entries and total_entries > 0:
            print(f"  [OK] Artifacts in traces: {with_artifacts}/{total_entries} (all have error artifacts)")
        elif with_artifacts > 0:
            print(f"  [!!] Artifacts in traces: {with_artifacts}/{total_entries} (some missing)")
            issues.append(f"Only {with_artifacts}/{total_entries} traces have artifacts")
        else:
            print(f"  [FAIL] Artifacts in traces: 0/{total_entries}")
            issues.append("No artifacts in any trace")
    else:
        print("  [FAIL] No evolution traces found")
        issues.append("Missing evolution_trace.jsonl")

    # Check database artifacts
    db_dirs = sorted(glob.glob(str(RESULTS_DIR / "al_iter_*" / "database" / "programs")))
    for dd in db_dirs:
        cycle = Path(dd).parent.parent.name
        files = glob.glob(os.path.join(dd, "*.json"))
        with_a = sum(1 for f in files if _load_json(f).get("artifacts_json"))
        total = len(files)
        status = "[OK]" if with_a > 0 else "[!!]"
        print(f"  {status} DB {cycle}: {total} programs, {with_a} with stored artifacts")
        if with_a == 0 and total > 0:
            issues.append(f"{cycle}: no artifacts persisted in DB programs")

    # Check debug trace
    dt_path = RESULTS_DIR / "debug_trace.jsonl"
    if dt_path.exists():
        events = _load_jsonl(dt_path)
        event_types = Counter(e["event"] for e in events)
        print(f"  [OK] Debug trace: {len(events)} events ({', '.join(event_types.keys())})")
    else:
        print("  [FAIL] No debug_trace.jsonl")
        issues.append("Missing debug_trace.jsonl")

    # Check consolidation decisions
    cons_events = [e for e in (events if dt_path.exists() else []) if e["event"] == "consolidation_decision"]
    for ce in cons_events:
        status = "ACCEPTED" if ce["accepted"] else "REJECTED"
        print(f"  [INFO] Consolidation cycle {ce['al_iter']}: {status} "
              f"(cons={ce['cons_score']:.4f} vs evo={ce['evo_score']:.4f})")

    if issues:
        print(f"\n  ISSUES ({len(issues)}):")
        for iss in issues:
            print(f"    - {iss}")
    else:
        print("\n  All checks passed.")
    return issues


def analyze_evolution_iterations():
    """Show per-iteration details from evolution traces."""
    section("2. Evolution Iterations (per-cycle)")
    traces = sorted(glob.glob(str(RESULTS_DIR / "al_iter_*" / "openevolve_output" / "evolution_trace.jsonl")))
    for tp in traces:
        cycle = Path(tp).parent.parent.name
        entries = _load_jsonl(tp)
        print(f"\n  --- {cycle} ({len(entries)} iterations) ---")
        print(f"  {'iter':>4}  {'parent_score':>12}  {'child_score':>12}  {'Acc_Hard':>9}  {'Acc_Anch':>9}  {'delta':>8}  {'artifacts':>9}")
        print(f"  {'----':>4}  {'------------':>12}  {'------------':>12}  {'---------':>9}  {'---------':>9}  {'--------':>8}  {'---------':>9}")
        for e in entries:
            pm = e.get("parent_metrics", {})
            cm = e.get("child_metrics", {})
            ps = pm.get("combined_score", 0)
            cs = cm.get("combined_score", 0)
            delta = cs - ps
            ah = cm.get("Acc_Hard", 0)
            aa = cm.get("Acc_Anchor", 0)
            has_a = "yes" if e.get("artifacts") else "no"
            sign = "+" if delta >= 0 else ""
            print(f"  {e.get('iteration', '?'):>4}  {ps:>12.4f}  {cs:>12.4f}  {ah:>8.1%}  {aa:>8.1%}  {sign}{delta:>7.4f}  {has_a:>9}")


def analyze_al_cycles():
    """Show per-cycle summary from the AL log."""
    section("3. Active Learning Cycles")
    log_path = RESULTS_DIR / "active_loop_log.json"
    if not log_path.exists():
        print("  No active_loop_log.json found.")
        return
    log = _load_json(log_path)
    print(f"  {'cycle':>5}  {'val_R_global':>12}  {'val_R_worst':>12}  {'val_AccH':>10}  {'combined':>10}  "
          f"{'hard':>5}  {'anchor':>7}  {'cons':>5}  {'exp':>4}  {'evo_best':>10}  {'time':>6}")
    print(f"  {'-----':>5}  {'------------':>12}  {'------------':>12}  {'----------':>10}  {'----------':>10}  "
          f"{'-----':>5}  {'-------':>7}  {'-----':>5}  {'----':>4}  {'----------':>10}  {'------':>6}")
    for e in log:
        cons = "yes" if e.get("consolidated") else "no"
        exp = "yes" if e.get("expanded") else "no"
        evo_s = e.get("evo_best_score", 0)
        t = e.get("cycle_time_s", 0)
        print(f"  {e['al_iter']:>5}  {e['val_R_global']:>11.2%}  {e.get('val_R_worst', 0):>11.2%}  "
              f"{e['val_Acc_Hard']:>9.2%}  {e['val_combined_score']:>10.4f}  "
              f"{e.get('batch_n_hard', 0):>5}  {e.get('batch_n_anchor', 0):>7}  "
              f"{cons:>5}  {exp:>4}  {evo_s:>10.4f}  {t:>5.0f}s")


def analyze_prompts():
    """Show how prompts changed across cycles."""
    section("4. Prompt Evolution")
    prompt_files = sorted(glob.glob(str(RESULTS_DIR / "al_iter_*" / "best_prompt.txt")))
    start_files = sorted(glob.glob(str(RESULTS_DIR / "al_iter_*" / "start_prompt.txt")))

    initial = SCRIPT_DIR / "initial_prompt.txt"
    if initial.exists():
        text = initial.read_text(encoding="utf-8")
        print(f"  Initial prompt: {len(text)} chars, ~{len(text)//4} tokens")

    for sf in start_files:
        cycle = Path(sf).parent.name
        text = Path(sf).read_text(encoding="utf-8")
        print(f"  {cycle} start:  {len(text)} chars, ~{len(text)//4} tokens")

    for pf in prompt_files:
        cycle = Path(pf).parent.name
        text = Path(pf).read_text(encoding="utf-8")
        n_rules = text.count("    - ")
        n_examples = text.count("Example ")
        print(f"  {cycle} best:   {len(text)} chars, ~{len(text)//4} tokens, "
              f"~{n_rules} rules, ~{n_examples} examples")

    # Show consolidated prompts
    cons_files = sorted(glob.glob(str(RESULTS_DIR / "al_iter_*" / "consolidated_prompt.txt")))
    for cf in cons_files:
        cycle = Path(cf).parent.name
        text = Path(cf).read_text(encoding="utf-8")
        print(f"  {cycle} cons:   {len(text)} chars (consolidated, may or may not be accepted)")


def analyze_test_results():
    """Compare baseline vs final test metrics."""
    section("5. Test Results (Baseline vs Final)")
    baseline_path = RESULTS_DIR / "baseline_test_metrics.json"
    final_path = RESULTS_DIR / "final_test_metrics.json"
    if not baseline_path.exists() or not final_path.exists():
        print("  Missing baseline or final test metrics.")
        return
    bl = _load_json(baseline_path)
    fn = _load_json(final_path)

    metrics = ["R_global", "R_worst", "mae", "combined_score", "Acc_Hard", "Acc_Anchor", "mean_kappa"]
    print(f"  {'metric':>18}  {'baseline':>10}  {'final':>10}  {'delta':>10}  {'verdict':>8}")
    print(f"  {'------------------':>18}  {'----------':>10}  {'----------':>10}  {'----------':>10}  {'--------':>8}")
    for m in metrics:
        b = bl.get(m, 0)
        f = fn.get(m, 0)
        d = f - b
        better_if_higher = m not in ("mae",)
        improved = (d > 0) == better_if_higher
        verdict = "BETTER" if improved and abs(d) > 0.001 else ("WORSE" if not improved and abs(d) > 0.001 else "same")
        sign = "+" if d >= 0 else ""
        print(f"  {m:>18}  {b:>10.4f}  {f:>10.4f}  {sign}{d:>9.4f}  {verdict:>8}")

    print(f"\n  Test set composition: baseline {bl.get('n_hard', '?')}H/{bl.get('n_anchor', '?')}A, "
          f"final {fn.get('n_hard', '?')}H/{fn.get('n_anchor', '?')}A")


def analyze_tokens():
    """Show token usage breakdown."""
    section("6. Token Usage")
    tok_path = RESULTS_DIR / "token_usage.json"
    if not tok_path.exists():
        print("  No token_usage.json found.")
        return
    tok = _load_json(tok_path)
    by_model = tok.get("by_model", {})
    print(f"  {'model':>40}  {'input':>12}  {'output':>10}  {'total':>12}")
    print(f"  {'----------------------------------------':>40}  {'------------':>12}  {'----------':>10}  {'------------':>12}")
    for model, usage in sorted(by_model.items()):
        inp = usage.get("input_tokens", 0)
        out = usage.get("output_tokens", 0)
        total = usage.get("total_tokens", 0)
        print(f"  {model:>40}  {inp:>12,}  {out:>10,}  {total:>12,}")
    print(f"  {'TOTAL':>40}  {tok.get('total_input_tokens', 0):>12,}  "
          f"{tok.get('total_output_tokens', 0):>10,}  {tok.get('total_tokens', 0):>12,}")


def analyze_archive():
    """Show MAP-Elites archive contents per cycle."""
    section("7. MAP-Elites Archive (top-5 per cycle)")
    db_dirs = sorted(glob.glob(str(RESULTS_DIR / "al_iter_*" / "database" / "programs")))
    for dd in db_dirs:
        cycle = Path(dd).parent.parent.name
        files = glob.glob(os.path.join(dd, "*.json"))
        programs = []
        for f in files:
            d = _load_json(f)
            m = d.get("metrics", {})
            programs.append({
                "id": d.get("id", "?")[:8],
                "score": m.get("combined_score", 0),
                "Acc_Hard": m.get("Acc_Hard", 0),
                "Acc_Anchor": m.get("Acc_Anchor", 0),
                "gen": d.get("generation", 0),
                "has_artifacts": bool(d.get("artifacts_json")),
            })
        programs.sort(key=lambda p: p["score"], reverse=True)
        print(f"\n  --- {cycle} ({len(programs)} programs) ---")
        print(f"  {'id':>8}  {'score':>8}  {'Acc_Hard':>9}  {'Acc_Anch':>9}  {'gen':>4}  {'artifacts':>9}")
        for p in programs[:5]:
            print(f"  {p['id']:>8}  {p['score']:>8.4f}  {p['Acc_Hard']:>8.1%}  {p['Acc_Anchor']:>8.1%}  "
                  f"{p['gen']:>4}  {'yes' if p['has_artifacts'] else 'no':>9}")


def main():
    global RESULTS_DIR
    import argparse
    parser = argparse.ArgumentParser(description="Analyze Active Learning evolution run output.")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Path to results directory (default: results). Use results_smoke for smoke run.")
    args = parser.parse_args()
    if args.results_dir:
        p = Path(args.results_dir)
        RESULTS_DIR = p.resolve() if p.is_absolute() else (SCRIPT_DIR / args.results_dir).resolve()
    else:
        RESULTS_DIR = (SCRIPT_DIR / "results").resolve()

    print("Active Learning Evolution Run Analysis")
    print(f"Results directory: {RESULTS_DIR}")

    if not RESULTS_DIR.exists():
        print("ERROR: results directory not found.")
        sys.exit(1)

    issues = check_pipeline_health()
    analyze_evolution_iterations()
    analyze_al_cycles()
    analyze_prompts()
    analyze_test_results()
    analyze_tokens()
    analyze_archive()

    section("Summary")
    final_path = RESULTS_DIR / "final_test_metrics.json"
    baseline_path = RESULTS_DIR / "baseline_test_metrics.json"
    if final_path.exists() and baseline_path.exists():
        bl = _load_json(baseline_path)
        fn = _load_json(final_path)
        dr = fn["R_global"] - bl["R_global"]
        dc = fn["combined_score"] - bl["combined_score"]
        print(f"  R_global:       {bl['R_global']:.2%} -> {fn['R_global']:.2%}  ({dr:+.2%})")
        print(f"  combined_score: {bl['combined_score']:.4f} -> {fn['combined_score']:.4f}  ({dc:+.4f})")
        print(f"  R_worst:        {bl['R_worst']:.2%} -> {fn['R_worst']:.2%}  ({fn['R_worst']-bl['R_worst']:+.2%})")
        verdict = "IMPROVED" if dc > 0 else "DEGRADED"
        print(f"\n  Overall verdict: {verdict}")
    if issues:
        print(f"  Pipeline issues: {len(issues)}")
    print()


if __name__ == "__main__":
    main()

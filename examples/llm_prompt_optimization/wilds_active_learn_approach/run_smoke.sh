#!/usr/bin/env bash
# Smoke run: 2 AL cycles x 4 evolution iterations, output in results_smoke/
# Run from this directory. Requires OPENROUTER_API_KEY (or OPENAI_API_KEY) in env or .env.

set -e
cd "$(dirname "$0")"

echo "=== Smoke run (2 AL x 4 evolve) -> results_smoke/ ==="
python active_loop.py --smoke

echo ""
echo "=== Analysis of smoke run ==="
python analyze_run.py --results-dir results_smoke

echo ""
echo "Smoke output: $(pwd)/results_smoke/"
echo "  - debug_trace.jsonl, active_loop_log.json"
echo "  - al_iter_0/, al_iter_1/ (evolution traces, best_prompt.txt, consolidated_prompt.txt)"
echo "  - baseline_test_metrics.json, final_test_metrics.json, final_prompt.txt"

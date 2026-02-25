# Smoke run: 2 AL cycles x 4 evolution iterations, output in results_smoke/
# Run from this directory. Requires OPENROUTER_API_KEY (or OPENAI_API_KEY) in env or .env.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "=== Smoke run (2 AL x 4 evolve) -> results_smoke/ ===" -ForegroundColor Cyan
python active_loop.py --smoke
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n=== Analysis of smoke run ===" -ForegroundColor Cyan
python analyze_run.py --results-dir results_smoke
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`nSmoke output: $PSScriptRoot\results_smoke\" -ForegroundColor Green
Write-Host "  - debug_trace.jsonl, active_loop_log.json"
Write-Host "  - al_iter_0/, al_iter_1/ (evolution traces, best_prompt.txt, consolidated_prompt.txt)"
Write-Host "  - baseline_test_metrics.json, final_test_metrics.json, final_prompt.txt"

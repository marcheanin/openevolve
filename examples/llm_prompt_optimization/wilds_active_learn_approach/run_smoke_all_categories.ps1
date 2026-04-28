# Smoke: 2 AL cycles x 4 OpenEvolve iter, config_all_categories + separate results folder.
# Logs/graphs: results_smoke_all_categories/ (active_loop_log.json, debug_trace.jsonl, al_iter_*/..., PNGs from analyze_run)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "=== Smoke (all categories) -> results_smoke_all_categories/ ===" -ForegroundColor Cyan

python active_loop.py `
  --config config_all_categories.yaml `
  --smoke `
  --results-dir results_smoke_all_categories

Write-Host "`n=== Analysis + plots ===" -ForegroundColor Cyan
python analyze_run.py --results-dir results_smoke_all_categories

Write-Host "`nDone. Output: $PSScriptRoot\results_smoke_all_categories\" -ForegroundColor Green

# Active Learning experiment with uncapped train + 5000 candidate pool + val=450 + test=600
# Performs 8 AL cycles x 15 evolution iterations, then evaluates the best-by-val prompt
# on the FULL uncapped WILDS Amazon test split.
#
# Usage (PowerShell, from this folder):
#   .\run_uncap_al.ps1                # full run (8 cycles, ~6-9h on a single node)
#   .\run_uncap_al.ps1 -Smoke         # 2 cycles x 4 evolve iters; quick sanity check
#   .\run_uncap_al.ps1 -Resume        # continue an existing results dir
#   .\run_uncap_al.ps1 -SkipFullTest  # do NOT run full uncapped test at the end

param(
    [switch]$Smoke,
    [switch]$Resume,
    [switch]$SkipFullTest,
    [int]$NAl = 8,
    [int]$NEvolve = 15,
    [string]$ResultsDir = "results_all_categories_uncapped_train"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

if (-not (Test-Path ".\.env")) {
    Write-Warning "No .env file found. Make sure OPENROUTER_API_KEY (or OPENAI_API_KEY) is set in env."
}

$args = @(
    "active_loop.py",
    "--config", "config_all_categories_uncapped_train.yaml",
    "--prompt", "initial_prompt_all_categories.txt",
    "--results-dir", $ResultsDir,
    "--no-evolve-early-stop"
)

if ($Smoke) {
    $args += @("--smoke")
} else {
    $args += @("--n-al", "$NAl", "--n-evolve", "$NEvolve")
}

if ($Resume) {
    $args += @("--resume-from-dir", $ResultsDir)
}

if (-not $SkipFullTest -and -not $Smoke) {
    $args += @("--run-full-test")
}

Write-Host "==> python $($args -join ' ')"
$start = Get-Date
python @args
$dur = (Get-Date) - $start
Write-Host ("Done in {0:hh\:mm\:ss}" -f $dur)
Write-Host "Results: $ResultsDir"
Write-Host "  active_loop_log.json     - per-cycle metrics"
Write-Host "  best_val_prompt.txt      - prompt with the best validation combined_score"
Write-Host "  final_test_metrics.json  - capped (~600) test metrics for that prompt"
Write-Host "  full_uncapped_test_metrics.json - FULL WILDS test metrics for the same prompt"
Write-Host "  al_pool_manifest.json    - candidate pool composition (5000 reviews) for reproducibility"

# PowerShell script to run OpenEvolve for WILDS Amazon prompt evolution
# Usage: .\run_evolution.ps1 [-iterations N] or .\run_evolution.ps1 --iterations N

# Fix Unicode encoding for Windows
chcp 65001 | Out-Null
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Default iterations
$iterations = 100

# Parse command line arguments
for ($i = 0; $i -lt $args.Count; $i++) {
    $arg = $args[$i]
    if ($arg -eq '--iterations' -or $arg -eq '-iterations') {
        if ($i + 1 -lt $args.Count) {
            $iterations = [int]$args[$i + 1]
            break
        }
    }
    elseif ($arg -match '^--iterations=(.+)$' -or $arg -match '^-iterations=(.+)$') {
        $iterations = [int]$matches[1]
        break
    }
}

# Set the environment variable for the evaluator
$promptFile = "wilds_prompt.txt"
$env:OPENEVOLVE_PROMPT = (Resolve-Path $promptFile).Path

Write-Host "Starting WILDS Amazon prompt evolution..." -ForegroundColor Green
Write-Host "Experiment: Office_Products sentiment classification" -ForegroundColor Cyan
Write-Host "Prompt file: $promptFile" -ForegroundColor Cyan
Write-Host "Iterations: $iterations" -ForegroundColor Cyan
Write-Host ""

# Get the path to openevolve-run.py
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$openevolveDir = Split-Path (Split-Path (Split-Path $scriptDir -Parent) -Parent) -Parent
$openevolveScript = Join-Path $openevolveDir "openevolve-run.py"

if (-not (Test-Path $openevolveScript)) {
    Write-Host "Error: Could not find openevolve-run.py at $openevolveScript" -ForegroundColor Red
    exit 1
}

Write-Host "Found openevolve-run.py at: $openevolveScript" -ForegroundColor Gray

# Run OpenEvolve
Write-Host "Running OpenEvolve..." -ForegroundColor Yellow
python $openevolveScript $promptFile evaluator.py --config config.yaml --iterations $iterations

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Evolution completed successfully!" -ForegroundColor Green
    Write-Host "Results are in: openevolve_output/" -ForegroundColor Cyan
    Write-Host "Evolution trace: openevolve_output/evolution_trace.jsonl" -ForegroundColor Cyan
    Write-Host "Best prompt: openevolve_output/best/best_program.txt" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "  1. Run: python visualize_evolution.py" -ForegroundColor White
    Write-Host "  2. Run: python analyze_improvements.py" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "Evolution failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}


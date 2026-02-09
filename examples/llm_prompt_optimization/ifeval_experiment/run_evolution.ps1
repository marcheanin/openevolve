# PowerShell script to run OpenEvolve for IFEval prompt evolution
# Usage: .\run_evolution.ps1 [-iterations N] or .\run_evolution.ps1 --iterations N

# Default iterations
$iterations = 100

# Parse command line arguments to support both -iterations and --iterations formats
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
$promptFile = "ifeval_prompt.txt"
$env:OPENEVOLVE_PROMPT = (Resolve-Path $promptFile).Path

Write-Host "Starting IFEval prompt evolution..." -ForegroundColor Green
Write-Host "Prompt file: $promptFile" -ForegroundColor Cyan
Write-Host "Iterations: $iterations" -ForegroundColor Cyan
Write-Host ""

# Get the path to openevolve-run.py (relative to this script)
# From ifeval_experiment: ../../../openevolve/openevolve-run.py
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
# Go up 3 levels: hotpotqa_experiment -> llm_prompt_optimization -> examples -> openevolve
$openevolveDir = Split-Path (Split-Path (Split-Path $scriptDir -Parent) -Parent) -Parent
$openevolveScript = Join-Path $openevolveDir "openevolve-run.py"

if (-not (Test-Path $openevolveScript)) {
    Write-Host "Error: Could not find openevolve-run.py at $openevolveScript" -ForegroundColor Red
    Write-Host "Expected path: openevolve/openevolve-run.py" -ForegroundColor Yellow
    Write-Host "From ifeval_experiment: ../../../openevolve/openevolve-run.py" -ForegroundColor Yellow
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
} else {
    Write-Host ""
    Write-Host "Evolution failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}


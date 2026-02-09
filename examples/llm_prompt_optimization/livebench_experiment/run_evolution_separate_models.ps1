# PowerShell script to run OpenEvolve for LiveBench IF with SEPARATE models
# Usage: .\run_evolution_separate_models.ps1 [-iterations N]

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
$promptFile = "livebench_prompt.txt"
$env:OPENEVOLVE_PROMPT = (Resolve-Path $promptFile).Path

Write-Host "Starting LiveBench IF prompt evolution (SEPARATE MODELS)..." -ForegroundColor Green
Write-Host "Prompt file: $promptFile" -ForegroundColor Cyan
Write-Host "Iterations: $iterations" -ForegroundColor Cyan
Write-Host "Config: config_separate_models.yaml" -ForegroundColor Cyan
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

# Run OpenEvolve with separate models config
Write-Host "Running OpenEvolve with separate models..." -ForegroundColor Yellow
python $openevolveScript $promptFile evaluator_separate_models.py --config config_separate_models.yaml --iterations $iterations

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Evolution completed successfully!" -ForegroundColor Green
    Write-Host "Results are in: openevolve_output/" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Run visualization:" -ForegroundColor Yellow
    Write-Host "  python visualize_evolution.py" -ForegroundColor White
    Write-Host "  python analyze_improvements.py" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "Evolution failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}


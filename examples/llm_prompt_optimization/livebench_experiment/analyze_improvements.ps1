# PowerShell script to analyze improvements from evolution trace
# Usage: .\analyze_improvements.ps1

Write-Host "Analyzing LiveBench IF prompt improvements..." -ForegroundColor Green
Write-Host ""

# Check if evolution trace exists
$traceFile = "openevolve_output/evolution_trace.jsonl"
if (-not (Test-Path $traceFile)) {
    Write-Host "Error: Evolution trace not found at $traceFile" -ForegroundColor Red
    Write-Host "Please run the evolution first using run_evolution.ps1" -ForegroundColor Yellow
    exit 1
}

# Run visualization script
Write-Host "Generating visualizations..." -ForegroundColor Yellow
python visualize_evolution.py

Write-Host ""
Write-Host "Analyzing improvements..." -ForegroundColor Yellow
python analyze_improvements.py

Write-Host ""
Write-Host "Analysis complete!" -ForegroundColor Green
Write-Host "Results saved to: openevolve_output/visualizations/" -ForegroundColor Cyan


# PowerShell script to analyze prompt improvements
# Usage: .\analyze_improvements.ps1

Write-Host "Analyzing prompt improvements from evolution trace..." -ForegroundColor Green
Write-Host ""

# Check if evolution has been run
$tracePath = "openevolve_output/evolution_trace.jsonl"
if (-not (Test-Path $tracePath)) {
    Write-Host "Error: Evolution trace not found at $tracePath" -ForegroundColor Red
    Write-Host "Run the evolution first: .\run_evolution.ps1" -ForegroundColor Yellow
    exit 1
}

# Step 1: Visualize evolution curves
Write-Host "Step 1: Generating learning curves..." -ForegroundColor Cyan
python visualize_evolution.py

# Step 2: Analyze improvements
Write-Host ""
Write-Host "Step 2: Analyzing prompt improvements..." -ForegroundColor Cyan
python analyze_improvements.py

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Analysis completed successfully!" -ForegroundColor Green
    Write-Host "Results are in: openevolve_output/" -ForegroundColor Cyan
    Write-Host "  - learning_curves.png - Evolution learning curves" -ForegroundColor Cyan
    Write-Host "  - evolution_summary.txt - Summary statistics" -ForegroundColor Cyan
    Write-Host "  - visualizations/improvements_analysis.txt - Full analysis report" -ForegroundColor Cyan
    Write-Host "  - visualizations/improvements/ - Individual improvement files" -ForegroundColor Cyan
    Write-Host "  - visualizations/improvements_summary.json - JSON summary" -ForegroundColor Cyan
    Write-Host "  - best/best_program.txt - Best evolved prompt" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "Analysis failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}


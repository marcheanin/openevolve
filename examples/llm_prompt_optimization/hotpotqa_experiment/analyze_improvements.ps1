# PowerShell script to analyze prompt improvements
# Usage: .\analyze_improvements.ps1

Write-Host "Analyzing prompt improvements from evolution trace..." -ForegroundColor Green
Write-Host ""

# Run the analysis script
python analyze_improvements.py

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Analysis completed successfully!" -ForegroundColor Green
    Write-Host "Results are in: openevolve_output/visualizations/" -ForegroundColor Cyan
    Write-Host "  - improvements_analysis.txt - Full analysis report" -ForegroundColor Cyan
    Write-Host "  - improvements/ - Individual improvement files" -ForegroundColor Cyan
    Write-Host "  - improvements_summary.json - JSON summary" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "Analysis failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}


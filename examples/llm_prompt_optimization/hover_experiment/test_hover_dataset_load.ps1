Param(
    [string]$Split = "train",
    [string]$CachePath = "$PSScriptRoot\hover_dataset_cache",
    [switch]$NoCache,
    [switch]$ForceHF,
    [switch]$Streaming,
    [int]$Sample = 3
)

Write-Host "Testing HoVer dataset loading..." -ForegroundColor Cyan

Write-Host "Environment versions:"
python -c "import sys; print(f'python={sys.version.split()[0]}')"
python -c "import importlib, sys; 
try:
 import datasets; print(f'datasets={datasets.__version__}')
except Exception as e:
 print(f'datasets=NOT INSTALLED ({e})')
try:
 import pyarrow; print(f'pyarrow={pyarrow.__version__}')
except Exception as e:
 print(f'pyarrow=NOT INSTALLED ({e})')
"

$argsList = @("--split", $Split, "--cache-path", $CachePath, "--sample", $Sample)
if ($NoCache) { $argsList += "--no-cache" }
if ($ForceHF) { $argsList += "--force-hf" }
if ($Streaming) { $argsList += "--streaming" }

Write-Host "Running: python test_hover_dataset_load.py $($argsList -join ' ')" -ForegroundColor Yellow
python "$PSScriptRoot\test_hover_dataset_load.py" @argsList
$exitCode = $LASTEXITCODE

if ($exitCode -eq 0) {
    Write-Host "HoVer dataset load: SUCCESS" -ForegroundColor Green
} else {
    Write-Host "HoVer dataset load: FAILED (exit code $exitCode)" -ForegroundColor Red
    Write-Host "Hints:" -ForegroundColor DarkYellow
    Write-Host "  - Ensure compatible versions: pip install `"datasets==2.14.0`" `"pyarrow<15.0`""
    Write-Host "  - Or create a local cache and place it at: $CachePath"
}

exit $exitCode



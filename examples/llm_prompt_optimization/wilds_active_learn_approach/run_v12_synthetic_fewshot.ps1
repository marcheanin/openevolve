param(
  [int]$NAL = 7,
  [int]$NEvolve = 20,
  [string]$ResultsDir = "results_v12_synth"
)

$ErrorActionPreference = "Stop"

Write-Host "Starting v12 synthetic-fewshot experiment..."
Write-Host "AL cycles: $NAL | Evolve iterations: $NEvolve | Results: $ResultsDir"

python active_loop.py --n-al $NAL --n-evolve $NEvolve --results-dir $ResultsDir

Write-Host "Done. Results in: $ResultsDir"


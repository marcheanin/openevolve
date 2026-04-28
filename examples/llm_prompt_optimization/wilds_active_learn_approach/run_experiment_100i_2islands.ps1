# 100 evolution iterations = 2 AL cycles x 50; 2 MAP-Elites islands; pool refresh each cycle; consolidation after each cycle.
# Requires: OPENROUTER_API_KEY (or OPENAI_API_KEY) in environment / .env

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

python .\active_loop.py `
  --config .\config_experiment_100i_2islands.yaml `
  --prompt .\initial_prompt.txt `
  --results-dir .\results_experiment_100i_2islands `
  --n-al 2 `
  --n-evolve 50 `
  --no-evolve-early-stop

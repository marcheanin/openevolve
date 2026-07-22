# Experiment template

## Hypothesis
Describe what this experiment tests.

## Command
```bash
cd prime_v2_group_robust
python -m prime.cli --config experiments/_template/config.yaml --smoke
```

## Expected artifacts
- `results/<experiment>/seed<N>_<timestamp>/config_used.yaml`
- `results/.../run_metadata.json` (git hash, seed)
- `results/.../events.jsonl`
- `results/.../al_iter_*/active_batch.json`, `best_prompt.txt`
- `results/.../summary.json` (final test metrics)

## How to read results
Compare `CVaR_cluster`, `R_worst`, and `R_global` in `summary.json` and per-cycle `val_metrics` in `events.jsonl`.

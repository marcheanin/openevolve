# PRIME v2 — Implementation Guide

Modular Python package for **group-robust prompt evolution** under WILDS OOD shifts. See [README.md](README.md) for research context.

## Package layout

```
prime_v2_group_robust/
  prime/
    config.py              # YAML → typed dataclasses, includes merge
    controller.py          # Thin AL orchestration loop
    cli.py                 # python -m prime.cli --config ...
    data/
      wilds_loader.py      # Official WILDS Amazon train/val/test (user-disjoint)
      cache.py             # Pickle + in-memory caches (splits, embeddings)
      profiles.py          # User style profiles
      clustering.py        # k-means clusters, serialization
    workers/ensemble.py    # LLM workers, majority vote, parallel predict
    acquisition/         # scoring policies, batch builder, seen/unseen pool
    fitness/               # metrics (R_worst, CVaR), configurable objective
    evolution/             # OpenEvolve adapter + error artifacts
    consolidation/         # pool carryover between cycles
    experiment/            # run context, JSONL logs, bootstrap stats
  configs/                 # Shared base YAML fragments
  experiments/             # Per-experiment configs + README (config beside results)
  scripts/                 # build_clusters.py, aggregate_results.py
  tests/
```

## Code style

- `from __future__ import annotations`, type hints, `@dataclass` for configs/results.
- Pure logic (scoring, CVaR, metrics) has no I/O; loaders and `experiment/` handle files.
- No magic numbers in fitness/acquisition — all weights in YAML (`FitnessCfg`, `AcquisitionCfg`).
- Determinism: `active_learning.seed` logged in `run_metadata.json`.
- Modules target <~300 lines; controller only orchestrates.

## Configuration

Load a single experiment YAML (with optional `includes:`):

```yaml
includes:
  - ../../configs/base.yaml
experiment:
  name: my_run
fitness:
  mode: cvar
  w_cvar: 0.7
```

```python
from pathlib import Path
from prime.config import load_config
cfg = load_config(Path("experiments/E1_pilot_cvar_vs_global/config_cvar.yaml"))
cfg.validate()
```

## Data flow (one AL cycle)

1. `wilds_loader.load_amazon_splits` — official `get_subset('train'|'val'|'test')`, user-disjoint checks.
   - **Disk cache**: `{data_root}/.prime_cache/wilds_amazon_{split}_cat{...}.pkl` (raw extraction, no caps).
   - **Memory cache**: keyed by caps + seed for the lifetime of the Python process.
2. `clustering.fit_style_clusters` — train user profiles → k-means; val/test mapped to centroids.
   - Embedding arrays cached under `.prime_cache/embeddings_*.pkl`.
3. `acquisition.pool.AcquisitionPool` — seen/unseen, hard/anchor, expansion.
4. `acquisition.batch_builder.build_active_batch` — policy score → top-k + diversity + group quotas.
5. `evolution.openevolve_adapter.run_evolution` — inner loop (or mock in smoke).
6. `fitness.objective.compute_fitness` — `a*CVaR + b*global + c*kappa - length`.
7. `consolidation.pool_carryover` — top-k prompts into next cycle.
8. `experiment.run_context.RunContext` — snapshots config + git hash into `results/<exp>/<run>/`.

## How to add an experiment

1. Copy `experiments/_template/` → `experiments/MyExp/`.
2. Edit `config.yaml` (hypothesis-specific fitness/acquisition arms).
3. Write `README.md` (hypothesis, command, artifacts).
4. Run: `python -m prime.cli --config experiments/MyExp/config.yaml`
5. Results: `results/<experiment.name>/seed<N>_<timestamp>/`

## How to add an acquisition policy

1. Add policy name to `AcquisitionCfg` validation in `config.py`.
2. Implement branch in `acquisition/scoring.py::score_examples`.
3. Add unit test in `tests/test_scoring.py`.

## How to add a dataset

1. Implement `prime/data/<name>_loader.py` returning `Dict[str, ReviewSplit]`.
2. Add `configs/dataset_<name>.yaml`.
3. Branch in `controller.py::_load_data`.

## How to add a fitness mode

1. Extend `FitnessCfg.mode` validation.
2. Implement in `fitness/objective.py::compute_fitness`.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/build_clusters.py` | Offline cluster artifacts → `data/clusters.json` |
| `scripts/aggregate_results.py` | Mean±std + bootstrap CI tables across seeds |

## Tests

```bash
cd prime_v2_group_robust
pip install -r requirements.txt
python -m pytest tests/ -q
```

## Caching

| Layer | Location | What is cached |
|-------|----------|----------------|
| Raw WILDS split | `{data_root}/.prime_cache/*.pkl` | texts/labels/user_ids per official split (pre-cap) |
| Capped splits | in-process dict | final `ReviewSplit` after user caps |
| Embeddings | `{data_root}/.prime_cache/embeddings_*.pkl` | sentence-transformer outputs |

Disable with `dataset.use_cache: false`. Custom path: `dataset.cache_dir: /path/to/cache`.

First run parses WILDS CSV (~2–3 min); subsequent runs load pickles instantly (same as v1).

## Smoke vs full runs

- **Smoke** (`experiment.smoke: true` or `--smoke`): capped examples, mock ensemble if no API key, synthetic data fallback if WILDS unavailable.
- **Full**: requires `OPENROUTER_API_KEY`, WILDS Amazon download, optional OpenEvolve config path for real inner-loop.

## E1 pilot (go/no-go)

Compare `experiments/E1_pilot_cvar_vs_global/config_global.yaml` vs `config_cvar.yaml` on `R_worst` and `CVaR_cluster` in `summary.json`.

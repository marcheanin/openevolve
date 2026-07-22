# Full uncapped test comparison

**Eval:** Amazon-WILDS all categories, **34 533** test reviews (uncapped).

Sources: `full_uncapped_test_metrics.json` per run (`results_evoprompt_full_seed42/`, `results_opro_full/`, `results_apo_full/`); per-worker rows from `results_all_categories_evolve_subsample/fulltest_al_iter_5/predict_progress/predict_progress_complete.json` (same PRIME prompt, AL iter 5).

| Experiment | Models | R_worst | R_global | combined | mae | mean_kappa |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **PRIME best prompt (AL iter 5)** | 3-model ensemble | 55.91% | **75.41%** | **0.838** | **0.256** | **0.882** |
| PRIME best prompt + **GPT-4o-mini only** | single | **56.80%** | 74.40% | 0.748 | 0.267 | — |
| PRIME best prompt + **Gemini 2.5 Flash only** | single | 52.99% | 75.01% | 0.739 | 0.262 | — |
| PRIME best prompt + **Claude 3.5 Haiku only** | single | 53.25% | 74.24% | 0.736 | 0.269 | — |
| **EvoPrompt GA** (seed 42, evolved prompt) | single GPT-4o-mini | 56.50% | 73.92% | 0.744 | 0.277 | — |
| **APO / ProTeGi** (seed 42, val-selected prompt) | single GPT-4o-mini | 52.90% | 72.60% | 0.727 | 0.291 | — |
| **OPRO** (seed 42, val-selected prompt) | single GPT-4o-mini | 50.98% | 72.05% | 0.719 | 0.300 | — |
| **All-categories plain evolved** † | 3-model ensemble | 54.44% | 72.11% | 0.819 | 0.290 | 0.888 |
| **LISA baseline** (Amazon WILDS) | DistilBERT | 54.70% | 71.30% | n/a | n/a | n/a |
| **Baseline ensemble** (initial prompt) | 3-model ensemble | 50.00% | 71.30% | 0.782 | 0.308 | 0.697 |
| **Zero-shot single GPT-4o-mini** (initial prompt) | single | 53.18% | 71.86% | 0.724 | 0.302 | — |

† Row label from earlier notes; metrics match `results_all_categories_uncapped_train/fulltest_al_iter_5/` (PRIME uncapped train, not plain evolution without AL).

Per-worker details: `results_all_categories_evolve_subsample/fulltest_al_iter_5/per_worker_vs_ensemble.md`

## Reference: Amazon WILDS baselines (OOD)

| Method | 10th-percentile accuracy (OOD) |
| --- | ---: |
| ERM | 53.8% |
| CORAL | 52.9% |
| IRM | 52.4% |
| GroupDRO | 53.3% |

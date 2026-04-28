# Last 4 experiments — full test comparison

Sources:
- `results_single_gpt4o_mini/full_uncapped_test_metrics.json`
- `results_all_categories_fulltest_baseline_ensemble/full_uncapped_test_metrics.json`
- `results_all_categories_evolve_subsample/fulltest_al_iter_5/full_uncapped_test_metrics.json`
- LISA (Amazon WILDS): user-provided `R_worst=54.7%` (other metrics not provided)

| Experiment | R_worst | R_global | combined_score | mae | mean_kappa |
|---|---:|---:|---:|---:|---:|
| **All-categories evolved prompt (AL iter 5 best prompt)** | 55.91% | 75.41% | 0.8383 | 0.2562 | 0.8815 |
| **LISA baseline on Amazon WILDS (reported)** | 54.70% | n/a | n/a | n/a | n/a |
| **All-categories baseline ensemble prompt** | 50.00% | 71.30% | 0.7819 | 0.3076 | 0.6973 |
| **Single model: GPT‑4o‑mini prompt** | 53.18% | 71.86% | 0.7243 | 0.3019 | 0.0000 |

## Reference: Amazon WILDS baselines (OOD)

| Метод    | 10th percentile accuracy (OOD) |
| -------- | ------------------------------ |
| ERM      | 53.8%                          |
| CORAL    | 52.9%                          |
| IRM      | 52.4%                          |
| GroupDRO | 53.3%                          |


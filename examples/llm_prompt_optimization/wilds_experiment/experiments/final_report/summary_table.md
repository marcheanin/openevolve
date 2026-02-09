# Final Experiment Report: WILDS Amazon Sentiment Classification

Generated: 2026-01-28 10:47:58

## Summary Table (Unified Metrics)

| Experiment | Setup | R_global | R_worst | MAE | Combined (Unified) | Old Combined | Kappa (κ) | Users | Examples |
|------------|-------|----------|---------|-----|-------------------|--------------|-----------|-------|----------|
| Exp 1: Single Model Baseline | 1 model (gpt-oss-120b), no evolution | 51.02% | 31.03% | 0.704 | 0.544 | 0.492 | N/A | 102 | 1566 |
| Exp 2: Single Model + Evolution | 1 model (gpt-oss-120b), with OpenEvolve | 54.61% | 34.55% | 0.640 | 0.574 | 0.529 | N/A | 25 | 542 |
| Exp 3a: Ensemble Baseline | 3 models (yandexgpt, gemma3-27b, gpt-oss-120b), majority vote, no evolution | 65.13% | 44.01% | 0.362 | 0.717 | 0.334 | 0.528 | 25 | 542 |
| Exp 3b: Ensemble + Evolution | 3 models (yandexgpt, gemma3-27b, gpt-oss-120b), majority vote, with OpenEvolve | 68.27% | 47.69% | 0.338 | 0.736 | 0.373 | 0.470 | 25 | 542 |

### Unified Formula Explanation

The **Combined (Unified)** score uses a unified formula for fair comparison:

**Base formula (for all experiments):**
```
base_score = 0.4 * R_global + 0.3 * R_worst + 0.3 * (1 - MAE/4)
```

**For ensemble experiments, adds consistency bonus (Cohen's Kappa):**
```
kappa_score = max(0, mean_kappa)
consistency_bonus = 0.1 * kappa_score
combined_score = base_score + consistency_bonus
```

κ uses Landis & Koch scale. This ensures comparable scores and rewards real inter-annotator agreement (kappa), not just low disagreement.
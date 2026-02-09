# Final Experiment Report: WILDS Amazon Sentiment Classification

Generated: 2026-01-21 10:57:30

## Summary Table (Unified Metrics)

| Experiment | Setup | R_global | R_worst | MAE | Combined (Unified) | Old Combined | Kappa (κ) | Users | Examples |
|------------|-------|----------|---------|-----|-------------------|--------------|-----------|-------|----------|
| Exp 1: Single Model Baseline | 1 model (gpt-oss-120b), no evolution | 50.19% | 30.77% | 0.721 | 0.539 | 0.484 | N/A | 102 | 1566 |
| Exp 2: Single Model + Evolution | 1 model (gpt-oss-120b), with OpenEvolve | 58.67% | 35.74% | 0.539 | 0.601 | 0.571 | N/A | 25 | 542 |
| Exp 3a: Ensemble Baseline | 3 models (yandexgpt, gemma3-27b, gpt-oss-120b), majority vote, no evolution | 65.50% | 44.01% | 0.360 | 0.717 | 0.339 | 0.511 | 25 | 542 |
| Exp 3b: Ensemble + Evolution | 3 models (yandexgpt, gemma3-27b, gpt-oss-120b), majority vote, with OpenEvolve | 67.53% | 47.09% | 0.343 | 0.723 | 0.383 | 0.432 | 25 | 542 |

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

κ uses Landis & Koch scale. This ensures comparable scores and rewards real inter-annotator agreement (kappa).
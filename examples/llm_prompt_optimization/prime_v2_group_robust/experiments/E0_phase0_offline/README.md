# Phase 0 — Offline bounds (no API)

Prerequisite before another live Amazon run. Answers three go/no-go questions
using only cached predictions from
`results/E1_pred_profile_cvar_lowvar/seed42_20260730_114314` (30 prompts × 360
D_select examples) plus noise repeats from the earlier cvar pair run.

## Run

```bash
python scripts/phase0_offline_bounds.py \
  --run-dir results/E1_pred_profile_cvar_lowvar/seed42_20260730_114314 \
  --noise-dir results/E1_pred_profile_cvar_lex/seed42_20260729_213313/exp_A_predictions
```

Outputs land in `<run>/analysis/phase0/` (`PHASE0_REPORT.md`, `phase0_report.json`).

## Questions

| ID | Question | Go threshold |
|----|----------|--------------|
| E0a | Does a cluster-routed portfolio beat the best single prompt? | LOO Δ ≥ +5 pp (optimistic D_select) |
| E0b | Does family-DRO rank differently *and* safely vs R_global? | Spearman < 0.9 **and** DRO-top R_global ≈ best R_global |
| E0c | Is there a demotion-Pareto zone that lifts 4★ without killing 5★? | ≥1 candidate in good zone |

## Locked decisions (see report)

- Portfolio on Amazon user-shift: **DROP** (LOO −3.6 pp; oracle fit only +2.2 pp).
- DRO / CVaR_bal in fitness: **REJECT** (different ranking ≠ better heir; both pick demotion).
- Amazon user-shift as paper headline: **NO** (empty good zone; use as negative case).
- Phase 1 constraint live: **GO** (mechanics check only).

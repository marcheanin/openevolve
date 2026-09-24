# Phase 0 findings (2026-08-01)

Offline analysis of 30 prompts cached during
`E1_pred_profile_cvar_lowvar/seed42_20260730_114314`, scored on the same
360-example D_select (median ensemble). Noise SDs from the earlier pair run's
`exp_A_predictions` (240-ex repeats). Full tables in `PHASE0_REPORT.md`.

## Headline

**On Amazon user-shift, the lowvar candidate pool contains no transferable
robustness improvement under any of the three proposed method pillars.**

1. **Portfolio (E0a):** best single prompt on raw R_global is already `initial`
   (0.714). Cluster-specialist LOO is **−3.6 pp**; oracle (in-sample) fit only
   **+2.2 pp**. Below the +5 pp optimistic go threshold → drop portfolio on this
   shift.
2. **Family-DRO fitness (E0b):** ranks differently from R_global (Spearman 0.12)
   but its top pick scores R_global 0.667 vs best 0.714 — same demotion trap as
   class-balanced CVaR. DRO_family_mean correlates 0.98 with R_global (redundant).
   Keep robustness as a **constraint**, not a DRO scalar.
3. **Demotion Pareto (E0c):** 0/30 candidates in the good or weak zone
   (lift 4★ without large 5★ damage). corr(demotion, R_global)=−0.95,
   corr(demotion, acc_5)=−1.0.

## Implication for the paper method

| Pillar | On Amazon user-shift | Next action |
|--------|----------------------|-------------|
| Certified non-regression gate | Still needed (caught lowvar C1) | Phase 1 live: `reject` + `fitness=global` |
| Portfolio of specialists | No headroom here | Deprioritize until CivilComments/category-shift |
| DRO-in-fitness | Unsafe heir selection | Constraint only |
| Headline claim | Not this split | CivilComments-WILDS / Amazon category-shift |

## Phase 1 expectation

Mechanics check only: final prompt must not significantly regress vs initial on
240-user paired test. Large worst-group gains are **not** expected.

# S9 execution status

**Updated:** 2026-08-12 ~13:40 — matrix complete; **S9 demoted to exploratory**
after the power audit and the R15/R16 controls. Live protocol is
[`PREREGISTRATION_S10.md`](PREREGISTRATION_S10.md).

## Done

| Item | Status |
|------|--------|
| Seed42–44 baselines | **done** |
| PRIME 42/43/44 matrix attach | **done** |
| Same-session stable ×3 (31/31) | **done** |
| `RESULTS.md` mean±sd + method setup | **done** |
| Power & confound audit ([`POWER_AUDIT.md`](POWER_AUDIT.md)) | **done** |
| `test_fixed_large` n=5251, superset of the 1800 | **done** |
| R15 calibration control (12 one-line edits) | **done** |
| R16 selection control + S9-pool replication | **done** |
| Selection-statistic ablation (k, τ, shrink) | **done** |
| `d_dev_targeted` + dev-budget reallocation (F9) | **done** |
| S10 preregistration locked (selection amended to targeted dev) | **done** |

Artifacts: `results/E5_s9_matrix/stable_session/` (+ `stable_aggregate.json`),
`results/E5_selection_control/{strictness_sweep,s9_pool,dev_targeted}/`.  
Table: [`RESULTS.md`](RESULTS.md) (exploratory).  
Findings: [`R15_CALIBRATION_CONTROL.md`](R15_CALIBRATION_CONTROL.md).

## Open

- Re-select each method from its own archive on `d_dev_targeted` — F9 says the
  matrix numbers are partly a dev-allocation artifact.
- Allocation ablation (2×225 / 3×150 / 4×112 / 9×50) to find the
  precision-vs-coverage turnover, and to test the blind-spot risk at the
  concentrated end.
- Second validation draw, to rule out a split-specific artifact.

## Interrupted (archived)

- `seed44_20260811_100905` — reboot mid C2; superseded by `…110051`
- first stable batch (per-method seed×3) — killed; complete jobs kept
- intentional pause 2026-08-11 ~18:30 — resumed 2026-08-12

# Phase 2b diagnostics - CivilComments seed

**Decision: `GO`**

Sample: n=180 equal-group val comments, repeats=3, seed=42.
Ensemble: google/gemma-3-12b-it, qwen/qwen3.7-flash, google/gemma-4-26b-a4b-it.

## Go / no-go

- worst-group gap 0.350 >= 0.05
- residual group effect resid_spread=0.350 (R²=-0.03)
- identity worst 'black' (0.550) << none (0.700)
- identity share of error mass ~ 89%
- prompt-only headroom (1 - oracle_workers)~0.111; agg-only~+0.189

- Global (weighted) acc: **0.700**
- Best group: `male` 0.900 (n=20)
- Worst group: `black` 0.550 (n=20)
- Gap: **0.350**

## Per-group profile (repeat 0)

| group | n | acc | gold tox | pred tox |
|---|---:|---:|---:|---:|
| none | 20 | 0.700 | 0.50 | 0.20 |
| male | 20 | 0.900 | 0.50 | 0.50 |
| female | 20 | 0.800 | 0.50 | 0.30 |
| LGBTQ | 20 | 0.600 | 0.50 | 0.20 |
| christian | 20 | 0.750 | 0.50 | 0.25 |
| muslim | 20 | 0.600 | 0.50 | 0.30 |
| other_religions | 20 | 0.750 | 0.20 | 0.05 |
| black | 20 | 0.550 | 0.50 | 0.35 |
| white | 20 | 0.650 | 0.50 | 0.15 |

## Noise floor (M15)

| metric | mean | SD | range |
|---|---:|---:|---:|
| CVaR_cluster | 0.5667 | 0.0167 | 0.0333 |
| CVaR_cluster_shrunk | 0.6519 | 0.0093 | 0.0185 |
| R_global | 0.6944 | 0.0056 | 0.0111 |
| R_global_ex | 0.6944 | 0.0056 | 0.0111 |
| R_worst | 0.0000 | 0.0000 | 0.0000 |
| R_worst_group | 0.5500 | 0.0000 | 0.0000 |
| mean_kappa | 0.1659 | 0.0042 | 0.0074 |

Prediction flips vs repeat 0: [6, 5] (['3.3%', '2.8%'])

## Aggregation headroom

- Ensemble 0.700; best single 0.733; oracle-over-workers 0.889
- Agg-only headroom +0.189
- Errors: 54 (some worker right 34, all wrong 20)

## Class-mix residual (C10-style)

- R2=-0.033; obs_spread=0.350; resid_spread=0.350

## Power analysis (M13)

Bootstrap SD at n=180: R_global=0.0345 (MDE~0.097); R_worst_group=0.0799 (MDE~0.224).

Examples needed for 80% power on R_global (scale bootstrap SD ~ 1/sqrt(n); target effects):

- effect 0.03: ~1870 examples
- effect 0.05: ~673 examples
- effect 0.08: ~263 examples

Recommended D_select / test caps for E4: **404** / **673** (target MDE 0.05 on R_global; worst-group needs more).


# Results — `E2_category_shift_books/seed42_20260802_052137`

Amazon **category-shift** (train Books / eval exclude Books) with the same cheap ensemble + `global_tail_mix` as E2 user-OOD.  
Wall ~3.4 h. Exit 0. Seed checkpoints under `run_dir/seed_c*` (O27 fix).

> Contaminated parallel attempt `seed42_20260801_205605` is invalid — ignore.

## Protocol

| Role | Filter |
|------|--------|
| Train / AL / D_select / D_anchor | `train_category_id: 0` (Books) |
| Val + Test | `eval_exclude_category_ids: [0]` |
| Caps after filter | train 2351 / val 1052 / test **1742** (240 users) |

Fitness / gate / OE / workers: identical to `E2_cheap_ensemble_global_tail`.

## Headline

**Small positive move on non-Books test — borderline.**

| Metric | Seed | Final | Δ | boot CI95 | p |
|--------|-----:|------:|--:|-----------|--:|
| R_global (ex-level) | 0.656 | 0.665 | **+0.009** | — | — |
| R_global (user boot) | 0.662 | 0.671 | **+0.009** | [+0.000, +0.017] | **0.045** |
| R_tail (q=0.2) | 0.329 | 0.339 | +0.010 | [−0.005, +0.026] | 0.25 |
| R_worst (p10) | 0.375 | 0.375 | 0 | — | — |
| CVaR_q40 | 0.559 | 0.575 | +0.016 | — | — |
| MAE | 0.373 | 0.361 | better | — | — |

McNemar: seed-only 26 / final-only **41**, p=0.086 (n.s. at 0.05).

Opposite sign vs user-OOD M22 (there −0.027***). Here the cheap-ensemble + tail mix does **not** demote; gain is small and sits near the noise floor (MDE≈0.041 for 80% power).

## In-loop

| Cycle | best fitness | notes |
|------:|-------------:|-------|
| 1 | 0.5095 | OE `changed=True` but best ≈ seed text |
| 2 | 0.5145 | longer prompt (→4312 chars) |
| 3 | 0.5145 | frozen at C2 champion |

## Cost

Workers ~6.0–7.1M tokens each → **~$2.3–2.6** (same stack as user-OOD). Stray kimi/235b lines again ~0.23M.

## Interpretation

1. Category-shift is a **more interesting** substrate than user-OOD for this method: seed is weaker (0.656), direction of evolution positive.
2. `global_tail_mix` is still not a clear win — R_tail lift n.s.; mean gain barely crosses p=0.05 on user bootstrap.
3. Do not overclaim; next step if pursuing mix: lex `global` floor + maximize `R_tail`, or oracle category groups in the constraint.

Artifacts: `analysis/seed_vs_final.json`, `analysis/paired.txt`.

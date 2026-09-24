# E6 Amazon category-shift — Preregistration (controls + top-3)

Locked 2026-08-17 **before** scoring E6 finals / top-3 methods.

## Substrate

- WILDS Amazon, **Books → non-Books** (`train_category_id=0`, `eval_exclude_category_ids=[0]`).
- Scorer: single `openai/gpt-4o-mini`, T=0.
- Groups: `pred_profile` K=6 fitted on train seed-preds (label-free). Do not claim style robustness.

## Fixed sets

| Set | Target | Design |
|-----|--------|--------|
| `test_fixed` | ~1800–2400 | cluster × collapsed-rating `{1–2, 3, 4–5}`, fingerprinted |
| `d_dev` | ~900 | same design on validation |
| `d_dev_targeted` | 900 | same n, concentrate on k=3 worst clusters by **seed on uniform d_dev** |

## Metrics

- **Primary:** CVaR@25% over **macro-per-class accuracy within each pred_profile cluster**.
- **Always report:** `R_global`, `R_tail`, mean macro, `|pred_mean − gold_mean|` (operating-point proxy).
- **Selection rules (controls):** `mean_macro`, `worst_class` (min over 5 classes), `cvar25`, `hard_min`, `softmin_shrunk`, `R_tail`.

## Phase A — controls (confirm F1–F9)

Harshness pool (R15 analogue): 12 one-line edits to seed DynamicRules.  
Optimizer pool: existing E2 category + E1 global heirs + seed (no re-evolution).  
Replication success: same **sign pattern** as CivilComments on ≥4 of {F1, F3, F5, F9}.

## Phase B — top-3 from CivilComments S9

Methods (by S9 mean `R_worst_gba`): **GPO**, **EvoPrompt-DE**, **PRIME-main**.  
Seeds: 42 (primary); 43/44 if budget allows.  
Selection: Top-1 on uniform `d_dev` by primary metric (or method-native then report primary).  
Report: 1 repeat on `test_fixed`; paired bootstrap vs seed; Holm across 3 methods.  
Mandatory baseline in table: harshness-pool best under same dev rule (R15 analogue).

## Out of scope

Full S9 clone; user-OOD as primary; inventing new fitness for Amazon wins.

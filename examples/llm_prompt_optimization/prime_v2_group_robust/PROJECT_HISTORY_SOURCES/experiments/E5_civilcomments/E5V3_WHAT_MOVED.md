# E5v3 — what moved (and what did not)

Run: `results/E5_civilcomments_prime_main_v3/seed42_20260809_193731`

## Goal of v3
Ship the mutator-feedback freeze (M37/O31): gen1 must see `PRIMARY TARGETS` + FP/FN `POLARITY` after few-shot inject.

## Verified
- Smoke + main: `mutator_artifacts_frozen.txt` written every cycle (C1–C4).
- OE / seed checkpoints carry `artifacts_json.error_examples` with `## PRIMARY TARGETS` and `POLARITY` (see `analysis/mutator_primary_targets_check.json`).
- Observation **M39**.

## Shipped headline
- `final_prompt.txt` **equals seed** (1488 chars).
- `test_fixed` raw worst-GBA **0.620** (seed S8 same-day ~0.643; E5v2 shipped 0.645).
- `best_evo_score` rose to **0.7099** on D_select (C2–C4), but `val_selection_key` stayed at seed’s D_dev softmin **0.6865** every cycle → D_dev Top-1 never accepted an OE heir.
- `stable_final` is `null` (post-run stable eval did not persist; likely API contention with parallel S8).

## Interpretation
Mutator wiring is fixed; **selection still vetoes evolution** (same family as M32). Do not credit v3 for a test lift — there is none vs seed. Next lever is selection / F8 diagnostics on why 0.7099 D_select heirs lose on D_dev, not more mutator text.

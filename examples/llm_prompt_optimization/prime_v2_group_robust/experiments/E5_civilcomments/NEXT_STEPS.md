# Phase 3 — next steps (after CivilComments S9 core)

Locked after S9 preregistration. Do **not** promote these to headline until separately measured.

## 1. OPRO (R13)

- Port [`wilds_active_learn_approach/baselines/opro_original_style.py`](../../../wilds_active_learn_approach/baselines/opro_original_style.py) onto `baselines.api.Task`.
- Same Scorer / F7 / D_dev Top-1 / `test_fixed`.
- Run CivilComments Regime-Shift × seeds 42/43/44.
- Native-ish: meta-prompt with score trajectory; budget-matched scorer calls.

## 2. MIPROv2 (R14)

- DSPy MIPROv2 adapter wrapping our Scorer as metric.
- Binary toxicity signature; `{review}` → Label 0/1.
- Prefer seed42 first if budget tight; then 43/44.
- Install `dspy` in the experiment env; pin version in RESULTS.

## 3. Amazon WILDS (§7.2)

- Ordinal within-group macro-per-class accuracy (not binary GBA).
- Fixed sets + harness reuse of S9 runner.
- Reduced matrix: `{Seed, APE, APO, GPO, PRIME, Random-AL} × 3` (+ OPRO/MIPROv2 if ready).
- Sanity: methods indistinguishable ⇒ measurement stack OK; spurious winners ⇒ revisit metric.

## 4. Final table

- Single paper/RESULTS table: CivilComments full R0–R12 (+ R13/R14) and Amazon reduced.
- Same-session stable eval footnote; Yelp→Flipkart gate status; bootstrap CI + Holm.
- Separate **preregistered** vs **exploratory** columns (E5v3 debt, A1–A5, gemma-4 transfer).

## Order

`OPRO → MIPROv2 → Amazon fixed sets → Amazon matrix → unified RESULTS / paper table`

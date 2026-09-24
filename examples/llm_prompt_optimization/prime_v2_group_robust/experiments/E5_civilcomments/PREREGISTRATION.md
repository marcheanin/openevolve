# E5 CivilComments — Preregistration (Phase 3 / S9)

> **Superseded for headline claims by [`PREREGISTRATION_S10.md`](PREREGISTRATION_S10.md)
> (2026-08-12).** S9 is demoted to exploratory: the primary metric and budget were
> revised after its results were seen ([`POWER_AUDIT.md`](POWER_AUDIT.md)). This
> file is kept unmodified as the record of what was actually locked.

Locked before S9 matrix claims. Source: `ROADMAP_PHASE3.md` §7.3.

## Headline

- **Primary metric:** `R_worst_gba` on fingerprinted `test_fixed` (n=1800).
- **Selection:** Top-1 by `R_soft_min_gba` on `D_dev` (n=900). Methods may also report their native ensemble protocol; headline remains Top-1.
- **Scorer:** `google/gemma-3-12b-it`, T=0. Same parser / F7 prompt contract for all methods.
- **Optimizer LLM (mutator):** `deepseek/deepseek-v4-pro` for all prompt-generation methods.

## Regime

Regime-Shift (§6.1): `S_source=360` labeled easy groups; `U_target=4000` unlabeled hard groups; label budget `L=240` for AL methods; `test_fixed` never seen during optimization.

## Methods in matrix (R0–R12)

| ID | Method | Seeds |
|----|--------|-------|
| R0 | Seed prompt | 42/43/44 |
| R1 | ALL-ZERO / ALL-ONE (offline floors) | — |
| R2 | APE (K=6, N=36) | 42/43/44 |
| R3 | APE budget-matched (K=48) | 42/43/44 |
| R4 | APO/ProTeGi (6 rounds, beam=4) | 42/43/44 |
| R5 | APE-ut | 42/43/44 |
| R6 | GPO (K=6, T=0.83) | 42/43/44 |
| R7 | EvoPrompt-GA | 42/43/44 |
| R8 | EvoPrompt-DE | 42/43/44 |
| R9 | GEPA | 42/43/44 |
| R10 | PRIME-main | 42/43/44 |
| R11 | Random-AL (APO + 240 random target labels) | 42/43/44 |
| R12 | Oracle upper bound (APO on fully labeled target) | **42 only** |

**R10 protocol (this matrix):** seed **42** uses the shipped E5v2 artifact
`results/E5_civilcomments_prime_main_v2/seed42_20260809_080918` (not E5v3).
Seeds **43/44** are new PRIME-main runs on `config_prime_main.yaml` (same v2-era selection
protocol: D_dev Top-1). E5v3 mutator / selection debug is **exploratory**, not R10.

OPRO (R13) and MIPROv2 (R14) are **out of this preregistration**; they are next-step work.

## Statistics

- Primary contrast: method vs seed on `R_worst_gba`, paired **bootstrap over examples**
  (10 000 iters, stratified by identity×label cells) → CI95 on Δ.
- Secondary: McNemar on `R_global`.
- Aggregate across seeds: mean ± SD; count of seeds with Δ>0.
- Multiplicity: Holm over number of primary method-vs-seed contrasts.
- Final measurements: same-session re-score of seed + all finals, `stable_final_repeats=3`
  (M29/M38).

## Success / failure criteria (PRIME)

- **Success:** mean `ΔR_worst_gba ≥ +0.05` vs seed, CI95 excludes 0, **and**
  `≥ +0.02` vs the best of {APO, GPO, Random-AL}.
- **Honest failure:** if PRIME ≤ Random-AL on headline, record as M-entry in OBSERVATIONS
  and revise the scientific claim (not the metric).

## Gate prerequisite

S7 Yelp→Flipkart reimplementation gate must pass before interpreting S9 foreign-method
wins: GPO > APE on Flipkart target; APO > APE on Yelp source (K=6, N=36, T=0.83).

## Exploratory (not headline)

- E5v3 selection-bug diagnosis (M40).
- A1–A5 ablations, gemma-4 transfer, Amazon §7.2.
- OPRO / MIPROv2 until separately preregistered.

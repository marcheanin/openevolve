# E5v3 selection debt (M40 follow-up) — diagnosis

**Not a D_dev rejection bug.** Seed shipped because within-cycle heir pick never left the incumbent.

## Evidence (run `seed42_20260809_193731`)

1. `best_evo_score ≈ 0.71` on C2–C4 is the **seeded gen0 program** inside OE (byte-identical to seed MD5), not an evolved win.
2. All four `dev_gate.json`: `accepted: true`, `reason: candidate_is_champion`, `n_rejected: 0`. F8 never compared seed vs OE.
3. `smoke_trace` / `_pick_heir`: `incumbent_is_front_best_no_gate_needed` every cycle.
4. Near-miss OE `3512d7dc`: raw softmin **↑** (~0.713 vs ~0.700) but length penalty (~0.017 at 367 words, `len_penalty_start=300`) drops **fitness** below seed (0.703 vs 0.710) → never becomes heir → never reaches D_dev.

## Root cause

Heir ranking uses **length-penalized `fitness`**, so softmin lifts that grow the prompt lose before F8. Reporting island-best (gen0 seed) as `best_evo_score` falsely looks like “evolution to 0.71 then D_dev veto.”

## Minimal fix (for a later PRIME patch, not R10 blocker)

1. Rank `_pick_heir` on `R_soft_min_gba` / `base_score`, length as secondary lex key (`prime/controller.py`).
2. Report `best_mutant_score` excluding gen0 / cycle-entry (`openevolve_adapter.py`).
3. Optional: monitor-mode score top OE-by-softmin on D_dev even when incumbent wins.

R10 in S9 uses **E5v2** artifact per `PREREGISTRATION.md`. This debt is exploratory.

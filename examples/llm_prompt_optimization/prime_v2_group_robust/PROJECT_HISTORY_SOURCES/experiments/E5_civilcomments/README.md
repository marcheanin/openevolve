# E5 CivilComments — Phase 3

See `../../ROADMAP_PHASE3.md` for the full plan.

## Status

**E5v2 live result (seed42_20260809_080918):** first shipped non-seed prompt under Phase3;
raw worst-GBA on `test_fixed` **0.615 → 0.645 (+3.0 pp)** vs seed. Decisive fix = **D_dev
selection** (v1 search worked but full-val Top-1 kept seed). Full write-up:
[`E5V2_WHAT_MOVED_EVOLUTION.md`](E5V2_WHAT_MOVED_EVOLUTION.md). Observations **M32–M37**, **O31**.

**Landed:**
- F1–F4 fail-closed, GBA, soft_min_lex
- F5 Scorer + **F6 wired** into AL batch / expansion (self-consistency uncertainty)
- F7 prompt contract
- **F8 D_dev gate** (`dev_gate_mode=reject`) + `generalization_gap` in summary
- **F9** `test_fixed` / `D_dev` builders (`scripts/build_e5_fixed_sets.py`)
- **F10** `budget_report.json` with `scorer_calls` / `optimizer_calls` (parent undercount: O30)
- Selection Top-1 on **D_dev**; final eval on **test_fixed**
- APE / APO / GPO / APE-ut / EvoPrompt / GEPA / Random-AL / Oracle bodies
- S8 pilot + **S9 matrix runner** (`scripts/run_e5_s9_matrix.py`)
- **PREREGISTRATION.md**; **NEXT_STEPS.md** (OPRO, MIPROv2, Amazon)
- E5v3 selection debt diagnosed: [`E5V3_SELECTION_DEBT.md`](E5V3_SELECTION_DEBT.md) (M41)

**Still open / in flight:**
- Finish S9 3-seed matrix + same-session stable eval
- §6.4 Yelp→Flipkart gate artifact (`gpo_gate.json`) must show pass
- Fix budget charging into OE child (O30)
- Next: OPRO, MIPROv2, Amazon §7.2, final table

## Commands

```bash
# Fixed sets + ALL-ZERO floor check
python scripts/build_e5_fixed_sets.py

# Same-session seed vs candidate on D_dev + test_fixed
python scripts/compare_prompts_fixed.py \
  --seed-prompt prompts/initial_prompt_civilcomments.txt \
  --candidate-prompt results/.../al_iter_2/best_prompt.txt \
  --out-dir results/.../compare_c2_vs_seed

# Sensitivity gate (S5)
python scripts/e5_prompt_sensitivity.py

# PRIME main
python scripts/run_e5_prime_main.py

# S8 pilot (seed + APE)
python scripts/run_e5_s8_pilot.py

# S7 Yelp→Flipkart gate
python scripts/run_gpo_yelp_flipkart_gate.py

# S9 matrix (one seed / method subset)
python scripts/run_e5_s9_matrix.py --seed 42 --methods ape_ut,ape_k48,gepa
python scripts/run_e5_s9_matrix.py --seed 42 --methods prime --prime-run results/E5_civilcomments_prime_main_v2/seed42_20260809_080918

# Stable batch after matrix prompts exist
python scripts/run_e5_s9_stable_batch.py --seeds 42,43,44 --repeats 3
```

## Headline

`R_worst_gba` on fingerprinted `test_fixed` (1800). Selection / gate on `D_dev` (900).
See [`RESULTS.md`](RESULTS.md) and [`PREREGISTRATION.md`](PREREGISTRATION.md).

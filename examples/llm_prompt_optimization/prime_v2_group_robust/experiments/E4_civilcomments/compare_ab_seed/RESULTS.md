# E4 A/B vs seed (test n=800)

Completed runs only. Arm C interrupted after cycle 1 (no `summary.json` / no test).

Runs:
- seed baseline: `experiments/E4_civilcomments/baseline_initial_prompt_test/`
- A: `results/E4_civilcomments_arm_a_global/seed42_20260805_104411`
- B: `results/E4_civilcomments_arm_b_min_group/seed42_20260805_135157`
- C (aborted mid C1→C2): `results/E4_civilcomments_arm_c_style/seed42_20260805_172744`

## Headline

| arm | fitness | R_global | R_macro | R_worst_group | Acc tox | CVaR | kappa |
|---|---|---:|---:|---:|---:|---:|---:|
| seed | — | 0.8638 | 0.6630 | 0.6364 | 0.410 | 0.674 | 0.187 |
| A_global | global | 0.8662 | 0.6644 | 0.5909 | 0.410 | 0.701 | 0.184 |
| B_min_group_lex | min_group_lex | 0.8712 | 0.6725 | 0.6818 | 0.422 | 0.732 | 0.194 |

### Δ vs seed

| arm | ΔR_global | ΔR_macro | ΔR_worst_group | ΔAcc tox |
|---|---:|---:|---:|---:|
| A_global | +0.0025 | +0.0014 | -0.0455 | +0.000 |
| B_min_group_lex | +0.0075 | +0.0095 | +0.0455 | +0.012 |

### Paired flips vs seed (same 800 examples)

| pair | seed→wrong / wrong→right | net | McNemar one-sided p |
|---|---:|---:|---:|
| A vs seed | 6 / 8 | +2 | 0.40 |
| B vs seed | 3 / 9 | +6 | **0.073** |
| B vs A | 3 / 7 | +4 | 0.17 |

B's global lift is **borderline / likely noise** at n=800 (need ~more seeds or larger test for claim).

## Per-oracle-group (test)

| group | seed | A | B | B−seed | B−A |
|---|---:|---:|---:|---:|---:|
| none | 0.881 | 0.883 | 0.885 | +0.004 | +0.002 |
| male | 0.884 | 0.860 | 0.872 | -0.012 | +0.012 |
| female | 0.812 | 0.828 | 0.828 | +0.016 | +0.000 |
| LGBTQ | 0.842 | 0.842 | 0.842 | +0.000 | +0.000 |
| christian | 0.938 | 0.954 | 0.954 | +0.015 | +0.000 |
| muslim | 0.750 | 0.786 | 0.786 | +0.036 | +0.000 |
| other_religions | 0.636 | 0.727 | 0.727 | +0.091 | +0.000 |
| black | 0.842 | 0.842 | 0.842 | +0.000 | +0.000 |
| white | 0.636 | 0.591 | 0.682 | +0.045 | +0.091 |

Group nets seed→B are mostly **±1 example** on tiny cells (`white` n=22, `other_religions` n=11). One flip moves R_worst_group by ~4–9 pp.

## Prompt forensics (important)

- Seed prompt hash `99066a101f5a917e` = B `initial_prompt.txt`.
- B selected heir (`al_iter_2/best_prompt.txt`) is **strip-identical to seed** (only trailing whitespace / 1488→1487).
- B evolution never beat cycle-entry lex fitness on C1 (OE 0.674 < entry 0.683); C2 "OE best 0.683" is recovering the same prompt after D_select rotation made entry look like 0.601.
- A did accept a different OE prompt (C2); that is the one that **hurt** `white` (0.636→0.591).

So B did not discover a better prompt text — it **kept the seed** against worse mutants. The test ΔB−seed is re-scoring noise / stochastic ensemble, not a content win.

## Arm A cycles

| c | entry fitness | OE best | heir | val key | gate |
|---:|---:|---:|---|---:|---|
| 1 | 0.8690 | 0.8833 | cycle_entry@0.8690 | 0.6967 | REJ:significant_regression drop=0.15 |
| 2 | 0.8619 | 0.8690 | oe@0.8690 | 0.6990 | OK:within_tolerance drop=0.01 |
| 3 | 0.8786 | 0.9000 | oe@0.9000 | 0.6709 | OK:tolerated_noise drop=0.02 |

Gate: 4 evals, 2 rejects (`significant_regression`×2). Correctly killed majority-collapse 0.883.

## Arm B cycles

| c | entry fitness | OE best | heir | val key | gate |
|---:|---:|---:|---|---:|---|
| 1 | 0.6828 | 0.6739 | cycle_entry@0.6828 | 0.6486 | no better cand |
| 2 | 0.6005 | 0.6828 | oe@0.6828 (~seed) | 0.6757 | OK:within_tolerance drop=0.01 |
| 3 | 0.6512 | 0.6828 | carried@0.6828 | 0.6757 | carried |

Gate barely fired (1 accept). Lex + rotate D_select: entry fitness swings a lot; selection recovers seed-level prompt.

## Arm C (interrupted)

Stopped after C1 consolidation: style K≈4–6, entry lex fitness 0.660, OE best 0.645 → heir=`cycle_entry`. No test. Needs full restart (or resume if checkpoint wiring exists).

## Verdict

1. **Fitness choice (directional):** `global` (A) is the wrong objective here — can select prompts that worsen identity worst-group vs seed. `min_group_lex` (B) at least **did not regress** and blocked worse OE kids.
2. **Not yet a method win:** B ≈ seed on prompt content; test +4.5 pp R_worst_group is **not** a clean content gain (small n, p≈0.07, strip-same prompt).
3. **Budget too lean** (3×3) for OE to move lex above seed on this seed prompt — need longer B (or richer seed-fail) before claiming lex finds better rules.
4. **Next:** finish Arm C for inferred-vs-oracle; if repeating B, raise `n_evolve_iterations` / cycles rather than trusting this Δ.

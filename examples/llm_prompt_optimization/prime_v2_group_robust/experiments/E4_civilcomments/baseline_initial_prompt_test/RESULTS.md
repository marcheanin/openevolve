# E4 baseline: initial prompt on official test

Same test slice as Arm A final (`n=800`, seed42 cap), same 3-worker ensemble.
Prompt: `prompts/initial_prompt_civilcomments.txt` (hash `99066a101f5a917e`).

Artifacts: this directory + mirror under
`results/E4_civilcomments_arm_a_global/seed42_20260805_104411/evals/initial_prompt/`.

## Ensemble (test)

| metric | value |
|---|---:|
| R_global | **0.8638** |
| R_macro | 0.6630 |
| R_worst_group | **0.6364** (`other_religions` / `white`, both 0.636) |
| Acc toxic / non-toxic | 0.410 / 0.916 |
| CVaR_cluster | 0.674 |
| mean_kappa | 0.187 |

## vs Arm A final (cycle-2 heir)

| metric | seed | Arm A final | Δ |
|---|---:|---:|---:|
| R_global | 0.8638 | 0.8663 | +0.0025 |
| R_macro | 0.6630 | 0.6644 | +0.0014 |
| R_worst_group | 0.6364 | 0.5909 | **−0.045** |
| Acc toxic | 0.410 | 0.410 | 0 |

Arm A global fitness did not beat seed beyond noise on global/macro; worst-group got slightly worse.

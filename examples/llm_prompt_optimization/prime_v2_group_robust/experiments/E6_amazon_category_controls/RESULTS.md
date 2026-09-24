# E6 Amazon category-shift — RESULTS

Date: 2026-08-17. Scorer: `openai/gpt-4o-mini`. Groups: pred_profile K=6.  
Primary: **CVaR@25% of macro-per-class within cluster**.

## Fixed sets

| Set | n | fingerprint |
|-----|--:|------------|
| test_fixed | 1376 | 9bab5002e8c1 |
| d_dev | 648 | 862cced55882 |
| d_dev_targeted | 570 | dac1f521e4ea (clusters 4,0,5) |

Cell sparsity vs plan (~1800/900): caps raised to 400/400/900 users; still ~17/18 cells nonempty.

## Phase A — controls

### Harshness / selection (15 candidates)

Oracle on test: **`deflate_positive`** (cvar25=0.458).  
Most d_dev rules pick **`lenient_max`** → test cvar25=0.431 (regret ≈0.027).

Spearman(dev rule → test cvar25):

| rule | ρ |
|------|--:|
| hard_min | **+0.40** |
| cvar25 | +0.26 |
| R_tail | +0.15 |
| softmin_shrunk | +0.10 |
| worst_class | +0.09 |
| mean_macro | −0.01 |
| R_global | **−0.16** |

Unlike CivilComments, **hard_min is not anti-correlated** here; **R_global is**.

Existing optimizer heirs (no re-evo): E2 category heir test cvar25=0.451; E1 global=0.419; seed≈0.422.

### Power (paired bootstrap vs seed_anchor)

**3/13** global contrasts resolvable at 95% (file: `selection_control/power_audit.json`). Same “few resolvable” sign as CC F1/F3.

### F9 targeted d_dev (rule=`hard_min`)

| | pick | test cvar25 | regret vs oracle |
|--|------|------------:|-----------------:|
| uniform d_dev | lenient_max | 0.419* | **0.025** |
| targeted d_dev | strict_5 | 0.437* | **0.007** |
| Δ regret | | | **−0.018** |

\*F9 re-scored (T=0 still has small run-to-run noise vs selection_control table).  
**Same sign as CivilComments F9:** concentrating budget on seed-worst clusters cuts selection regret.

## Phase B — top-3 (CivilComments S9 order)

| Method | test cvar25 | R_global | op_shift | status |
|--------|------------:|---------:|---------:|--------|
| seed | 0.432 | 0.579 | 0.055 | done |
| **GPO** | **0.458** | 0.576 | 0.110 | done |
| EvoPrompt-DE | — | — | — | **blocked**: OpenRouter key limit mid-run (gen 6/8) |
| PRIME | — | — | — | evolution **finished** (`seed42_20260817_103306`); **test_fixed score pending** (same key limit) |

Harshness-pool mandatory baseline (same table): `deflate_positive` ≈ **0.458** — ties GPO on primary.

PRIME run: 3 AL cycles; early-stop OE each cycle; `final_prompt.txt` archived to `results/E6_top3_matrix/seed42/prime/best_prompt.txt`.  
Bugfix during run: ordinal `pred_pos_rate` degenerate guard was rejecting all fitness (−1e9); skipped for non-binary gold in `prime/fitness/objective.py`.

## Replication checklist vs CivilComments

| Finding | CC | Amazon E6 |
|---------|----|-----------|
| Few resolvable paired contrasts | yes | **yes** (3/13) |
| Min-based selection anti-corr | yes | **no** (hard_min ρ=+0.40) |
| R_global poor for robust pick | yes | **yes** (ρ=−0.16) |
| F9 targeted reduces regret | yes | **yes** (0.025→0.007) |
| Harshness/edit competitive with optimizers | yes | **yes** (deflate ≈ GPO) |

## Resume after key top-up

```powershell
cd prime_v2_group_robust
python scripts/run_e6_top3_matrix.py --methods evoprompt_de --seed 42
python scripts/run_e6_top3_matrix.py --methods prime --prime-run results/E6_amazon_category_controls/seed42_20260817_103306 --seed 42
```

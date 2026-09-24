# E6 Amazon category-shift — replication log

Status: **Phase A complete; Phase B partial** (2026-08-17).  
Blocked: OpenRouter **key limit exceeded** mid EvoPrompt-DE / PRIME test scoring.

## Locked substrate

- Books → non-Books (`train_category_id=0`, `eval_exclude_category_ids=[0]`)
- Scorer: `openai/gpt-4o-mini` (single, fail-closed)
- Groups: `pred_profile` K=6
- Primary: CVaR@25% of macro-per-class within cluster

## Artifacts

| Piece | Path |
|-------|------|
| Config | `experiments/E6_amazon_category_controls/config.yaml` |
| Prereg | `experiments/E6_amazon_category_controls/PREREGISTRATION.md` |
| Results | `experiments/E6_amazon_category_controls/RESULTS.md` |
| Harshness pool | `experiments/E6_amazon_category_controls/pools/harshness_sweep/` |
| Fixed sets | `experiments/E6_amazon_category_controls/fixed_sets/` |
| Selection control | `results/E6_amazon_category_controls/selection_control/` |
| F9 realloc | `results/E6_amazon_category_controls/f9_realloc/` |
| Top-3 matrix | `results/E6_top3_matrix/seed42/` |
| PRIME live | `results/E6_amazon_category_controls/seed42_20260817_103306/` |

## Phase A checklist (sign pattern vs CivilComments)

- [x] F1/F3: few resolvable paired contrasts (3/13)
- [ ] F5: min-based anti-correlated — **not replicated** (hard_min ρ=+0.40)
- [x] F9: targeted d_dev reduces regret (0.025→0.007)
- [x] Harshness-pool competitive with optimizer heirs / GPO

## Phase B

- [x] seed (cvar25=0.432)
- [x] GPO (cvar25=0.458)
- [ ] EvoPrompt-DE — resume after key top-up
- [ ] PRIME test_fixed score — prompt ready; score after key top-up
- [x] harshness best ≈ GPO on primary

## Resume

```powershell
python scripts/run_e6_top3_matrix.py --methods evoprompt_de --seed 42
python scripts/run_e6_top3_matrix.py --methods prime --prime-run results/E6_amazon_category_controls/seed42_20260817_103306 --seed 42
```

# E4 — CivilComments Phase 2c (live)

Group-robust prompt evolution on CivilComments-WILDS. Hypothesis: identity worst-groups
are recognizable / expressible / non-conflicting (ROADMAP_PHASE2 §1).

## Arms

| Arm | Config | Fitness | Groups |
|---|---|---|---|
| A (control) | `config_arm_a_global.yaml` | `global` + reject gate | oracle (diag only) |
| B lean | `config_arm_b_min_group.yaml` | `min_group_lex` + reject | oracle |
| B post-fix (M28) | `config_arm_b_min_group_2x20.yaml` | hard min, 2×20, D_select=720, **no rotate** | oracle |
| **B next (M30)** | `config_arm_b_soft_min_2x20.yaml` | **soft_min_lex** τ=0.08, 2×20, D_select=720, **rotate on**, length penalty | oracle |
| C (transfer) | `config_arm_c_style.yaml` | `min_group_lex` + reject | style (inferred) |

Caps: lean D_select≈420; post-fix B D_select=720 (~80/group); test≈800. See M26–M30.

## Launch

```bash
# Lean A/B/C (legacy)
python scripts/run_e4_civilcomments.py --arm all --dry-run
python scripts/run_e4_civilcomments.py --arm ab

# Post-fix hard-min B (M27/M28) — completed; regressed on test (M30)
python scripts/run_e4_b_2x20.py --dry-run

# Next live: soft_min + rotate + length (M30)
python scripts/run_e4_b_soft_min_2x20.py --dry-run
python scripts/run_e4_b_soft_min_2x20.py          # + same-day test noise vs seed
```

Same-day re-score of all table prompts:

```bash
python scripts/e4_rescore_all_prompts.py --force
```

Offline (~0 API):

```bash
python scripts/phase2c_offline_bounds.py
pytest tests/test_soft_min_lex.py tests/test_m27_stale_seed_metrics.py tests/test_phase2a_fixes.py -q
```

## Headline metrics

On CivilComments use **`R_worst_group`** (min Acc over identity/style groups), not
`R_worst` (user p10 — collapses when each comment is a synthetic user).
Compare arms **same-day paired** (M29); use `scripts/e4_test_noise.py`.

## Status

- Deep forensics: `deep_analysis_20260806.md`
- A lean + B lean + B 4×20 + B 2×20 completed; B 2×20 validated M27, lost test (M30)
- Next live: B soft_min 2×20 → (if signal) larger test / 3 seeds; Arm C frozen until then
- Pointers: `last_E4_civilcomments_arm_*.json`, `baseline_initial_prompt_test/`, `compare_all_test/`

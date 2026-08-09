# E2 — Cheap ensemble + `global_tail_mix` (Amazon user-OOD)

## Hypothesis

On official WILDS Amazon user-OOD, with a **weaker cheap ensemble** (more prompt headroom than Phase 1’s strong trio), maximizing

```text
fitness = 0.5·R_global + 0.5·R_tail(D_select) − length
```

under the Phase 1 skeleton (`anchor_gate=reject`, consolidation off, 15×3) improves test `R_global` and/or `R_tail` vs the seed prompt without demotion.

`R_tail` = mean accuracy over the worst 20% of users (continuous; not p10 `R_worst`).

## Diff vs Phase 1

| Knob | Phase 1 | This run |
|------|---------|----------|
| `fitness.mode` | `global` | **`global_tail_mix`** |
| Workers | deepseek-v4 / kimi / qwen-235b | **4o-mini / flash-lite / qwen3-32b** |
| Mutator | glm-5 | **deepseek-v4-pro** |
| OE budget | 8×3 | **15×3** |
| Cluster pin | Phase0 artifact | **refit** |

## Command

```bash
cd prime_v2_group_robust
python scripts/run_e1_wilds_live.py --config experiments/E2_cheap_ensemble_global_tail/config.yaml
```

## Success criteria

- Process exits 0; `summary.json` present.
- Paired seed→final on test: no significant demotion (McNemar / boot CI).
- Wake-up (optional): Δ R_global or R_tail clearly above noise.

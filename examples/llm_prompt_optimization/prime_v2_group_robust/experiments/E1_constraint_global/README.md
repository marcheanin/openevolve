# E1 Phase 1 — Constraint + raw global fitness (mechanics check)

Live Amazon-WILDS run that validates **one claim** from Phase 0:

> With robustness as a **reject gate** (not a CVaR fitness) and raw `R_global`
> as the objective, the search must not produce a significant test regression
> vs the initial prompt.

This is **not** the paper headline. Phase 0 showed Amazon user-shift has no
bidirectional-calibration good zone and no portfolio headroom; headline moves
to CivilComments / category-shift after this mechanics check passes.

## Diff vs lowvar

| Knob | lowvar (failed) | this run |
|------|-----------------|----------|
| `fitness.mode` | `cvar_lex` + `class_balanced` | `global` |
| `anchor_gate_mode` | `monitor` | **`reject`** |
| consolidation | every 2 cycles | **disabled** |
| selection key | balanced/shrunk CVaR | R_global / shrunk CVaR monitor only |
| D_select | 360 fixed | 360 (same size; rotation deferred) |

## Success criteria

1. Paired test vs initial: R_global Δ ≥ −0.02 (not a significant regression).
2. Anchor gate fires on demotion candidates (rejection_rate > 0 if such candidates appear).
3. `best_selection_cycle` may be 1 — acceptable if final ≈ initial (non-harm).

Failure (significant R_global drop again) → problem is deeper than objective
misspecification; stop Amazon live and move to CivilComments scaffolding.

## Launch

```bash
python scripts/run_e1_wilds_live.py --config experiments/E1_constraint_global/config.yaml
```

Dry-run preflight:

```bash
python scripts/run_e1_wilds_live.py --config experiments/E1_constraint_global/config.yaml --dry-run
```

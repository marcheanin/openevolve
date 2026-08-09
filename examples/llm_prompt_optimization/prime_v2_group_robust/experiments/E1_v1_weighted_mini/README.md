# E1 v1-weighted mini — head-to-head vs Phase 1

Same Amazon-WILDS OOD substrate as `E1_constraint_global` (workers, mutator,
caps, seed, pinned clusters, reject gate, consolidation off), but OE fitness is
authentic **v1 Hard/Anchor**:

```
fitness = 0.5·Acc_Hard + 0.3·Acc_Anchor + 0.2·κ − length_penalty
```

evaluated on the active batch (not D_select).

## Diff vs Phase 1

| Knob | Phase 1 constraint | this mini |
|------|--------------------|-----------|
| `fitness.mode` | `global` on D_select | **`v1_weighted` on Hard/Anchor batch** |
| OE iters | 8 | **4** (~½ OE budget) |
| AL cycles | 3 | 3 |
| gate / consolidation | reject / off | same |
| synth few-shot | none | none (second step if this wakes up) |

## Fair comparison protocol

1. Same initial prompt, same test 240 users.
2. Paired Δ vs seed-prompt test ensemble (cluster_assign_test).
3. Compare to `results/E1_constraint_global/seed42_20260801_002853`.
4. In-loop scores are **not** comparable across arms (batch vs D_select) — only the final paired test is.

## Launch

```bash
# Optional: reuse Phase 1 cluster-assign caches to skip ~40 min of seed eval
$env:PRIME_SEED_PRED_CACHE = "results/E1_constraint_global/seed42_20260801_002853/pred_cache"

python scripts/run_e1_wilds_live.py --config experiments/E1_v1_weighted_mini/config.yaml --dry-run
python scripts/run_e1_wilds_live.py --config experiments/E1_v1_weighted_mini/config.yaml
```

## Wake-up criteria

- `best_evo` / Acc_Hard rises across cycles (search not frozen), **or**
- paired test Δ R_global clearly above Phase 1's noise (+0.004 n.s. band), **or**
- prompt grows real 4↔5 rules (qualitative).

If still flat → problem is OOD/workers/seed headroom, not CVaR vs Hard/Anchor.

## Result (`seed42_20260801_061506`)

**No wake-up.** Test R_global 0.722→0.724 (+0.002 n.s.) vs Phase 1's 0.726.
Final prompt = seed. C1: all mutations lost on-batch. C2/C3 OE contaminated by
stale Pareto-seed metrics (O26). Full write-up: [`RESULTS.md`](RESULTS.md).

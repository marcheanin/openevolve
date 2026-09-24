# S9 execution status

**Resumed:** 2026-08-11. PRIME43 done (seed ship); PRIME44 live.

## Done

| Item | Status |
|------|--------|
| Seed42 R0–R12 (PRIME-E5v2) | **done** |
| Seed43/44 baselines | **done** |
| PRIME seed43 live `seed43_20260811_074014` | **done** (final==seed; E5v3 selection debt) |
| Matrix attach seed43 prime | **done** |

### Seed43 PRIME (matrix-comparable score)

| metric | value |
|--------|------:|
| R_worst_gba | 0.643 |
| R_worst_group | 0.620 |
| R_global | 0.702 |

(= seed; selection never left incumbent. Internal `final_test` path reported different numbers — use matrix attach for table.)

## In flight

1. **PRIME seed44** `seed44_20260811_100905`
2. Attach seed44 → matrix
3. `run_e5_s9_stable_batch.py --seeds 42,43,44 --repeats 3`
4. Refresh RESULTS.md / canvas

## Resume

```powershell
$env:PYTHONIOENCODING='utf-8'
python scripts/run_e5_prime_main.py --config experiments/E5_civilcomments/config_prime_main_seed44.yaml
# after finish:
python scripts/run_e5_s9_matrix.py --seed 44 --methods prime --prime-run results/E5_civilcomments_prime_main_s44/seed44_YYYYMMDD_HHMMSS
python scripts/run_e5_s9_stable_batch.py --seeds 42,43,44 --repeats 3
```

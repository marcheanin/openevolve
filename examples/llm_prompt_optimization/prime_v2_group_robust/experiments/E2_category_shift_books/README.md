# E2 — Amazon category-shift (Books → non-Books)

## Hypothesis

Training / AL / D_select on **Books** (WILDS category id 0) and evaluating on **all other categories** gives a domain shift where prompt evolution + `global_tail_mix` can move test metrics more than on user-OOD alone.

Same cheap ensemble and fitness as `E2_cheap_ensemble_global_tail`.

## Protocol

| Role | Filter |
|------|--------|
| Train / AL pool / D_select / D_anchor | `train_category_id: 0` (Books) |
| Val + Test | `eval_exclude_category_ids: [0]` |

Clusters refit on Books train seed-preds. Headline: paired Δ on non-Books test vs seed.

## Command

```bash
cd prime_v2_group_robust
python scripts/run_e1_wilds_live.py --config experiments/E2_category_shift_books/config.yaml
```

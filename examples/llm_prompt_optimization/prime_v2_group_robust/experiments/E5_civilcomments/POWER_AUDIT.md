# E5 S9 — power & confound audit

**Date:** 2026-08-12. Computed from the cached `stable_session` predictions
(no new scoring). Reproduce with:

```powershell
python scripts/e5_metric_variance_audit.py --n-boot 1500
```

Raw output: `results/E5_s9_matrix/stable_session/metric_variance_audit.json`.

## Verdict

The S9 matrix cannot distinguish any method from the seed prompt, and the
headline metric is the main reason. Three findings, in order of how much they
change what we do next.

## F1 — no contrast survives, and the metric is the bottleneck

Paired bootstrap over `test_fixed` examples (resampled within `group×label`
cells, both prompts see the same rows, majority-of-3 predictions):

- **0 of 31** method×seed contrasts are resolvable at 95% under hard-min GBA.
- Best observed: GPO seed42, Δ = +0.035, CI95 `[-0.005, +0.075]`.
- Under CVaR@25% the best becomes GPO seed42 Δ = +0.0375, CI95 `[+0.003, +0.062]`
  — nominally non-zero but **p_Holm = 0.28** across the 11 methods.

Ranks invert across matrix seeds (best method is `gpo` / `evoprompt_ga` /
`random_al` on 42 / 43 / 44), which is what a pure-noise ordering looks like.

Sanity check that confirms it: **oracle** (APO with the target fully labeled,
an upper bound) scores 0.615 vs seed 0.620.

## F2 — repeats were the wrong place to spend budget

Spread of the metric over 3 repeats of the *same* prompt vs the paired
bootstrap sd over examples:

| metric | repeat sd | bootstrap sd | ratio |
|--------|----------:|-------------:|------:|
| hard-min GBA | 0.0014 | ~0.0185 | ~13× |
| CVaR@25% | 0.0008 | ~0.0131 | ~16× |

Scorer nondeterminism contributes **~1/250 of the variance** (ratio squared).
`stable_final_repeats=3` therefore bought almost nothing, while costing 3×.

Relative CI width and the implied effective-data multiplier vs hard-min:

| metric | mean CI width | effective data | resolved at 95% |
|--------|--------------:|---------------:|----------------:|
| hard-min GBA | 0.0724 | 1.00× | 0/31 |
| **CVaR@25%** | **0.0514** | **1.98×** | 1/31 |
| CVaR@50% | 0.0363 | 3.97× | 4/31 |
| softmin (τ=0.1) | 0.0304 | 5.66× | 6/31 |
| mean GBA | 0.0249 | 8.42× | 7/31 |
| worst-class acc | 0.0434 | 2.78× | 24/31 |

**Free win:** at constant scorer cost, `1 repeat × 5400 examples` with CVaR@25%
gives a CI half-width of ≈0.015, versus ≈0.036 today (`3 repeats × 1800`,
hard-min). Keep 2 repeats only as a scorer-drift guard on the seed prompt.

Budget to resolve a given effect under CVaR@25%, single repeat:

| target Δ | needed n | note |
|---------:|---------:|------|
| 0.030 | ~3,000 | below current 3-repeat cost |
| 0.020 | ~6,700 | ≈ current cost |
| 0.015 | ~12,000 | ~2× current cost, finalists only |

## F3 — the dominant axis is the operating point, not group structure

Across all 34 evaluated prompts (seed + 11 methods × 3 seeds):

- `corr(CVaR@25%, mean GBA)` = **+0.83**; regressing CVaR@25% on mean GBA gives
  **R² = 0.687**, and adding `|recall − specificity|` raises it by **0.000**.
- `corr(CVaR@25%, |recall − specificity|)` = **−0.67**.
- The bottom of the ranking is uniformly recall-skewed: `ape@42`, `random_al@42`,
  `apo@{42,43}` sit at `|recall − spec|` = 0.30–0.32 (recall ≈ 0.85, spec ≈ 0.53).
  The top is balanced or specificity-leaning: `evoprompt_de@44` 0.003,
  `prime@44` 0.017, `gpo@42` 0.084.

**Interpretation.** On this benchmark, worst-group score is almost entirely a
function of average quality, and average quality is largely a function of where
the prompt puts the toxicity threshold. Prompt optimizers here are mostly moving
the operating point; the failing ones over-flag. No method demonstrably closes a
*group gap* beyond what its mean improvement predicts.

**Required new baseline (R15, calibration control).** Sweep only the strictness
wording of the seed prompt to trace an operating-point curve, select the point by
the same dev rule as every other method, and report it in the matrix. If R15
matches PRIME/GPO, the group machinery is not what is producing the gains, and
that must be stated.

> **Run, and it does match.** See [`R15_CALIBRATION_CONTROL.md`](R15_CALIBRATION_CONTROL.md):
> 12 one-line edits place 4th of 35 on identical rows, no S9 cell beats them at
> Holm < 0.05, and the choice of *dev selection rule* alone moves the outcome by
> 0.030 — with the worst-group rules turning out to be anti-correlated with the
> test worst-group score.

## Metric decisions taken

- **Primary:** CVaR@25% over the 8 identity GBAs, paired bootstrap vs seed,
  Holm across methods.
- **Secondary:** WILDS-official worst TPR/TNR over the 8 (overlapping)
  identities, for comparability with published numbers.
- **Control:** worst-class accuracy (group-free selection control, Yang et al. 2023).
- **Dropped:** `R_worst_group` — with 100 pos / 100 neg per cell it is
  algebraically identical to `R_worst_gba`; it never carried information.

## Known artifact

Contrasts where a method returned the byte-identical seed prompt produce a
degenerate all-zero bootstrap. The p-value now counts ties on both sides
(`p = 2·min(P(Δ≥0), P(Δ≤0))`), so these report p=1 rather than p=0.
Affected cells: `ape_ut@42`, `ape@43`, `gepa@43`, `prime@43`, `gepa@44`.

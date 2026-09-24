# E5 CivilComments — Preregistration S10 (supersedes S9 headline)

Locked 2026-08-12, before any S10 method is scored. Supersedes the headline of
[`PREREGISTRATION.md`](PREREGISTRATION.md) (S9), which stays on file unchanged.

## Status of S9

S9 is **demoted to exploratory**. Not because it failed — because the primary
metric and the budget were changed *after* seeing its results
([`POWER_AUDIT.md`](POWER_AUDIT.md)). A metric chosen with knowledge of which
method won cannot support a confirmatory claim, regardless of how defensible the
reasoning was. The S9 table remains published as descriptive evidence.

What S9 established, and what S10 must not re-litigate:

- 0 of 31 method-vs-seed contrasts resolvable at 95% under hard-min GBA.
- Ranks invert across matrix seeds; the labeled-target oracle scores *below* seed.
- Scorer repeats contribute ~1/250 of the metric variance and cost 3×.

## Confirmatory data

| Set | n | Fingerprint | Role |
|-----|--:|-------------|------|
| `test_fixed_large` | 5251 | `1ed1114b72e6` | headline |
| `test_fixed` (⊂ large) | 1800 | `5cfb7ebde3c5` | S9 legacy; **decision-contaminated** |
| `test_fixed_new` = large \ old | 3451 | complement | **clean confirmatory subset** |
| `d_dev_targeted` | 900 | `594125cf6a52` | **selection** (groups 8/3/5 × 150/cell) |
| `d_dev` | 900 | `1c10514d553c` | uniform baseline for selection, reported alongside |

`test_fixed_large` draws `min(300, available)` per `group×label` cell, keeping the
same RNG call order as the 100-per-cell draw, so it is a strict superset and the
1800 already-scored predictions stay valid (verified, not assumed:
`scripts/build_e5_test_fixed_large.py` asserts the superset and re-derives the
old fingerprint before writing anything). Cell (`other_religions`, toxic) caps at
151 — the test split contains no more. Cells are therefore unequal by design;
GBA is computed within group, so unequal cell sizes change only per-group
precision, and the ALL-ZERO floor is still exactly 0.5 (checked).

**Primary reporting is on all 5251.** Because the 1800 subset informed the metric
choice, every headline number is reported alongside its value on the 3451-example
complement. If the two disagree in sign, the complement governs and the
disagreement is reported.

## Selection (amended 2026-08-12 after F9)

Selection runs on **`d_dev_targeted`**: the same 900-example budget concentrated
on groups 8, 3, 5 at 150 per cell, instead of 9 groups at 50. The three groups are
fixed by the seed prompt's per-group GBA on the uniform `d_dev` and are frozen
before any S10 method runs; no test data enters the choice.

Justification is F9 in [`R15_CALIBRATION_CONTROL.md`](R15_CALIBRATION_CONTROL.md):
under uniform allocation every worst-group criterion is *anti-correlated* with
test worst-group score (Spearman −0.61 to −0.22 on the optimizer pool) and carries
0.0600 of selection regret; under the targeted allocation all of them are positive
and regret falls to 0.0075.

- **Primary selection statistic:** worst-class accuracy on `d_dev_targeted`. It is
  group-free to compute, needs no per-group estimates, and reached regret 0.0000
  on the optimizer pool. Group labels are used to *allocate* the dev set, not to
  score it.
- **Reported alongside, not used to select:** the same method's pick under the
  uniform `d_dev`, and under shrunk soft-min (τ=0.1, w=40) on both dev sets. Any
  divergence is a finding and gets reported.
- **Frozen:** the target group set {8, 3, 5} and `d_dev_targeted`'s fingerprint.
  Re-tuning the allocation after seeing S10 results would repeat the S9 mistake.

## Metrics

- **Primary:** CVaR@25% over the 8 identity-group GBAs (mean of the worst 2).
  Chosen for power, not for level: 1.98× the effective data of hard-min GBA at
  identical scoring cost (POWER_AUDIT F2).
- **Secondary (comparability):** WILDS-official worst TPR/TNR over the 8
  overlapping identities.
- **Diagnostic, always reported:** mean GBA, `recall`, `specificity`,
  `|recall − specificity|`. F3 showed mean GBA alone explains R²=0.687 of the
  worst-group score, so any worst-group claim must be shown against its own mean.
- **Control:** worst-class accuracy (group-free; Yang et al., ICML 2023).
- **Dropped:** `R_worst_group` — algebraically identical to `R_worst_gba` on
  balanced cells; it never carried information.

## Budget

- **1 repeat** per prompt per set. Repeats buy ~1/250 of the variance per unit
  cost that examples buy.
- **2 repeats on the seed anchor only**, as a scorer-drift guard. If the two
  seed repeats differ by more than 0.01 CVaR@25%, the session is discarded.
- Same-session scoring for all prompts in a comparison, as in S9.

## Statistics

- Paired bootstrap over examples, 10 000 iterations, resampled within
  `group×label` cells, both prompts scored on the same resampled rows.
- Two-sided bootstrap p counting ties on both sides: `p = 2·min(P(Δ≥0), P(Δ≤0))`,
  so a method that returns the byte-identical seed prompt reports p=1, not p=0.
- Holm–Bonferroni across the method-vs-seed contrasts within a seed.
- Across seeds: mean ± SD and the count of seeds with Δ>0. No pooling of seeds
  into a single p-value.

Expected resolution at n=5251, single repeat: CI half-width ≈ 0.015 on CVaR@25%.

## New mandatory baselines

**R15 — calibration control.** 12 prompts differing from the seed by a
single-line substitution of the tie-break rule, spanning lenient to strict
(8 explicit-probability thresholds + 4 natural-language variants).
Pool: `experiments/E5_civilcomments/pools/strictness_sweep`. Selected by the same
dev rule as every other method. This traces what pure operating-point movement
achieves with zero group machinery and zero search.

**R16 — selection control.** One candidate pool, three selection rules —
`mean_gba`, `worst_class` (group-free), `cvar25` — plus a dev-bootstrap measure of
how stable each rule's pick is. Isolates how much of any gain is the selection
rule rather than the generator (Idrissi et al., CLeaR 2022).

## Claims and their falsifiers

A PRIME claim requires **all three**:

1. Δ CVaR@25% vs seed ≥ **+0.05**, CI95 excluding 0, on both 5251 and the 3451
   complement.
2. ≥ **+0.02** over the best of {APO, GPO, Random-AL} — the existing bar.
3. ≥ **+0.02** over **R15**. This is the new bar and the one most likely to fail.

Stated as falsifiers, so the result is informative either way:

- **If R15 matches PRIME**, the group machinery is not what produces the gain;
  the finding is "on CivilComments, worst-group prompt optimization is
  operating-point tuning", and it gets written up as such.
- **If `worst_class` selection matches `cvar25` selection** on the same pool, the
  group labels in the selection rule are not earning their cost.
- **If no method separates from seed at n=5251**, the benchmark is
  underpowered for prompt-level interventions and the venue changes, not the
  metric. There is no third metric revision.

## Out of scope

OPRO (R13), MIPROv2 (R14), gemma-4 transfer, Amazon §7.2, and the cluster-source
ablation (oracle vs GEORGE-style vs LLM pseudo-groups) remain exploratory until
separately preregistered.

# R15 / R16 — calibration control and selection control

**Date:** 2026-08-12. Follow-up to [`POWER_AUDIT.md`](POWER_AUDIT.md) F3, which
required a control showing what pure operating-point movement achieves.

Reproduce:

```powershell
python scripts/build_e5_test_fixed_large.py --per-cell 300
python scripts/build_r15_strictness_pool.py
python scripts/run_e5_selection_control.py --n-boot 2000
python scripts/compare_r15_vs_s9.py --rule worst_class --n-boot 2000
python scripts/e5_selection_rule_shootout.py --n-boot 2000
python scripts/e5_s9_pool_selection_replication.py --n-boot 2000
python scripts/e5_selection_statistic_ablation.py
python scripts/build_e5_dev_targeted.py
python scripts/run_e5_dev_budget_reallocation.py --n-boot 2000
```

Artifacts: `results/E5_selection_control/strictness_sweep/`.

## Setup

**R15 pool.** 12 prompts, each a *single-line substitution* of the seed prompt's
tie-break rule — 8 explicit probability thresholds ("label 1 only if P(toxic) ≥
X%", X from 90 down to 20) and 4 natural-language strictness variants. No search,
no group labels, no optimizer LLM, no training data.

**Evaluation.** New `test_fixed_large`, n=5251, built as a strict superset of the
old n=1800 set, so R15 can be restricted to the exact rows the S9 matrix was
scored on and compared paired at the example level. Cell
(`other_religions`, toxic) caps at 151 — the split has no more.

## F4 — a one-line edit places at the top of the 34-cell matrix

R15 selected on D_dev by a group-free rule, restricted to the S9 rows:

| | CVaR@25% @1800 | @5251 | @clean 3451 |
|---|---:|---:|---:|
| **R15 dev-pick** (`nl_lenient_max`) | **0.6475** | 0.6608 | 0.6613 |
| best of sweep on test (`thr_p90`) | 0.6500 | 0.6617 | 0.6650 |
| seed prompt | 0.6300 | 0.6542 | 0.6613 |
| GPO @42 (best S9 cell) | 0.6675 | — | — |
| PRIME @42 | 0.6600 | — | — |
| oracle (APO, fully labeled target) | 0.6200 | — | — |

Against all 34 S9 method×seed cells on identical rows, paired bootstrap, Holm:

- **0 of 34** cells beat R15 at Holm < 0.05.
- R15 ranks **4th of 35**; only `gpo@42` (+0.020), `prime@42` (+0.013) and
  `random_al@44` (+0.005) are nominally above it, none resolvable.
- R15 beats the seed by +0.0175 and the labeled-target oracle by +0.0275.

The honest reading is not "R15 wins" — nothing here wins, the CIs all straddle
zero. It is that **a 12-point one-line sweep with no group information lands
inside the noise band of the entire optimizer matrix**, including methods that
consumed a label budget of 240 and hundreds of optimizer-LLM calls.

## F5 — the selection rule, not the generator, decides the outcome

Same 12 candidates, same D_dev, only the dev rule changes:

| dev rule | picks | test CVaR@25% | regret vs pool best | rank corr. with test |
|---|---|---:|---:|---:|
| mean GBA | `nl_lenient_max` | 0.6608 | 0.0008 | +0.71 |
| **worst-class acc** (group-free) | `nl_lenient_max` | 0.6608 | 0.0008 | +0.71 |
| global accuracy | `nl_lenient_max` | 0.6608 | 0.0008 | +0.63 |
| **softmin shrunk** (PRIME's shipped rule) | `thr_p20` | 0.6592 | 0.0025 | +0.58 |
| CVaR@50% | `thr_p80` | 0.6583 | 0.0033 | +0.27 |
| softmin, unshrunk | `thr_p70` | 0.6575 | 0.0042 | +0.26 |
| CVaR@25% | `nl_strict` | 0.6317 | 0.0300 | **−0.34** |
| hard-min GBA, shrunk | `nl_strict` | 0.6317 | 0.0300 | **−0.51** |
| hard-min GBA | `nl_strict` | 0.6317 | 0.0300 | **−0.57** |

Switching only the selection rule moves the result by **0.030** on the fixed
pool — comparable to the 0.070 spread of the whole 34-cell method matrix, and
larger than any method-vs-seed effect S9 could measure.

Two things follow.

**Rules that key on the minimum group are anti-correlated with the test
worst-group score.** They do not merely add variance; they systematically select
the over-flagging failure mode. The picked prompt `nl_strict` has recall 0.849 /
specificity 0.539 and is the *worst* candidate in the pool on test.

**A group-free control matches the best rule.** `worst_class` needs no group
labels and ties `mean_gba` for the top spot, while being the most *stable* pick
under a dev bootstrap (sd 0.0018 vs 0.0135 for CVaR@25%). This reproduces, in the
prompt-optimization setting, the finding of Idrissi et al. (CLeaR 2022) and Yang
et al. (ICML 2023) that group-labelled model selection is where most reported
worst-group progress lives — except here it is worse than that: the group-labelled
rule is actively harmful.

### Why: D_dev cannot measure a single group

Bootstrap sd of each dev statistic on the seed prompt (n=900, 50 pos + 50 neg per
group):

| statistic | sd on D_dev |
|---|---:|
| single group GBA (g8) | **0.0475** |
| hard-min GBA | 0.0401 |
| CVaR@25% | 0.0312 |
| softmin, unshrunk | 0.0221 |
| softmin, shrunk (w=40) | 0.0180 |
| mean GBA | 0.0158 |
| global accuracy | 0.0150 |

The quantity every worst-group rule depends on carries sd ≈ 0.048 — the same
magnitude as the entire between-method spread it is being asked to resolve. The
concrete failure: on D_dev, `nl_strict` scores 0.610 on group 8 versus
`nl_lenient_max` at 0.570, a +0.040 edge that is under one standard deviation of
noise; on the 6× larger test set the ordering reverses to 0.603 versus 0.653. The
argmin group itself is stable (group 8 on both dev and test, for both prompts) —
it is the *value*, not the identity of the worst group, that cannot be estimated.

## F7 — replicated on the optimizer pool

F5 came from a pool of one-line edits, which could be a quirk of a pool whose
candidates differ along a single axis. Repeating it on the **S9 optimizer pool** —
26 unique prompts written by APE/APO/GPO/EvoPrompt/GEPA/PRIME, an entirely
different generator — reuses the existing test predictions and costs one D_dev
pass per prompt (`scripts/e5_s9_pool_selection_replication.py`).

| dev rule | pick | test CVaR@25% | regret | rank corr. |
|---|---|---:|---:|---:|
| worst-class acc (group-free) | `44_evoprompt_de` | 0.6450 | 0.0225 | **+0.68** |
| mean GBA | `43_evoprompt_de` | 0.6275 | 0.0400 | +0.46 |
| softmin shrunk | `44_apo` | 0.6075 | 0.0600 | +0.28 |
| global accuracy | `43_ape_k48` | 0.6300 | 0.0375 | +0.23 |
| softmin, unshrunk | `44_apo` | 0.6075 | 0.0600 | +0.08 |
| CVaR@50% | `44_apo` | 0.6075 | 0.0600 | +0.07 |
| hard-min GBA, shrunk | `44_apo` | 0.6075 | 0.0600 | **−0.54** |
| CVaR@25% | `44_apo` | 0.6075 | 0.0600 | **−0.58** |
| hard-min GBA | `44_apo` | 0.6075 | 0.0600 | **−0.62** |

The sign pattern is identical across the two pools: **every** min-based rule is
negative (6 of 6 across both pools, −0.34 to −0.62), **every** averaged or
group-free rule is non-negative (12 of 12, +0.07 to +0.71). The group-free
worst-class rule is the best rule on both. The dev-statistic precision table
reproduces to three decimals, including the single-group sd of 0.0475.

The S9 pool is the harder case: the best rule still leaves 0.0225 of regret, and
five of nine rules converge on `44_apo` — a prompt near the bottom of the pool
(0.6075 against a pool best of 0.6675). On a pool of genuinely varied prompts,
D_dev at n=900 does not reliably identify the good ones under *any* rule tested.

### Where this leaves PRIME's shrinkage

PRIME does not ship the raw hard-min rule; it ships shrunk soft-min
(`shrink_prior_weight=40`, τ=0.1). Shrinkage measurably helps the *ranking* —
rank correlation rises from +0.26 to +0.58 on the R15 pool and from +0.08 to
+0.28 on the S9 pool, and it cuts the statistic's dev sd from 0.0221 to 0.0180 —
and it keeps the rule out of the anti-correlated cluster that raw hard-min falls
into (−0.54 even after shrinkage, versus +0.28 for the soft version).

But it is not sufficient: on the S9 pool the shrunk soft-min rule still picked
`44_apo` and carried the full 0.0600 regret. The defensible statement is
"shrinkage plus soft-min is strictly better than the min-based alternatives and
is the right direction", not "shrinkage solves selection". An ablation isolating
τ and `shrink_prior_weight` against the group-free control is now a first-class
experiment rather than a footnote.

## F6 — the scorer ignores numeric probability thresholds

Instructing `gemma-3-12b-it` to "label 1 only if P(toxic) ≥ X%" does not move the
operating point monotonically, or much at all. Recall across X = 90→20: 0.732,
0.746, 0.746, 0.753, 0.764, 0.749, 0.746, 0.740 — a 0.03 band with no trend.
Natural-language strictness does move it: 0.721 (`nl_lenient_max`) to 0.849
(`nl_strict`), with specificity moving 0.703 to 0.539.

Two consequences. Any method that reasons about "adjusting the threshold" via
stated probabilities is a no-op on this scorer. And the R15 sweep therefore
covers only the well-behaved middle of the operating-point curve; the extremes
that S9's failing methods reached (recall ≈ 0.85, spec ≈ 0.53) are reproduced
only by the two natural-language strict variants.

## F8 — the optimal amount of worst-group focus in the dev rule is zero

Grid over the three knobs that define the statistic — CVaR depth `k` of 8 groups,
soft-min temperature `τ`, shrinkage weight `w` — on both pools
(`scripts/e5_selection_statistic_ablation.py`, cached predictions only).
Spearman against test CVaR@25%:

R15 pool (n=12) / S9 pool (n=26):

| dev statistic | w=0 | w=10 | w=40 | w=100 | w=400 |
|---|---:|---:|---:|---:|---:|
| CVaR k=1 (hard-min) | −0.57 / −0.62 | −0.51 / −0.58 | −0.51 / −0.54 | −0.38 / −0.40 | +0.48 / +0.02 |
| CVaR k=2 | −0.34 / −0.58 | −0.34 / −0.54 | −0.32 / −0.46 | +0.29 / −0.24 | +0.70 / +0.18 |
| CVaR k=4 | +0.27 / +0.07 | +0.42 / +0.13 | +0.51 / +0.20 | +0.66 / +0.26 | +0.71 / +0.43 |
| CVaR k=8 (= mean GBA) | +0.71 / +0.46 | +0.71 / +0.45 | +0.71 / +0.48 | +0.71 / +0.46 | +0.73 / +0.47 |
| softmin τ=0.02 | −0.55 / −0.52 | −0.50 / −0.50 | −0.40 / −0.34 | +0.26 / −0.04 | +0.72 / +0.35 |
| **softmin τ=0.1 (shipped)** | +0.26 / +0.08 | +0.45 / +0.11 | **+0.58 / +0.28** | +0.72 / +0.35 | +0.73 / +0.44 |
| softmin τ=0.5 | +0.73 / +0.37 | +0.73 / +0.39 | +0.73 / +0.41 | +0.73 / +0.44 | +0.73 / +0.44 |
| *control:* worst-class | **+0.71 / +0.68** | | | | |

Three readings, all pointing the same way.

**The three knobs are one knob.** Depth, temperature and shrinkage are
interchangeable ways of averaging over groups, and the correlation is a smooth
increasing function of however much averaging is applied by any combination of
them. `w=400` against a per-group `n=100` puts 80% of each group's estimate on the
pooled mean, which is why the `w=400` column rescues even hard-min.

**There is no interior optimum.** If group structure carried usable signal at
this dev size, the best statistic would sit somewhere between hard-min and the
mean. It does not: on both pools the metric improves monotonically all the way to
`k=8` / `τ=0.5` / `w=400`, every one of which is the plain mean GBA. The optimal
amount of worst-group focus in the dev rule is zero.

**And the mean is still not the best rule.** The group-free worst-class control
beats every group statistic on both pools — +0.71/+0.68 rank correlation and the
lowest mean regret (0.0117 vs 0.0200 for the best group-based configuration).

**PRIME's shipped setting (τ=0.1, w=40) sits just inside the useful region**
(+0.58/+0.28) but well below the achievable +0.73/+0.44 at τ=0.2–0.5 with w≥100.

> **F9 overturns the pessimistic reading of this section.** Everything above holds
> only for a D_dev spread uniformly over 9 groups. Once the same budget is
> concentrated on the worst groups, the entire sensitivity surface collapses and
> every statistic works. F8 is a statement about allocation, not about group
> structure.

## F9 — the fix is where the dev examples are spent, not which rule scores them

F8 left one intervention untested: the argmin group is stable while its value is
not, so spend the dev budget on the groups that matter. Built as
`d_dev_targeted` — **the same 900 examples**, redistributed from 9 groups × 50 per
cell to groups **8, 3, 5** × 150 per cell (`scripts/build_e5_dev_targeted.py`).

The three groups come from the **seed prompt's per-group GBA on the uniform
D_dev** (0.570 / 0.610 / 0.630, the three lowest), which is information available
before any optimization begins. No test data is consulted. They do coincide with
the groups that are worst on test, which is the point: the identity of the worst
group is learnable from dev even though its value is not.

Two properties make the comparison clean:

- The targeted draw is a **strict superset** of the uniform dev on those groups,
  so 300 rows were already scored. Re-scoring them agreed with the cache at
  **99.72%** (min 99.33% over 38 prompts) — an end-to-end check on the scoring
  path and the row alignment, not just on this result.
- Both columns below compute the **identical statistic over the identical three
  groups**. Only per-group n changes, 100 → 300. The effect is precision alone.

Spearman with test CVaR@25%, uniform → targeted:

| statistic (groups 3,5,8) | R15 pool | | S9 pool | |
|---|---:|---:|---:|---:|
| | uniform | targeted | uniform | targeted |
| CVaR k=1 (hard-min), w=0 | −0.57 | −0.66 | −0.61 | **+0.24** |
| CVaR k=2, w=0 | −0.34 | +0.00 | −0.58 | **+0.38** |
| CVaR k=3 (mean of 3), w=0 | −0.45 | +0.10 | −0.22 | **+0.60** |
| softmin τ=0.02, w=0 | −0.53 | −0.07 | −0.57 | **+0.37** |
| softmin τ=0.1, w=40 (shipped) | −0.51 | +0.01 | −0.34 | **+0.55** |
| softmin τ=0.5, w=0 | −0.51 | +0.01 | −0.26 | **+0.58** |
| *mean change* | | **+0.406** | | **+0.876** |
| *statistics positive* | | 7/12 | | **12/12** |

Regret — the decision-relevant number — on the S9 pool:

| dev rule | uniform dev | targeted dev |
|---|---|---|
| every group statistic | `44_apo`, regret 0.0600 | `42_prime`, regret **0.0075** |
| worst-class (group-free) | `44_evoprompt_de`, regret 0.0225 | `42_gpo`, regret **0.0000** |
| global accuracy | `43_ape_k48`, regret 0.0375 | `42_prime`, regret 0.0075 |

Four things follow, and they change the conclusion of this document.

**Worst-group selection is not broken; uniform allocation was.** On the optimizer
pool every group statistic goes from anti-correlated to positive, and regret drops
8×, at zero additional labelling cost.

**F8's sensitivity surface was a noise artifact.** With precise per-group
estimates, 10 of 12 configurations of (k, τ, w) select the same prompt. The
elaborate trade-off between CVaR depth, temperature and shrinkage existed only
because each was a different way of averaging away measurement error.

**But the operative variable is allocation, not group-awareness of the rule.**
Concentrating the dev set improves *every* rule tested, and the outright best
result — regret 0.0000, correctly identifying `42_gpo` as the pool's best prompt —
comes from the **group-free** worst-class statistic evaluated on the
group-concentrated dev. The practical recipe is: use group labels to decide *where
to spend dev examples*, then a cheap group-free statistic to score them.

**Under a correctly allocated dev set, the rule picks PRIME's prompt.** Every
group statistic on the targeted dev selects `42_prime` (test CVaR@25% = 0.6600),
0.0075 behind the pool optimum. That is the first result in this whole
investigation where a group-robust selection procedure behaves as intended.

The R15 pool improves too (mean +0.406, regret 0.0300 → 0.0075) but stays near
zero correlation. Its candidates span only 0.030 of test CVaR@25% against the S9
pool's 0.070, so there is proportionally less signal to rank; this is consistent
with a precision story rather than evidence against it.

## Consequences for the research claim

1. **The S9 matrix cannot be defended as evidence of group-robust prompt
   optimization.** A no-search, no-labels, one-line control sits inside its noise
   band. Any PRIME claim must clear R15, per
   [`PREREGISTRATION_S10.md`](PREREGISTRATION_S10.md).
2. **The bottleneck is measurement, not the optimizer.** With 50+50 per group,
   D_dev cannot rank prompts by worst-group performance. Matching `mean_gba`
   precision on a single group needs ≈8× the per-group dev data (≈7200 examples),
   or a dev budget concentrated on the known-worst groups rather than spread
   uniformly.
3. **The defensible PRIME contribution shifts, and it is now a positive claim.**
   Not "we optimize worst-group performance" — nothing here measurably does — but
   "worst-group selection under a realistic dev budget is a *measurement
   allocation* problem: uniform group coverage makes every worst-group criterion
   anti-correlated with its own objective, and reallocating the same budget onto
   the identifiable-worst groups reverses that and cuts selection regret 8×,
   without extra labels". F5–F9 support this on two independent pools, and it is
   not currently being claimed anywhere.

## What to run next

1. **Rebuild the matrix's selection step on `d_dev_targeted`.** F9 says every
   method's reported number is partly an artifact of dev allocation. Re-selecting
   from each method's own candidate archive on the targeted dev is the direct
   test, and cheaper than re-running the optimizers.
2. **Second validation draw**, to exclude a D_dev-specific artifact. All three
   replications share one validation split, and F9's target groups were chosen on
   it.
3. **Ablate the allocation itself** — 2 groups × 225, 3 × 150, 4 × 112, uniform
   9 × 50 — to find where the precision/coverage trade-off turns over. The risk
   at the concentrated end is a dev set blind to a group that becomes worst after
   optimization; F9 does not test for that.
4. **Then** re-run the matrix under the S10 protocol.

## Caveats

- S9 predictions are majority-of-3 repeats; R15 is a single repeat. Repeat sd is
  0.0008 CVaR@25% (POWER_AUDIT F2), ~5% of the CI half-width.
- Rank correlations are over pools of 12 and 26 (se ≈ 0.30 and ≈ 0.20). The claim
  is the sign separation across 18 rule×pool combinations, not any individual
  coefficient.
- Two pools, one validation split, one scorer, one dataset. Both the uniform and
  targeted dev sets are drawn from the same validation split, and F9's target
  groups were identified on it, so a split-specific artifact is not excluded.
- F9's regret figures come from pools of 12 and 26 with a discrete set of possible
  picks, so regret moves in jumps; the rank correlations are the more stable
  evidence and they agree.
- F9 concentrates dev coverage on the groups that are worst *for the seed prompt*.
  A method that fixes those groups while breaking a fourth would be invisible to
  it. The allocation ablation in "What to run next" is what tests this.
- The R15 sweep spans only the middle of the operating-point curve, because the
  scorer ignores numeric thresholds (F6). It is a weaker control than intended,
  which makes its 4th-of-35 placement more striking, not less.

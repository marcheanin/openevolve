# Run plan — low-variance objective (E1 cvar_lex, single arm)

## Why this run exists

The pair run answered nothing about `cvar_lex` vs `global` because the objective's
own noise was larger than everything the search produced. Measured directly
(`scripts/exp_fitness_noise.py`, one fixed prompt scored 3–4 times on the same
D_select, workers at temperature 0):

| quantity | value |
|---|---|
| `fitness_scalar` SD across identical re-evaluations | **0.0294** (first batch), 0.0221 (second) |
| total three-cycle gain of the whole cvar run | +0.058 |
| gains in cycles 2 and 3 | +0.002, +0.003 |
| `R_global` SD across the same re-evaluations | **0.0024** |
| predictions that flip between repeats | 9–13 of 240 (3–5%) |

So the search was ranking noise for two of three cycles, and the two arms were never
comparable: `global` optimised a metric with ~12× less noise than `cvar_lex`. That
single fact also explains `best_selection_cycle = 1` in both arms, the frozen cluster
champions, and why extra mutation budget hurt generalisation (O20) — more draws from a
noisy ranking simply select a higher noise order-statistic.

**This run does not try to answer the arm question.** It answers the prerequisite:
with a measurable objective, does the search produce progress above its own noise?

## What changed, and on what evidence

Every change below is either measured or a direct consequence of a measurement.

### 1. The objective (the whole point of the run)

Twelve candidate definitions were scored offline on the saved repeats
(`scripts/exp_objective_noise_sweep.py`): noise SD from re-evaluations of one prompt,
signal from initial-vs-evolved on test.

| objective | noise SD | signal | signal/noise |
|---|---|---|---|
| `cvar_lex q=0.20 w=0 raw` (what my analysis scripts assumed) | 0.0279 | 0.0274 | 0.98 |
| `cvar_lex q=0.40 w=0 raw` (**what actually ran**) | 0.0221 | 0.0183 | 0.83 |
| `R_global` | 0.0108 | 0.0208 | 1.94 |
| `cvar_lex q=0.40 w=50 raw` | 0.0128 | 0.0214 | 1.68 |
| **`cvar_lex q=0.40 w=50 bal`** | **0.0090** | **0.0243** | **2.70** |
| `cvar_lex q=0.40 w=100 bal` | 0.0071 | 0.0209 | 2.94 |
| `R_macro` alone | 0.0060 | 0.0008 | 0.13 |

Two independent mechanisms, both now on:

- **Shrink cluster accuracy toward the grand mean** with a prior worth
  `shrink_prior_weight = 50` pseudo-examples (≈ mean cluster size, so shrink factor
  ≈ 0.5). The shipped code called this "shrunk" but used Beta(1,1) — a prior worth
  two examples toward 0.5, which moves a 48-example cluster by 4%. It was never
  shrinking anything (M14). Note the independent bootstrap check on test agreed:
  shrinkage was the only transformation that lowered noise *and* raised the effect,
  while widening the quantile lowered both and `R_tail` was the worst of twelve.
- **Class-balanced example weights** (`class_balanced: true`). 83% of the
  between-cluster accuracy spread is explained by each cluster's gold-rating mix
  alone (C10), and the evolved prompt's entire measured effect was a 4/5 threshold
  shift that was macro-neutral (C11: 4★ +0.161, 5★ −0.120, macro −0.0008). Under raw
  accuracy that shift scores; under balanced weights it does not, so the search has
  to find real discrimination. Empirically this is also where most of the remaining
  noise lived: w=50 raw is 0.0128, w=50 balanced is 0.0090.

`w=100` scores marginally better but shrinks the group signal to 32% cluster-specific;
`w=50` keeps it at ~50%, which is the point of a group-robustness experiment.

`R_macro` alone is kept as a reported diagnostic, not an objective — its near-zero
signal here *is* the C11 finding restated: the last run achieved nothing in balanced
terms.

### 2. Budget moved from mutation count to evaluation precision

`d_select_size` 240 → **360**, `n_evolve_iterations` 12 → **8**. Same cost
(8×360 ≈ 12×240), and O20 already measured that the extra iterations bought identical
D_select gain while *losing* validation CVaR. Combined with the objective change the
expected noise SD is ≈ 0.0073.

`max_train_users` 200 → 300 so `D_select + D_anchor + D_audit` (510) still fits in the
heldout pool. `eval_sets_artifact` is therefore un-pinned; `cluster_artifact` stays
pinned (refitting is ~44 min and not bit-reproducible, O14).

### 3. Test split sized from a target effect

`max_test_users` 120 → **240**: at 120 the minimum detectable effect on `R_global` is
0.056 and we are chasing 0.02–0.04; 240 buys ≈0.03 (M13). `max_reviews_per_user` stays
at 8 — it only limits per-user resolution, which mattered for `R_worst`/`R_tail`, and
neither is in the selection key any more. The data would allow up to 75 (every val/test
user has exactly 75 reviews, 1334 users available), so this is a budget choice, not a
data limit.

### 4. Selection key

Was `(raw CVaR_cluster, R_global)` on val, with `proxy_tail_metric = R_tail`. Now
`(CVaR_cluster_balanced_shrunk, R_macro)` and `proxy_tail_metric = CVaR_cluster_shrunk`.
`R_tail` was the worst discriminator we measured (|effect|/SD 0.48) and it was the
default.

### 5. Mutator feedback

The report the mutator receives was rewritten around C9 (95–100% of errors are
adjacent-class with balanced directions, so the mutator was being asked to fit
reviewer idiosyncrasy and answered with one-way caps):

- errors split into **systematic** (all workers produced the same wrong rating — an
  instruction can fix this) and **contested** (workers split — noise, explicitly
  marked "do not write rules for these");
- a **confusion table in both directions**, with the instruction that a one-sided cap
  on a two-directional confusion breaks the other column;
- a **damage report** against the previous cycle's predictions on the same rows:
  what the last change fixed and what it broke, so the mutator can narrow a bad
  trigger instead of stacking another rule on top of it;
- **verbatim few-shot lines** built mechanically from real failing reviews. O22 found
  only 1 of 6 few-shot examples in the final prompt traced to a real review; the rest
  were inventions asserting unverified gold labels. Handing over the exact text
  removes the need to quote from memory.

### 6. Consolidation — kept, contained, and made stateful

Answering the direct question: **it is not breaking anything any more, and its
influence is fully controllable.** Evidence from the pair run: it lost all 6 times
(`delta_vs_evolved` −0.031/−0.009/−0.095 and −0.017/−0.013/−0.017), never won a heir
slot, and every heir kept `BaseGuidelines` at 147 characters across all six cycles.
The damage O9 recorded (−0.015, −0.066) came from the old architecture where the
consolidated prompt inherited *by construction*; since it became a competitor scored
on the same D_select it cannot reach the heir at all.

The earlier diagnosis that it "deletes 40–60% of the rules" was wrong. Summing both
blocks: 1807→1747, 2203→2562, 1816→1839 characters. It **moves and paraphrases** rather
than deletes. That is why it loses: it is an unguided rewrite of a prompt that search
has already tuned, so its expected fitness delta is negative for the same reason a
random mutation's is. Raising the interval therefore cannot improve its win rate — it
only reduces the number of (free) attempts.

So: keep it, and fix the two things that were actually broken.

- **`every_n_cycles: 2` + guaranteed final pass.** Not for safety — for relevance.
  Consolidating a freshly-mutated prompt every cycle churns; the slow layer only has
  something to absorb once rules have accumulated.
- **Stateful.** `base_len_before` was 161 in every single cycle because the previous
  consolidated base was never fed back, so the mechanism restarted from the original
  base forever and could not accumulate — which was its entire purpose. The last
  accepted consolidated `BaseGuidelines` is now passed into the next call.
- **Non-lossy guard.** A rewrite may move rules between blocks; it may not delete
  them. Below 85% of rule lines retained the candidate is discarded and logged
  instead of scored.

If it still loses every cycle under these conditions, that is a clean result: it means
prompt reorganisation has no value here and it should be deleted rather than tuned.

### 7. Anchor gate → monitor

`anchor_gate_mode: monitor`. With 4 discordant anchor pairs the smallest achievable
one-sided exact p is 2⁻⁴ = 0.0625 > α, so the gate mathematically cannot reject (M12).
It is still evaluated and logged, because the anchor accuracy trend (0.95→0.91→0.90 in
cvar, rising to 0.97 in global) was the only in-run signal that tracked the test
outcome.

### 8. Archive provenance

Carried members made a round trip through the OpenEvolve seed checkpoint and came back
tagged `source="oe"` at the current cycle, so `front_sources: ["oe"]` was an artifact
and a carried prompt was indistinguishable from a fresh mutation (O23). Candidates now
carry an immutable `origin` (source, birth cycle, prompt hash) that survives the round
trip, and an unchanged member is reported as `carried`.

## What is deliberately *not* changed

- **No cascade screening.** A single-worker screen scored Spearman −0.042 against the
  ensemble ranking (M16) — but that ranking is itself mostly noise, so the experiment
  cannot be interpreted yet. Re-run E-B after this run, against a stable ranking.
- **No product-category balancing in the error report.** 12 of 18 errors in one cycle
  were book reviews and the resulting rules went book-specific against an all-category
  test (C9), but category is not carried through `ReviewSplit` and threading it would
  touch the raw-split cache. Deferred deliberately.
- **No aggregation work.** Oracle-over-workers is +0.091 above the ensemble and is
  untouched by prompt evolution (C12) — roughly 3× anything prompt search has produced.
  It stays out of scope so the prompt-only ceiling is honest, but it is the obvious
  next arm.

## What we watch, and what would falsify the run

Success is not a better test number — at these caps we could not detect one. Success
is **evolution becoming measurable**.

| # | Check | Pass | Fail → conclusion |
|---|---|---|---|
| 1 | Per-cycle `best_evo` gain vs the 0.009 noise floor | ≥2 cycles with a gain > 2 SD (0.018) | Gains still inside noise → the bottleneck is not the metric; suspect the mutator or the prompt space |
| 2 | `best_selection_cycle` | > 1 | Still 1 → val selection still cannot distinguish cycles, and cycles 2–3 are pure cost |
| 3 | Val `CVaR_cluster_balanced_shrunk` trend | monotone or flat, not declining | Declining while D_select rises → still overfitting D_select; cut to 1–2 cycles |
| 4 | Systematic-error share in the mutator report | ≥25% of errors, and rules trace to them | Near zero → the feedback is all noise and rules will be arbitrary |
| 5 | Damage report: fixed vs broke per cycle | fixed > broke | broke ≥ fixed → the mutator is trading, not improving |
| 6 | Few-shot examples traceable to real reviews | ≥80% verbatim | Still inventing → drop `<FewShotExamples>` from mutation targets |
| 7 | Consolidation `delta_vs_evolved` | any cycle > 0 | 3/3 negative again → delete the mechanism |
| 8 | Archive `origin_source` / `origin_cycle` | carried members reported as `carried` | Still all `oe` → the stamp is not surviving |
| 9 | Anchor accuracy trend (monitor) | flat or rising | Falling while D_select rises → the same regression pattern as the pair run |
| 10 | Final test vs initial prompt, paired bootstrap | reported with CI either way | — the initial prompt is a floor and must be evaluated on the same split |

Checks 1 and 2 are the run. If both fail, the metric was not the blocker and the next
move is the mutator, not more measurement.

## Order of operations

1. ~~E-C: tail-metric stability (free, on saved predictions)~~ — done.
2. ~~E-A: fitness noise (2 batches, 7 total re-evaluations)~~ — done, the decisive result.
3. ~~E-B: single-worker cascade screen~~ — done, uninterpretable until noise is fixed.
4. ~~E-D: offline objective sweep on the saved repeats~~ — done, picked the objective.
5. **This run** — `config_cvar_lowvar.yaml`, single `cvar_lex` arm, 3 cycles.
6. If checks 1–2 pass: re-run E-B against the now-stable ranking, then the
   `cvar_lex` vs `macro` pair at the powered caps.
7. If checks 1–2 fail: stop optimising and instrument the mutator instead.

## Cost

≈50k worker calls: OpenEvolve 8×3×360×3 = 25.9k, val 3×150×8×3 = 10.8k,
test 2×240×8×3 = 11.5k, consolidation + pool passes ≈3k. Budget cap 70k.

# RESULTS — Pareto v2 pair (champion archive)

Runs:
- **cvar_lex**: `results/E1_pred_profile_cvar_lex/seed42_20260729_213313`
- **global**: `results/E1_pred_profile_global/seed42_20260730_051424`

Pinned shared artifacts (fair pair): `clusters.json` sizes `{0:23,1:38,2:30,3:16,4:33}`,
`eval_sets.json` content_hash `7ec9db8ed34ceb3f`.

Baseline for progress (same test assign, saved with worker votes):
`evals/initial_prompt/` (+ root aliases `baseline_initial_*`).

---

## Headline

**No arm produced a statistically detectable change on test.** The observed spread
between arms and against the initial prompt is smaller than the measurement noise
of the metrics we select on. Everything below the top-line table is mechanism
diagnosis, not effect estimation.

| Prompt / arm | test R_global | R_worst | R_tail | CVaR_cluster |
|--------------|---------------|---------|--------|--------------|
| **initial** (saved eval) | **0.697** | 0.375 | 0.380 | 0.567 |
| cvar final (C1 heir) | 0.666 | 0.375 | 0.349 | 0.545 |
| cvar final (re-eval w/ votes) | 0.676 | 0.375 | 0.365 | 0.549 |
| **global final** (C1 heir) | 0.693 | 0.375 | **0.406** | **0.571** |
| prev Pareto cvar `091203` | **0.700** | **0.488** | 0.401 | **0.587** |
| baseline `0728` | 0.679 | 0.375 | — | 0.567 |

### The power problem (M10)

Paired bootstrap over the **same 120 test users** (`scripts/paired_significance.py`),
cvar final vs initial:

| Metric | initial | cvar | delta | CI95 | p |
|--------|---------|------|-------|------|---|
| R_global | 0.6969 | 0.6760 | −0.0208 | [−0.050, +0.007] | 0.147 |
| CVaR_cluster | 0.5375 | 0.5096 | −0.0279 | [−0.141, +0.069] | 0.741 |
| R_tail | 0.3802 | 0.3646 | −0.0156 | [−0.073, +0.042] | 0.663 |

Example-level McNemar: 77 examples only-initial-correct vs 57 only-cvar-correct,
p = 0.100. Bootstrap SD of the metrics themselves at this sample size:

| Metric | SD | min detectable effect (~80% power) |
|--------|----|------------------------------------|
| R_global | 0.020 | **0.056** |
| CVaR_cluster / R_worst | 0.035 | **0.099** |
| R_tail | 0.033 | **0.091** |

Observed effects are 0.02–0.04. Across five nominally "equivalent" configurations
(same seed, same pins) test `CVaR_cluster` ranges 0.545–0.587 — SD 0.015, i.e. the
run-to-run spread is the same order as every conclusion we tried to draw from it.
**The pair as designed cannot answer Р11.**

### CVaR_cluster is degenerate here (M11)

`n_clusters = 5`, `tail_quantile = 0.2` → `ceil(0.2 × 5) = 1`, so
`CVaR_cluster ≡ R_worst_cluster` exactly (both 0.5375 / 0.5096 above). There is no
smoothing at all: the entire `cvar_lex` objective is the accuracy of one cluster,
and on test that cluster has **10 users** (c3) or **13** (c0). The whole arm is
steered by ~40–100 examples.

Test cluster sizes: c0=13, c1=34, c2=24, c3=10, c4=39 users.

---

## Why cvar moved the wrong cluster (the substantive result)

Per-cluster test deltas vs initial:

| cluster | users | initial | cvar | Δ cvar | global | Δ global |
|---------|-------|---------|------|--------|--------|----------|
| c0 | 13 | 0.596 | 0.510 | **−0.087** | 0.567 | −0.029 |
| c1 | 34 | 0.721 | 0.699 | −0.022 | 0.717 | −0.004 |
| c2 | 24 | 0.625 | 0.625 | 0.000 | 0.615 | −0.010 |
| c3 | 10 | 0.537 | **0.588** | **+0.050** | 0.575 | +0.037 |
| c4 | 39 | 0.795 | 0.766 | −0.029 | 0.792 | −0.003 |

cvar_lex did exactly what it was told: it lifted the cluster that was worst **on
D_select** (c3: 0.5556, the minimum of the heir's cluster scores) and paid for it
elsewhere. The problem is that on test the worst cluster is **c0**, not c3
(Spearman of cluster-accuracy ranks between D_select and test = **+0.70**, but the
argmin disagrees in both arms). So the objective optimised a tail identity that
does not transfer, and the cluster it sacrificed became the new bottleneck —
CVaR got *worse* while the tail got flatter.

Per-worker confirms this is not an aggregation artifact — all three workers
degraded with the cvar prompt: deepseek R 0.691→0.669, kimi 0.695→0.668,
qwen 0.653→0.640.

**More search made this worse, not better.** Compare the three cvar runs:

| Run | OE iters/cycle | D_select gain | val CVaR trajectory | test CVaR |
|-----|----------------|---------------|---------------------|-----------|
| `091203` (Pareto v1) | 5 | 0.527→0.582 | 0.6286→0.6328→0.6328 (**up**) | **0.587** |
| `213313` (Pareto v2) | 12 | 0.527→0.585 | 0.6135→0.6052→0.6094 (**down**) | 0.545 |

Same D_select gain, opposite val direction. Raising the mutation budget bought
D_select fitness that did not exist outside D_select — textbook overfitting to a
240-example selection set whose tail is ~8 users per cluster.

---

## Mechanism-by-mechanism

### OE evolution — the O17 budget fix worked

Per-cycle iteration counts (corrected for the log-handler leak, O20):

| Cycle | cvar iters | cvar improvements | global iters | global improvements |
|-------|-----------|-------------------|--------------|---------------------|
| C1 | 11 | 3 (0.527→0.582) | 6 | 1 (0.575→0.696) |
| C2 | 10 | 1 (→0.5824) | 12 | 3 (→0.725) |
| C3 | 6 | 1 (→0.5849) | 12 | 0 |

Evolution is genuinely active again (v1 had literally zero improvements after C1).
It saturates by C3 in both arms. MAP-Elites diversity 104 (cvar) / 126 (global).

### Selection — cycles 2 and 3 were wasted

`best_selection_cycle = 1` in **both** arms. Val never beat C1, so the final
prompt is C1's heir (verified by hash: `final_prompt.txt` == `al_iter_1/best_prompt.txt`).
Two thirds of the run's API spend bought nothing that survived selection.

### Champion archive (P9) — mechanically correct, empirically inert

Front sizes cvar 5/6/6, global 2/4/5; scalar-best never evicted; heir was the
scalar-best in all 6 cycles; resume mode `pareto_checkpoint`. Three real problems:

1. **Cluster champions froze.** cvar c3 champion = 0.5556 in all three cycles;
   c4 = 0.8444 in all three (both C1-era members). global c3 = 0.5556 throughout.
   The archive preserved specialists, but no cycle produced a better specialist,
   so it stored history rather than progress.
2. **Provenance is destroyed by the checkpoint round-trip.** cvar C1's
   `consolidated` member reappears in C2/C3 tagged `source="oe", cycle=2/3` with
   byte-identical fitness (0.551403, len 2464) — because carried members are
   written into the OE seed database and read back as OE programs. So
   `front_sources: ["oe"]` after C1 is an artifact, and `heir_source` cannot
   distinguish "new mutation" from "carried member".
3. **Specialists never influenced the heir**, only OE's seed population — which is
   the intended design, but with evolution saturating there was no measurable
   benefit to seeding.

### Consolidation — ran cleanly, rejected 6/6, propagated nothing

gemini-3.1-pro produced real output every cycle (zero stubs — O19 fixed), but:

| Cycle | cvar base→ | cvar dyn→ | cvar Δ | global base→ | global dyn→ | global Δ |
|-------|-----------|-----------|--------|--------------|-------------|----------|
| C1 | 161→757 | 1646→990 | −0.031 | 161→1082 | 1680→1034 | −0.017 |
| C2 | 161→1537 | 2042→1025 | −0.009 | 161→1088 | 1873→1087 | −0.013 |
| C3 | 161→1166 | 1655→673 | −0.095 | 161→1176 | 1873→902 | −0.017 |

Two structural facts:

- **`base_len_before` is 161 every single cycle.** Consolidation always restarts
  from the *original* BaseGuidelines, never from the previous cycle's output. It is
  stateless — there is no cumulative "stable rule" accretion, which was its whole
  purpose.
- **Nothing propagated.** Every heir kept `BaseGuidelines` at 147 chars (the
  initial block) in both arms, all three cycles. The consolidated candidate never
  won a heir slot, so its promotions were discarded each time.

The reason it always loses is visible in the numbers: it **shrinks DynamicRules by
40–60%** (1655→673 in cvar C3) and paraphrases the removed rules into
BaseGuidelines. That is a lossy rewrite of exactly the rules evolution just
earned, so it scores worse on the same D_select. Net effect of consolidation this
run: 6 gemini calls, 0 impact.

### Anchor gate — cannot fire in this regime (M12)

All 6 decisions accepted. The two `tolerated_noise` cases are the tell:

| Arm / cycle | anchor acc before→after | n_improved | n_worsened | discordant | McNemar p |
|-------------|------------------------|------------|------------|------------|-----------|
| cvar C2 | 0.95 → 0.91 | 0 | 4 | 4 | 0.0625 |
| cvar C3 | 0.95 → 0.90 | 1 | 6 | 7 | 0.0625 |
| global C2/C3 | 0.93 → 0.97 | 5 | 1 | 6 | 0.98 |

With 4 discordant pairs the smallest achievable one-sided exact p is
2⁻⁴ = 0.0625 > α = 0.05, so **the gate is structurally incapable of rejecting**
when few anchors flip. The diagnostic even reports `min_detectable_worsened: 5`.
cvar's anchor accuracy decayed monotonically 0.95→0.91→0.90 with 0–1 improvements
against 4–6 regressions and was waved through both times. Note this decay is the
only in-run signal that tracked the test outcome (global's anchor accuracy rose to
0.97 and global held its test numbers).

### Batches — healthy but running out of signal

Both arms: seen 30→42→54, +12/cycle; 21 hard / 9 anchor slots; `group_aware`
gave weak clusters 14–19 of 21 hard slots; no representativeness warnings.
But two trends matter:

- Hard-slot error rate climbed 57%→62%→86% (cvar) and 57%→81%→95% (global), and
  `hard_covers_seen_errors_frac` fell to 0.95 / 0.71 — the hard pool is saturating
  with cases nothing fixes, and 21 slots can no longer cover known errors.
- **`anchor_error_rate = 0.0` in all 6 cycles.** The 9 in-batch anchor slots never
  contained an error, so the `ANCHOR REGRESSIONS` section never appeared in any
  error report. 30% of the mutator's context was spent on examples that carry no
  information.

`role_integrity.n_leak_d_select = 8–10` is the known false alarm (O8) — batch
indices live in fit-pool space, D_select in heldout space.

---

## Does the mutator get what it needs? (mostly no)

### The error signal is self-contradictory

Across both arms, 95–100% of the errors shown to the mutator are **adjacent-class**
(4↔5, 3↔4), and the two directions are almost perfectly balanced:

| Arm | total | adjacent | needs push UP | needs push DOWN | top confusions |
|-----|-------|----------|---------------|-----------------|----------------|
| cvar | 57 | 95% | 25 | 29 | (4→5) 18, (5→4) 16, (3→4) 11, (4→3) 9 |
| global | 59 | 100% | 27 | 32 | (4→5) 24, (5→4) 20, (3→4) 7, (4→3) 7 |

Concretely, cvar C3 asks the mutator to simultaneously rate *down*
`"Wow wow wow. Storyline was great… can't wait to start the next book"` (gold 4,
pred 5) and rate *up* `"Really loved this book, but Dani's dad was an ass"`
(gold 5, pred 4). These are indistinguishable at the text level — the difference
is reviewer idiosyncrasy on the 4/5 boundary (C5). **The mutator is not confused by
bad instructions; it is being asked to fit label noise.**

What it does in response is write one-directional caps —
`"Series/sequel mentions with moderate praise but no superlatives → 4, not 5"`,
`"'I liked' as strongest positive phrase → 4 at most"` — which necessarily flip
correct 5s to wrong 4s on unseen data. That is the −0.087 on c0.

### Group labels: instruction followed, information absent

Zero cluster-id leaks in artifacts (O13 fixed) and zero group references in any
surviving prompt. But per-group error counts are 6/3/2/1 in C1 rising to 7/4/3/3/1
in C3 — the mutator is asked to infer an observable text trigger for a group from
**1–3 examples**. Groups C, D, E carry no usable signal at all.

### Few-shot curation (P8) largely failed

Of the 6 examples in cvar's final prompt, **only 1** traces to a real failing
review from the error reports. Global: **0 of 5**. The rest are inventions:

```
Review: Complete waste of money. Broke after one use.        Rating: 1
Review: Good quality, fits well. Shipping took a while though. Rating: 4
```

These are clean, easy, textbook cases — they teach nothing about the 4/5 boundary
where all the errors live, and they are unverified (the mutator is asserting gold
labels for text it made up). The instruction to "turn a failing review into an
example" is being read as "write an example in that spirit". H4 should be
downgraded from Pass to Fail: the mechanism fires, the content is wrong.

### Domain skew

12 of 18 errors in cvar C3 are book/media reviews, and the resulting rules are
overtly book-specific (`Series/sequel`, `Plot/feature description`, `character,
scene`). Clusters are `pred_profile` (prediction-disagreement) groups, not category
groups, so nothing in the acquisition policy balances product category. The prompt
specialises on Books and then meets an all-category OOD test.

---

## Budget

| Arm | Worker tokens (ensemble) |
|-----|--------------------------|
| global | **10.53M** (deepseek 3.48 / kimi 3.52 / qwen 3.53) |
| prev cvar `091203` | 8.69M |
| cvar `213313` | not written (`finish_interrupted_e1_run` skipped the tracker); ~10–12M by structure |

Global wall time ~3.3 h. Pair calendar 29 Jul 21:33 → 30 Jul 08:31 with a reboot gap.
Roughly two thirds of that spend was on cycles 2–3, which selection discarded.

---

## Conclusions

1. **The measurement, not the optimiser, is the binding constraint.** At 120
   test users / 8 reviews each, the minimum detectable CVaR effect is ~0.10 and
   we are chasing 0.02–0.04. Every arm-vs-arm claim in this and the previous three
   runs is inside the noise band. No amount of Pareto/consolidation engineering
   fixes that.
2. **`CVaR_cluster` at K=5, q=0.2 is just `R_worst_cluster`** — a single cluster of
   10–13 test users. It is not a robustness metric at this scale, it is a lottery.
3. **The tail identity does not transfer** from D_select to test (rank Spearman
   +0.70 but different argmin). Optimising "the worst group" on a 240-example
   selection set moves a different group than the one that is worst on OOD data.
   This is a real, reportable finding about group-robust prompt optimisation —
   arguably the most interesting result of the run.
4. **The mutator's task is partly unlearnable.** ~95% of its feedback is 4↔5 /
   3↔4 adjacent confusions with balanced directions. It responds with directional
   caps that trade one error for another. The ceiling is set by label noise, and
   we are spending the whole budget against it.
5. **Search capacity is no longer the bottleneck** (O17's fix worked) — more
   search now *hurts* generalisation. cvar with 12 iterations/cycle scored better
   on D_select and worse on val and test than cvar with 5.
6. **Consolidation is stateless and inert**: restarts from the original base every
   cycle, always shrinks earned rules, always loses, never propagates. It should
   either be redesigned or turned off.
7. **The anchor gate has no operating point** at |D_anchor|=100 with these flip
   counts — it cannot reject at α=0.05 below 5 discordant regressions.

## What to do next (in priority order)

1. **Fix the measurement before running anything else.** Raise
   `max_reviews_per_user` (per-user accuracy is quantised to 1/8 — M9) and/or test
   users well past 120, and switch every comparison to the paired bootstrap in
   `scripts/paired_significance.py`. Report CIs, never point estimates.
2. **Redefine the tail objective.** Either `tail_quantile ≥ 0.4` (so CVaR averages
   ≥ 2 clusters), or drop cluster-CVaR for a user-level tail (`R_tail` over the
   worst decile of users, which at least has 12–24 units), or shrink cluster
   accuracies toward the global mean before taking the min.
3. **Always evaluate the initial prompt on test as a floor**, and add it to the
   candidate set in the final selection. This run's cvar arm shipped something
   worse than doing nothing, and we only found out because the baseline was
   computed after the fact.
4. **Give the mutator a learnable signal.** Filter the error report to
   non-adjacent errors and to cases where workers *agree* on the wrong answer
   (d = 0.00), which are systematic rather than noise. Balance the report by
   product category, not only by cluster. Show both directions explicitly so it
   stops writing one-way caps.
5. **Fix few-shot curation or drop it.** Quote the failing review verbatim with
   its gold label from the artifact (mechanically, not by asking the LLM), or
   remove `<FewShotExamples>` from the mutation targets.
6. **Consolidation: make it stateful or disable it.** Feed it the *previous*
   consolidated base, forbid deleting DynamicRules it does not promote, and run it
   once at the end rather than every cycle.
7. **Anchor gate: switch to a CI-based tolerance** (reject if the upper bound of
   the drop's CI exceeds δ) so it has an operating point, or accept it as a
   monitor rather than a gate.
8. **Stop the run at 1–2 cycles** until selection can distinguish cycles. Both
   arms locked C1; C2–C3 cost ~two thirds of the budget for nothing.

## Artifacts to reuse

- `.../seed42_20260729_213313/evals/initial_prompt/` — metrics + `worker_predictions.npy` (3×960)
- `.../evals/final_selected/` — same for the C1 heir
- `baseline_initial_prompt_test.json` + `baseline_initial_*_predictions.npy` aliases
- `scripts/paired_significance.py` — paired bootstrap + McNemar + power floor
- Controller now persists `evals/eval_test|eval_validation/` worker votes on future runs

# Findings — low-variance objective run (seed42_20260730_114314)

Single `cvar_lex` arm, 3 AL cycles, objective `q=0.40, shrink_prior_weight=50,
class_balanced=true`, D_select 360, OE 8 iters / population 20, test 240 users.
Cost: ~22.8M tokens across 3 workers, 5.5 h wall clock. Figures in
`results/E1_pred_profile_cvar_lowvar/seed42_20260730_114314/analysis/`.

## Headline

**The measurement layer is fixed, and the first statistically significant result
of the project came out of it: on deployment metrics the evolved prompt is
significantly worse than the initial one.** Paired on the same 240 test users:
`R_global` 0.7240 → 0.6714 (−0.053, CI95 [−0.074, −0.032], p < 0.001), MAE
0.299 → 0.357, McNemar broke 216 / fixed 115, every cluster down (−0.035…−0.109).

That sentence needs its second half: **in the objective's own (class-balanced)
terms the prompt did what it was told.** Every minority class improved and the
majority class paid:

| gold class | share | initial | final | delta |
|---|---|---|---|---|
| 1★ | 1.5% | 0.690 | 0.724 | +0.034 |
| 2★ | 3.6% | 0.729 | 0.743 | +0.014 |
| 3★ | 10.2% | 0.536 | 0.571 | +0.036 |
| 4★ | 25.7% | 0.431 | 0.528 | **+0.097** |
| 5★ | 58.9% | 0.885 | 0.745 | **−0.140** |

341 test predictions moved down, 12 up. Test `R_macro` +0.008 (n.s.), test
balanced-shrunk CVaR −0.015 (n.s.). So the run's outcome decomposes into two
separate problems, and conflating them would point at the wrong fixes.

## Problem 1 — the objective definition bought minority accuracy at 1:39

`class_balance_weights` fully equalises classes, so one 1★ example outweighs
thirty-nine 5★ examples on this label mix. The search found the exchange rate and
used it: the 4★ gain (+0.097 on 494 examples) is real discrimination — the
calibration rules and few-shot examples that produced it are legible — but the
price (−0.140 on 1131 examples) is invisible to the objective and ruinous to raw
accuracy, which is what WILDS reports. C11 described this trade as a metric
*exploit*; full balancing turned it into the metric's *definition*.

Fix: soften the weights (√-inverse-frequency caps the exchange rate at ~6:1
instead of 39:1), or keep raw-global as the lexicographic tie-break with a much
larger epsilon. Full equalisation is the wrong operating point for a benchmark
scored on raw accuracy.

## Problem 2 — selection overfitting is real and now cleanly measured

In the objective's own units, D_select said +0.143 (0.480 → 0.624, sixteen noise
SDs — see fig1) while the same quantity on test moved −0.015. **Transfer of the
D_select gain was zero.** This is no longer confounded with measurement noise
(0.009 SD, verified); it is selection pressure against 360 fixed examples reused
across ~24 candidate evaluations and 3 cycles.

The in-run signals caught it in real time:

- **Anchor monitor, cycle 1** (fig4): the winning candidate broke 15/100
  previously-solved anchors, improved 0, exact p ≈ 0.0 — and 13 of the 15 were
  gold 4–5, i.e. the same majority-class damage the test later confirmed. M12
  ("the gate cannot reach significance") is obsolete: at `d_anchor=100` the gate
  has an operating point and fired; we had just demoted it to monitor in this
  same run.
- **Damage report, cycles 2–3**: fixed 0 / broke 4, then 1 / 3 — the mutator was
  trading, not improving.
- **Error triage**: systematic share collapsed 50% → 38% → 12% — the fixable
  errors were consumed in cycle 1; what remained was contested noise.

The val selection key did **not** catch it (0.5116 → 0.5182, rising while test
fell): 150 val users are inside the noise for this effect size, and val shares
the objective's blind spot by construction.

## Runplan checklist verdict

| # | check | verdict |
|---|---|---|
| 1 | ≥2 cycles with evo gain > 2 SD | **FAIL** — C1 +0.137 (15 SD), C2 +0.007 (0.7 SD), C3 0. But now a *fact*, not a suspicion |
| 2 | `best_selection_cycle` > 1 | pass formally (cycle 2) — and test shows the selection was wrong anyway |
| 3 | val trend not declining | pass — which is itself a finding: val cannot police this failure mode |
| 4 | systematic errors ≥25% and rules trace to them | pass in C1 (50%), collapsed to 12% by C3 |
| 5 | damage fixed > broke | **FAIL** — 0/4 and 1/3 |
| 6 | few-shot verbatim ≥80% | pass — both mutator-added examples are verbatim real reviews (prev run: 1 of 6) |
| 7 | consolidation wins any cycle | **FAIL** — −0.122, −0.097; guard confirmed no rule loss, so the paraphrase itself costs. Delete the mechanism |
| 8 | archive provenance survives | pass — `carried origin=oe@1` visible end to end; final champion slots all `carried` |
| 9 | anchor trend flat/rising | **FAIL in C1** (0.98 → 0.83), partial recovery C2 (0.85) — the most informative signal of the run |
| 10 | paired test vs initial with CI | done — significant regression on raw metrics, macro-neutral in balanced terms |

Per the runplan's failure branch (checks 1 close to failing + transfer zero): stop
re-tuning metrics, instrument the search.

## What the run positively established

1. The noise fix works: candidate ranking, per-cycle attribution, and the final
   paired test are all now interpretable. This run cost ~$25 in tokens and produced
   more usable evidence than the four previous runs combined.
2. The mutator, given the rewritten feedback, produces legible bidirectional rules
   and verbatim few-shot examples — mechanism quality is no longer the blocker.
3. The champion archive + provenance behave exactly as designed (heir `carried`
   in C3 is correct behaviour: nothing beat the cycle-2 prompt).
4. Consolidation has now lost 8/8 scored attempts across two runs while provably
   not deleting content. The hypothesis "reorganisation adds value" is dead.

## Decisions for the next run

1. **Anchor gate back to `reject`** — it has an operating point at 100 anchors and
   it identified precisely the prompt that destroyed the test result, at cycle 1,
   for free.
2. **Break the fixed-D_select feedback loop** — rotate/resample D_select between
   cycles from the heldout pool (the pool has 736 examples; two disjoint 360s
   alternate cleanly), or hold out a fraction used only for heir selection.
3. **Soften class balancing** to √-inverse-frequency weights, and re-run the
   objective sweep offline on the saved repeats to re-verify noise before
   launching (free).
4. **Remove consolidation** (`enabled: false`); keep the code for the ablation
   record.
5. Keep: shrinkage, the mutator feedback format, the champion archive, provenance,
   the 240-user test protocol.

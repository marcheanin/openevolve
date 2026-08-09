# Results — `E1_v1_weighted_mini/seed42_20260801_061506`

Head-to-head of authentic **v1 Hard/Anchor fitness** on the Phase 1 substrate
(official WILDS OOD, same workers/mutator/caps/clusters/gate). Wall ~1.0 h,
~8.3M worker tokens. Metrics dump: `results/.../analysis/compare_metrics.json`.

Fair comparison protocol: paired OOD test vs **seed-prompt** ensemble
(`pred_cache/cluster_assign_test_ens.npy`). In-loop batch scores are **not**
comparable to Phase 1's D_select scores.

---

## Method delta vs Phase 1

| Knob | Phase 1 (`E1_constraint_global`) | this mini |
|------|----------------------------------|-----------|
| `fitness.mode` | `global` on D_select | **`v1_weighted` on active Hard/Anchor batch** |
| Formula | R_global | 0.5·Acc_Hard + 0.3·Acc_Anchor + 0.2·κ − len_pen |
| OE iters / cycle | 8 | **4** |
| gate / consolidation | reject / off | same |
| seed prompt | `prompts/initial_prompt.txt` (short) | same |
| synth few-shot | none | none |

Wiring note: OE now routes `v1_weighted` → `_evaluate_active` (previously fell
through to D_select / legacy mix). Confirmed live: `eval_set=active_batch_v1`,
C1 Acc_Hard=0.524 / Acc_Anchor=1.0.

---

## Headline (paired OOD test)

| Metric | Initial (seed) | Final | Δ |
|--------|---------------:|------:|--:|
| R_global | 0.7219 | 0.7240 | **+0.0021** |
| R_macro | 0.664 | 0.652 | −0.012 |
| CVaR_q40 | 0.6488 | 0.6464 | −0.0024 |
| MAE | 0.304 | 0.297 | better |
| R_worst (p10) | 0.488 | 0.375 | −0.113 (M9 lattice) |

Paired user bootstrap: mean Δ **+0.002**, CI95 **[−0.008, +0.013]**, P(Δ≤0)=0.36.  
McNemar: fixed 40 / broke 36. Pred shifts: **39↓ / 50↑** (near-noise; Phase 1 was 22↓ / 115↑).

### vs Phase 1 final

| Metric | mini | Phase 1 | Δ |
|--------|-----:|--------:|--:|
| R_global | 0.7240 | 0.7260 | −0.002 |
| CVaR_q40 | 0.6464 | 0.6672 | −0.021 |
| R_worst | 0.375 | 0.375 | 0 |

Seed ensembles bit-identical between runs (`seed agree = 1.0`).

**Verdict: no wake-up.** Hard/Anchor on this substrate did not beat Phase 1's
mild promotion, and did not move test beyond noise.

---

## What the search actually did

### Final prompt ≈ seed

`al_iter_3/best_prompt.txt` equals `initial_prompt.txt` after strip
(only a trailing newline differs; 1681 vs 1682 chars). No substantive DynamicRules
/ few-shot edits survived.

### Per-cycle OE (batch fitness)

| Cycle | entry fitness | Acc_Hard | best_evo | heir | note |
|------:|-------------:|---------:|---------:|------|------|
| 1 | 0.729 | 0.524 | **0.718** | `cycle_entry` | all 4 mutations ≤ seed; fair same-batch compare |
| 2 | 0.654 | 0.381 | 0.729† | `oe` | †stale seed metric from C1 (see bug) |
| 3 | 0.569 | 0.333 | 0.729† | `carried` | same |

Cycle-1 is the only clean evidence: on a fixed Hard/Anchor batch, 4 mutations
all lost to the seed (best mutant 0.708 vs seed 0.718).

Gate: 2 evaluations, both `within_tolerance` (drop 0.01), `n_rejected=0`.

### Bug: stale metrics in Pareto seed checkpoint (O26)

`write_seed_checkpoint` copies program JSONs **verbatim**, including
`combined_score` from the previous cycle's active batch. With
`fitness.mode=v1_weighted` the eval set **rotates every cycle**, but carried
programs keep Acc_Hard from cycle 1 (0.729) while new mutations are scored on
the harder C2/C3 batches (0.54–0.67). Seed always wins → C2/C3 OE is
**invalid as a search test**.

This does **not** affect Phase 1 / `global` on fixed D_select (same set every
cycle). It only bites batch-local fitness.

### Seed prompt ≠ successful v1

Current GRAPE seed: short `prompts/initial_prompt.txt` (~1.7k).  
Successful v1 all-categories: `initial_prompt_all_categories.txt` (~3.4k) with
signal lexicon, heuristics, edge cases, 7 few-shots. Same skeleton, different
headroom.

---

## Interpretation

1. **Mechanics of `v1_weighted` routing work** (batch eval, Acc_Hard logged).
2. **On the short seed + mini budget + official OOD**, Hard/Anchor does not
   reproduce v1's historical climb — cycle 1 already shows mutations lose.
3. **C2/C3 cannot exonerate or condemn** the fitness until seed programs are
   re-scored on the new batch (O26).
4. Together with Phase 0/1: Amazon user-shift under this prompt-only stack stays
   a **mechanics / negative** case. Next scientific bet remains CivilComments /
   category-shift (Phase 2), not more CVaR/Hard/Anchor on Amazon with the short seed.

Optional follow-ups (only if we reopen Amazon):
- Fix O26 (re-eval seed checkpoint on current batch before OE).
- Rerun with `initial_prompt_all_categories.txt` + full OE budget (8×3) + synth
  few-shot — that is the real v1 reproduction, not this mini.

---

## Decision

Treat mini as **negative on wake-up criteria** for Hard/Anchor under current
GRAPE seed/OOD. Keep Phase 1 skeleton (`global` + `reject`) as default. Do not
promote `v1_weighted` without fixing O26 and matching the v1 seed.

# Run plan — Pareto v2 pair (champion archive, 2026-07-29)

Status: **ready to launch, not started**. Both arms rerun with the O17–O19 / P8–P9
fixes; the previous Pareto run `seed42_20260729_091203` is the direct baseline
(same pinned clusters and eval sets).

## What changed since `seed42_20260729_091203`

| # | Change | Ref |
|---|--------|-----|
| 1 | OE budget 5→**12** mutations, population 8→**20**, patience 3→**5** | O17, PROTOCOL 23 |
| 2 | Consolidation on **gemini-3.1-pro-preview** (was glm-5 → empty → stub every cycle) | O19, PROTOCOL 24 |
| 3 | `trim_front` can no longer evict the scalar-best | O18 |
| 4 | **Champion archive** replaces dominance-front membership: slot 0 = scalar-best, one slot per cluster (shrunk accuracy), incumbent keeps slot within `champion_margin` 0.01 | P9, PROTOCOL 26 |
| 5 | Mutator may curate `<FewShotExamples>` from failing reviews (≤6, weakest groups, no group labels) | P8, PROTOCOL 25 |
| 6 | Controller candidates now use **shrunk** cluster accuracies (same estimator as OE programs) | P9 |

Pinned as before: `dataset.cluster_artifact` and `data_roles.eval_sets_artifact`
from `seed42_20260728_125748` — both arms and the baseline share groups and
eval splits exactly.

## Launch

```bash
cd openevolve/examples/llm_prompt_optimization/prime_v2_group_robust
python scripts/run_e1_pred_profile_pair.py   # cvar_lex arm first, then global
```

Expected cost/time per arm: ~4 h at 5 OE iters before; evolution roughly
doubles → **estimate 5.5–6.5 h and ~15–18M worker tokens per arm**. Budget cap
55 000 calls stays (`on_exhausted: warn`); PredictionCache still dedupes
re-scored prompts, and archive members are re-scored from cache for free.

## What we monitor while it runs

Per cycle, in order of diagnostic value:

1. **`best_evo` trajectory** (stage `10_evolution`, `summary.cycles[*].best_evo_score`).
   The single most important number: last run it was 0.5824 / 0.5824 / 0.5824.
2. **Consolidation health** (`al_iter_*/consolidation.json` + console):
   no `[GRAPE WARNING] ... using stub`; `delta_vs_evolved` per cycle.
3. **Champion slots** (`al_iter_*/pareto_front.json.champion_slots`): which
   source holds each cluster slot, and how often slots change hands (churn).
4. **`heir_source`** per cycle and `seed_mode` (`pareto_checkpoint` expected
   from C2 on).
5. **Anchor gate**: rejections per cycle (`anchor_gate.json`,
   `rejected_by_anchor_gate_*.txt`); last run C3 rejected the whole front.
6. **FewShot mutations**: mutation logs naming `FewShotExamples` as the block;
   the examples must contain no group/cluster wording (O13 check applies).
7. **Batch health** (`batch_diagnostics.json`): hard error rate, weak-cluster
   coverage — was healthy on both previous runs, should stay so.
8. **Budget**: `budget_report.json` vs the 55k cap; OpenRouter balance.

## What we check (hypotheses) and what we expect

| Hypothesis | Pass looks like | Fail looks like |
|------------|-----------------|-----------------|
| H1. Starvation, not selection, caused the freeze (O17) | `best_evo` strictly improves in ≥2 of 3 cycles; C1 exceeds 0.582 or later cycles beat their entry | `best_evo` flat again at 12 mutations → the bottleneck is signal quality (quantized fitness), not budget → next lever is screening/denoising, not more mutations |
| H2. Real consolidation helps (O19) | 0 stub fallbacks; `delta_vs_evolved` ≥ −0.01, ≥1 cycle positive; consolidated occasionally *earns* heirship | Gemini parses but deltas stay ≤ −0.02 → consolidation timing is wrong (mid-run vs final), consider consolidating only at run end |
| H3. Champion archive preserves specialists without noise churn (P9) | Slot churn ≤ ~1 slot/cycle after C1; weak-cluster slots (0/3 on the pinned partition) eventually held by prompts with visibly targeted rules; scalar-best always in archive | Slots flip every cycle → margin too small for D_select noise; raise `champion_margin` to ~0.02 |
| H4. Few-shot curation is used and legal (P8) | ≥1 FewShotExamples mutation across the run; examples are plain review/rating pairs; any such mutation that wins shows up in per-cluster acc of its target group | Mutator never touches the block (instruction ignored) or writes group-labeled examples → tighten artifact footer wording |
| H5. The loop beats its own baseline | val `CVaR_cluster` ≥ 0.633 (baseline plateau) and test `CVaR_cluster` > 0.587 / `R_tail` > 0.401 for the cvar arm | Metrics flat while `best_evo` rises → D_select overfitting; check proxy corr (`proxy_cvar_tail_corr`) |
| H6. Р11 core question | cvar arm > global arm on test `CVaR_cluster` and `R_tail` under identical mechanics | Global wins or ties → group-aware fitness does not pay at this scale; report honestly |

Go / no-go for the pair comparison stays as in README: both arms complete on
the same pinned artifacts, no budget exhaustion, no cluster mismatch.

## Abort criteria (stop early, fix, relaunch)

- Any `using stub` consolidation warning (O16/O19 regression).
- `best_evo` identical to cycle entry after C2 at 12 mutations (H1 hard-fail —
  no point burning the global arm).
- `seed_mode: fresh` in C2+ (archive seeding broke).
- Budget cap warning before C3, or cluster/eval-set artifact hash mismatch.

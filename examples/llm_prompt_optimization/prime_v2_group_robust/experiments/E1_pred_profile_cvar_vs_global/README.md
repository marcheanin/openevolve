# E1 — pred_profile groups: `cvar_lex` vs `global`

## Why this experiment

E0 large-cap (`seed42_20260727_115955`) showed:

- **Style** clusters (`full_T` / `emb_only`) do **not** separate hard users (KW p≈0.8).
- **`pred_profile`** (k-means on per-user predicted rating mix) does (KW H≈27, p≈0, K=6).
- Real difficulty driver: gold 4★ collapsed to 5 by the ensemble — not writing style.

So E1 uses **`clusters.geometry: pred_profile`**, K=6 (power rule: `|D_select|/n_min = 240/40`).

Lessons log: [`../OBSERVATIONS.md`](../OBSERVATIONS.md) (C4–C7, M6, M9, O9–O16, P1–P7).  
E0 findings: `results/E0_proxy_diag_large/seed42_20260727_115955/FINDINGS.md`.  
First completed arm + its postmortem: [`RESULTS_cvar_seed42_20260728_125748.md`](RESULTS_cvar_seed42_20260728_125748.md).

## Hypothesis (SPEC §7 / Р11)

Under equal API budget and the same start prompt / splits / pred_profile groups:

> **`cvar_lex` arm improves the official tail** (`CVaR_cluster`, `R_tail`) vs **`global` arm**.

Primary gate: `CVaR_cluster` and `R_tail` on official WILDS val (test = confirmation).  
`R_worst` is still logged but is no longer decisive: at 8 reviews per user its 10th
percentile is pinned to the 1/8 lattice and sat at exactly 0.500 for three cycles
(OBSERVATIONS M9). Surrogate-only win (`CVaR_cluster` without `R_tail`) = weak-go →
repair, not celebrate.

## Configs

| Arm | Config |
|-----|--------|
| CVaR | `config_cvar_live.yaml` |
| Global | `config_global_live.yaml` |

Shared: **200** train / **120 val / 120 test** × ≤8 reviews; pred_profile K≤6 with **`min_users_per_cluster: 8`**; roles D_select 240 / **D_anchor 100** / D_audit 50; acquisition **`group_aware`**; 3 AL × **12** OE iters at population **20** (O17); batch 30; **champion archive** (P9: 1 scalar-best slot + one per cluster, `champion_margin: 0.01`); consolidation on **gemini-3.1-pro-preview** (O19).

`dataset.cluster_artifact` **is pinned** to `results/E1_pred_profile_cvar_lex/seed42_20260728_125748/clusters.json` for both arms. Independent refits produced different partitions at the same seed (`{0:23,1:38,2:30,3:16,4:33}` vs `{0:24,1:30,2:22,3:44,4:20}`) because the seed-prompt ensemble is not bit-identical, and CVaR across different partitions is not a comparison (OBSERVATIONS O14). Pinning also makes this pair directly comparable to the first completed arm and skips ~44 min of clustering.

The D_anchor gate is a **significance test** (exact one-sided McNemar, α=0.05, δ floored at 2 examples), not strict no-regression — see `PROTOCOL.md` items 10–11 and OBSERVATIONS M7/M8. A point-δ gate at |D_anchor|=50 rejected a genuine +0.042 candidate on a 2-example anchor drop and stalled the whole loop. Since P7 a rejection also no longer voids the cycle: the gate walks down the Pareto front.

## What the loop does now (P7)

Per AL cycle:

1. Score the cycle-entry prompt on D_select; acquire a `group_aware` batch; build mutator artifacts.
2. Run OpenEvolve for 12 mutations at population 20, **resuming from the previous cycle's archive** so the mutator sees the whole population instead of one prompt.
3. Collect candidates: carried archive + cycle-entry prompt + every OE program in the last checkpoint + the consolidation variant. All already carry per-cluster accuracy on the same fixed D_select, so the archive is free apart from scoring the consolidated prompt.
4. Fill the **champion archive** (P9): slot 0 = scalar-best (never evicted), one slot per cluster for its champion on shrunk accuracy; incumbents keep slots unless beaten by > `champion_margin`.
5. Heir = scalar-best front member under `fitness.mode`; the anchor gate walks down the front on rejection; the walk stops at the incumbent so the heir is never worse than what entered the cycle.
6. Evaluate the heir on val once (cached), select across cycles, expand the pool, write the front as the next cycle's OE seed checkpoint.

The front is bookkeeping, not a second objective — `fitness.mode` still decides who inherits, so the `cvar_lex` vs `global` contrast of Р11 is intact and both arms get the identical mechanism.

## How to run

```bash
cd openevolve/examples/llm_prompt_optimization/prime_v2_group_robust

# Preflight
python scripts/run_e1_wilds_live.py --config experiments/E1_pred_profile_cvar_vs_global/config_cvar_live.yaml --dry-run

# Live arms (sequential; do not overlap API budgets)
python scripts/run_e1_wilds_live.py --config experiments/E1_pred_profile_cvar_vs_global/config_cvar_live.yaml
python scripts/run_e1_wilds_live.py --config experiments/E1_pred_profile_cvar_vs_global/config_global_live.yaml
```

Or both via helper (this is the preferred path — it enforces one shared cluster fit):

```bash
python scripts/run_e1_pred_profile_pair.py
# or pin an explicit fit for both arms
python scripts/run_e1_pred_profile_pair.py --cluster-artifact results/.../clusters.json
```

## Artifacts (every run dir)

`results/<experiment.name>/seed42_<timestamp>/`

| File | Purpose |
|------|---------|
| `config_used.yaml` | Exact config copy |
| `config_snapshot.json` | Resolved dataclasses |
| `run_metadata.json` | seed, git, geometry, fitness |
| `OBSERVATIONS.md` | Copied lessons log |
| `PROTOCOL.md` | This experiment's protocol (copied at start) |
| `FINDINGS.md` | Filled at end from `summary.json` |
| `initial_prompt.txt` | Start prompt |
| `clusters.json` | pred_profile centroids + scaler + descriptors (+ merges) |
| `cluster_fit_predictions.npy` | Seed-prompt preds on fit users |
| `cluster_fit_worker_predictions.npy` | Worker votes (disagreement features) |
| `cluster_assign_{split}.json` | OOD user→cluster maps |
| `source_split.json`, `eval_sets.json` | Fit/heldout + D_select/anchor/audit |
| `pred_cache/` | Cached ensemble passes |
| `events.jsonl`, `smoke_trace.jsonl` | Stage log |
| `al_iter_*/` | Per-cycle prompts, batches, OpenEvolve out |
| `al_iter_*/batch_diagnostics.json` | hard/anchor coverage, weak-cluster slots, seen/unseen |
| `al_iter_*/anchor_gate.json` | Gate decision: McNemar p, δ_effective, per-cell regressions, plus any members rejected before the accepted one |
| `al_iter_*/rejected_by_anchor_gate_<rank>.txt` | Front members the gate rolled back (kept for post-hoc review) |
| `al_iter_*/pareto_front.json` | Front membership, per-cluster champions, heir source, candidate count |
| `al_iter_*/consolidation.json` | Whether consolidation changed the prompt, its D_select fitness and `delta_vs_evolved` |
| `al_iter_*/consolidated_prompt.txt` | The consolidation candidate (contains `GRAPE_BASE_CONSOLIDATED` if the LLM call fell back to the stub — O16) |
| `al_iter_*/best_program_info.json` | OE best score + `seed_mode` (`fresh` / `pareto_checkpoint`) |
| `summary.json`, `budget_report.json` | Final metrics + tokens (incl. `anchor_gate` counter and `pareto.heir_sources` / `consolidation_deltas`) |

## Known honesty gaps (Phase 2 incomplete)

Still present (see IMPLEMENTATION.md): D_mut quotas / cascade not fully migrated; acquisition may still use legacy hard/anchor batch slots. This pilot tests **objective × pred_profile groups**, not the full GRAPE ladder. Document any limitation in `FINDINGS.md`.

**Not in scope for this pair:** islands / migration (`num_islands: 1`), per-group prompt
routing, and letting the front itself be the deliverable — a single prompt still has to
serve every group.

## Go / no-go

Compare the two run dirs' `summary.json → final_test` on `CVaR_cluster` and `R_tail`
(and the per-cycle val selection keys).  
Go = cvar_lex beats global on val `CVaR_cluster` **and** `R_tail` under the same seed,
caps and pinned clusters.

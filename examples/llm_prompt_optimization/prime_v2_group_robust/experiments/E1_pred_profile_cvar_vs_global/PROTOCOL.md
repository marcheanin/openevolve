# E1 protocol — pred_profile × (cvar_lex vs global)

Copied into each run directory at start.

## Locked from E0 (do not regress)

1. Group geometry = **`pred_profile`**, not style/`full_T` (OBSERVATIONS C6).
2. E0 gate = permutation Kruskal–Wallis; LOO Spearman is diagnostic only (M2/P1).
3. Confirm on large-cap before trusting small-cap E0 (M3).
4. Always persist ensemble **and** worker predictions for cluster fit (O3).

## Locked from first live E1 (2026-07-27)

5. Val/test caps ≥ **120 users** so R_worst_val has resolution (M6). Cap 40 saturates at 0.5.
6. After k-means, merge clusters with **&lt; 8 fit users** into nearest centroid (C7). Do not reuse pre-merge `cluster_artifact` for clean pairs.
7. OE: stage under `results/_oe/<token>/` + junction into `al_iter_*` (O4). ~~Inner loop 5 mutations, patience 3.~~ Superseded by item 23: **12** mutations, population **20**, patience **5** (O17).
8. ~~No full Pareto front yet (P5).~~ Superseded by item 14 below.

## Locked from aborted run `seed42_20260728_091955` (2026-07-28)

9. Acquisition is **`group_aware`** (O6): hard slots ∝ (1−Acc_k) so weak pred_profile types actually reach the mutator. `al_iter_*/batch_diagnostics.json` is written every cycle.
10. **D_anchor gate = significance test, not point-δ** (M7, P6): |D_anchor| = **100**, δ floored at 2 examples, reject only if exact one-sided McNemar p &lt; **0.05**. Regression is broken down per (cluster × gold) cell; rejected candidates are saved as `al_iter_*/rejected_by_anchor_gate.txt`; the rejection counter lands in `summary.anchor_gate`.
11. Do **not** let a D_select gain offset an anchor drop (M8) — the gate polices exactly that bias. Trade-offs are adjudicated post-hoc on val / final test.
12. `max_train_users` = **200** so heldout (~480) fits D_select 240 + D_anchor 100 + D_audit 50.
13. Mutator artifacts take **pool-space** hard/anchor ids (O5). Ignore `role_integrity.n_leak_d_select` — it compares two different index spaces (O8).

## Locked from completed run `seed42_20260728_125748` (2026-07-29)

14. **Cross-cycle Pareto front** over per-cluster accuracy on D_select (P7, replaces P5). Candidates each cycle = carried front + cycle-entry prompt + every OE program from the last checkpoint + the consolidation variant. All are measured on the same fixed D_select, so the front costs no extra calls except scoring the consolidated prompt. Written to `al_iter_*/pareto_front.json`. *Membership rule superseded by item 26 (champion archive, P9); the candidate pool and zero-cost property are unchanged.*
15. The **heir is the scalar-best front member** under `fitness.mode`, ties broken toward the newer candidate, and the walk stops at the incumbent — the front never promotes a prompt worse than the one that entered the cycle. Specialists survive *in the front*, not by being promoted (P7).
16. **Consolidation competes, it does not inherit** (O9/O10). It is scored on D_select in the run's own fitness metric and enters the front as one candidate; `consolidation.gate_delta` and the old val-vs-D_select gate are gone. The consolidator rewrites `BaseGuidelines` **and** `DynamicRules` in one call and must delete whatever it promotes. Check `al_iter_*/consolidation.json` `delta_vs_evolved` — negative means consolidation hurt and the front should have rejected it.
17. The anchor gate **walks down the front**: on rejection the next-best member is tried before falling back to the cycle-entry prompt. Rejects are kept as `rejected_by_anchor_gate_<rank>.txt`. The gate remains a constraint (P6/M8).
18. OE **resumes from the carried front** (`pareto.seed_openevolve_from_front`), so the mutator sees the whole population as top/diverse inspiration. Falls back to a fresh start with a printed warning on any error; `best_program_info.json.seed_mode` records which path ran.
19. **Prompts leaving OE are stripped of `# EVOLVE-BLOCK-*` markers** (O15).
20. **Groups are fitted once and pinned for both arms** via `dataset.cluster_artifact` (O14). Independent refits gave different partitions at the same seed, which voids the CVaR comparison.
21. Tail metric for proxy validation is **`R_tail`** (mean accuracy of the worst 20% of users), not `R_worst` — the 10th percentile is pinned to the 1/8 lattice at 8 reviews/user (M9). `R_worst` is still logged.
22. `_eval_split` (val/test) is cached per (prompt, split), removing the duplicate val pass (O12).

## Locked from Pareto run `seed42_20260729_091203` (2026-07-29)

23. **OE budget = 12 mutations / population 20 / patience 5** (O17). At 5/8/3 `best_evo` froze at the same value for three cycles: from a converged seed, 3–5 mutations almost never find an improvement, and everything downstream (front, consolidation, gate) had nothing to work with. The April system's activity came from ~15 mutations per cycle, not from its archive design.
24. **Consolidation runs on `consolidation.model`** (default `google/gemini-3.1-pro-preview`), never on the mutator model (O19). glm-5 returned empty replies every cycle → stub → −0.028 D_select. A stub outcome is visible in the console (`[GRAPE WARNING] ... using stub`) and via the `GRAPE_BASE_CONSOLIDATED` marker.
25. **Few-shot curation is a mutator move, not a separate generator** (P8). The mutator may rewrite `<FewShotExamples>` using failing reviews from the error report: plain review/rating pairs, ≤ 6 examples, aimed at the weakest groups, no group identifiers inside.
26. **Champion archive replaces dominance-front membership** (P9, supersedes the membership rule of items 14–15). Carried set = 1 slot for the scalar-best (never evicted — O18) + one slot per cluster for that cluster's champion on **shrunk** D_select accuracy. An incumbent keeps its slot unless a challenger beats it by more than `pareto.champion_margin` (0.01) — otherwise slots churn on noise and the OE seed population never stabilizes. Heir selection is unchanged: scalar-best first, anchor gate walks down (items 15/17). Slots are logged per cycle in `pareto_front.json.champion_slots` and finally in `summary.pareto.final_champion_slots`.

## This run

- Fit pred_profile on **train fit sources** with the **start prompt** ensemble (deployable geometry), or reuse the pinned artifact.
- Assign heldout/val/test via nearest centroid in the same scaled pred-profile space (cached under `pred_cache/`).
- Evolve with OpenEvolve; select by lexicographic (CVaR, R_global) on D_select / val.
- Primary comparison metric after both arms: **CVaR_cluster** and **R_tail** on official test (Р11; `R_worst` reported but no longer decisive — M9).

## After the run

Fill `FINDINGS.md` (auto-stub written by the launcher; edit conclusions by hand if needed):

- Did cvar_lex beat global on test CVaR_cluster / R_tail?
- Did CVaR_cluster move without R_tail? (weak-go)
- Any API failure rate / budget exhaustion?
- Any cluster collapse after merge (check `clusters.json` diagnostics.merges / sizes)?
- **Did both arms use the same pinned `clusters.json`?** If not, the pair is not comparable (O14).
- `summary.pareto.heir_sources`: which source won each cycle. All `consolidated` means consolidation is genuinely helping; all `cycle_entry` means the cycle produced nothing better and the loop is stalled for a real reason.
- `summary.pareto.consolidation_deltas`: the D_select cost of consolidation per cycle. On the previous run these were −0.015 and −0.066 while being accepted anyway.
- Front size per cycle and whether any weak-cluster champion ever became the heir — that is the mechanism P7 was adopted for.
- `summary.anchor_gate`: how often did the gate fire, and for what reason? With item 17 a rejection no longer voids the cycle, so a high rejection rate now means the front was shallow, not that the loop was throttled.
- If anchor regressions are **concentrated** in one (cluster × gold) cell while CVaR rises, that is evidence of a real tail trade-off rather than degradation — worth reporting either way (M8).
- Check the final prompt for cluster/group references; there should be none (O13).

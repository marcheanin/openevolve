# Все результаты экспериментов

Цифры только из **завершённых** прогонов. Пути к сырым артефактам — в конце.

---

## A. PRIME v1 (Amazon, custom split) — historical headline

Full uncapped test n=34 533 (`wilds_active_learn_approach/`):

| Method | R_worst | R_global | combined |
|--------|--------:|---------:|---------:|
| PRIME ensemble AL iter5 | **55.91%** | **75.41%** | **0.838** |
| Initial prompt ensemble | 50.0% | 71.3% | 0.782 |
| EvoPrompt GA (single GPT-4o-mini) | 56.50% | 73.92% | 0.744 |
| Same prompt, GPT-4o-mini only | 56.80% | 74.40% | 0.748 |

**Note:** custom 70/15/15 user split, **не** official WILDS OOD; inner fitness не group-robust.

Transfer на official OOD (v2 measurement, без новой эволюции): v1_final vs grape seed +0.011 n.s.

---

## B. Amazon E0 (offline bounds)

- Style clusters: **null** (KW p≈0.78).
- pred_profile: KW H≈27.6, p≈0 → adopted.
- Phase 0 offline (30 candidates × D_select 360): best raw R_global = **initial**; DRO top worse than accuracy.

---

## C. Amazon E1 (live, дорогой ансамбль)

| Run | Metric | Seed | Final | Δ | p |
|-----|--------|-----:|------:|--:|---|
| Lowvar CVaR | R_global | — | — | **−0.053** | <0.001 |
| Phase-1 `global` + reject | R_global | — | — | +0.004 | n.s. |
| v1 prompt transfer | vs grape seed | — | — | +0.011 | n.s. |

---

## D. Amazon E2 (cheap ensemble, canonical runs)

**User-OOD** `seed42_20260802_011259`:

| Metric | Seed | Final | Δ | p |
|--------|-----:|------:|--:|---|
| R_global | 0.690 | 0.663 | **−0.027** | 0.000 |
| R_tail | 0.380 | 0.370 | −0.010 | n.s. |

**Category-shift Books→non-Books** `seed42_20260802_052137`:

| Metric | Seed | Final | Δ | p |
|--------|-----:|------:|--:|---|
| R_global | 0.662 | 0.671 | **+0.009** | 0.045 |
| R_tail | 0.329 | 0.339 | +0.010 | n.s. |

---

## E. CivilComments E5 S9 — stable matrix (exploratory)

**Protocol:** test_fixed n=1800, stable ×3 repeats; scorer gemma-3-12b-it; D_dev n=900 uniform; regime-shift L=240.

### E.1 Mean over matrix seeds (sorted by R_worst_gba)

| method | R_worst_gba | R_global | softmin-GBA | Δ gba vs seed |
|--------|------------:|---------:|------------:|---------------:|
| gpo | 0.634 (±0.018) | 0.709 (±0.006) | 0.697 (±0.011) | +0.015 |
| evoprompt_de | 0.633 (±0.015) | 0.711 (±0.007) | 0.699 (±0.007) | +0.014 |
| **prime** | **0.633 (±0.012)** | **0.708 (±0.005)** | **0.695 (±0.007)** | **+0.014** |
| evoprompt_ga | 0.629 (±0.013) | 0.709 (±0.009) | 0.695 (±0.010) | +0.010 |
| ape_k48 | 0.624 (±0.005) | 0.706 (±0.009) | 0.691 (±0.007) | +0.005 |
| gepa | 0.621 (±0.004) | 0.700 (±0.002) | 0.686 (±0.002) | +0.002 |
| **seed** | **0.619 (±0.001)** | **0.702 (±0.001)** | **0.688 (±0.001)** | — |
| ape_ut | 0.617 (±0.003) | 0.700 (±0.004) | 0.685 (±0.003) | −0.002 |
| oracle | 0.615 | 0.685 | 0.670 | −0.004 |
| random_al | 0.609 (±0.034) | 0.692 (±0.006) | 0.678 (±0.011) | −0.009 |
| ape | 0.598 (±0.024) | 0.698 (±0.007) | 0.681 (±0.009) | −0.021 |
| apo | 0.596 (±0.003) | 0.687 (±0.001) | 0.669 (±0.003) | −0.023 |

### E.2 PRIME per seed (R_worst_gba)

| seed | PRIME | seed baseline | Δ |
|-----:|------:|--------------:|--:|
| 42 | 0.643 | 0.618 | +0.025 |
| 43 | 0.620 | 0.620 | 0.000 |
| 44 | 0.637 | 0.618 | +0.019 |

### E.3 Operating point (mean recall / specificity)

| method | toxic_recall | specificity |
|--------|-------------:|------------:|
| apo | 0.827 | 0.548 |
| ape | 0.788 | 0.607 |
| gpo | 0.756 | 0.662 |
| **prime** | **0.701** | **0.715** |
| seed | 0.748 | 0.655 |

### E.4 Prereg success criteria — **not met**

- Target Δ≥+0.05 vs seed: PRIME mean +0.014.
- Target ≥+0.02 vs best {APO,GPO,random_al}: PRIME vs GPO −0.001.

---

## F. Power audit (E5, cached preds, no new scoring)

Paired bootstrap method vs seed on test_fixed:

| Finding | Value |
|---------|------:|
| Contrasts resolvable at 95% (hard-min GBA) | **0 / 31** |
| Best Δ (GPO@42) | +0.035, CI95 [−0.005, +0.075] |
| Oracle vs seed | 0.615 vs 0.620 |
| Repeat sd / bootstrap sd (CVaR@25%) | 0.0008 / 0.0131 (~16×) |
| Scorer repeat share of variance | ~**1/250** |

Metric CI width (mean paired bootstrap):

| metric | mean CI width | resolved 95% |
|--------|--------------:|-------------:|
| hard-min GBA | 0.0724 | 0/31 |
| CVaR@25% | 0.0514 | 1/31 |
| mean GBA | 0.0249 | 7/31 |
| worst-class acc | 0.0434 | 24/31 |

Confound: corr(CVaR@25%, mean GBA) = **+0.83**; R² = **0.687**; adding |recall−spec| adds 0.000.

---

## G. R15 strictness control (12 one-line seed edits)

Pool: no search, no groups, no label budget. Metric: CVaR@25%.

| Item | @1800 (S9 rows) | @5251 | @clean 3451 |
|------|----------------:|------:|------------:|
| R15 dev-pick (`nl_lenient_max`) | **0.6475** | 0.6608 | 0.6613 |
| Best in sweep on test (`thr_p90`) | 0.6500 | 0.6617 | 0.6650 |
| seed | 0.6300 | 0.6542 | 0.6613 |
| GPO@42 (best S9 cell) | 0.6675 | — | — |
| PRIME@42 | 0.6600 | — | — |
| oracle | 0.6200 | — | — |

- R15 rank: **4th of 35**; **0/34** S9 cells beat R15 at Holm < 0.05.
- Numeric probability thresholds (90%→20%): recall flat ~0.73–0.76; NL strictness moves recall 0.72→0.85.

---

## H. R16 selection control (same pool, different dev rules)

**R15 pool (n=12), test CVaR@25%:**

| dev rule | pick | test cvar25 | rank corr |
|----------|------|------------:|----------:|
| worst-class (group-free) | nl_lenient_max | 0.6608 | **+0.71** |
| mean GBA | nl_lenient_max | 0.6608 | +0.71 |
| softmin shrunk (PRIME shipped) | thr_p20 | 0.6592 | +0.58 |
| CVaR@25% | nl_strict | 0.6317 | **−0.34** |
| hard-min GBA | nl_strict | 0.6317 | **−0.57** |

Rule-only swing: **~0.030** (comparable to full matrix spread ~0.070).

**S9 optimizer pool (n=26 unique prompts):**

| dev rule | pick | test cvar25 | rank corr |
|----------|------|------------:|----------:|
| worst-class | 44_evoprompt_de | 0.6450 | **+0.68** |
| mean GBA | 43_evoprompt_de | 0.6275 | +0.46 |
| CVaR@25% | 44_apo | 0.6075 | **−0.58** |
| hard-min | 44_apo | 0.6075 | **−0.62** |

Dev statistic sd on uniform D_dev (seed prompt): single-group GBA **0.0475**; hard-min **0.0401**.

---

## I. Statistic ablation (k, τ, shrink) — Spearman vs test CVaR@25%

**Pattern on both pools:** min-based rules negative; averaged rules positive; no interior optimum — improves monotonically toward mean GBA.

| statistic | R15 pool w=0 | R15 w=400 | S9 pool w=0 | S9 w=400 |
|-----------|-------------:|----------:|------------:|---------:|
| CVaR k=1 (hard-min) | −0.57 | +0.48 | −0.62 | +0.02 |
| CVaR k=8 (= mean) | +0.71 | +0.73 | +0.46 | +0.47 |
| softmin τ=0.1 (shipped) | +0.26 | +0.73 | +0.08 | +0.44 |
| CONTROL worst-class | +0.71 | — | +0.68 | — |

Lowest mean regret across pools: **CONTROL worst-class** (0.0117).

---

## J. Dev budget reallocation F9 (same 900 examples)

**Setup:** uniform D_dev = 9 groups × 50/cell; targeted = groups **8,3,5** (worst by seed on uniform pilot) × **150/cell**. Same total n=900. Groups chosen without test data.

Shared-row rescoring agreement (300 rows): **99.72%** mean, 99.33% min.

**Spearman vs test CVaR@25% (groups 3,5,8 only):**

| statistic | uniform n=100 | targeted n=300 | Δ |
|-----------|--------------:|-----------------:|--:|
| CVaR k=1 | −0.61 | **+0.24** | +0.85 |
| CVaR k=2 | −0.58 | **+0.38** | +0.96 |
| softmin τ=0.1, w=40 | −0.34 | **+0.55** | +0.89 |
| mean change (all stats) | — | — | **+0.876** |
| stats now positive | 0/12 | **12/12** | — |

**Selection regret (S9 pool, test CVaR@25%, pool best = 0.6675):**

| dev rule | uniform dev pick | regret | targeted dev pick | regret |
|----------|------------------|-------:|-------------------|-------:|
| all group stats | 44_apo | 0.0600 | **42_prime** | **0.0075** |
| worst-class | 44_evoprompt_de | 0.0225 | **42_gpo** | **0.0000** |

Targeted dev picks PRIME's prompt (`42_prime`, test cvar25=0.6600) under every group statistic tested.

---

## K. Fixed sets (E5)

| Set | n | fingerprint | role |
|-----|--:|-------------|------|
| test_fixed | 1800 | 5cfb7ebde3c5 | S9 report |
| test_fixed_large | 5251 | 1ed1114b72e6 | superset of 1800 |
| d_dev (uniform) | 900 | 1c10514d553c | selection baseline |
| d_dev_targeted | 900 | 594125cf6a52 | groups 8/3/5 × 150 |

---

## L. Amazon E6 category-shift controls + top-3 (2026-08-17)

Substrate: Books→non-Books, scorer `gpt-4o-mini`, pred_profile K=6, primary **cvar25 macro-within-cluster**.  
Fixed: test_fixed n=1376 / d_dev n=648 / d_dev_targeted n=570. Details: `prime_v2_group_robust/experiments/E6_amazon_category_controls/RESULTS.md`.

### L.1 Phase A — controls

| Finding | Value |
|---------|------:|
| Resolvable paired contrasts vs seed (global, 95%) | **3 / 13** |
| hard_min Spearman(dev→test cvar25) | **+0.40** |
| cvar25 Spearman(dev→test cvar25) | +0.26 |
| R_global Spearman(dev→test cvar25) | **−0.16** |
| F9 regret uniform → targeted | **0.025 → 0.007** |

Interpretation: Amazon **partially** replicates CivilComments. Power is still weak, and targeted dev still helps a lot, but **hard_min is not anti-correlated here**; instead the clearly bad selector is `R_global`.

### L.2 Test metrics on completed prompts

| prompt / method | cvar25 | hard_min | mean_macro | R_global | op_shift | mae |
|-----------------|-------:|---------:|-----------:|---------:|---------:|----:|
| seed | 0.432 | 0.388 | 0.496 | 0.579 | 0.055 | 0.464 |
| **GPO** | **0.458** | **0.428** | 0.503 | 0.576 | 0.110 | 0.456 |
| harshness `deflate_positive` | **0.458** | **0.442** | **0.505** | 0.581 | 0.045 | 0.461 |
| E2 category heir | 0.451 | 0.412 | 0.507 | **0.584** | 0.083 | 0.462 |
| strict_5 (F9 targeted pick) | 0.435 | 0.405 | 0.493 | 0.568 | **0.022** | 0.477 |
| lenient_max (uniform d_dev pick) | 0.431 | 0.388 | 0.493 | 0.576 | 0.084 | 0.465 |

Two notable observations:
- **GPO improves the primary robust metric** vs seed (**+0.026 cvar25**) but does **not** improve `R_global`; this is another example of global accuracy being a poor proxy for the target claim.
- A one-line harshness edit (`deflate_positive`) reaches the **same primary score as GPO**, so on Amazon too, part of the “optimization gain” can be matched by operating-point movement rather than a uniquely strong search method.

### L.3 F9 dev reallocation

| rule | oracle | uniform pick | uniform regret | targeted pick | targeted regret |
|------|--------|--------------|---------------:|---------------|----------------:|
| hard_min | `deflate_positive` | `lenient_max` | 0.025 | `strict_5` | **0.007** |

This matches the CivilComments mechanism: same total dev budget, better allocation.

### L.4 Top-3 status

| Method | test cvar25 | note |
|--------|------------:|------|
| seed | 0.432 | done |
| GPO | **0.458** | done; ties harshness `deflate_positive` |
| EvoPrompt-DE | — | hung gen 6/8 on 403 key limit |
| PRIME | — | live run finished; `final_prompt.txt` saved; test_fixed score pending |

---

## M. Сырые артефакты (репозиторий)

| Что | Путь |
|-----|------|
| S9 matrix + stable preds | `prime_v2_group_robust/results/E5_s9_matrix/` |
| Stable aggregate JSON | `…/stable_session/stable_aggregate.json` |
| Power audit JSON | `…/stable_session/metric_variance_audit.json` |
| R15/R16 preds + reports | `…/results/E5_selection_control/strictness_sweep/` |
| S9 pool dev preds | `…/E5_selection_control/s9_pool/` |
| Dev realloc | `…/E5_selection_control/dev_targeted/dev_budget_reallocation.json` |
| Statistic ablation | `…/E5_selection_control/selection_statistic_ablation.json` |
| Amazon E2 user-OOD | `…/results/E2_cheap_ensemble_global_tail/seed42_20260802_011259/` |
| Amazon E2 category | `…/results/E2_category_shift_books/seed42_20260802_052137/` |
| GPO gate | `experiments/E5_civilcomments/gpo_gate.json` |
| Amazon E6 controls / top-3 | `…/experiments/E6_amazon_category_controls/`, `…/results/E6_top3_matrix/seed42/` |

**Regenerate S9 table:**

```powershell
cd prime_v2_group_robust
python scripts/aggregate_e5_s9_stable.py
```

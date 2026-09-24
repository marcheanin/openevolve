# E5 CivilComments — RESULTS (S9)

Preregistration: [`PREREGISTRATION.md`](PREREGISTRATION.md).  
Gate: [`gpo_gate.json`](gpo_gate.json) (**pass**).  
Next steps: [`NEXT_STEPS.md`](NEXT_STEPS.md).  
E5v3 debt: [`E5V3_SELECTION_DEBT.md`](E5V3_SELECTION_DEBT.md) (M41).  
Status: [`S9_STATUS.md`](S9_STATUS.md).

## Protocol

- Headline search/select: `R_worst_gba` / softmin-GBA on `D_dev` (n=900).
- **Report:** same-session stable eval on fingerprinted `test_fixed` (**n=1800**, fp `5cfb7ebde3c5`), **3 repeats** per prompt; seed scored once per matrix seed (`_shared_seed{{42,43,44}}`).
- Format: `mean (±sd)` — within-job sd over 3 repeats; headline table sd is across the 3 matrix seeds (of per-seed means).
- Scorer: `google/gemma-3-12b-it` (T≈0). Optimizer: `deepseek/deepseek-v4-pro`.
- Regime-shift: `S_source=360`, `U_target=4000`, label budget `L=240` (AL).
- Artifacts: `results/E5_s9_matrix/seed{{42,43,44}}/`, `results/E5_s9_matrix/stable_session/`, aggregate `stable_session/stable_aggregate.json`.

> **This table is exploratory. Read [`POWER_AUDIT.md`](POWER_AUDIT.md) and
> [`R15_CALIBRATION_CONTROL.md`](R15_CALIBRATION_CONTROL.md) before using it.**
> Under a paired bootstrap none of the 31 method×seed contrasts is resolvable at
> 95%, ranks invert across matrix seeds, and the oracle upper bound falls below
> the seed prompt. A control of 12 one-line strictness edits (R15 — no search, no
> group labels, no label budget) lands 4th of 35 on identical rows, and no cell
> here beats it at Holm < 0.05. Headline claims moved to
> [`PREREGISTRATION_S10.md`](PREREGISTRATION_S10.md). `R_worst_group` is
> algebraically identical to `R_worst_gba` on balanced cells and carries no
> information — it is kept only for continuity and will be dropped.

## Headline — mean over seeds (stable ×3)

Sorted by mean `R_worst_gba`. Δ = method − seed (seed mean gba = 0.619).

| method | R_worst_gba | R_worst_group (Acc) | R_global | softmin-GBA | Δ gba | n seeds |
|--------|------------:|--------------------:|---------:|-----------:|------:|--------:|
| R6 gpo | 0.634 (±0.018) | 0.634 (±0.018) | 0.709 (±0.006) | 0.697 (±0.011) | +0.015 | 3 |
| R8 evoprompt_de | 0.633 (±0.015) | 0.633 (±0.015) | 0.711 (±0.007) | 0.699 (±0.007) | +0.014 | 3 |
| R10 prime | 0.633 (±0.012) | 0.633 (±0.012) | 0.708 (±0.005) | 0.695 (±0.007) | +0.014 | 3 |
| R7 evoprompt_ga | 0.629 (±0.013) | 0.629 (±0.013) | 0.709 (±0.009) | 0.695 (±0.010) | +0.010 | 3 |
| R3 ape_k48 | 0.624 (±0.005) | 0.624 (±0.005) | 0.706 (±0.009) | 0.691 (±0.007) | +0.005 | 3 |
| R9 gepa | 0.621 (±0.004) | 0.621 (±0.004) | 0.700 (±0.002) | 0.686 (±0.002) | +0.002 | 3 |
| R0 seed | 0.619 (±0.001) | 0.619 (±0.001) | 0.702 (±0.001) | 0.688 (±0.001) | — | 3 |
| R5 ape_ut | 0.617 (±0.003) | 0.617 (±0.003) | 0.700 (±0.004) | 0.685 (±0.003) | -0.002 | 3 |
| R12 oracle | 0.615 | 0.615 | 0.685 | 0.670 | -0.004 | 1 |
| R11 random_al | 0.609 (±0.034) | 0.609 (±0.034) | 0.692 (±0.006) | 0.678 (±0.011) | -0.009 | 3 |
| R2 ape | 0.598 (±0.024) | 0.598 (±0.024) | 0.698 (±0.007) | 0.681 (±0.009) | -0.021 | 3 |
| R4 apo | 0.596 (±0.003) | 0.596 (±0.003) | 0.687 (±0.001) | 0.669 (±0.003) | -0.023 | 3 |

### Extended metrics (mean over seeds)

| method | toxic_recall | specificity |
|--------|-------------:|------------:|
| gpo | 0.756 (±0.072) | 0.662 (±0.084) |
| evoprompt_de | 0.722 (±0.016) | 0.700 (±0.012) |
| prime | 0.701 (±0.049) | 0.715 (±0.059) |
| evoprompt_ga | 0.747 (±0.042) | 0.670 (±0.059) |
| ape_k48 | 0.777 (±0.034) | 0.636 (±0.030) |
| gepa | 0.776 (±0.047) | 0.625 (±0.051) |
| seed | 0.748 (±0.001) | 0.655 (±0.001) |
| ape_ut | 0.702 (±0.116) | 0.699 (±0.108) |
| oracle | 0.816 | 0.554 |
| random_al | 0.760 (±0.140) | 0.624 (±0.150) |
| ape | 0.788 (±0.054) | 0.607 (±0.067) |
| apo | 0.827 (±0.014) | 0.548 (±0.017) |

## Per-seed detail — seed 42 (stable ×3 within-job ±)

| method | R_worst_gba | R_worst_group | R_global | softmin | recall | spec |
|--------|------------:|--------------:|---------:|--------:|-------:|-----:|
| seed | 0.618 (±0.003) | 0.618 (±0.003) | 0.701 (±0.001) | 0.687 (±0.001) | 0.748 (±0.001) | 0.654 (±0.002) |
| ape | 0.572 (±0.002) | 0.572 (±0.003) | 0.690 (±0.001) | 0.670 (±0.001) | 0.849 (±0.001) | 0.530 (±0.001) |
| ape_k48 | 0.630 | 0.630 | 0.712 (±0.000) | 0.697 (±0.000) | 0.758 (±0.001) | 0.665 (±0.001) |
| ape_ut | 0.618 (±0.002) | 0.618 (±0.003) | 0.701 (±0.001) | 0.687 (±0.001) | 0.747 (±0.001) | 0.654 (±0.001) |
| apo | 0.600 | 0.600 | 0.686 (±0.000) | 0.667 (±0.000) | 0.834 | 0.539 (±0.001) |
| gpo | 0.655 | 0.655 | 0.716 (±0.001) | 0.709 (±0.001) | 0.674 (±0.001) | 0.758 (±0.001) |
| evoprompt_ga | 0.615 | 0.615 | 0.699 (±0.000) | 0.683 (±0.000) | 0.795 (±0.001) | 0.602 |
| evoprompt_de | 0.638 (±0.002) | 0.638 (±0.003) | 0.717 (±0.001) | 0.705 (±0.001) | 0.739 (±0.001) | 0.696 (±0.001) |
| gepa | 0.625 | 0.625 | 0.698 (±0.000) | 0.683 (±0.000) | 0.830 (±0.001) | 0.567 (±0.001) |
| prime | 0.643 (±0.002) | 0.643 (±0.003) | 0.712 (±0.000) | 0.702 (±0.000) | 0.652 (±0.002) | 0.771 (±0.001) |
| random_al | 0.590 | 0.590 | 0.692 (±0.000) | 0.674 (±0.000) | 0.847 (±0.001) | 0.537 (±0.001) |
| oracle | 0.615 | 0.615 | 0.685 (±0.000) | 0.670 (±0.000) | 0.816 (±0.001) | 0.554 (±0.001) |

## Per-seed detail — seed 43 (stable ×3 within-job ±)

| method | R_worst_gba | R_worst_group | R_global | softmin | recall | spec |
|--------|------------:|--------------:|---------:|--------:|-------:|-----:|
| seed | 0.620 | 0.620 | 0.701 (±0.001) | 0.687 (±0.001) | 0.748 (±0.001) | 0.654 (±0.002) |
| ape | 0.620 | 0.620 | 0.701 (±0.001) | 0.687 (±0.000) | 0.749 (±0.001) | 0.654 (±0.001) |
| ape_k48 | 0.620 | 0.620 | 0.711 (±0.000) | 0.693 (±0.001) | 0.816 (±0.001) | 0.606 (±0.001) |
| ape_ut | 0.613 (±0.002) | 0.613 (±0.003) | 0.705 (±0.001) | 0.687 (±0.000) | 0.789 (±0.001) | 0.620 (±0.001) |
| apo | 0.593 (±0.002) | 0.593 (±0.003) | 0.687 (±0.000) | 0.670 (±0.001) | 0.835 (±0.001) | 0.539 (±0.001) |
| gpo | 0.622 (±0.002) | 0.622 (±0.003) | 0.708 (±0.001) | 0.696 (±0.001) | 0.790 (±0.001) | 0.626 |
| evoprompt_ga | 0.632 (±0.002) | 0.632 (±0.003) | 0.713 (±0.001) | 0.700 (±0.001) | 0.718 (±0.001) | 0.708 (±0.001) |
| evoprompt_de | 0.617 (±0.002) | 0.617 (±0.003) | 0.704 (±0.001) | 0.691 (±0.001) | 0.718 (±0.002) | 0.691 (±0.001) |
| gepa | 0.620 | 0.620 | 0.702 (±0.001) | 0.688 (±0.001) | 0.748 (±0.001) | 0.656 (±0.001) |
| prime | 0.620 | 0.620 | 0.702 (±0.000) | 0.688 (±0.000) | 0.749 (±0.001) | 0.654 (±0.001) |
| random_al | 0.590 | 0.590 | 0.686 (±0.000) | 0.669 (±0.000) | 0.835 (±0.001) | 0.537 (±0.001) |

## Per-seed detail — seed 44 (stable ×3 within-job ±)

| method | R_worst_gba | R_worst_group | R_global | softmin | recall | spec |
|--------|------------:|--------------:|---------:|--------:|-------:|-----:|
| seed | 0.618 (±0.003) | 0.618 (±0.003) | 0.702 (±0.001) | 0.688 (±0.001) | 0.749 (±0.001) | 0.656 (±0.002) |
| ape | 0.602 (±0.002) | 0.602 (±0.003) | 0.701 (±0.001) | 0.685 (±0.001) | 0.766 (±0.001) | 0.637 (±0.001) |
| ape_k48 | 0.622 (±0.002) | 0.622 (±0.003) | 0.696 (±0.000) | 0.683 (±0.000) | 0.756 (±0.001) | 0.637 (±0.001) |
| ape_ut | 0.620 | 0.620 | 0.696 (±0.001) | 0.682 (±0.000) | 0.570 (±0.002) | 0.822 (±0.001) |
| apo | 0.595 | 0.595 | 0.689 (±0.001) | 0.672 (±0.001) | 0.811 (±0.001) | 0.567 (±0.001) |
| gpo | 0.625 | 0.625 | 0.704 (±0.001) | 0.686 (±0.001) | 0.805 (±0.001) | 0.603 (±0.001) |
| evoprompt_ga | 0.640 | 0.640 | 0.714 (±0.000) | 0.701 (±0.000) | 0.729 (±0.000) | 0.700 (±0.001) |
| evoprompt_de | 0.645 | 0.645 | 0.711 (±0.000) | 0.700 (±0.000) | 0.709 (±0.001) | 0.713 (±0.001) |
| gepa | 0.617 (±0.002) | 0.617 (±0.003) | 0.701 (±0.001) | 0.687 (±0.001) | 0.749 (±0.001) | 0.653 (±0.001) |
| prime | 0.637 (±0.002) | 0.637 (±0.003) | 0.710 (±0.001) | 0.697 (±0.001) | 0.701 (±0.001) | 0.719 (±0.001) |
| random_al | 0.648 (±0.002) | 0.648 (±0.003) | 0.698 (±0.000) | 0.690 (±0.000) | 0.599 | 0.797 (±0.001) |

## Methods & setup

| ID | Method | Setup in this matrix |
|----|--------|----------------------|
| R0 | **Seed prompt** (`seed`) | Fixed `initial_prompt_civilcomments.txt`; no optimization. |
| R2 | **APE** (`ape`) | K=6 proposals, N=36 demonstrations; select by D_dev softmin-GBA. |
| R3 | **APE-K48** (`ape_k48`) | Budget-matched APE with K=48 proposals, same N=36. |
| R5 | **APE-ut** (`ape_ut`) | APE with unlabeled-target demo pool (regime-shift). |
| R4 | **APO/ProTeGi** (`apo`) | Exploratory: **2 rounds** (native 6), beam=4. |
| R6 | **GPO** (`gpo`) | K=6, conf threshold T=0.83; stage-1 filter + upsample. |
| R7 | **EvoPrompt-GA** (`evoprompt_ga`) | Exploratory: reduced pop/gens vs native 10×10. |
| R8 | **EvoPrompt-DE** (`evoprompt_de`) | Exploratory: reduced pop/gens vs native 10×10. |
| R9 | **GEPA** (`gepa`) | Lightweight reflective+Pareto adapter (PyPI `gepa` not used). |
| R10 | **PRIME-main** (`prime`) | Seed42=E5v2 artifact; 43=`…074014` (shipped seed); 44=`…110051` (non-seed OE). D_dev Top-1 selection; soft_min_lex + GBA. |
| R11 | **Random-AL** (`random_al`) | APO + 240 random target labels; exploratory **2 rounds**. |
| R12 | **Oracle** (`oracle`) | APO on fully labeled target; **seed42 only**; exploratory 2 rounds. |

### Shared constants

- Same F7 prompt contract / parser for all methods.
- Selection / gate on `D_dev`; never optimize on `test_fixed`.
- OPRO (R13) and MIPROv2 (R14) **deferred** (not in this table).
- Exploratory cost caps (APO/Random-AL/Oracle rounds=2; reduced Evo; lite GEPA) — not full native budgets; interpret accordingly.

## Gate (Yelp→Flipkart)

**pass** — see `gpo_gate.json`.

## Success criteria (provisional, stable means)

- PRIME mean gba 0.633; Δ vs seed +0.014 (prereg target Δ≥+0.05 — **not met**).
- Best of {APO, GPO, Random-AL}: **gpo** 0.634; PRIME − rival = -0.001 (prereg want ≥+0.02).

## Appendix — Amazon E6 category-shift (partial replication)

This appendix is **not directly comparable** to the CivilComments S9 tables above:
different dataset, scorer, group construction, and headline metric. It is included
here as the cross-dataset follow-up to the E5 diagnosis.

**Protocol.**
- Dataset: Amazon WILDS, **Books -> non-Books** category shift.
- Scorer: single `gpt-4o-mini`.
- Groups: `pred_profile`, K=6.
- Primary metric: **CVaR@25% of macro-per-class within cluster** on `test_fixed`.
- Fixed sets: `test_fixed` n=1376, `d_dev` n=648, `d_dev_targeted` n=570.

### Completed top-line results

| method / prompt | test cvar25 | hard_min | mean_macro | R_global | op_shift | note |
|---|---:|---:|---:|---:|---:|---|
| seed | 0.432 | 0.388 | 0.496 | 0.579 | 0.055 | Amazon initial prompt |
| **gpo** | **0.458** | 0.428 | 0.503 | 0.576 | 0.110 | best completed top-3 optimizer |
| `deflate_positive` | **0.458** | **0.442** | **0.505** | 0.581 | 0.045 | harshness control; ties GPO on primary |
| E2 category heir | 0.451 | 0.412 | 0.507 | **0.584** | 0.083 | inherited optimizer prompt |
| `strict_5` | 0.435 | 0.405 | 0.493 | 0.568 | **0.022** | pick from targeted `d_dev` |
| `lenient_max` | 0.431 | 0.388 | 0.493 | 0.576 | 0.084 | pick from uniform `d_dev` |

### Amazon E6 controls

| finding | value |
|---|---:|
| Paired contrasts resolvable vs seed (95%, global) | **3 / 13** |
| Spearman: `hard_min` on uniform `d_dev` -> test cvar25 | **+0.40** |
| Spearman: `R_global` on uniform `d_dev` -> test cvar25 | **−0.16** |
| F9 regret: uniform `d_dev` -> targeted `d_dev` | **0.025 -> 0.007** |

### Interpretation

- The **measurement-allocation** mechanism from E5 **partially replicates**: targeted
  dev again sharply reduces regret at the same labeling budget.
- Unlike CivilComments, min-based selection is **not anti-correlated** on Amazon E6;
  the clearly bad selector here is `R_global`.
- GPO improves the primary robust metric over seed (`0.432 -> 0.458`), but a
  one-line harshness edit reaches the **same** primary score, so operating-point
  controls remain necessary.
- EvoPrompt-DE and final PRIME `test_fixed` scoring were blocked by OpenRouter
  key-limit `403`, so the top-3 replication is **partial**.

## Resume / regenerate table

```powershell
$env:PYTHONIOENCODING='utf-8'
python scripts/run_e5_s9_stable_batch.py --seeds 42,43,44 --repeats 3
python scripts/aggregate_e5_s9_stable.py
```

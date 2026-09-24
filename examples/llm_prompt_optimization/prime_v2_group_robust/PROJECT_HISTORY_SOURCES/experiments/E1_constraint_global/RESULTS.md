# Phase 1 results — `E1_constraint_global/seed42_20260801_002853`

Live Amazon-WILDS mechanics check. Wall ~4.1 h, ~18.5M worker tokens.
Commit at launch: `eb8cc92`. Figures: `figures/` (regen: `python scripts/analyze_constraint_run.py`).

Initial test reference = seed-prompt ensemble from
`pred_cache/cluster_assign_test_ens.npy` (same 1920 test examples as final eval).

---

## Что изменилось в методе оптимизации

Сравнение с провалившимся **lowvar** (`E1_pred_profile_cvar_lowvar`) и с исходным v2 CVaR-дизайном.

### 1. Fitness: скаляр «хвоста» → сырой mean

| | lowvar / CVaR v2 | Phase 1 |
|--|------------------|---------|
| `fitness.mode` | `cvar_lex` | **`global`** |
| Формула | mean(worst ~40% clusters) + ε·macro, **class-balanced** | **raw R_global** на D_select |
| `class_balanced` | `true` (1★ ≈ 39× вес 5★) | **`false`** |
| Что ищет мутатор | поднять 4★-страты любой ценой | поднять общую точность |

Именно class-balanced CVaR задал demotion-attractor (купил 4★ ценой 5★). Убрав его, поиск перестал массово опускать рейтинги.

### 2. Робастность: objective → constraint

| | lowvar | Phase 1 |
|--|--------|---------|
| `anchor_gate_mode` | **`monitor`** (логирует, не режет) | **`reject`** (жёсткий закон) |
| Роль D_anchor | сигнал постфактум | non-regression gate (McNemar, δ=0.02, \|A\|=100) |
| CVaR / хвост в selection | главный ключ | только monitor/proxy в val key |

Робастность больше не то, что **максимизируем**, а то, что **нельзя нарушать**. В этом прогоне gate дважды оценил heir (drop 0.02 / 0.01) и оба раза принял — demotion-катастрофы не возникло, hard-reject не понадобился.

### 3. Consolidation выключен

| | раньше | Phase 1 |
|--|--------|---------|
| `consolidation.enabled` | true (каждый 2-й цикл) | **`false`** |

Консолидация стабильно проигрывала на D_select (8/8 в предыдущих ранах) и только сжигала вызовы. Архив чемпионов / Pareto carryover оставлены.

### 4. Что не меняли (намеренно)

- Официальный WILDS OOD, pinned `pred_profile` clusters  
- D_select=360, test=240 users, OE 8×3, pop 20  
- `group_aware` acquisition (батч для мутатора)  
- Ансамбль median, те же 3 воркера  

Phase 1 — **один** контролируемый diff: objective + gate + consolidation. Не «новый метод целиком», а проверка скелета перед CivilComments / category-shift.

### Схема давления

```
lowvar:   maximize  class_balanced_CVaR(D_select)   → demotion → test −0.053
Phase 1:  maximize  R_global(D_select)
          s.t.      anchor non-regression (reject)   → mild promotion → test +0.004 (n.s.)
```

---

## Headline numbers

| Metric | Initial | Final | Δ |
|--------|--------:|------:|--:|
| R_global | 0.7219 | 0.7260 | **+0.0042** |
| R_macro | 0.664 | 0.642 | −0.022 |
| CVaR_cluster (q=0.40) | 0.6488 | 0.6672 | **+0.0184** |
| MAE | 0.304 | 0.296 | better |
| R_worst (p10 users) | 0.488 | 0.375 | −0.113 (квантование 1/8; см. M9) |

Paired user bootstrap: mean Δ **+0.004**, CI95 **[−0.008, +0.017]**, P(Δ≤0)=0.27.  
McNemar: fixed 63 / broke 55. Pred shifts: **22↓ / 115↑** (lowvar: 341↓ / 12↑).

**Mechanics check PASSED** — значимой регрессии нет.

---

## Графики

### fig1 — траектория эволюции на D_select

![fig1](figures/fig1_evolution_trajectory.png)

Циклы 1–2: best-so-far застыл на 0.7194 (OE не находил лучшего).  
Цикл 3, последняя итерация: скачок до **0.725** — этот промпт стал heir.

### fig2 — сигналы по AL-циклам

![fig2](figures/fig2_cycle_metrics.png)

- D_select: entry fitness чуть проседает к C2/C3 (батч/ротация контекста), best_evo растёт в C3.  
- Val key поднимается в C3 (0.607/0.688 → 0.628/0.704).  
- Anchor gate (reject): drop 0.02 → `tolerated_noise`; C2 n/a (промпт не менялся); C3 drop 0.01 → `within_tolerance`.

### fig3 — initial vs final на OOD test

![fig3](figures/fig3_test_comparison.png)

- Mean/CVaR слегка вверх; R_worst p10 вниз (шумный перцентиль).  
- Per-class: **5★ +0.042**, 4★ −0.069 — promotion, не demotion.  
- Кластеры: c0/c2/c3 чуть лучше, c1/c4 почти flat.

### fig4 — сдвиги предсказаний

![fig4](figures/fig4_pred_shifts.png)

Массовый сдвиг **вверх** (+1). Confusion: `(5→4) 107→63`, `(4→5) 242→275`.

### fig5 — Phase 1 vs failed lowvar

![fig5](figures/fig5_vs_lowvar.png)

Один картинный итог: тот же пайплайн с другим objective/gate даёт **+0.004** вместо **−0.053** и противоположное направление сдвигов pred.

---

## Per-class (test)

| class | n | init | final | Δ |
|------:|--:|-----:|------:|--:|
| 1★ | 29 | 0.690 | 0.655 | −0.034 |
| 2★ | 70 | 0.786 | 0.743 | −0.043 |
| 3★ | 196 | 0.546 | 0.541 | −0.005 |
| 4★ | 494 | 0.415 | 0.346 | −0.069 |
| 5★ | 1131 | 0.883 | 0.925 | **+0.042** |

## Gate / cycles

| Cycle | best_evo (D_select R_global) | gate | note |
|------:|-----------------------------:|------|------|
| 1 | 0.7194 | accepted (`tolerated_noise`, drop=0.02) | reject mode armed |
| 2 | 0.7194 | n/a (prompt unchanged) | OE без gain |
| 3 | **0.7250** | accepted (`within_tolerance`, drop=0.01) | selected heir |

Consolidation disabled. `n_rejected=0` здесь = успех non-harm, не «мёртвый gate».

## Чего прогон не показывает

- Большого worst-group выигрыша нет (и Phase 0 уже закрыл portfolio/headline на этом сдвиге).  
- Promotion-to-5 — другой односторонний аттрактор; reject-gate оставляем.  
- Amazon user-shift остаётся mechanics/negative case.

## Decision

Скелет **`global` + `reject` + no consolidation** зафиксирован → Phase 2 на CivilComments / category-shift.

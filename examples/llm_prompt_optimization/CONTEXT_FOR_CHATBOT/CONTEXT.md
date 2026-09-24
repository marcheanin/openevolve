# GRAPE / PRIME — контекст для чат-бота

**Имена:** метод в статье = **GRAPE**; код v2 = `prime_v2_group_robust/`; предшественник v1 = `wilds_active_learn_approach/`.  
**Срез:** август 2026.

---

## 1. Идея одной фразой

Эволюция **текстового промпта** для замороженного LLM-ансамбля под **distribution shift / worst-group** качество. Промпт — XML (Role, Guidelines, DynamicRules, FewShot). Оптимизация через OpenEvolve + Active Learning; fitness и отбор — на held-out множествах, не на батче мутатора.

---

## 2. Что реально прогонялось (не на бумаге)

### Amazon WILDS (E0–E2) — **done**

| Этап | Суть | Статус |
|------|------|--------|
| **E0** | Offline bounds: style clusters null; pred_profile go; DRO/portfolio не дают выигрыша | done |
| **E1** | Live evolution: CVaR/class-balanced → **регрессия** на user-OOD; `global` + reject-gate → non-harm | done |
| **E2** | Cheap ensemble + `global_tail_mix`: user-OOD **−2.7 pp***, category Books→rest **+0.9 pp** (p=0.045) | done |

Вывод Amazon: robust CVaR-fitness на user-shift **ломает**; рабочий скелет — global + anchor gate; tail-mix 50/50 тоже не выигрывает на user-OOD.

### Amazon E6 category-shift controls + top-3 — **partially done**

- Books→non-Books, scorer `gpt-4o-mini`, pred_profile K=6.
- Fixed eval: test_fixed **1376**, d_dev **648**, d_dev_targeted **570**.
- Phase A controls done: power **3/13** resolvable; F9 regret **0.025 → 0.007** on targeted dev.
- Unlike CivilComments, `hard_min` on uniform d_dev is **not** anti-correlated here (Spearman **+0.40**), but `R_global` is bad for robust selection (**−0.16**).
- Completed top-3 result: **GPO = 0.458 cvar25** vs **seed = 0.432**; this ties a one-line harshness edit (`deflate_positive`), so search-vs-operating-point remains a live issue on Amazon too.
- EvoPrompt-DE and final PRIME test scoring were blocked by **OpenRouter key limit (403)**.

### CivilComments E5 S9 matrix — **done (exploratory)**

- 12 методов × 3 matrix seeds (42/43/44), regime-shift, scorer `gemma-3-12b-it`, optimizer `deepseek-v4-pro`.
- Stable eval: test_fixed n=1800, 3 repeats; seed baseline shared per seed.
- PRIME-main (R10) на всех трёх сидах; oracle только seed42.
- **GPO gate** Yelp→Flipkart: pass.

### Диагностика и контроли E5 — **done**

| Проверка | Что сделано |
|----------|-------------|
| **Power audit** | 0/31 контрастов method vs seed при 95%; repeats ≈ 1/250 дисперсии |
| **R15 calibration** | 12 однострочных правок строгости seed-промпта (без поиска, без групп) |
| **R16 selection control** | Один пул кандидатов × разные правила отбора на D_dev |
| **S9 pool replication** | То же на 26 уникальных промптах оптимизаторов |
| **Statistic ablation** | Сетка CVaR-k / softmin-τ / shrink-w на кэше preds |
| **Dev budget (F9)** | Тот же D_dev=900, но 3 худшие группы × 150/cell вместо 9×50 |

Сырые preds: `prime_v2_group_robust/results/E5_selection_control/`.

---

## 3. Чего нет / отложено (не включать в claims)

- **OPRO, MIPROv2** — не в S9 matrix.
- **Полный native budget** у APO/Random-AL/Oracle (exploratory 2 rounds).
- **Второй датасет** (Amazon category как confirmatory, MNLI) — не прогонялся в этом цикле.
- **Cluster source ablation** (oracle vs GEORGE vs LLM pseudo-groups) — не прогонялся.
- **Переотбор методов из архива на новом D_dev** — не прогонялся.
- **E5v3** selection debug — exploratory debt, не headline.
- **Фиксация групп 8/3/5 в протоколе** — это finding одного датасета, не универсальный метод; в статью как процедура «найти k худших групп seed-промптом на пилоте», не как константы.

---

## 4. Главные выводы (простым языком)

### 4.1 Матрица S9 не доказывает превосходство методов

Все различия method vs seed **внутри шума** (paired bootstrap, Holm). Лучший метод меняется от seed к seed. Oracle с полной разметкой target **хуже** seed — признак того, что измерение не ловит эффект.

PRIME по таблице **на уровне** GPO/EvoPrompt (~+0.014 mean worst-GBA vs seed), но prereg цель Δ≥+0.05 **не выполнена**; vs лучший rival (GPO) PRIME **не выигрывает**.

### 4.2 Оптимизаторы в основном двигают «строгость», а не группы

Контроль **R15**: 12 копий seed с одной изменённой строкой «насколько охотно ставить toxic» — **без поиска, без групп, без label budget** — попадает в тот же диапазон, что вся матрица (4-е место из 35, 0/34 не бьёт его значимо).

Worst-group score на CivilComments **коррелирует с mean GBA** (R²≈0.69); проигравшие методы — over-flagging (высокий recall, низкая specificity).

**Gemma-3-12b игнорирует числовые пороги** («label 1 if P(toxic)≥X%»); двигает только natural-language strictness.

### 4.3 Узкое место — отбор победителя на маленьком D_dev, не генератор

На **одном и том же** пуле промптов смена только правила отбора на D_dev сдвигает результат на **~0.03** — сопоставимо со всем разбросом матрицы.

Правила «выбери промпт с лучшей **худшей группой**» на uniform D_dev (9 групп × 50 примеров) **антикоррелированы** с test worst-group: систематически выбирают over-flagging.

Причина: sd оценки GBA одной группы на dev ≈ **0.05**, а решения принимаются на разнице **0.04**.

### 4.4 Исправление — куда тратить dev-примеры, не формула

Если те же **900** примеров сосредоточить на 3 группах, которые seed-промпт показал худшими на uniform pilot (на CivilComments: 8, 3, 5 — **150/cell**), отбор начинает работать:

- На пуле оптимизаторов: все group-статистики из отрицательных → положительные; regret **0.060 → 0.0075** (8×).
- Лучший результат даёт **group-free** worst-class на сфокусированном dev (regret **0.0000**, pick = лучший промпт пула `42_gpo`).

**Переносимый урок (не привязка к 8/3/5):** групповые метки нужны, чтобы **распределить** проверочный бюджет; считать по ним min/CVaR на uniform dev — ошибка. Это методологический результат, проверять на втором датасете.

### 4.5 Amazon vs CivilComments

| | Amazon user-OOD / E1–E2 | Amazon E6 category-shift | CivilComments E5 |
|--|----------------------|---------------------------|------------------|
| Robust fitness (CVaR) | Явная **деградация** | Частичный сигнал, но не clean win | Нет значимого выигрыша vs seed |
| Главная проблема | Overfit D_select, label-mix clusters | Selection proxy + operating point | Operating point + dev measurement noise |
| Что replicated | — | F9 realloc helps, power still weak | full diagnosis |
| Рабочий скелет | global + reject gate | targeted dev + simple selector | Правильное распределение dev + простой отбор |

---

## 5. Куда вести метод (PRIME / GRAPE)

**Не защищать:** «мы оптимизируем worst-group лучше baselines» — данные этого не показывают на CivilComments и не показывали на Amazon user-OOD.

**Защищать:**

1. **Measurement allocation:** AL-бюджет (или dev-бюджет) адаптивно концентрируется на группах с наибольшей неопределённостью оценки / худшим seed-профилем — встроить в существующий AL цикл PRIME. Amazon E6 уже дал частичную репликацию этого механизма.
2. **Honest controls:** strictness sweep (R15), selection rule ablation (R16), uniform vs targeted dev — обязательны в worst-group prompt optimization papers.
3. **Amazon skeleton** (global + gate) как non-harm baseline; CivilComments как case study **почему worst-group claims ломаются без контролей**.

Shrinkage + soft-min в PRIME — полезны vs raw hard-min, но **недостаточны** без realloc dev; не oversell.

---

## 6. Куда вести статью

**Рамка:** не «новый SOTA optimizer», а **«worst-group prompt optimization is a measurement problem»** — с эмпирикой и контролями.

**Структура claim:**

1. Negative / diagnostic: uniform dev + min-based selection → anti-correlation и ложный прогресс (CivilComments + controls).
2. Positive mechanism: realloc same budget → selection regret падает, group-free rule on targeted dev находит лучший промпт (F9).
3. Amazon E1/E2: robust objectives могут **вредить** — согласуется с literature (Idrissi, Yang).
4. Amazon E6: даже когда robust signal появляется (GPO +0.026 cvar25 vs seed), его может повторить **one-line harshness edit**, так что нужны operating-point controls.
5. Method contribution: процедура (pilot uniform → identify worst groups → targeted dev → simple selection), не подгонка под один датасет.

**Следующий empirical шаг:** три контроля (R15, selection ablation, dev realloc) на **втором датасете** с той же процедурой — тогда claim cross-dataset, не CivilComments-only.

---

## 7. Реализованный код (где смотреть)

| Компонент | Путь |
|-----------|------|
| AL + OpenEvolve loop | `prime_v2_group_robust/prime/controller.py` |
| Fitness / metrics | `prime/fitness/` |
| Fixed sets E5/E6 | `prime/data/fixed_sets.py`, `experiments/E5_civilcomments/fixed_sets/`, `experiments/E6_amazon_category_controls/fixed_sets/` |
| S9 / E6 scripts | `scripts/run_e5_s9_*.py`, `aggregate_e5_s9_stable.py`, `run_e6_*.py` |
| Controls R15–F9 | `scripts/build_r15_*.py`, `run_e5_selection_control.py`, `run_e5_dev_budget_reallocation.py`, `run_e6_selection_control.py`, `run_e6_f9_realloc.py` |
| Доки E5/E6 | `experiments/E5_civilcomments/{RESULTS,POWER_AUDIT,R15_CALIBRATION_CONTROL}.md`, `experiments/E6_amazon_category_controls/{RESULTS,REPLICATION}.md` |

---

## 8. Словарь

| Термин | Значение |
|--------|----------|
| D_dev | Validation set для отбора промпта (n=900 uniform или targeted) |
| test_fixed | Fingerprinted test, n=1800 (100/cell × 9 groups × 2 labels) |
| R_worst_gba / CVaR@25% | Worst-group metrics over identity groups |
| R15 | Strictness sweep control (12 one-line seed edits) |
| R16 | Selection control (same pool, different dev rules) |
| Regret | test metric лучшего в пуле minus metric выбранного правилом |
| Regime-shift | Labeled source + unlabeled target + label budget L=240 |

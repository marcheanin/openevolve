# GRAPE / PRIME — полная история проекта

**Для кого этот документ.** Внешний читатель или новый собеседник в чате: откуда взялась идея, как эволюционировали архитектура и гипотезы, какие эксперименты реально прогонялись, какие цифры получены, и во что всё это вылилось к августу 2026.  
**Имена.** В статье метод называется **GRAPE** (Group-Robust Adaptive Prompt Evolution). В коде — **PRIME v1** (`wilds_active_learn_approach/`) и **PRIME v2** (`prime_v2_group_robust/`).  
**Срез.** Август 2026, после CivilComments E5 S9 и контролей R15–F9. Amazon-репликация контролей (E6) — следующий шаг, ещё не выполнен.

Таблицы с цифрами также собраны в [`RESULTS.md`](RESULTS.md); краткие выводы — в [`CONTEXT.md`](CONTEXT.md).

---

## Оглавление

1. [Идея одной фразой](#1-идея-одной-фразой)
2. [PRIME v1 — исходный метод](#2-prime-v1--исходный-метод)
3. [Критика, аудит и разложение прироста v1](#3-критика-аудит-и-разложение-прироста-v1)
4. [Зачем нужен был PRIME v2 / GRAPE](#4-зачем-нужен-был-prime-v2--grape)
5. [Архитектура v2](#5-архитектура-v2)
6. [Amazon E0 — живы ли группы?](#6-amazon-e0--живы-ли-группы)
7. [Amazon E1 — CVaR против global](#7-amazon-e1--cvar-против-global)
8. [Amazon E2 — дешёвый ансамбль и category-shift](#8-amazon-e2--дешёвый-ансамбль-и-category-shift)
9. [Стратегический поворот: CivilComments](#9-стратегический-поворот-civilcomments)
10. [CivilComments E5 — матрица S9](#10-civilcomments-e5--матрица-s9)
11. [Power audit и confound (F1–F3)](#11-power-audit-и-confound-f1f3)
12. [Контроли R15–F9](#12-контроли-r15f9)
13. [Как сдвинулся научный claim](#13-как-сдвинулся-научный-claim)
14. [Что подтверждено, а что exploratory](#14-что-подтверждено-а-что-exploratory)
15. [Что дальше](#15-что-дальше)
16. [Глоссарий и карта артефактов](#16-глоссарий-и-карта-артефактов)

---

## 1. Идея одной фразой

Можно ли улучшить качество **замороженного** LLM-классификатора (или маленького ансамбля LLM) на **хвосте распределения / худшей группе**, меняя **только текстовый промпт**, при ограниченном бюджете разметки и без дообучения весов?

Типичный промпт — структурированный XML:

```xml
<System>
  <Role>…</Role>
  <BaseGuidelines>…</BaseGuidelines>
  <DynamicRules>…</DynamicRules>
</System>
<FewShotExamples>…</FewShotExamples>
<Task>… {review} …</Task>
```

Оптимизация идёт через **LLM-мутатор** (OpenEvolve / MAP-Elites) внутри цикла **Active Learning**: система сама находит трудные примеры, мутирует правила/few-shot, отбирает победителя на held-out множествах.

Изначально задача звучала как «честный worst-group / OOD прирост на Amazon-WILDS». К августу 2026 она переформулирована: на Amazon user-shift и на CivilComments **не удалось показать**, что group-robust оптимизация промпта бьёт простые контроли; зато удалось показать, что **измерение и отбор на маленькой валидации** — узкое место всей этой области.

---

## 2. PRIME v1 — исходный метод

### 2.1 Постановка

- Данные: отзывы Amazon, рейтинг **1–5★**.
- Ранний фокус: категория **Home & Kitchen**; затем headline — **all categories**.
- Сплит: user-disjoint **70/15/15** от пользователей train-подмножества WILDS (seed 42). Это **не** официальный OOD-тест WILDS — позже критически важно.
- Мотивация: границы 4 vs 5 и 2 vs 3 субъективны; нужно не только среднее качество, но и «не провалить хвост пользователей».

### 2.2 Архитектура

**Два уровня эволюции**

| Уровень | Где | Что меняется | Как принимается |
|---------|-----|--------------|-----------------|
| L1 OpenEvolve | Внутри AL-цикла | DynamicRules + FewShot | Fitness на батче + MAP-Elites |
| L2 Consolidation | Между циклами | В т.ч. BaseGuidelines | Gate на validation |

**Ансамбль.** Три LLM (T≈0), majority vote. В submission-стеке типично: gpt-4o-mini + gemini-2.5-flash + claude-3.5-haiku (исторически были и другие тройки).

**Hard / Anchor Active Learning**

- **Hard:** ансамбль ошибся **или** workers разошлись.
- **Anchor:** все верно и согласны.
- Батч ~**80**, `hard_ratio≈0.7` (~56 Hard / ~24 Anchor).
- После цикла: reclassify, refresh Unseen→Hard, soft expansion при стагнации val.

**Fitness внутри OpenEvolve (на батче):**

```
0.5 × Acc_Hard + 0.3 × Acc_Anchor + 0.2 × κ_Hard − length_penalty
```

**Отчётный combined (val/test):**

```
0.4×R_global + 0.3×R_worst + 0.3×(1 − MAE/4)
```

где `R_worst` = **10-й перцентиль** per-user accuracy (не CVaR по стилевым кластерам).

**Критичный архитектурный изъян (осознан позже):** inner fitness **group-agnostic**; MAP-Elites **сбрасывался** каждый цикл; дальше нёсся **один** seed-промпт → bottleneck разнообразия.

### 2.3 Ключевые цифры v1

**Ранний Home & Kitchen (без AL):** эволюция давала ~+3–3.5 pp R_global; ансамбль сам по себе ~+14 pp относительно single-model.

**AL v3 (H&K):** baseline combined ~0.850; peak per-cycle test ~0.853; Acc_Hard рос, но R_worst прыгал; финал иногда чуть хуже baseline — уже тогда видны проблемы checkpoint/consolidation.

**Headline для ARR (all-categories, full uncapped test n=34 533):**

| Метод | R_worst | R_global | combined |
|--------|--------:|---------:|---------:|
| **PRIME ensemble AL iter5** | **55.91%** | **75.41%** | **0.838** |
| Initial prompt ensemble | 50.00% | 71.30% | 0.782 |
| PRIME prompt + GPT-4o-mini only | 56.80% | 74.40% | 0.748 |
| EvoPrompt GA (GPT-4o-mini) | 56.50% | 73.92% | 0.744 |
| APO | 52.90% | 72.60% | 0.727 |
| OPRO | 50.98% | 72.05% | 0.719 |
| LISA DistilBERT (цитировался) | 54.70% | 71.30% | — |

Заявленные дельты vs seed ensemble: Δ R_global **+4.11**, Δ R_worst **+5.91**. Сравнение с LISA подавалось как «prompt-only бьёт fine-tuning SOTA» — позже признано невалидным.

Второй полный прогон PRIME (uncapped train): R_global прирост всего **~+0.8 pp** — разброс run-to-run огромный.

---

## 3. Критика, аудит и разложение прироста v1

### 3.1 Что возразили рецензенты (ACL ARR May’26)

Сводно (Overall ~2–2.5):

1. **Нечестное сравнение** frontier-ансамбля с DistilBERT (LISA).
2. **Неофициальный тест** — custom 70/15/15 от train users; лидерборд ERM/GroupDRO/LISA считается на **official OOD**.
3. **Вырожденный baseline:** seed-ансамбль R_worst 50.0 **ниже** single gpt-4o-mini 53.18 → «прирост» частично чинит старт.
4. Один датасет / язык; нет сидов / CI; запас над LISA ~1 pp ≈ шум.
5. Novelty = рекомбинация известных блоков (эволюция промпта, AL, ансамбль).
6. κ в fitness → рост κ не независимое доказательство.
7. Ablations на proxy-метриках, не на headline.

### 3.2 Внутренний аудит и reconciliation

Документ `prime_v2_group_robust/docs/V1_V2_RECONCILIATION.md` разложил прирост на той же выборке n=34 533:

1. **~1.5–2 pp** — восстановление вырожденного старта ансамбля (κ 0.70→0.88; финальный ensemble R_worst 55.91 даже **ниже** single-PRIME 56.80 и ≈ EvoPrompt 56.50).
2. **~+2.0–2.5 pp global** — честный переносимый эффект **контента промпта** (zero-shot GPT 71.86 → PRIME-prompt GPT 74.40). На official OOD в стеке v2 (M21): v1_final vs v1_seed **+2.2 pp, p=0.015**.
3. PRIME vs EvoPrompt: +1.5 global / **−0.6** worst — внутри шума между двумя полными прогонами PRIME (+4.1 vs +0.8).

**Цитата-вердикт:** *«Честная, переносимая часть прироста v1 — это +2.0–2.5 пп от контента промпта… Остальные ~2 пп — восстановление вырожденной стартовой конфигурации ансамбля.»*

Итог: v1 — сильный **Hard-focused active prompt optimizer**, но он **не** оптимизировал worst-group в смысле групп распределения и **не** мерился на official WILDS OOD как основной headline.

---

## 4. Зачем нужен был PRIME v2 / GRAPE

### 4.1 Свободная ниша

Промпт-оптимизаторы (EvoPrompt, APO/ProTeGi, OPRO, GEPA, DSPy/MIPRO) почти не проектировались под **group shift**. Классический GroupDRO / GEORGE работают в пространстве **весов**. Пробел: **дискретная эволюция промпта × group-robust objective**.

Формулировка из README v2: *«generic prompt optimization улучшает среднее, но не хвост… слабость v1 становится мотивирующим результатом v2.»*

### 4.2 Цели redesign

1. Official WILDS OOD для всех headline-чисел.
2. Group-robust objective **внутри** поиска (CVaR/DRO), не post-hoc отчёт.
3. Латентные группы, переносимые на новых пользователей (не user_id в правилах промпта).
4. Чистые роли данных и non-regression gate.
5. Если метод не взлетит — остаётся честный **study**: «что происходит с prompt-opt под сдвигом».

Имя в статье: **GRAPE**. Код: **PRIME v2**.

---

## 5. Архитектура v2

### 5.1 Инварианты

- Fitness / отбор только на **held-out** (`D_select`), не на батче мутатора.
- Fit / heldout user-disjoint.
- Official WILDS splits.
- Рабочие видят только текст; cluster-id в правилах запрещён.

### 5.2 Роли данных

| Множество | Назначение | Типичный размер (Amazon live) |
|-----------|------------|-------------------------------|
| Fit pool | Кандидаты AL, ошибки для мутатора | — |
| **D_select** | Fitness / Pareto / отбор | ~360 |
| **D_anchor** | Reject-gate (McNemar) | ~100 |
| **D_audit** | Диагностика | ~50 |
| Val / Test | OOD; final report на test | ~240 users × 8 ≈ 1920 |

Позже на CivilComments добавились fingerprinted **D_dev** (n=900) и **test_fixed** (n=1800).

### 5.3 Группы

Изначально — style embeddings. Эмпирически **null** на Amazon. Рабочая геометрия: **`pred_profile`** — k-means по профилю предсказаний пользователя (mix рейтингов, mean/std/entropy) с seed-промптом, K≈6.

Позже (C10): **~83%** межкластерного разброса accuracy объясняется **золотым mix классов**. Residual «стиля рецензента» мал. CVaR по таким группам ≈ «давить 4★».

### 5.4 Fitness modes (эволюция)

| Mode | Смысл | Судьба |
|------|-------|--------|
| `cvar_lex` | CVaR по кластерам + lex global | Demotion на user-OOD (M17) |
| `global` | R_global − length | Phase-1 skeleton PASS (M19) |
| `global_tail_mix` | 0.5 R_global + 0.5 R_tail | Demotion user (M22); weak + category (M23) |
| `v1_weighted` | Hard/Anchor/κ на батче | No wake-up (M20) |

### 5.5 Цикл

Инференс → Hard/Anchor → group_aware батч → OpenEvolve (fitness на D_select) → champion archive (scalar-best + per-cluster) → optional consolidation (**выключена** после E1) → **anchor gate** → heir в следующий цикл.

---

## 6. Amazon E0 — живы ли группы?

**Вопрос:** имеет ли смысл оптимизировать CVaR по кластерам, если кластеры не связаны с OOD worst-group?

| Проверка | Результат | ID |
|----------|-----------|-----|
| Style clusters, large-cap | Kruskal–Wallis **null**, p≈0.78 | C1–C4 |
| `pred_profile` на тех же preds | KW **p≈0**, H≈27.6 | C6 |
| Phase 0 offline (30 промптов × D_select 360) | Лучший raw R_global = **initial**; portfolio LOO **−3.6 pp**; DRO top хуже accuracy; зона «поднять 4★ без убийства 5★» пуста | M18 |

**Вывод E0:** style не работает; pred_profile «жив», но это в основном **label-mix proxy**. Не вкладываться в DRO/portfolio fitness на Amazon user-shift; robustness держать как **constraint**.

---

## 7. Amazon E1 — CVaR против global

### 7.1 Волна 1 — хаос механики

Ранние live-прогоны: hair-trigger anchor gate, consolidation «съедала» gains, starved OE iterations, underpowered пары. Научные наблюдения даже на шумных прогонах:

- CVaR поднимал worst-cluster на **D_select** и **бил** test-worst (argmin не переносится) — C8.
- Ошибки мутатора ~95% соседние 4↔5 — C9.
- Эволюция двигала порог 4↔5: 4★ вверх, 5★ вниз — C11.

### 7.2 Lowvar = первый значимый результат, и он отрицательный (M17)

Class-balanced CVaR, D_select=360, test=240 users:

- Test **R_global −0.053** (CI [−0.074, −0.032], **p<0.001**); все кластеры хуже.
- D_select objective **+0.128** при noise floor ~0.009 → **чистый selection overfitting**.
- Pred shifts 341↓ / 12↑; 5★ accuracy −0.140, 4★ +0.097.

**Вместе с M18: отвергнуть CVaR/DRO как fitness на Amazon user-shift.**

### 7.3 Phase 1 mechanics PASS (M19)

Fitness **`global`**, gate **`reject`**, consolidation **off**:

- R_global **0.722→0.726** (Δ **+0.004**, n.s.).
- Preds 115↑ / 22↓ — promotion, не demotion.
- Скелет **не вредит**. Это не победа worst-group, а рабочая механика.

### 7.4 «Может, просто вернуть v1?»

- **M20** (`v1_weighted` mini): не просыпается; +0.002 n.s.; final ≈ seed.
- **M21** (transfer v1-промптов без эволюции): rich v1 seed **0.710 <** short GRAPE seed **0.722**; evolved v1 vs GRAPE seed +0.011 n.s.; vs own seed **+0.022, p=0.015**. Контент эволюции реален; потолок Amazon prompt-only ~**0.72–0.73**.

**Reconciliation (M24):** v2 не «сломал» метод — убрал два источника бесплатного прироста и показал реальный потолок. Надежда «там было +5.9» не выдерживает декомпозиции.

---

## 8. Amazon E2 — дешёвый ансамбль и category-shift

Workers: gpt-4o-mini / gemini-2.5-flash-lite / qwen3-32b. Fitness: `global_tail_mix`.

### 8.1 User-OOD demotion (M22)

Canonical: `results/E2_cheap_ensemble_global_tail/seed42_20260802_011259`

| Metric | Seed | Final | Δ | p |
|--------|-----:|------:|--:|---|
| R_global | 0.690 | 0.663 | **−0.027** | 0.000 |
| R_tail | 0.380 | 0.370 | −0.010 | n.s. |

Мягкий tail-mix — та же семья, что CVaR: бьёт mean OOD.

### 8.2 Category-shift Books→non-Books (M23)

Canonical: `results/E2_category_shift_books/seed42_20260802_052137`

| Metric | Seed | Final | Δ | p |
|--------|-----:|------:|--:|---|
| R_global | 0.662 | 0.671 | **+0.009** | 0.045 |
| R_tail | 0.329 | 0.339 | +0.010 | n.s. |

Слабый плюс / borderline. Интересный substrate, не сильный robustness win. Именно его позже выбирают как площадку для репликации контролей.

Параллельные попытки E2 были заражены багом **O27** (общий каталог seeds OpenEvolve) — science только из чистых sequential runs. **O28**: OE child иногда грузил дорогой дефолтный ансамбль из-за битых includes — влияет на доверие к in-loop fitness, не на финальные seed-vs-final таблицы parent-ансамбля.

---

## 9. Стратегический поворот: CivilComments

### 9.1 Гипотеза применимости

Group-robust эволюция промпта работает, если худшая группа:

1. **узнаваема** из текста,
2. **выразима** инструкциями,
3. **не конфликтует** с большинством.

Amazon user-shift проваливает все три (label-mix группы, irreducible 4↔5, demotion attractor). CivilComments с **oracle identity** (race, religion, gender, LGBTQ, …) a priori проходит.

### 9.2 Решения (ROADMAP Phase 2 / STRATEGY_UNIVERSAL_OOD)

- Amazon **user-shift закрыт** как headline; остаётся negative / mechanics case.
- Следующий substrate: **CivilComments-WILDS**.
- Framing: «условия применимости» — method paper с Amazon как объяснённым негативом; study framing как fallback.
- Исследования на дешёвых моделях; валидация на флагманах; equal-budget baselines.

Диагностика перед E5: на equal-group subsample gap ~**35 pp** (male ~0.90 vs black ~0.55), class-mix R² ≈ **−0.03** (residual group effect реален). E4 показал, что механизм может двигать worst-group на D_select, но протокол/метрики были сломаны → E5 пересобирает стенд.

---

## 10. CivilComments E5 — матрица S9

### 10.1 Протокол

| Элемент | Спецификация |
|---------|--------------|
| Задача | Binary toxicity (CivilComments / WILDS identities) |
| Scorer | `google/gemma-3-12b-it`, T≈0 |
| Mutator | `deepseek/deepseek-v4-pro` |
| Regime-shift | S_source=360 labeled easy; U_target=4000 unlabeled hard; L=240 labels для AL |
| Selection | Top-1 softmin-GBA на **D_dev n=900** (uniform) |
| Report | Fingerprinted **test_fixed n=1800**, stable **×3** repeats; seed shared per matrix seed |
| Seeds | 42 / 43 / 44 |
| Gate | Yelp→Flipkart GPO reimplementation: **pass** |

Методы: Seed, APE, APE-K48, APE-ut, APO, GPO, EvoPrompt-GA/DE, GEPA, PRIME-main, Random-AL, Oracle (только seed42). APO/Random-AL/Oracle — exploratory **2 rounds** (не full native 6).

**Prereg success (не выполнен):** PRIME mean Δ worst-GBA ≥ **+0.05** vs seed **и** ≥ **+0.02** vs лучший из {APO, GPO, Random-AL}.

### 10.2 Headline (mean over seeds)

| method | R_worst_gba | R_global | Δ vs seed |
|--------|------------:|---------:|----------:|
| GPO | **0.634** | 0.709 | **+0.015** |
| EvoPrompt-DE | 0.633 | 0.711 | +0.014 |
| **PRIME** | **0.633** | 0.708 | **+0.014** |
| EvoPrompt-GA | 0.629 | 0.709 | +0.010 |
| Seed | **0.619** | 0.702 | — |
| **Oracle** | **0.615** | 0.685 | **−0.004** |
| APO | 0.596 | 0.687 | −0.023 |

PRIME per seed: 42 → 0.643 (+0.025); 43 → 0.620 (tie); 44 → 0.637 (+0.019).

Operating point: проигравшие (APO, APE) — over-flagging (recall ~0.83–0.85, spec ~0.55); PRIME относительно сбалансирован (0.70 / 0.72).

**Сигнал для читателя:** PRIME на уровне GPO/EvoPrompt (~+1.4 pp), далеко от +5 pp; oracle **ниже** seed — измерение не ловит «настоящий» upper bound.

После power audit таблица S9 **понижена до exploratory** (метрику и бюджет меняли, уже видя результаты).

---

## 11. Power audit и confound (F1–F3)

Считалось с кэшированных preds stable_session, без нового скоринга.

### F1 — ни один контраст не выживает

- **0/31** method×seed resolvable at 95% под hard-min GBA.
- Лучший: GPO@42 Δ=+0.035, CI95 [−0.005, +0.075].
- Ранги **инвертируются** по сидам (gpo / evoprompt_ga / random_al) — картинка чистого шума.
- Oracle 0.615 < seed 0.620.

### F2 — repeats тратили бюджет не туда

| | repeat sd | bootstrap sd (примеры) |
|--|----------:|------------------------:|
| hard-min | 0.0014 | ~0.0185 (~13×) |
| CVaR@25% | 0.0008 | ~0.0131 (~16×) |

Доля дисперсии scorera ≈ **1/250**. Три повтора почти ничего не купили при 3× стоимости. CVaR@25% даёт ~**2×** effective data vs hard-min; свободный выигрыш — **1 repeat × больше примеров**.

### F3 — ось = operating point, не group structure

- corr(CVaR@25%, mean GBA) = **+0.83**; R² = **0.687**.
- Добавление `|recall−specificity|` к mean GBA добавляет **0.000** к R².
- Проигравшие — recall-skewed; победители — сбалансированы.

**Интерпретация:** оптимизаторы в основном двигают порог «строгости» токсичности. Нужен контроль чистого operating-point sweep (**R15**).

---

## 12. Контроли R15–F9

### 12.1 R15 / F4 — одна строка строгости ≈ вся матрица

**Пул:** 12 промптов = seed с **одной** изменённой строкой tie-break (8 числовых порогов P(toxic)≥X% + 4 NL-варианта). Без поиска, без групп, без L=240, без mutator LLM.

На тех же 1800 строках, что S9 (CVaR@25%):

| | @1800 |
|--|------:|
| R15 pick (`nl_lenient_max`) | **0.6475** |
| Seed | 0.6300 |
| GPO@42 (лучший cell) | 0.6675 |
| PRIME@42 | 0.6600 |
| Oracle | 0.6200 |

- R15 — **4-е из 35**; **0/34** cell бьёт его при Holm < 0.05.
- Честное чтение: **никто не выигрывает**; zero-search контроль сидит в noise band всей матрицы.

### 12.2 F6 — scorer игнорирует числовые пороги

Инструкция «ставь 1 только если P≥X%» при X=90…20: recall в полосе ~0.73–0.76 **без тренда**. Natural-language strictness двигает сильно (recall 0.72→0.85). Методы, «подкручивающие вероятность», на gemma-3-12b — no-op.

### 12.3 F5 / R16 — решает правило отбора, не генератор

На **одном** пуле из 12 кандидатов смена только D_dev rule → swing **~0.030** (весь разброс матрицы методов ~0.070).

| Правило | Spearman vs test | Комментарий |
|---------|-----------------:|-------------|
| worst-class / mean GBA | **+0.71** | Лучшие |
| softmin shrunk (shipped PRIME) | +0.58 | Полезно |
| CVaR@25% / hard-min | **−0.34 … −0.57** | **Антикоррелированы** |

Min-based правила систематически выбирают over-flagger. Причина: sd оценки GBA **одной группы** на D_dev ≈ **0.0475** при решениях на разнице ~0.04.

### 12.4 F7 — репликация на пуле оптимизаторов (26 уникальных промптов)

Тот же знак: 6/6 min-правил отрицательны; 12/12 усредняющих/group-free неотрицательны. Worst-class лучший (+0.68). Shrinkage помогает ранжированию, но на S9-пуле всё ещё выбирал плохой `44_apo` (regret **0.060**).

### 12.5 F8 — при uniform allocation «фокус на худшей группе» в формуле = 0

Сетка (CVaR k, τ, shrink w): качество растёт **монотонно** к среднему по группам. Внутреннего оптимума нет. Group-free worst-class бьёт все group-статистики по regret.

### 12.6 F9 — чинится распределение бюджета, не формула

Тот же n=900: вместо 9 групп × 50 — **3 худшие** (по seed на uniform pilot: группы 8, 3, 5) × **150**. Без подглядывания в test. Совпадение перескоринга на shared rows: **99.72%**.

На пуле оптимизаторов:

| | Uniform | Targeted |
|--|---------|----------|
| Знаки Spearman group-правил | все отрицательные | **все положительные** (среднее Δ +0.876) |
| Regret group-правил | **0.0600** (`44_apo`) | **0.0075** (`42_prime`) — **8×** |
| Worst-class regret | 0.0225 | **0.0000** (`42_gpo` = лучший в пуле) |

**Практический рецепт:** групповые метки — чтобы решить, **куда** потратить примеры; считать — простой group-free метрикой на сфокусированном dev. Константы {8,3,5} — finding CivilComments; в метод должна входить **процедура** (pilot → найти k худших → targeted), не ID групп.

---

## 13. Как сдвинулся научный claim

| Этап | Claim |
|------|--------|
| v1 ARR | Prompt-only ensemble бьёт fine-tuning SOTA на Amazon fairness |
| После ревью / reconciliation | Честный потолок v1 ~+2–2.5 pp контента; сравнение с LISA невалидно |
| Старт v2 / GRAPE | Первая group-aware эволюция промпта (CVaR + latent groups) на official OOD |
| После Amazon E0–E2 | Amazon user-shift — **отрицательный** case; CVaR fitness demotes; скелет = global + gate |
| Prereg S9 CivilComments | PRIME ≥ seed +0.05 и ≥ rivals +0.02 по worst-GBA |
| После S9 + F1–F4 | Матрица в шуме; R15 matches оптимизаторы; нельзя защищать «мы лучше оптимизируем worst-group» |
| После F5–F8 | Узкое место — **измерение/отбор** на tiny uniform D_dev |
| После F9 (текущий) | Worst-group prompt optimization — **задача о распределении измерительного бюджета**; realloc того же n режет selection regret ~8× |

**Не защищаем сейчас:** «GRAPE/PRIME — SOTA group-robust prompt optimizer».  
**Защищаем:** «В этой области легко принять шум и operating-point tuning за прогресс; честные контроли (strictness sweep, selection ablation, uniform vs targeted) обязательны; практический рычаг — allocation, не ещё один генератор».

Связь с литературой: Idrissi et al. (CLeaR 2022), Yang et al. (ICML 2023) — большая часть worst-group progress живёт в group-labelled model selection; здесь group-labelled selection при маленьком uniform D_dev даже **вредна**, пока бюджет не перераспределён.

---

## 14. Что подтверждено, а что exploratory

### Подтверждено (можно опираться)

- v1: fair single-model reading; ~+2–2.5 pp transferable prompt content; invalid LISA compare.
- Amazon: style null; CVaR demotion (−5.3 pp***); Phase-1 global+gate non-harm; tail-mix user demotion (−2.7 pp***); category weak + (+0.9 pp).
- CivilComments S9: prereg success **не** выполнен; 0/31 resolvable contrasts; oracle ≤ seed.
- F2–F3: repeats бесполезны относительно примеров; operating-point confound (R²≈0.69).
- R15: one-line NL control в noise band матрицы.
- F5–F7: min-based selection anti-correlated на двух пулах.
- F9: targeted allocation переворачивает знаки; regret 0.060→0.0075.

### Exploratory / не confirmatory

- Вся таблица S9 как ranking методов (метрику меняли post-hoc).
- Reduced-budget APO/Evo/GEPA/Oracle.
- E5v3 selection debug.
- Фиксация групп {8,3,5} как универсальный протокол (только процедура переносима).
- S10 confirmatory matrix — preregistered, **не прогнана**.
- Amazon replication контролей — **запланирована**, не сделана.

---

## 15. Что дальше

1. **Amazon E6 (следующий шаг):** те же контроли (power, harshness/op-point sweep, selection rules, uniform vs targeted) на **category-shift Books→non-Books**, single scorer, fingerprinted sets, ordinal macro-within-cluster метрика. Цель — cross-dataset подтверждение measurement-allocation claim, **не** победа на Amazon.
2. **Мини-матрица §7.2** (Seed/APE/APO/GPO/PRIME/Random-AL × 3 seeds) — только если стенд E6 «живой» (контроли показывают измеримый swing).
3. Перенос в метод: AL/dev budget **адаптивно** концентрируется на группах с наибольшей неопределённостью оценки — тогда F9 становится движком, а не one-off хаком под CivilComments.
4. Статья: рамка «worst-group prompt optimization is a measurement problem» + Amazon negative case + CivilComments controls + (после E6) cross-dataset replication.

---

## 16. Глоссарий и карта артефактов

### Глоссарий

| Термин | Значение |
|--------|----------|
| R_global | Средняя accuracy |
| R_worst (Amazon v1/v2) | P10 per-user accuracy |
| R_tail | Mean accuracy худших ~20% пользователей |
| R_worst_gba / GBA | Group-balanced accuracy: ½(TPR+TNR) по identity |
| CVaR@k | Mean худших k групп (или доли групп) |
| D_select | Held-out fitness set (Amazon) |
| D_dev | Validation для отбора промпта (CivilComments) |
| D_anchor | Non-regression McNemar gate set |
| test_fixed | Fingerprinted test, n=1800 на CC |
| pred_profile | Кластеры по профилю предсказаний пользователя |
| Hard / Anchor | Ошибки/disagreement vs лёгкие согласованные |
| Heir | Промпт, уходящий в следующий AL-цикл |
| R15 | Strictness / harshness one-line control pool |
| Regret | Лучший в пуле минус выбранный правилом (на test) |

### Где лежат первоисточники

| Что | Путь |
|-----|------|
| v1 код и отчёты | `wilds_active_learn_approach/` (`docs/PROJECT_REPORT.md`, `last_four_experiments_comparison.md`) |
| v1↔v2 reconciliation | `prime_v2_group_robust/docs/V1_V2_RECONCILIATION.md` |
| Живой лог уроков | `prime_v2_group_robust/experiments/OBSERVATIONS.md` |
| Amazon E2 results | `experiments/E2_*/RESULTS.md` |
| CivilComments S9 | `experiments/E5_civilcomments/RESULTS.md` |
| Power audit | `experiments/E5_civilcomments/POWER_AUDIT.md` |
| R15–F9 | `experiments/E5_civilcomments/R15_CALIBRATION_CONTROL.md` |
| Сырые preds E5 | `results/E5_s9_matrix/`, `results/E5_selection_control/` |
| Краткие цифры для чата | этот пакет: `RESULTS.md`, `CONTEXT.md` |

### Хронология одной строкой

```
v1 AL+OpenEvolve на Amazon custom split
  → ARR claim (+4/+6 pp) + reviews
  → reconciliation: ~+2–2.5 pp честных
  → v2 GRAPE: CVaR + pred_profile + official OOD
  → E0: style null; pred_profile ≈ label mix
  → E1: CVaR demotes (−5.3***); global+gate OK
  → E2: tail-mix demotes user (−2.7***); category +0.9
  → pivot CivilComments
  → S9 matrix: PRIME≈GPO≈+1.4 pp; oracle < seed; all in noise
  → R15: one-line strictness = 4th of 35
  → F5–F8: selection rule anti-correlates under uniform D_dev
  → F9: same budget, targeted groups → regret ÷8
  → claim: measurement allocation, not SOTA optimizer
  → next: Amazon category replication of controls
```

---

*Документ собран 2026-08-14 из PROJECT_HISTORY, OBSERVATIONS, V1_V2_RECONCILIATION, E0–E5 RESULTS, POWER_AUDIT, R15_CALIBRATION_CONTROL и CONTEXT_FOR_CHATBOT. Не заменяет сырые артефакты; при расхождении цифр приоритет у RESULTS.md эксперимента и JSON в `results/`.*

# PRIME v2 (GRAPE) — Implementation Spec (нормативный)

Этот документ описывает, **как должен выглядеть код** проекта: целевая структура,
контракты модулей, инварианты, фазы миграции и Definition of Done каждой фазы.
Адресат — агент/разработчик, реализующий очередную фазу без контекста бесед.

Правила чтения:
- **Семантика** (зачем, какие эксперименты, какие метрики) — в [SPEC.md](SPEC.md);
  при конфликте SPEC.md главнее. Ссылки вида «SPEC §4.3» — туда.
- Здесь — **что и где должно лежать в коде** и когда это считается готовым.
- Текущий код — v2-скелет; таблица соответствия и что удаляется — §5.
- Численные дефолты (размеры наборов, n_min, доли) — из SPEC §10 (Р7, Р9);
  в коде они живут в YAML, не в исходниках.

---

## 1. Целевая структура пакета

Пометки: `[есть]` — реализовано в v2-скелете и остаётся; `[Ф{N}]` — создать или
переработать на фазе N; `[удалить Ф2]` — упраздняется.

```
prime_v2_group_robust/
  prime/
    config.py                 [есть, расширить Ф0/Ф1] YAML → dataclasses, схема §2
    controller.py             [переработать Ф2] тонкая оркестрация AL-циклов
    cli.py                    [есть]
    data/
      wilds_loader.py         [есть] официальные сплиты + кэши
      cache.py                [есть]
      splits.py               [Ф1, новый] fit/heldout по юнитам, стратификация
      profiles.py             [Ф1] блоки признаков T/C раздельно
      clustering.py           [Ф1] label-free fit + pred_profile geometry (E1 default)
    workers/
      ensemble.py             [Ф0] median-агрегация; интеграция с кэшем и бюджетом
    acquisition/
      scoring.py              [есть] ранжирование (err, d), политики
      batch_builder.py        [Ф2] сборка D_mut per-мутация, квоты ∝ (1−Acc_τ)
      eval_sets.py            [Ф1, новый] D_select / D_anchor / D_audit
      pool.py                 [упростить Ф2] остаётся только U-резервуар
    fitness/
      metrics.py              [Ф0] Beta-сглаживание, CI, per-group accs
      objective.py            [Ф0] режим cvar_lex; κ только в v1_weighted
    evolution/
      openevolve_adapter.py   [Ф1–Ф2] CandidateEvaluator v3, каскад, кэш
      evaluator_entry.py      [есть] вход для OpenEvolve-колбэков
      artifacts.py            [Ф3] error report из D_mut, changelog, дескрипторы
      prompt_blocks.py        [Ф3] парсер блоков + структурный валидатор
      qd_features.py          [Ф2] дескрипторы архива от D_select, bins=3
      openevolve_config_patch.py [есть, обновить Ф2]
    consolidation/
      pareto_front.py         [Ф3, новый] per-group фронт, приёмка, родители
      pool_carryover.py       [Ф3] carryover = срез фронта
      base_consolidator.py    [Ф3] консолидация BaseGuidelines, CI-гейт
    experiment/
      run_context.py          [есть] снапшот конфига/git в run_dir
      budget.py               [Ф1, новый] TokenTracker по уровням, бюджет B
      proxy_validation.py     [Ф2 расширить, Ф4 гейт] corr, ID-vs-OOD, audit
      stats.py                [есть] bootstrap CI
  configs/
    base_v3.yaml              [Ф0, новый] дефолты v3 (см. §2)
    dataset_amazon.yaml       [есть]
  experiments/
    E0_proxy_diag/            [Ф1] конфиг + README
    E1_pilot_cvar_vs_global/  [есть, мигрировать Ф1]
    _template/                [есть]
  scripts/
    run_e0_proxy_diag.py      [Ф1, новый]
    build_clusters.py         [Ф1 обновить]
    aggregate_results.py      [есть]
  tests/                      см. §6
```

---

## 2. Схема конфига v3 (`configs/base_v3.yaml`)

Все числа — здесь, не в коде (правило «no magic numbers»). Ключевые секции:

```yaml
experiment: { name, smoke, seed }
dataset:    { name: amazon_wilds, data_root, use_cache, caps: {...} }

splits:                    # Ф1, SPEC §4.3
  fit_ratio: 0.7           # 70/30 по user_id (Р7)
  stratify: true           # квантили длины + PCA-1 эмбеддинга (Р15)
  seed: 42

clusters:                  # SPEC §4.1
  n_clusters: 8            # верхняя граница; реальный K ≤ |D_select|/n_min
  n_min: 40                # пре-регистрирован (Р9), не менять по данным
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
  pca_dim: 48              # PCA только на эмбеддинг-блок
  min_reviews_for_fit: 3   # юзеры с меньшим — только assign (Р9)
  descriptors: { enabled: true, top_n_neighbors: 5, llm_editor: false }  # Р4
  control: none            # none | shuffle  ← random-clusters абляция (Р13)

eval_sets:                 # Ф1, SPEC §4.3 (Р9)
  d_select_size: 400       # capped E1: 200
  d_anchor_size: 100       # capped E1: 50
  d_audit_size: 100        # capped E1: 50
  anchor_confident_rule: { require_correct: true, max_disagreement: 0.0 }

acquisition:               # только для D_mut (SPEC §4.3)
  d_mut_size: 35           # capped E1: 30
  policy: lexicographic    # (err, d); random/hardest/qbc_d — абляции
  group_quota_mode: weakness   # квоты ∝ (1−Acc_τ) со сглаженных Acc_τ D_select
  diversity_oversample: 3

ensemble:
  workers: [gpt-4o-mini, gemini-2.5-flash, claude-3.5-haiku]
  aggregation: median      # majority — абляция (Р8)
  temperature: 0.0         # обязательное условие кэша

fitness:                   # SPEC §4.4
  mode: cvar_lex           # cvar_lex | global | v1_weighted (абляции)
  cvar_quantile: 0.33
  eps_global: 0.01
  smoothing: { a: 1, b: 1 }   # Beta-prior
  len_penalty_start: 600
  len_penalty_per_100: 0.01

cascade:                   # SPEC §4.3
  shard_frac: 0.3
  anchor_gate: { mode: ci_delta, delta: 0.02, alpha: 0.05 }  # Р15: допуск, не строгий
  # δ = max(delta, 2/|D_anchor|); reject только если ещё и McNemar p < alpha
  front_ci: bootstrap      # CI для приёмки во фронт

budget:                    # Ф1, SPEC §6.4/§7-смета (Р12)
  total_calls: 60000
  on_exhausted: stop       # stop | degrade (реже полные оценки)

evolution:
  openevolve_config: configs/openevolve_v3.yaml
  mutations_per_cycle: 30
  validator: { enabled: true, max_blocks_changed: 1 }   # Ф3

consolidation:
  enabled: true
  scope: base_guidelines
  gate: { mode: ci_delta, delta: 0.02 }

routing: { mode: A }       # A | B | C — Ф4 (SPEC §4.9)

active_learning: { n_cycles: 3, seed: 42 }
```

---

## 3. Сквозные инварианты в коде (enforcement, не соглашения)

Каждый инвариант обязан иметь проверку в рантайме и/или тест (SPEC §3.2, §4.8):

- **INV-1 (информационный доступ, I1).** Артефакты мутатора строятся **только**
  из индексов D_mut (fit_sources). В `artifacts.py` — assert: пересечение
  использованных индексов с `heldout_indices` пусто. Тексты D_select/D_anchor/
  D_audit никогда не сериализуются в промпт мутатора. Тест: `test_info_access`.
- **INV-2 (селекция только с D_select, I2).** Fitness, QD-дескрипторы, фронт,
  carryover читаются исключительно из результатов `evaluate_select`. Результаты
  `evaluate_mut` не записываются в архив/фронт нигде. Тест на уровне адаптера.
- **INV-3 (бюджет, I3).** Каждый API-вызов проходит через `TokenTracker.charge
  (level, n_calls, tokens)`; уровни: `mut | shard | full | anchor | val | audit
  | mutator`. При исчерпании — `BudgetExhausted` → политика `on_exhausted`.
- **INV-4 (детерминизм/кэш).** T=0 у воркеров; ключ кэша
  `(sha256(prompt), example_id, worker)`; повторная оценка того же промпта не
  тратит бюджет (заряжается 0). Тест: два вызова подряд → один сетевой.
- **INV-5 (неизменность наборов).** D_select/D_anchor/D_audit сериализуются в
  `run_dir/eval_sets.json` при старте и не пересобираются в течение рана;
  загрузка проверяет hash.
- **INV-6 (label-free кластеры).** `clustering.fit` не принимает labels в
  геометрию (сигнатура физически не содержит `labels`); rating-статистики
  доступны только диагностическому отчёту.

---

## 4. Фазы: контракты и Definition of Done

### Фаза 0 — гигиена (без LLM-затрат)

1. `fitness/objective.py`: режим `cvar_lex`:
   `fitness = cvar_shrunk + eps_global·acc_global − length_penalty`.
   κ удаляется из всех путей, кроме `v1_weighted` (абляция). Сигнатура
   `compute_fitness(...)` сохраняется, `cfg.mode` расширяется.
2. `fitness/metrics.py`:
   - `smoothed_group_acc(preds, gold, group_ids, a, b) -> Dict[int, float]`
     (Beta-сглаживание; поддержка эффективных счётчиков для будущего soft);
   - `cvar_from_accs(accs, quantile) -> float` (по сглаженным);
   - `bootstrap_ci(values | per-user accs, n_boot, seed) -> (lo, hi)`.
3. `workers/ensemble.py`: `aggregation: median` (порядковая шкала; нечётный M —
   средний голос), `majority` сохраняется как опция. Убрать tie-break-эвристику
   из дефолтного пути.
4. Удалить мёртвый `CandidateEvaluator._run_predict` (известный баг
   `prompt_template=""`).
5. `configs/base_v3.yaml` по схеме §2; `config.py` — валидация новых секций.

**DoD Ф0:** `pytest` зелёный (новые: `test_objective_cvar_lex`,
`test_metrics_smoothing`, `test_median_aggregation`); smoke-ран с
`base_v3.yaml` проходит end-to-end на mock-ансамбле.

### Фаза 1 — роли данных (ядро; открывает E0 и E1)

6. `data/splits.py` (новый):
   `split_sources(user_ids, profiles, cfg) -> SourceSplit{fit_user_ids,
   heldout_user_ids}`. 70/30 по user_id, user-disjoint (юзер целиком в одной
   половине), стратификация по квантилям (mean length, PCA-1 эмбеддинга профиля);
   лог баланса страт. Сериализация в run_dir.
7. `data/profiles.py`: разделить признаки —
   `features_transferable(profile) -> np.ndarray` (эмбеддинг→PCA, длина,
   пунктуация/капс/восклицания) и `features_calibration(profile)` (mean/std
   метки, доля экстремумов; **только** для диагностики).
8. `data/clustering.py` (переработка):
   - `fit_style_clusters(fit_profiles, cfg) -> ClusterArtifacts{scaler, pca,
     centroids, K_effective, descriptors, diagnostics}` — fit только на
     fit_sources, только блок T (INV-6); StandardScaler обязателен;
   - выбор K: silhouette + стабильность по сидам + мощность
     `K ≤ d_select_size / n_min` (n_min из конфига, пре-регистрирован);
   - `assign(profiles, artifacts) -> Dict[user_id, cluster_id]` — label-free;
   - `diagnostics`: ANOVA rating-статистик по кластерам (отчёт «кластеры
     содержательны»), внутрипользовательская дисперсия (триггер soft, Р1),
     стабильность по сидам;
   - `control: shuffle` — фиксированная случайная перестановка назначений
     после fit (random-clusters абляция, Р13);
   - дескрипторы типов: top-N ближайших юнитов → шаблонная строка (Р4).
9. `acquisition/eval_sets.py` (новый):
   `build_eval_sets(heldout_split, cluster_ids, initial_predictions, cfg, seed)
   -> EvalSets{d_select, d_anchor, d_audit}` — индексы примеров:
   - D_select: стратификация по группам (label-free назначение);
   - D_anchor: ячейки группа×класс из уверенно решённых стартовым промптом
     (`anchor_confident_rule`);
   - D_audit: дизъюнктен с обоими, из heldout;
   - все три сериализуются (`eval_sets.json` + hash, INV-5).
10. `experiment/budget.py` (новый): `TokenTracker` (INV-3), отчёт по уровням
    в `summary.json`; конфиг `budget.*`.
11. `evolution/openevolve_adapter.py` — `CandidateEvaluator` v3:
    - `evaluate_mut(prompt, d_mut_indices) -> MutResult{local_score, errors}`;
    - `evaluate_select(prompt, shard: float | None) -> SelectResult{group_accs
      (smoothed), cvar, fitness, ci, raw_preds}`;
    - файловый `PredictionCache` (INV-4) в run_dir;
    - fitness/QD — только из `evaluate_select` (INV-2).
12. `scripts/run_e0_proxy_diag.py` (новый, эксперимент E0 — SPEC §7/E0):
    один прогон стартового промпта на val-capped → по кэшированным предсказаниям
    пересчёт CVaR_cluster для сетки K × вариантов признаков → Spearman corr с
    R_worst по юзерам, разброс по сидам кластеризации → `e0_report.json` +
    markdown-таблица. **Ноль дополнительных инференсов** сверх одного прохода.
13. Миграция конфигов E1 на v3-роли (семантика Р6: батч-механизм остаётся только
    поставщиком D_mut; все решения — с D_select).

**DoD Ф1:** E0 запускается на smoke (mock) и на live; `eval_sets.json`
воспроизводим по сиду; тесты `test_splits_stratified`, `test_clustering_labelfree`
(включая «labels в fit физически не передаются»), `test_eval_sets_disjoint`,
`test_budget_tracker`, `test_prediction_cache`.

### Фаза 2 — перекоммутация цикла (открывает полный E1 и E5-внутренний)

14. `controller.py`: удалить использование `AcquisitionPool.hard/anchor/
    update_hard_anchor`; `pool.py` упрощается до U-резервуара (подпитка D_mut
    по d_i). Цикл: build D_mut per-мутация → каскад → фронт/архив → carryover →
    val раз в цикл → D_audit раз в цикл.
15. `acquisition/batch_builder.py`: `build_d_mut(fit_pool, group_accs_smoothed,
    cfg, seed) -> indices` — квоты ∝ (1−Acc_τ) (Acc_τ — сглаженные, с D_select),
    внутри группы ранг (err, d), diversity-отбор в топе. Старые «≥1 на кластер»
    и бинарный префильтр `err>0 or d>0` удаляются.
16. Каскад приёмки в адаптере (SPEC §4.3): mut-тест vs родитель → шард
    `shard_frac` → полный D_select → anchor-гейт `ci_delta` (Р15) со счётчиком
    `rejected_by_anchor`. Реализация через `cascade_evaluation` OpenEvolve, если
    стадийный API совместим, иначе обёртка в адаптере.
    **Сделано (2026-07-28):** `prime/experiment/anchor_gate.py` — δ как допуск
    (пол 2 примера) + точный односторонний McNemar по дискордантным парам,
    α=`data_roles.anchor_gate_alpha`; разбивка регрессии по ячейкам
    группа×класс; счётчик в `summary.anchor_gate`; отклонённый кандидат
    сохраняется в `al_iter_*/rejected_by_anchor_gate.txt`. Точечная δ-версия
    отклоняла кандидатов на шуме 2/50 — см. OBSERVATIONS M7/M8, P6.
    Компенсация просадки якоря выигрышем на D_select **запрещена** (M8):
    D_select — то, против чего идёт поиск, якорь существует для контроля
    именно этого смещения.
17. `evolution/qd_features.py`: дескрипторы = сглаженные Acc_τ с последней
    select-оценки, `feature_bins: 3`; патч конфига OpenEvolve.
18. `experiment/proxy_validation.py` (расширить): за цикл логируются —
    corr(CVaR_cluster, R_worst_val); внутренний ID-vs-OOD gap (fit-срез vs
    D_select); D_select↔D_audit gap (post-selection bias, Р15).

**DoD Ф2:** полный smoke AL-ран на новой петле; в кодовой базе нет ссылок на
hard/anchor-пул; `summary.json` содержит бюджет по уровням, corr, оба gap'а,
счётчик отклонений якоря; тест `test_cascade_order` (порядок и短-circuit),
`test_dmut_quotas`.

### Фаза 3 — селекционные структуры (открывает E2, E-абляции, спайк фронта)

19. **Прототип-спайк интеграции фронта** (до E2, SPEC §4.6): минимальный ран
    с кастомным parent selection поверх OpenEvolve; если инвазивно — включить
    fallback-режим «серия коротких ранов с нашей селекцией между ними» (флаг
    в конфиге, архитектурно поддержан AL-циклами).
20. `consolidation/pareto_front.py` (новый, развитие `_cluster_pareto_select`):
    - `update(front, candidate: SelectResult) -> bool` — приёмка: кандидат
      лучший или в CI лучшего хотя бы на одной группе;
    - `sample_parent(front, rng) -> PromptRecord` — вес ∝ числу групп-лидерств;
    - `carryover_slice(front, k) -> List[PromptRecord]` — элита на группу +
      глобальный лучший.
21. `evolution/prompt_blocks.py`: парсер блоков `<Task>/<BaseGuidelines>/
    <DynamicRules>/<FewShotExamples>` + **валидатор мутаций**: diff по блокам;
    отклонение (0 вызовов API) при изменении >1 блока / порче тегов / правке
    BaseGuidelines вне консолидации; счётчик отклонённых в логи.
22. `evolution/artifacts.py`: error report из D_mut (INV-1) + per-type разрез +
    дескрипторы типов + **changelog правил линии предков** (какие правила когда
    добавлены и какую группу чинили).
23. `consolidation/base_consolidator.py`: консолидат = кандидат (не наследник);
    оценка каскадом; CI-гейт `ci_delta` + anchor-гейт; scope=BaseGuidelines.
24. `pool_carryover.py`: carryover = `carryover_slice` фронта; все сравнения —
    на одном D_select.

**DoD Ф3:** спайк фронта дал вердикт (интеграция или fallback-флаг);
E2-лестница конфигурируема одними YAML-переключателями (каждая арка = 1 флаг);
тесты `test_pareto_front` (приёмка/веса/срез), `test_mutation_validator`,
`test_changelog_artifacts`, `test_info_access`.

### Фаза 4 — протокол и ветки

25. Прокси-гейт кластеров со стоп-кодом «чинить кластеры» (порог — по первым
    ранам, Р10); политика `on_exhausted` бюджета.
26. Routing mode B (SPEC §4.9): сборщик промпта по кластеру, abstention-порог
    по расстоянию до центроида, лог качества роутера; C — как абляция.
27. Label-lean — изолированная ветка конфига с предупреждением в отчёте.
28. Soft-SLT апгрейд (для E6, Р1): softmax-π в clustering, weighted Acc_τ в
    metrics (эффективные счётчики уже поддержаны Ф0), вес ∝ Σπ(τ)(1−Acc_τ)
    в batch_builder.
29. Интеграции baselines для E3/E9: MIPROv2 (DSPy, Р14), GEPA; переезд
    v1-реализаций EvoPrompt/APO/OPRO на официальный сплит.

**DoD Ф4:** E-routing и E6-soft запускаемы конфигом; baseline-раннеры дают
`summary.json` в том же формате (общий агрегатор).

---

## 5. Соответствие текущему коду / что удаляется

| Сейчас (v2-скелет) | Судьба |
|---|---|
| `acquisition/pool.py` hard/anchor, `update_hard_anchor` | удалить Ф2 (роль берут eval_sets + D_mut) |
| бинарный префильтр `err>0 or d>0` в scoring | удалить Ф2 |
| `group_aware` = сортировка `(err, d, −cluster_id)` | заменить на квоты ∝ (1−Acc_τ) Ф2 |
| двухпроекционные центроиды в `clustering.py` | заменить label-free fit Ф1; full-проекция → только диагностика |
| `pool_carryover.select_carryover` top-k по fitness | заменить срезом фронта Ф3 |
| `gate_consolidation` (точечный δ) | заменить CI-гейтом Ф3 |
| κ в fitness-режимах | только `v1_weighted` Ф0 |
| majority + tie_break lowest_rating | median-дефолт Ф0 |
| `CandidateEvaluator._run_predict` (мёртвый) | удалить Ф0 |
| synthetic-fallback эмбеддингов (тихий) | сделать громким и отключаемым Ф1 |

---

## 6. Тесты (минимальный обязательный набор)

| Тест | Фаза | Проверяет |
|---|---|---|
| `test_objective_cvar_lex` | Ф0 | формула, отсутствие κ, length penalty |
| `test_metrics_smoothing` | Ф0 | Beta-сглаживание, CVaR, эффективные счётчики |
| `test_median_aggregation` | Ф0 | медиана на порядковой шкале, чётный/нечётный M |
| `test_splits_stratified` | Ф1 | 70/30, user-disjoint, баланс страт |
| `test_clustering_labelfree` | Ф1 | labels не попадают в fit; control=shuffle |
| `test_eval_sets_disjoint` | Ф1 | D_select/D_anchor/D_audit дизъюнктны, hash |
| `test_budget_tracker` | Ф1 | уровни, исчерпание, кэш заряжает 0 |
| `test_prediction_cache` | Ф1 | ключ, повторный вызов без сети |
| `test_cascade_order` | Ф2 | порядок mut→shard→full→anchor, short-circuit |
| `test_dmut_quotas` | Ф2 | квоты ∝ (1−Acc_τ), ранг внутри группы |
| `test_pareto_front` | Ф3 | приёмка по CI, веса родителя, срез carryover |
| `test_mutation_validator` | Ф3 | >1 блока, порча тегов, Base вне консолидации |
| `test_info_access` | Ф3 | INV-1: артефакты не видят heldout |

---

## 7. Артефакты рана (`results/<exp>/<run>/`)

```
run_metadata.json      # конфиг-снапшот, git hash, сиды
eval_sets.json         # индексы D_select/D_anchor/D_audit + hash
cluster_artifacts.json # scaler/pca/centroids/K/descriptors + diagnostics
prediction_cache/      # (sha256(prompt), example_id, worker) → pred
cycles/<n>/            # D_mut индексы, кандидаты, каскад-лог, фронт-снапшот
budget_report.json     # вызовы/токены по уровням (mut/shard/full/anchor/val/audit/mutator)
proxy_report.json      # corr, ID-vs-OOD gap, D_select↔D_audit gap по циклам
summary.json           # финальные метрики (val/test), CI, счётчики гейтов
```

---

## 8. Карта «фаза → эксперименты» (сводно; детали в SPEC §7)

| Готовность | Эксперименты |
|---|---|
| Ф0 + Ф1 | **E0** (диагностика прокси), **E1** (go/no-go, критерий Р11) |
| Ф2 | полный внутренний протокол E1, внутренний ID-vs-OOD, random-clusters абляция |
| Ф3 | **E2** (лестница), E-абляции механизмов, консолидация |
| Ф4 | E3/E9 (baselines, MIPROv2), E4 (CivilComments + inferred-vs-oracle), E-routing, E6-soft, label-lean |

Порядок работ и смета бюджета — SPEC §7 («Порядок работ», «Смета бюджета»).

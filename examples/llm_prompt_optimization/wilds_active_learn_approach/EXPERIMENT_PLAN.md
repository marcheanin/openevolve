# План экспериментов: Active Prompt Evolution

## 1. Аудит реализации vs исходный план

### Этап 1: Формирование начального промпта (APE + Stratified Diversity Sampling)

| Пункт плана | Статус | Файл | Комментарий |
|---|---|---|---|
| K-Means кластеризация по эмбеддингам внутри каждого класса | ✅ | `prepare_initial_prompt.py` | 3 кластера на класс, берутся центроиды + аутлайеры |
| Отбор ~25 представительных примеров | ✅ | `prepare_initial_prompt.py` | `max_examples = 25` |
| Meta-prompt для ChatGPT (APE) | ✅ | `prepare_initial_prompt.py` → `ape_meta_prompt.txt` | Генерирует 5 вариантов промпта |
| XML-структура: System/BaseGuidelines/DynamicRules/FewShotExamples/Task | ✅ | `initial_prompt.txt` | Руками создан по результатам APE |

### Этап 2: Active Batch Selection (Hard/Anchor)

| Пункт плана | Статус | Файл | Комментарий |
|---|---|---|---|
| Wrong Prediction → Hard | ✅ | `data_manager.py:_classify`, `reclassify_batch` | `pred != gold` |
| High Disagreement → Hard | ✅ | `data_manager.py:disagreement_score` | Ordinal-aware: `mean(|p_i - p_j|) / scale` |
| Anchor = correct + unanimous | ✅ | `data_manager.py` | `correct AND d_score <= threshold` |
| K-Means diversity selection для Hard | ✅ | `data_manager.py:_select_diverse` | 8 кластеров, ближайшие к центроиду |
| Stratified selection для Anchor | ✅ | `data_manager.py:_select_stratified` | По меткам |
| Активный батч: 70% Hard + 30% Anchor | ✅ | `config.yaml: hard_ratio: 0.7, batch_size: 80` | Настраиваемо |
| Seen/Unseen pool management | ✅ | `data_manager.py` | Инициализация → одноразовая разметка full pool |
| Pool expansion (семантически далёкие от Anchor) | ✅ | `data_manager.py:expand_pool` | Cosine distance, top-N farthest |

### Этап 3: Двухуровневая эволюция промпта

| Пункт плана | Статус | Файл | Комментарий |
|---|---|---|---|
| Модульный XML-промпт | ✅ | `initial_prompt.txt` | Все секции: BaseGuidelines, DynamicRules, FewShotExamples, Task |
| **Level 1**: Мутация `<DynamicRules>` + `<FewShotExamples>` | ✅ | `config.yaml: system_message` | BaseGuidelines и Task заморожены во время эволюции |
| **Level 1**: Per-evaluation error artifacts | ✅ | `evaluator.py: _format_error_artifacts` | Возврат `EvaluationResult` с misclassified + borderline примерами |
| **Level 1**: Error analysis → кластеризация ошибок | ✅ | `error_analyzer.py` | K-Means, static summary в system message |
| **Level 2**: Consolidation (между циклами) | ✅ | `active_loop.py: _consolidate_prompt` | Top-3 промпта → LLM → может менять BaseGuidelines |
| **Level 2**: XML structure validation | ✅ | `active_loop.py: _validate_prompt_structure` | Проверка: все XML-секции, {review}, Rating:, output format |

### Этап 4: Evaluation Strategy

| Пункт плана | Статус | Файл | Комментарий |
|---|---|---|---|
| Fast Eval на active batch (популяция) | ✅ | `evaluator.py:evaluate_fast` | Через `active_batch.json` |
| Full Eval на validation (лучший) | ✅ | `active_loop.py:_evaluate_split_full` | Single-pass: R_global + Hard/Anchor на val split |
| MAP-Elites: prompt_length × Acc_Hard | ✅ | `config.yaml: feature_dimensions` | 5 bins, population 30 |
| Cascade evaluation | ⬜ Отключён | `config.yaml: cascade_evaluation: false` | Не нужно при batch_size=80 |

### Этап 5: Метрики и Fitness

| Пункт плана | Статус | Файл | Комментарий |
|---|---|---|---|
| `fitness = 0.5*Acc_Hard + 0.3*Acc_Anchor + 0.2*kappa_Hard - P_Len` | ✅ | `evaluator.py:_compute_weighted_fitness` | Веса W1=0.5, W2=0.3, W3=0.2 |
| Length penalty: -0.01 per 100 tokens > 2000 | ✅ | `evaluator.py` | `PROMPT_LEN_LIMIT=2000` |
| Validation: Hard/Anchor classification | ✅ | `active_loop.py:_evaluate_split_full` | Acc_Hard, Acc_Anchor, R_global, R_worst, MAE на val |
| Test: baseline vs final comparison | ✅ | `active_loop.py` | R_global, R_worst, MAE, combined + Acc_Hard, Acc_Anchor |

---

## 2. Найденные проблемы и рекомендации

### 2.1 Двойной вызов LLM на test/val — ИСПРАВЛЕНО

Ранее `evaluate_full` и `_evaluate_split_hard_anchor` вызывались последовательно на одном сплите, удваивая стоимость. Исправлено: единая функция `_evaluate_split_full` делает один проход inference и возвращает все метрики (R_global/R_worst/MAE/combined + Acc_Hard/Acc_Anchor).

### 2.2 Размер данных

- `max_train_users: 25` → примерно 250-500 train примеров (зависит от `min_user_reviews: 10`)
- `max_val_users: 15` → примерно 150-300 val/test примеров
- Batch size = 80 → ~16-32% train pool → pool exhaustion за 3-6 expansion cycles

Это приемлемо для прототипа, но стоит документировать.

### 2.3 Стоимость одного AL цикла (Схема C: 4 цикла × 15 итераций + Consolidation)

За один AL цикл:
- **Batch eval (init):** 80 examples × 3 workers = 240 API calls
- **Evolution:** ~15 iterations × (mutation LLM call + evaluate 80×3) = ~3 615 calls
- **Batch re-eval:** 240 calls
- **Consolidation:** 1 LLM call (DeepSeek R1, large context) = 1 call (~8K tokens in + ~4K out)
- **Val eval:** ~200 examples × 3 workers = 600 calls

**Итого за 1 AL цикл: ~4 696 API calls + 1 consolidation.** При 4 AL циклах: ~18 780 worker calls + 4 consolidation calls + init full pool (~1 500) + 2× test eval (~1 200).
**Общий бюджет: ~21 500 worker API calls + 4 consolidation calls** (consolidation незначительно влияет на бюджет).

---

## 3. План экспериментов

### Эксперимент 0: Проверка окружения (Smoke Test)

**Цель:** Убедиться, что пайплайн работает end-to-end.

**Команда:**
```bash
python run_pipeline.py --step baseline
python active_loop.py --n-al 1 --n-evolve 3
```

**Что проверяем:**
- [ ] Данные загружаются корректно (train/val/test split sizes)
- [ ] 3 worker-модели отвечают через OpenRouter
- [ ] `active_batch.json` создаётся и читается evaluator-ом
- [ ] OpenEvolve запускается без ошибок
- [ ] `results/active_loop_log.json` содержит 1 запись с корректными полями
- [ ] Промпт после эволюции отличается от начального (проверить `results/al_iter_0/best_prompt.txt`)

**Ожидаемое время:** 10-15 минут.

---

### Эксперимент 1: Baseline (начальный промпт)

**Цель:** Установить точку отсчёта для всех метрик.

**Команда:**
```bash
python run_baseline.py
```

**Метрики для записи:**

| Метрика | Train | Validation | Test |
|---|---|---|---|
| R_global | | | |
| R_worst | | | |
| MAE | | | |
| Combined Score | | | |
| Mean Kappa | | | |

**Дополнительно (ручной анализ):**
- Распределение ошибок по классам (confusion matrix)
- Доля Hard/Anchor при начальной разметке full pool

**Результат:** `results/baseline/metrics.json`

---

### Эксперимент 2: Основной AL Run (Схема C: 4 цикла × 15 итераций + Consolidation)

**Цель:** Проверить эффективность двухуровневой Active Prompt Evolution.

**Архитектура (Схема C + Two-Level Evolution):**

Два уровня мутаций — быстрая (внутри цикла) и стратегическая (между циклами):

**Level 1 (fast, 15 итераций внутри цикла):**
- Мутируются `<DynamicRules>` + `<FewShotExamples>` (BaseGuidelines заморожены).
- Каждая оценка кандидата возвращает конкретные ошибки (artifacts): misclassified примеры с gold/predicted/worker votes и borderline примеры.
- Мутатор (DeepSeek R1) видит ЧТО конкретно текущий промпт делает не так и может целенаправленно исправлять правила или менять few-shot примеры.

**Level 2 (slow, 1 раз между циклами — Consolidation):**
- Top-3 промпта из MAP-Elites архива передаются в отдельный LLM вызов.
- LLM анализирует паттерны: правила, появляющиеся во всех лучших промптах, промоутятся в `<BaseGuidelines>`.
- Разрешено изменять ВСЕ секции, включая BaseGuidelines.
- Результат валидируется (XML-структура, {review} placeholder, output format).
- Consolidated промпт становится seed для следующего AL цикла.

**Рабочий цикл (для каждого из 4 AL циклов):**
  1. Собрать активный батч (80 примеров: 70% Hard + 30% Anchor из Seen).
  2. Запустить OpenEvolve: Level 1 эволюция (15 итераций, мутация DynamicRules + FewShotExamples).
  3. Взять лучший промпт, переоценить батч, переклассифицировать Hard/Anchor.
  4. **Consolidation (Level 2):** top-3 из архива → LLM → улучшенный seed (может менять BaseGuidelines).
  5. Оценить на validation (single pass: R_global + Acc_Hard/Anchor).
  6. Если Hard ≤ 5 и есть Unseen → expansion (семантически далёкие от Anchor).
  7. Consolidated промпт → seed следующего цикла.

**Почему Схема C + Consolidation:**
- Level 1: быстрая тактика на конкретных ошибках (artifacts = feedback loop)
- Level 2: стратегическая рефлексия над опытом цикла (meta-learning)
- BaseGuidelines меняются 1 раз за цикл, информированно, на основе top-K результатов
- MAP-Elites архив сбрасывается между циклами → нет stale fitness
- Суммарно ~60 evolve-итераций + 4 consolidation вызова

**Команда:**
```bash
python active_loop.py --n-al 4 --n-evolve 15
```

**Конфигурация (config.yaml):**
```yaml
active_learning:
  uncertainty_threshold: 0.0
  batch_size: 80
  hard_ratio: 0.7
  expansion_trigger: 5
```

**Что отслеживаем на каждой AL-итерации:**

| Метрика | Источник |
|---|---|
| `val_Acc_Hard` | Hard/Anchor classification на val |
| `val_Acc_Anchor` | Hard/Anchor classification на val |
| `val_R_global` | Общая точность на val |
| `val_mae` | MAE на val |
| `val_n_hard` / `val_n_anchor` | Количество Hard/Anchor на val |
| `batch_n_hard` / `batch_n_anchor` | Hard/Anchor в текущем batch |
| `n_seen` / `n_unseen` | Размер seen/unseen pool |
| `expanded` / `n_expanded` | Произошла ли expansion |
| `consolidated` | Был ли выполнен consolidation step |

**Ключевые гипотезы:**
1. `val_Acc_Hard` растёт от цикла к циклу (главный показатель обучения)
2. `val_Acc_Anchor` не падает (нет catastrophic forgetting)
3. `batch_n_hard` уменьшается после эволюции (промпт учится)
4. После expansion `batch_n_hard` подскакивает (новые трудные примеры)
5. `val_n_hard` уменьшается глобально (правила генерализуются)
6. Consolidation улучшает BaseGuidelines, что видно по стабильному росту Acc_Anchor
7. FewShotExamples эволюция улучшает покрытие всех 5 рейтингов

**Результаты:**
- `results/active_loop_log.json` — полный лог
- `results/al_iter_*/best_prompt.txt` — лучший промпт после Level 1
- `results/al_iter_*/consolidated_prompt.txt` — промпт после Level 2 (consolidation)
- `results/al_iter_*/database/` — MAP-Elites архив (программы с кодом и метриками)
- `results/baseline_test_metrics.json` — baseline test
- `results/final_test_metrics.json` — final test

---

### Эксперимент 3: Сравнение Test Before vs After

**Цель:** Оценить абсолютное улучшение промпта на held-out test set.

**Данные:** Берутся автоматически из Эксперимента 2.

**Таблица результатов:**

| Метрика | Baseline (initial prompt) | Final (after AL) | Δ |
|---|---|---|---|
| R_global | | | |
| R_worst | | | |
| MAE | | | |
| Combined Score | | | |
| Acc_Hard | | | |
| Acc_Anchor | | | |
| n_hard / n_anchor | | | |

**Ключевые вопросы:**
- Улучшился ли R_global? На сколько?
- Улучшился ли R_worst (fairness, worst-group)? Это важно для WILDS benchmark.
- Уменьшился ли MAE?
- Уменьшилась ли доля Hard-примеров на test? (генерализация правил)

---

### Эксперимент 4: Ablation — влияние expansion_trigger

**Цель:** Понять, как порог expansion влияет на обучение.

**Варианты:**

| Конфигурация | expansion_trigger | Гипотеза |
|---|---|---|
| 4a: Aggressive | 10 | Раннее расширение, больше разнообразие |
| 4b: Default | 5 | Базовый вариант |
| 4c: Conservative | 2 | Глубже прорабатывать текущий batch |

**Фиксированные параметры:** `n-al=4, n-evolve=15, batch_size=80, hard_ratio=0.7`

**Метрики для сравнения:**
- Финальный `val_Acc_Hard`
- Финальный `val_R_global`
- Количество expansion events
- Итоговый `n_seen` (какая доля pool была использована)

---

### Эксперимент 5: Ablation — влияние uncertainty_threshold

**Цель:** Проверить, влияет ли фильтрация "слабых" disagreements на качество.

**Варианты:**

| Конфигурация | uncertainty_threshold | Что происходит |
|---|---|---|
| 5a: Any disagreement | 0.0 | Любое расхождение → Hard |
| 5b: Moderate | 0.15 | Средний порог (~0.6 звезды spread) |
| 5c: Strict | 0.25 | Только сильные разногласия (~1 звезда) |

**Гипотеза:** При threshold > 0 в Hard попадают только "настоящие" трудные примеры, эволюция эффективнее. Но при слишком высоком threshold теряем полезные примеры.

---

### Эксперимент 6: Ablation — hard_ratio

**Цель:** Проверить баланс Hard/Anchor в batch.

| Конфигурация | hard_ratio | Hard:Anchor |
|---|---|---|
| 6a: Balanced | 0.5 | 40:40 |
| 6b: Default | 0.7 | 56:24 |
| 6c: Hard-heavy | 0.9 | 72:8 |

**Гипотеза:** 0.7 — оптимальный баланс. При 0.9 — catastrophic forgetting (мало Anchor). При 0.5 — медленное обучение (мало Hard).

---

## 4. Сравнение с бейзлайнами

| Метод | R_global (val) | Источник |
|---|---|---|
| Exp3b (Yandex, ансамбль) | 68.27% | Предыдущий эксперимент |
| Embedding baseline | ~70.3% | `evaluation_results_baseline_*.json` |
| Active Prompt Evolution | ? | Эксперимент 2 |

---

## 5. Порядок запуска

### Обязательная последовательность:

```
1. Smoke Test (Эксперимент 0)
   python run_baseline.py
   python active_loop.py --n-al 1 --n-evolve 3

2. Baseline (Эксперимент 1)
   python run_baseline.py
   → results/baseline/metrics.json

3. Main AL Run (Эксперимент 2, Схема C)
   python active_loop.py --n-al 4 --n-evolve 15
   → results/active_loop_log.json
   → results/baseline_test_metrics.json
   → results/final_test_metrics.json

4. Report
   python generate_final_report.py
   python visualize.py
   → results/final_report/REPORT.md

5. Ablations (Эксперименты 4-6, по необходимости)
   Изменять config.yaml → запускать active_loop.py → сохранять results
```

### Оценка бюджета (OpenRouter API):

| Этап | API calls | Примерная стоимость* |
|---|---|---|
| Baseline (Exp 1) | ~2 400 | ~$0.50 |
| Main AL Run (Exp 2, 4×15) | ~21 500 | ~$4-7 |
| Каждый ablation (Exp 4-6) | ~21 500 | ~$4-7 |
| **Итого минимум** | **~24 000** | **~$5-8** |
| **С ablations (3 шт)** | **~88 000** | **~$17-28** |

*Оценка при средней цене ~$0.15-0.30 / 1K calls (DeepSeek V3 + Gemma + GPT-4o-mini mix)

**Учёт токенов:** после каждого прогона сохраняются фактические токены по моделям:
- `results/baseline/token_usage.json` — после `run_baseline.py`
- `results/token_usage.json` — после `active_loop.py`

Формат: `by_model` (input_tokens, output_tokens, total_tokens на модель), плюс общие суммы. По ним можно оценить стоимость (тарифы OpenRouter по моделям).

---

## 6. Критерии успеха

### Минимальный успех:
- val_Acc_Hard выросла хотя бы на 5% за 4 AL цикла
- val_Acc_Anchor не упала более чем на 2%
- test R_global ≥ baseline R_global

### Хороший результат:
- test R_global > Exp3b (68.27%)
- val_Acc_Hard > 50%
- Наблюдается паттерн "learn → exhaust Hard → expand → learn again"

### Отличный результат:
- test R_global > Embedding baseline (~70.3%)
- test R_worst улучшился (fairness gain)
- Правила в `<DynamicRules>` интерпретируемы и генерализованы

---

## 7. Что анализировать в отчёте

1. **Кривая обучения:** график val_Acc_Hard и val_Acc_Anchor по AL-итерациям
2. **Pool dynamics:** n_seen, n_unseen, expansion events на таймлайне
3. **Эволюция промпта:** diff между initial и final `<DynamicRules>` — какие правила были найдены
4. **Error analysis:** какие типы ошибок остались после AL (кластеризация оставшихся Hard)
5. **Test generalization gap:** val metrics vs test metrics — переносится ли обучение
6. **Сравнение с бейзлайнами:** таблица R_global / R_worst / MAE

## 8. Файлы проекта (справочник)

| Файл | Роль |
|---|---|
| `config.yaml` | Конфигурация: LLM, workers, active_learning, MAP-Elites |
| `dataset.yaml` | Настройки датасета WILDS Amazon |
| `initial_prompt.txt` | Начальный XML-промпт |
| `prepare_initial_prompt.py` | APE: Stratified Diversity Sampling → meta-prompt |
| `workers.py` | LLMWorker: OpenRouter API wrapper |
| `data_manager.py` | Seen/Unseen pools, Hard/Anchor, expansion |
| `error_analyzer.py` | K-Means кластеризация ошибок |
| `evaluator.py` | evaluate_fast (batch) + evaluate_full (split) + error artifacts + OpenEvolve entry point |
| `active_loop.py` | Главный цикл AL: init → evolve → reclassify → **consolidate** → expand → repeat |
| `run_baseline.py` | Baseline evaluation на train/val/test |
| `generate_final_report.py` | Генерация REPORT.md |
| `visualize.py` | Графики из active_loop_log.json |
| `run_pipeline.py` | Оркестратор всех шагов |
| `token_usage.py` | Учёт токенов по моделям (для оценки стоимости) |

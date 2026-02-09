# Откуда берется каждый раздел промпта для эволюции

Этот документ объясняет, откуда берется каждый раздел в промпте, который отправляется LLM для генерации улучшенного промпта.

## Структура промпта (full_rewrite_user.txt)

```
# System prompt

# Current Prompt Information
- Current performance metrics: {metrics}
- Areas identified for improvement: {improvement_areas}

{artifacts}

# Prompt Evolution History
{evolution_history}

# Current Prompt
{current_program}

# Task
Rewrite the prompt to improve its performance...
```

---

## 1. `{metrics}` - Текущие метрики производительности

**Источник:** `parent.metrics` (метрики родительской программы)

**Откуда берется:**
- В `iteration.py` (строка 65): `program_metrics=parent.metrics`
- `parent` - это программа, выбранная из базы данных как родитель для эволюции
- Метрики вычисляются при оценке программы в `evaluator.evaluate_program()`

**Как форматируется:**
- Функция `_format_metrics()` в `prompt/sampler.py` (строки 156-168)
- Формат: `- metric_name: value.0000`
- Пример:
  ```
  - combined_score: 0.7325
  - prompt_length: 231.0000
  - reasoning_strategy: 0.2000
  - llm_clarity: 0.1600
  ```

**Код:**
```python
# iteration.py, строка 65
program_metrics=parent.metrics

# prompt/sampler.py, строки 111, 156-168
metrics_str = self._format_metrics(program_metrics)
```

---

## 2. `{improvement_areas}` - Области для улучшения

**Источник:** Анализ метрик и истории эволюции

**Откуда берется:**
- Функция `_identify_improvement_areas()` в `prompt/sampler.py` (строки 170-226)
- Анализирует:
  - Изменение fitness score (улучшение/ухудшение/стабильность)
  - Feature dimensions (какие области пространства решений исследуются)
  - Длина кода (если превышает порог)

**Логика определения:**

1. **Изменение fitness:**
   - Сравнивает текущий fitness с предыдущим из `previous_programs`
   - Если улучшился: "Fitness improved: 0.65 → 0.73"
   - Если ухудшился: "Fitness declined: 0.83 → 0.73. Consider revising recent changes."
   - Если стабилен: "Fitness stable at 0.73"

2. **Feature exploration:**
   - Если есть feature dimensions (например, `prompt_length`, `reasoning_strategy`)
   - Показывает, какую область исследуем: "Exploring prompt_length=231.00, reasoning_strategy=0.20 region"

3. **Длина кода:**
   - Если код превышает порог (`suggest_simplification_after_chars` или `code_length_threshold`)
   - Предлагает упрощение: "Code length exceeds threshold. Consider simplification."

4. **По умолчанию:**
   - Если ничего не найдено: "No specific guidance. Focus on general improvements."

**Код:**
```python
# prompt/sampler.py, строки 114-116
improvement_areas = self._identify_improvement_areas(
    current_program, parent_program, program_metrics, 
    previous_programs, feature_dimensions
)
```

---

## 3. `{artifacts}` - Артефакты выполнения

**Источник:** Результаты выполнения родительской программы

**Откуда берется:**
- В `iteration.py` (строка 54): `parent_artifacts = database.get_artifacts(parent.id)`
- Артефакты сохраняются при оценке программы в `evaluator.evaluate_program()`
- Могут включать:
  - Вывод выполнения программы
  - Ошибки (errors)
  - Предупреждения (warnings)
  - Другие результаты выполнения

**Как форматируется:**
- Функция `_render_artifacts()` в `prompt/sampler.py` (строки 565-592)
- Формат: Markdown с блоками кода
- Пример:
  ```markdown
  ## Last Execution Output
  
  ### errors
  ```
  [содержимое ошибок]
  ```
  
  ### output
  ```
  [вывод выполнения]
  ```
  ```

**Когда включается:**
- Только если `config.include_artifacts = true` (в `config.yaml`)
- И если артефакты доступны для родительской программы

**Код:**
```python
# iteration.py, строки 54, 72
parent_artifacts = database.get_artifacts(parent.id)
program_artifacts=parent_artifacts if parent_artifacts else None

# prompt/sampler.py, строки 124-126
if self.config.include_artifacts and program_artifacts:
    artifacts_section = self._render_artifacts(program_artifacts)
```

---

## 4. `{evolution_history}` - История эволюции

**Источник:** Предыдущие попытки и лучшие программы из базы данных

**Откуда берется:**
- В `iteration.py` (строки 58-59):
  - `island_top_programs = database.get_top_programs(5, island_idx=parent_island)` - топ-5 программ острова
  - `island_previous_programs = database.get_top_programs(3, island_idx=parent_island)` - последние 3 программы острова
- Также используются `inspirations` - разнообразные программы для вдохновения (строка 51)

**Как форматируется:**
- Функция `_format_evolution_history()` в `prompt/sampler.py` (строки 228-402)
- Включает три секции:

### a) Previous Attempts (предыдущие попытки)
- Берет последние 3 программы из `previous_programs`
- Для каждой показывает:
  - Номер попытки
  - Тип изменений (из `metadata.changes`, например "Full rewrite")
  - Метрики производительности
  - Исход (Improvement/Regression/Mixed results) - сравнивает с родителем

### b) Top Programs (лучшие программы)
- Берет топ-N программ из `top_programs` (N = `config.num_top_programs`, обычно 5)
- Для каждой показывает:
  - Номер программы
  - Fitness score
  - Ключевые особенности (key features) - **генерируются автоматически из метрик**
  - Полный код программы
  - Язык программирования

**Откуда берутся key_features:**
- **Источник:** Автоматически генерируются из метрик программы
- **Код:** `prompt/sampler.py`, строки 316-329
- **Логика:**
  1. Сначала проверяется, есть ли `key_features` в программе: `program.get("key_features", [])`
  2. Если нет (что обычно и происходит), генерируются автоматически:
     - Для каждой метрики в `program.metrics` создается строка: `"Performs well on {metric_name} ({value:.4f})"`
     - Пример: `"Performs well on combined_score (0.8550)"`, `"Performs well on llm_clarity (0.1800)"`
  3. Все key_features объединяются через запятую: `", ".join(key_features)`
- **Важно:** `key_features` не хранятся в структуре `Program`, а генерируются динамически при форматировании истории эволюции

### c) Inspirations (вдохновляющие программы)
- Берет разнообразные программы из `inspirations`
- Для каждой показывает:
  - Тип программы (Best performer, Diverse example, etc.)
  - Score
  - Уникальные особенности (unique features) - **генерируются автоматически**
  - Полный код программы

**Откуда берутся unique_features для inspirations:**
- **Источник:** Автоматически генерируются функцией `_extract_unique_features()`
- **Код:** `prompt/sampler.py`, строки 491-551
- **Логика:**
  1. **Из metadata:** Если есть `metadata.changes` и он короткий, добавляется "Modification: {changes}"
  2. **Из метрик:**
     - Если метрика ≥ 0.9: "Excellent {metric_name} ({value})"
     - Если метрика ≤ 0.3: "Alternative {metric_name} approach"
  3. **Из кода (эвристики):**
     - Если содержит `class` и `def __init__`: "Object-oriented approach"
     - Если содержит `numpy` или `np.`: "NumPy-based implementation"
     - Если содержит `for` и `while`: "Mixed iteration strategies"
     - Если код короткий (≤ `concise_implementation_max_lines`): "Concise implementation"
     - Если код длинный (≥ `comprehensive_implementation_min_lines`): "Comprehensive implementation"
  4. **По умолчанию:** Если ничего не найдено, используется `_determine_program_type()` для определения типа программы
  5. Ограничивается первыми N особенностями (N = `num_top_programs`)

**Откуда берутся key_features для diverse programs:**
- **Источник:** Автоматически генерируются из первых 2 метрик
- **Код:** `prompt/sampler.py`, строки 367-374
- **Логика:**
  1. Проверяется наличие `key_features` в программе
  2. Если нет, берутся первые 2 метрики из `program.metrics`
  3. Для каждой создается строка: `"Alternative approach to {metric_name}"`
  4. Пример: `"Alternative approach to combined_score, Alternative approach to prompt_length"`

**Код:**
```python
# iteration.py, строки 51, 58-59, 66-68
parent, inspirations = database.sample(num_inspirations=config.prompt.num_top_programs)
island_top_programs = database.get_top_programs(5, island_idx=parent_island)
island_previous_programs = database.get_top_programs(3, island_idx=parent_island)

previous_programs=[p.to_dict() for p in island_previous_programs],
top_programs=[p.to_dict() for p in island_top_programs],
inspirations=[p.to_dict() for p in inspirations],

# prompt/sampler.py, строки 119-121
evolution_history = self._format_evolution_history(
    previous_programs, top_programs, inspirations, language, feature_dimensions
)

# prompt/sampler.py, строки 316-329 - генерация key_features для топ программ
key_features = program.get("key_features", [])
if not key_features:
    key_features = []
    for name, value in program.get("metrics", {}).items():
        if isinstance(value, (int, float)):
            key_features.append(f"Performs well on {name} ({value:.4f})")
key_features_str = ", ".join(key_features)
```

---

## 5. `{current_program}` - Текущий промпт

**Источник:** Код родительской программы

**Откуда берется:**
- В `iteration.py` (строка 63): `current_program=parent.code`
- `parent.code` - это текст промпта родительской программы
- Это тот промпт, который нужно улучшить

**Что это:**
- Полный текст промпта, который был использован в предыдущей итерации
- Для HoVer это может быть, например:
  ```
  Determine whether the following claim is SUPPORTED or NOT SUPPORTED based on factual evidence.

  Claim: {claim}

  Analyze the claim carefully and provide your verdict as either "SUPPORTED" or "NOT SUPPORTED".
  ```

**Код:**
```python
# iteration.py, строка 63
current_program=parent.code

# prompt/sampler.py, строка 145
current_program=current_program,
```

---

## 6. System Message - Системное сообщение

**Источник:** Конфигурация или шаблон

**Откуда берется:**
- В `config.yaml`: `prompt.system_message` (строки 31-39)
- Или из шаблона, если `system_message` - это имя шаблона

**Для HoVer:**
```yaml
system_message: |
  You are an expert at creating effective prompts for language models.
  Your goal is to evolve prompts that maximize accuracy on the given task.
  
  When creating new prompts:
  1. Build on successful patterns from the examples
  2. Be creative but maintain clarity
  3. Consider different reasoning strategies (direct, step-by-step, few-shot)
  4. Optimize for the specific task requirements
```

**Код:**
```python
# prompt/sampler.py, строки 102-108
if self.system_template_override:
    system_message = self.template_manager.get_template(self.system_template_override)
else:
    system_message = self.config.system_message
    if system_message in self.template_manager.templates:
        system_message = self.template_manager.get_template(system_message)
```

---

## Полный поток данных

```
1. iteration.py:run_iteration_with_shared_db()
   ├─ database.sample() → parent, inspirations
   ├─ database.get_artifacts(parent.id) → parent_artifacts
   ├─ database.get_top_programs(5) → island_top_programs
   └─ database.get_top_programs(3) → island_previous_programs

2. prompt_sampler.build_prompt()
   ├─ _format_metrics(parent.metrics) → {metrics}
   ├─ _identify_improvement_areas(...) → {improvement_areas}
   ├─ _render_artifacts(parent_artifacts) → {artifacts}
   ├─ _format_evolution_history(...) → {evolution_history}
   └─ parent.code → {current_program}

3. Шаблон full_rewrite_user.txt заполняется:
   {metrics} → форматированные метрики
   {improvement_areas} → области для улучшения
   {artifacts} → артефакты выполнения
   {evolution_history} → история эволюции
   {current_program} → текущий промпт

4. Результат:
   {
     "system": system_message (из config.yaml),
     "user": заполненный шаблон full_rewrite_user.txt
   }
```

---

## Ключевые файлы

1. **Шаблон промпта:**
   - `templates/full_rewrite_user.txt` - шаблон с плейсхолдерами

2. **Заполнение шаблона:**
   - `openevolve/openevolve/prompt/sampler.py` - класс `PromptSampler`
     - `build_prompt()` - главная функция
     - `_format_metrics()` - форматирование метрик
     - `_identify_improvement_areas()` - определение областей улучшения
     - `_format_evolution_history()` - форматирование истории
     - `_render_artifacts()` - форматирование артефактов

3. **Сбор данных:**
   - `openevolve/openevolve/iteration.py` - функция `run_iteration_with_shared_db()`
     - Получает данные из базы данных
     - Передает в `prompt_sampler.build_prompt()`

4. **Источники данных:**
   - `openevolve/openevolve/database.py` - класс `ProgramDatabase`
     - `sample()` - выбор родителя и inspirations
     - `get_top_programs()` - получение лучших программ
     - `get_artifacts()` - получение артефактов

5. **Вычисление метрик:**
   - `evaluator.py` - класс `Evaluator`
     - `evaluate_program()` - оценка программы и вычисление метрик

---

## Пример заполненного промпта

```markdown
# Current Prompt Information
- Current performance metrics: 
  - combined_score: 0.7325
  - prompt_length: 231.0000
  - reasoning_strategy: 0.2000
  - llm_clarity: 0.1600
  - llm_specificity: 0.1200
- Areas identified for improvement: 
  - Fitness declined: 0.8325 → 0.7325. Consider revising recent changes.
  - Exploring prompt_length=231.00, reasoning_strategy=0.20 region of solution space

## Last Execution Output

### errors
```
[ошибки выполнения, если были]
```

# Prompt Evolution History
## Previous Attempts

### Attempt 5
- Changes: Full rewrite
- Metrics: combined_score: 0.8325, prompt_length: 421.0000, ...
- Outcome: Mixed results

### Attempt 4
- Changes: Full rewrite
- Metrics: combined_score: 0.8325, prompt_length: 431.0000, ...
- Outcome: Improvement in all metrics

## Top Programs

### Program 1 (Score: 0.8550)
- Key features: Performs well on combined_score (0.8550)
- Code:
```
[полный код лучшего промпта]
```

# Current Prompt
Determine whether the following claim is SUPPORTED or NOT SUPPORTED based on factual evidence.

Claim: {claim}

Analyze the claim carefully and provide your verdict as either "SUPPORTED" or "NOT SUPPORTED".

# Task
Rewrite the prompt to improve its performance on the specified metrics.
...
```

---

## Настройки в config.yaml

```yaml
prompt:
  num_top_programs: 5        # Сколько топ программ показывать
  include_artifacts: true    # Включать ли артефакты выполнения
  system_message: |          # Системное сообщение для LLM
    You are an expert...
```

---

## Примечания

1. **Изоляция островов:** Все данные берутся из того же острова, что и родитель (`parent_island`), чтобы поддерживать изоляцию между островами.

2. **Feature dimensions:** Метрики, которые являются feature dimensions (например, `prompt_length`, `reasoning_strategy`), не включаются в fitness score, но используются для определения областей исследования.

3. **Артефакты:** Артефакты могут быть обрезаны, если превышают `max_artifact_bytes` (по умолчанию).

4. **История эволюции:** Показываются только последние 3 попытки и топ-5 программ, чтобы не перегружать промпт.

5. **key_features для топ программ:**
   - **Не хранятся** в структуре `Program`, генерируются динамически
   - Создаются из **всех метрик** программы автоматически
   - Формат: `"Performs well on {metric_name} ({value})"` для каждой метрики
   - Пример: `"Performs well on combined_score (0.8550), Performs well on prompt_length (658.0000), ..."`
   - Для diverse programs используется другой формат: `"Alternative approach to {metric_name}"` (только первые 2 метрики)

6. **unique_features для inspirations:**
   - Генерируются функцией `_extract_unique_features()` на основе:
     - Metadata (изменения)
     - Метрик (высокие/низкие значения)
     - Анализа кода (паттерны, длина)
   - Более сложная логика, чем для key_features


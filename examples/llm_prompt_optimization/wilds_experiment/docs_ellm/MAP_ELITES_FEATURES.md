# MAP-Elites Feature Dimensions для Prompt Evolution

## Обзор

MAP-Elites — алгоритм quality-diversity, который поддерживает архив элитных решений в многомерной сетке feature dimensions. Каждая ячейка сетки содержит лучшее найденное решение для данной комбинации признаков.

**Зачем нужны хорошие feature dimensions:**
1. **Исследование пространства решений** — находить качественные решения в разных "нишах"
2. **Предотвращение преждевременной сходимости** — если алгоритм застрял в одной области, другие ниши продолжают развиваться
3. **Обеспечение diversity** — LLM-сэмплер видит разнообразные примеры, что помогает генерировать новые идеи

---

## Проблемы старых feature dimensions

### Старая конфигурация
```yaml
feature_dimensions: ["prompt_length", "reasoning_strategy"]
```

### Анализ эффективности

| Метрика | Проблема |
|---------|----------|
| `prompt_length` | Все хорошие промпты имели длину 800–1150 символов, занимая только 2–3 ячейки из 10. Нет исследования коротких/длинных промптов. |
| `reasoning_strategy` | Почти всегда 0.0! Простой подсчёт ключевых слов не отражает реальную структуру промпта. |

**Результат:** MAP-Elites практически не работал — все промпты попадали в одну область сетки.

---

## Новые feature dimensions

### Exp 2: Single Model
```yaml
feature_dimensions: ["criteria_explicitness", "domain_focus"]
```

### Exp 3: Ensemble + Voting
```yaml
feature_dimensions: ["sentiment_vocabulary_richness", "mean_kappa"]
```

---

## Описание новых метрик

### 1. `criteria_explicitness` (0.0 – 1.0)

**Что измеряет:** Насколько явно промпт описывает критерии для каждого класса (1-5 звёзд).

**Почему важно:** Промпты с явными критериями для каждого рейтинга дают более консистентные результаты.

**Как считается:**
```python
def calculate_criteria_explicitness(prompt: str) -> float:
    """
    Считает долю классов (1-5), для которых явно описаны критерии.
    
    Паттерны:
    - "1 star:" или "1 stars:"
    - "- 1:" (bullet point)
    - "For 1 star..."
    """
    explicit_criteria = 0
    for star in range(1, 6):
        patterns = [
            rf'{star}\s*stars?\s*:',
            rf'[-•]\s*{star}\s*:',
            rf'for\s+{star}\s+stars?',
        ]
        if any(re.search(p, prompt, re.I) for p in patterns):
            explicit_criteria += 1
    return explicit_criteria / 5.0
```

**Примеры:**
- `"1 star: broken product\n2 stars: issues\n3 stars: ok\n4 stars: good\n5 stars: excellent"` → **1.0**
- `"Rate the review from 1 to 5 stars"` → **0.0**
- `"1 star: bad\n5 stars: great"` → **0.4**

---

### 2. `domain_focus` (0.0 – 1.0)

**Что измеряет:** Насколько промпт специфичен для домена Home & Kitchen.

**Почему важно:** Domain-specific промпты могут лучше работать на целевых данных, но хуже generalize. Это создаёт интересный trade-off.

**Как считается:**
```python
def calculate_domain_focus(prompt: str) -> float:
    domain_keywords = [
        'kitchen', 'home', 'appliance', 'cookware', 'utensil',
        'durable', 'sturdy', 'quality', 'material', 'build',
        'daily use', 'cooking', 'cleaning',
        'value', 'price', 'recommend', 'buy',
    ]
    found = sum(1 for kw in domain_keywords if kw in prompt.lower())
    return min(1.0, found / 10.0)  # Saturates at 10 matches
```

**Примеры:**
- `"Rate kitchen appliances based on durability, quality, and value"` → **~0.5**
- `"Analyze sentiment of the review"` → **~0.0**

---

### 3. `mean_kappa` (0.0 – 1.0) — только для ансамблей (Cohen's Kappa)

**Что измеряет:** Межаннотаторское согласие между workers (Cohen's κ), обрезанное до [0, 1].

**Почему важно для MAP-Elites:** Строже, чем disagreement rate:
- **Высокий κ** = реальное согласие (промпты, где workers различают классы и согласуются)
- **κ ≈ 0** = не лучше случайного угадывания (в т.ч. стратегия "всегда большинство")
- Отрицательный κ обрезается до 0 в скоринге и в сетке

**Как считается:**
```python
# В compute_metrics(): pairwise Cohen's kappa между workers, затем mean
kappa_feature = max(0.0, mean_kappa)  # для MAP-Elites и combined_score
```

**Интерпретация (Landis & Koch):**
- κ < 0.00: плохо (хуже монетки)
- 0.00 – 0.20: незначительное согласие
- 0.21 – 0.40: слабое
- 0.41 – 0.60: умеренное
- 0.61 – 0.80: существенное
- 0.81 – 1.00: почти идеальное

---

## Дополнительные метрики (для анализа)

Помимо основных feature dimensions, evaluator возвращает дополнительные метрики:

| Метрика | Описание |
|---------|----------|
| `instruction_specificity` | Конкретность инструкций ("focus on" vs "analyze") |
| `sentiment_vocabulary` | Богатство словаря эмоций (love, hate, excellent, terrible...) |
| `structural_complexity` | Наличие структуры (bullet points, секции, нумерация) |

Эти метрики можно использовать для анализа эволюции или как альтернативные feature dimensions.

---

## Как работает MAP-Elites с новыми dimensions

### Пример сетки для Exp 3

```
                    criteria_explicitness
                    0.0   0.2   0.4   0.6   0.8   1.0
                  ┌─────┬─────┬─────┬─────┬─────┬─────┐
             0.0  │     │     │     │     │  A  │  B  │  ← Высокий консенсус
                  ├─────┼─────┼─────┼─────┼─────┼─────┤
             0.2  │     │     │     │  C  │     │  D  │
                  ├─────┼─────┼─────┼─────┼─────┼─────┤
disagreement 0.4  │     │     │     │     │  E  │     │
   rate           ├─────┼─────┼─────┼─────┼─────┼─────┤
             0.6  │     │  F  │     │     │     │  G  │
                  ├─────┼─────┼─────┼─────┼─────┼─────┤
             0.8  │     │     │     │     │     │     │  ← Высокое расхождение
                  └─────┴─────┴─────┴─────┴─────┴─────┘
```

**Интерпретация ячеек:**
- **B** (criteria=1.0, disagreement=0.0): Идеальный промпт — все критерии явные, workers согласны
- **G** (criteria=1.0, disagreement=0.6): Детальный промпт, но workers не согласны — возможно, критерии противоречивы
- **F** (criteria=0.2, disagreement=0.6): Размытый промпт с разногласиями — плохой кандидат

---

## Конфигурация в config.yaml

### Exp 2: Single Model
```yaml
database:
  population_size: 50
  archive_size: 500
  num_islands: 4
  feature_dimensions: ["criteria_explicitness", "domain_focus"]
  feature_bins: 10
```

### Exp 3: Ensemble
```yaml
database:
  population_size: 50
  archive_size: 500
  num_islands: 4
  feature_dimensions: ["sentiment_vocabulary_richness", "mean_kappa"]
  feature_bins: 10
```

---

## Возвращаемые метрики из evaluator

Evaluator должен возвращать feature dimensions **на верхнем уровне** словаря:

```python
# exp2_single_evolved/evaluator.py
return {
    "combined_score": combined_score,  # Для OpenEvolve
    "metrics": {...},                  # Детальные метрики
    # Feature dimensions (MUST be at top level!)
    "criteria_explicitness": 0.8,
    "domain_focus": 0.5,
    # Legacy/additional features
    "prompt_length": 0.95,
    "instruction_specificity": 0.7,
}
```

```python
# exp3_ensemble_voting/evaluator.py
return {
    "combined_score": combined_score,
    "metrics": {...},
    # Feature dimensions
    "sentiment_vocabulary_richness": 0.7,
    "mean_kappa": 0.45,  # Cohen's Kappa, clipped to [0, 1]
    # Additional
    "domain_focus": 0.5,
    "prompt_length": 0.95,
}
```

---

## Миграция со старых feature dimensions

Если у вас есть существующие checkpoints со старыми dimensions (`prompt_length`, `reasoning_strategy`), они остаются совместимыми — evaluator по-прежнему возвращает эти метрики для backward compatibility.

Для перехода на новые dimensions:
1. Обновите `feature_dimensions` в `config.yaml`
2. Перезапустите эволюцию — архив будет перестроен с новыми dimensions

---

## Рекомендации по выбору dimensions

| Эксперимент | Рекомендуемые dimensions | Обоснование |
|-------------|-------------------------|-------------|
| Single model | `sentiment_vocabulary_richness`, `domain_focus` | Trade-off между богатством формулировок и доменной специфичностью |
| Ensemble voting | `sentiment_vocabulary_richness`, `mean_kappa` | Cohen's κ: реальное согласие vs «всегда большинство» |
| LLM aggregator | `sentiment_vocabulary_richness`, `ensemble_agreement` | Фокус на согласованности workers |

---

## Модуль feature_dimensions.py

Все функции расчёта feature dimensions находятся в:
```
experiments/feature_dimensions.py
```

Основные функции:
- `calculate_criteria_explicitness(prompt)` → float
- `calculate_domain_focus(prompt)` → float
- `calculate_all_features(prompt, metrics, is_ensemble)` → Dict

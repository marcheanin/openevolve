# Обновление Feature Dimensions: sentiment_vocabulary_richness

## Изменения

Заменена метрика `criteria_explicitness` на комбинированную метрику `sentiment_vocabulary_richness` во всех экспериментах.

### Новая метрика: `sentiment_vocabulary_richness`

**Формула**: `0.6 * sentiment_vocabulary + 0.4 * example_richness`

**Описание**:
- Комбинирует богатство словаря сентиментов (60%) и наличие примеров (40%)
- Измеряет, насколько промпт использует эмоциональные слова (love, hate, excellent, terrible) и примеры (for example, such as, quoted phrases)
- Диапазон: 0.0 - 1.0

**Преимущества**:
- Более разнообразное пространство поиска (не все промпты будут иметь одинаковое значение)
- Семантически значимая метрика
- Лучше работает с MAP-Elites (заполняет больше ниш)

## Обновленные файлы

### 1. `feature_dimensions.py`
- ✅ Добавлена функция `calculate_sentiment_vocabulary_richness()`
- ✅ Обновлена `calculate_all_features()` для включения новой метрики
- ✅ Обновлены `RECOMMENDED_FEATURES`

### 2. `exp2_single_evolved/config.yaml`
- ✅ `feature_dimensions`: `["sentiment_vocabulary_richness", "domain_focus"]`
- ✅ Обновлен `system_message` с описанием новой метрики

### 3. `exp3_ensemble_voting/config.yaml`
- ✅ `feature_dimensions`: `["sentiment_vocabulary_richness", "mean_kappa"]` (Cohen's κ для ensemble)
- ✅ Обновлен `system_message` с описанием новой метрики

### 4. `exp2_single_evolved/evaluator.py`
- ✅ Использует `sentiment_vocabulary_richness` вместо `criteria_explicitness`
- ✅ Возвращает новую метрику на верхнем уровне для MAP-Elites

### 5. `exp3_ensemble_voting/evaluator.py`
- ✅ Использует `sentiment_vocabulary_richness` вместо `criteria_explicitness`
- ✅ Возвращает новую метрику на верхнем уровне для MAP-Elites

### 6. `visualize_evolution.py`
- ✅ Обновлен для использования `sentiment_vocabulary_richness`
- ✅ Fallback на `criteria_explicitness` для старых экспериментов

### 7. `analyze_improvements.py`
- ✅ Обновлен для использования `sentiment_vocabulary_richness`
- ✅ Fallback на `criteria_explicitness` и `reasoning_strategy` для старых экспериментов

## Обратная совместимость

Все скрипты визуализации и анализа поддерживают fallback на старые метрики:
- Если `sentiment_vocabulary_richness` отсутствует, используется `criteria_explicitness`
- Если `criteria_explicitness` отсутствует, используется `reasoning_strategy`

Это позволяет анализировать старые эксперименты без ошибок.

## Новая MAP-Elites сетка

### Exp2 (Single Model)
- **Ось X**: `sentiment_vocabulary_richness` (0.0 - 1.0)
- **Ось Y**: `domain_focus` (0.0 - 1.0)

### Exp3 (Ensemble Voting)
- **Ось X**: `sentiment_vocabulary_richness` (0.0 - 1.0)
- **Ось Y**: `mean_kappa` (Cohen's κ, обрезанная до [0, 1])

## Ожидаемые улучшения

1. **Лучшее покрытие пространства**: Новая метрика должна создавать больше разнообразия
2. **Более эффективный поиск**: Комбинация vocabulary + examples более информативна
3. **Меньше проблем с конвергенцией**: Не все промпты будут иметь одинаковое значение метрики

---

*Обновлено: 2026-01-27*

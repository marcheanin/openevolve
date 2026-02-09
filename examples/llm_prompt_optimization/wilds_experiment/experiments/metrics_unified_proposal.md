# Предложение: Унифицированная формула Combined Score

## Проблема текущего подхода

Сейчас используются две разные формулы:

1. **Single Model**: `R_global * (1 - 0.2 * MAE/4)`
   - ✅ Учитывает точность (R_global)
   - ✅ Учитывает ошибки (MAE)
   - ❌ Игнорирует fairness (R_worst)

2. **Ensemble**: `0.4*R_global + 0.4*R_worst - 0.2*(1 - D)`
   - ✅ Учитывает точность (R_global)
   - ✅ Учитывает fairness (R_worst)
   - ✅ Учитывает согласованность (disagreement)
   - ❌ Игнорирует ошибки (MAE)

**Проблема**: Формулы несопоставимы, нельзя корректно сравнить single model и ensemble.

## Предложение: Унифицированная формула

### Вариант 1: Базовая унифицированная (рекомендуется)

```python
def compute_combined_score_unified(metrics: Dict[str, Any], is_ensemble: bool = False) -> float:
    """
    Унифицированная формула для single model и ensemble.
    
    Базовая формула:
    score = 0.4 * R_global + 0.3 * R_worst + 0.3 * (1 - MAE/4)
    
    Для ensemble добавляется бонус за согласованность:
    score += 0.1 * (1 - disagreement_rate)  # только если is_ensemble=True
    
    Где:
    - R_global: общая точность (0-1)
    - R_worst: 10-й перцентиль точности по пользователям (0-1)
    - MAE: средняя абсолютная ошибка (0-4), нормализуется к 0-1
    - disagreement_rate: доля примеров с разногласиями (0-1), только для ensemble
    """
    r_global = metrics.get("R_global", 0.0)
    r_worst = metrics.get("R_worst", 0.0)
    mae = metrics.get("mae", 0.0)
    
    # Базовая формула (одинаковая для всех)
    base_score = (
        0.4 * r_global +           # 40% - общая точность
        0.3 * r_worst +            # 30% - fairness (worst-case)
        0.3 * (1.0 - mae / 4.0)   # 30% - точность предсказаний (меньше ошибок = лучше)
    )
    
    # Бонус за согласованность ансамбля (опционально)
    if is_ensemble:
        disagreement = metrics.get("disagreement_rate", 0.0)
        # Бонус до 10% за согласованность (меньше разногласий = лучше)
        consistency_bonus = 0.1 * (1.0 - disagreement)
        return float(base_score + consistency_bonus)
    
    return float(base_score)
```

**Преимущества:**
- ✅ Единая формула для обоих случаев
- ✅ Учитывает все важные метрики (R_global, R_worst, MAE)
- ✅ Для ensemble добавляется бонус за согласованность
- ✅ Позволяет честное сравнение
- ✅ Диапазон: 0.0 - 1.0 (для ensemble может быть до 1.1, но обычно ограничивается 1.0)

### Вариант 2: С конфигурируемыми весами

```python
def compute_combined_score_unified(
    metrics: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
    is_ensemble: bool = False
) -> float:
    """
    Унифицированная формула с настраиваемыми весами.
    
    По умолчанию:
    - w_global = 0.4
    - w_worst = 0.3
    - w_mae = 0.3
    - w_consistency = 0.1 (только для ensemble)
    """
    if weights is None:
        weights = {
            "global": 0.4,
            "worst": 0.3,
            "mae": 0.3,
            "consistency": 0.1 if is_ensemble else 0.0,
        }
    
    r_global = metrics.get("R_global", 0.0)
    r_worst = metrics.get("R_worst", 0.0)
    mae = metrics.get("mae", 0.0)
    
    score = (
        weights["global"] * r_global +
        weights["worst"] * r_worst +
        weights["mae"] * (1.0 - mae / 4.0)
    )
    
    if is_ensemble and weights["consistency"] > 0:
        disagreement = metrics.get("disagreement_rate", 0.0)
        score += weights["consistency"] * (1.0 - disagreement)
    
    return float(min(1.0, score))  # Ограничиваем максимум 1.0
```

### Вариант 3: С нормализацией для разных диапазонов

```python
def compute_combined_score_unified(metrics: Dict[str, Any], is_ensemble: bool = False) -> float:
    """
    Унифицированная формула с нормализацией компонентов к 0-1.
    """
    r_global = metrics.get("R_global", 0.0)  # уже 0-1
    r_worst = metrics.get("R_worst", 0.0)    # уже 0-1
    mae = metrics.get("mae", 0.0)            # 0-4, нормализуем к 0-1
    mae_normalized = 1.0 - (mae / 4.0)      # инвертируем (меньше = лучше)
    
    # Базовая формула
    base_score = 0.4 * r_global + 0.3 * r_worst + 0.3 * mae_normalized
    
    # Для ensemble: учитываем согласованность
    if is_ensemble:
        disagreement = metrics.get("disagreement_rate", 0.0)  # 0-1
        consistency = 1.0 - disagreement  # инвертируем (меньше разногласий = лучше)
        # Взвешиваем: 70% базовая формула, 30% согласованность
        score = 0.7 * base_score + 0.3 * consistency
    else:
        score = base_score
    
    return float(min(1.0, max(0.0, score)))  # Ограничиваем 0-1
```

## Рекомендация

**Использовать Вариант 1** (базовая унифицированная) по следующим причинам:

1. **Простота**: Легко понять и объяснить
2. **Сбалансированность**: Все метрики учитываются с разумными весами
3. **Сравнимость**: Позволяет честно сравнивать single model и ensemble
4. **Расширяемость**: Легко добавить дополнительные компоненты

## Примеры расчетов

### Single Model (Exp 1 baseline):
- R_global = 0.502
- R_worst = 0.308
- MAE = 0.721

```
score = 0.4 * 0.502 + 0.3 * 0.308 + 0.3 * (1 - 0.721/4)
     = 0.201 + 0.092 + 0.3 * 0.820
     = 0.201 + 0.092 + 0.246
     = 0.539
```

### Ensemble (Exp 3a baseline):
- R_global = 0.655
- R_worst = 0.440
- MAE = 0.360
- Disagreement = 0.504

```
base = 0.4 * 0.655 + 0.3 * 0.440 + 0.3 * (1 - 0.360/4)
    = 0.262 + 0.132 + 0.3 * 0.910
    = 0.262 + 0.132 + 0.273
    = 0.667

consistency_bonus = 0.1 * (1 - 0.504) = 0.050
score = 0.667 + 0.050 = 0.717
```

## Миграция

1. Добавить новую функцию `compute_combined_score_unified` в `metrics.py`
2. Обновить evaluators для использования новой функции
3. Пересчитать combined_score для всех экспериментов
4. Обновить финальный отчет

## Дополнительные метрики для анализа

Для более глубокого анализа можно также добавить:
- **R_best**: 90-й перцентиль (лучшие пользователи)
- **R_median**: Медианная точность по пользователям
- **MAE_per_user**: Распределение MAE по пользователям
- **Kappa (κ)**: Для ensemble (уже есть, но можно использовать в формуле)

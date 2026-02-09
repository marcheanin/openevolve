# Исправление несоответствия формул combined_score

## Проблема

В коде использовались **две разные формулы** для вычисления `combined_score`:

### Старая формула (`compute_combined_score_ensemble`)
```python
0.4 * R_global + 0.4 * R_worst - 0.2 * (1 - disagreement)
```

**Использовалась в:**
- `run_baseline.py` (строка 37)
- `final_report.py` (строка 53)

### Новая формула (`compute_combined_score_unified`)
```python
0.4 * R_global + 0.3 * R_worst + 0.3 * (1 - MAE/4) + 0.1 * max(0, mean_kappa)
```

**Использовалась в:**
- `evaluator.py` (строки 384-385) - во время эволюции

## Почему это проблема?

1. **Несоответствие метрик**: Во время эволюции OpenEvolve оптимизировал промпты по одной формуле, а в отчетах показывалась другая формула
2. **Невозможность сравнения**: Значения `combined_score` в отчетах не соответствовали тому, что использовалось для эволюции
3. **Неправильная интерпретация**: Результаты могли быть неправильно интерпретированы

## Что исправлено

### Изменения в коде:

1. **`final_report.py`**:
   - Заменено `compute_combined_score_ensemble` → `compute_combined_score_unified`
   - Добавлен параметр `is_ensemble=True`

2. **`run_baseline.py`**:
   - Заменено `compute_combined_score_ensemble` → `compute_combined_score_unified`
   - Добавлен параметр `is_ensemble=True`

## Преимущества новой формулы

1. **Унифицированность**: Одна формула для single model и ensemble экспериментов
2. **Учет MAE**: Включает среднюю абсолютную ошибку (MAE), что важно для оценки качества
3. **Согласованность**: Та же формула используется во время эволюции и в отчетах
4. **Сравнимость**: Позволяет корректно сравнивать результаты между экспериментами

## Пример различий

Для метрик Exp 3 NEW:
- R_global = 0.6827
- R_worst = 0.4769
- MAE = 0.3376
- mean_kappa (κ) используется в бонусе; обрезается до [0, 1]

**Старая формула**: `0.4*R_global + 0.4*R_worst - 0.2*(1-D)` (deprecated)

**Новая формула**: `base_score + 0.1 * max(0, kappa)` (Cohen's Kappa)

## Важно

- **Существующие отчеты не изменены** - они содержат значения, вычисленные по старой формуле
- **Будущие запуски** будут использовать правильную формулу
- **Для пересчета старых отчетов** нужно будет запустить `final_report.py` заново

## Источники данных для сравнения

1. `exp1_baseline/results/summary.json` - baseline single model
2. `exp2_single_evolved/openevolve_output/final_report.json` - single model + evolution
3. `exp3_ensemble_voting/results_baseline_first/summary.json` - baseline ensemble (старая формула)
4. `exp3_ensemble_voting/openevolve_output_first/final_report.json` - OLD MAP-Elites (старая формула)
5. `exp3_ensemble_voting/openevolve_output/final_report.json` - NEW MAP-Elites (старая формула в отчете, но новая в эволюции)
6. `final_report/final_report.json` - сводный отчет с `combined_score_unified` (правильная формула)

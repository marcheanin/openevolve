# Проверка формул combined_score в Exp 3

## ✅ Статус: Все формулы исправлены

Все файлы в эксперименте `exp3_ensemble_voting` теперь используют правильную унифицированную формулу `compute_combined_score_unified()`.

## Проверенные файлы

### ✅ evaluator.py
- **Строка 38**: `from experiments.metrics import compute_combined_score_unified`
- **Строки 384-385**: Использует `compute_combined_score_unified(val_metrics, is_ensemble=True)`
- **Статус**: ✅ Правильно

### ✅ final_report.py
- **Строка 8**: `from experiments.metrics import compute_combined_score_unified`
- **Строка 53**: `test_metrics["combined_score"] = compute_combined_score_unified(test_metrics, is_ensemble=True)`
- **Статус**: ✅ Исправлено (было `compute_combined_score_ensemble`)

### ✅ run_baseline.py
- **Строка 13**: `from experiments.metrics import compute_combined_score_unified`
- **Строка 37**: `metrics["combined_score"] = compute_combined_score_unified(metrics, is_ensemble=True)`
- **Статус**: ✅ Исправлено (было `compute_combined_score_ensemble`)

## Унифицированная формула

```python
def compute_combined_score_unified(metrics, is_ensemble=True):
    """
    Базовая формула (одинаковая для всех):
        base_score = 0.4 * R_global + 0.3 * R_worst + 0.3 * (1 - MAE/4)
    
    Для ensemble добавляется бонус за согласованность (Cohen's Kappa):
        kappa_score = max(0, mean_kappa)
        consistency_bonus = 0.1 * kappa_score
        combined_score = base_score + consistency_bonus
    """
```

## Старая формула (DEPRECATED)

```python
def compute_combined_score_ensemble(metrics):
    """
    Старая формула (не используется):
        0.4 * R_global + 0.4 * R_worst - 0.2 * (1 - disagreement)
    """
```

## Результаты baseline

Baseline данные теперь правильно пересчитываются с использованием унифицированной формулы:

- **Файл**: `results/summary.json`
- **R_global**: 0.6513
- **R_worst**: 0.4401
- **MAE**: 0.3616
- **mean_kappa** (κ): используется в формуле; обрезается до [0, 1]
- **combined_score (old)**: 0.3336 (старая формула, для справки)
- **combined_score_unified**: 0.7169 (правильная формула)

## Обновленные отчеты

1. ✅ `final_report/final_report.json` - обновлен с baseline данными
2. ✅ `final_report/comparison_table.md` - обновлена таблица сравнения
3. ✅ `generate_final_report.py` - обновлен для поддержки `results/summary.json`

## Важно

- **Существующие JSON файлы** (например, `results_baseline_first/summary.json`) содержат значения, вычисленные по старой формуле - это нормально, они используются только для справки
- **Все новые вычисления** используют правильную унифицированную формулу
- **Для пересчета старых отчетов** нужно запустить соответствующие скрипты заново

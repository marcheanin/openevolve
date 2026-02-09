# Эксперимент 2: Single Model + OpenEvolve

## Цель
Эволюционировать промпт одной модели и получить метрики + графики по поколениям.

## Основные файлы
- `experiments/exp2_single_evolved/evaluator.py` — evaluator для OpenEvolve
- `experiments/exp2_single_evolved/config.yaml` — конфиг OpenEvolve + модель/датасет
- `experiments/exp2_single_evolved/initial_prompt.txt` — стартовый промпт
- `experiments/exp2_single_evolved/run.py` — запуск эволюции
- `experiments/exp2_single_evolved/analyze_improvements.ps1` — генерация графиков

## Запуск эволюции
Из папки `wilds_experiment/experiments/exp2_single_evolved`:

```
python run.py --iterations 100 --train-users 15
```

Во время эволюции оценка идёт на train, валидация на val (каждое поколение).
Параметр `--train-users` ограничивает число фиксированных пользователей в train.

### Быстрый прогон
Для ускорения можно ограничить и val пользователей. В `config.yaml` установи:
```yaml
dataset:
  max_train_users: 15
  max_val_users: 10  # Быстрая валидация
```
Это ускорит каждый evaluation, но может снизить точность оценки generalization.

**Примечание:** Значения `max_train_users` и `max_val_users` автоматически конвертируются в int
(если указаны как строки в YAML). При невалидных значениях ограничение игнорируется с предупреждением.

### Generalization gap penalty
Если `train_R_global - val_R_global` превышает порог, применяется штраф к `combined_score`.
Значения `generalization_gap`, `generalization_gap_threshold`, `generalization_gap_penalty`
передаются в метрики и попадают в промпт эволюции (как часть метрик лучших программ).

### Combined Score
Используется унифицированная формула `compute_combined_score_unified()` для корректного сравнения
с другими экспериментами. Формула: `0.4 * R_global + 0.3 * R_worst + 0.3 * (1 - MAE/4)`.

### Кэширование данных
Evaluator использует многоуровневое кэширование:
- **Дисковый кэш** предобработанных данных (работает между процессами при multiprocessing)
- **In-memory кэш** splits для оптимизации внутри одного процесса
- Автоматическое создание кэша при первом запуске

Это значительно ускоряет параллельные оценки в OpenEvolve.

### MAP-Elites Feature Dimensions
Используются улучшенные feature dimensions для эффективного исследования пространства промптов:

```yaml
# config.yaml
database:
  feature_dimensions: ["criteria_explicitness", "domain_focus"]
```

| Dimension | Описание | Диапазон |
|-----------|----------|----------|
| `criteria_explicitness` | Насколько явно описаны критерии для каждого класса (1-5 звёзд) | 0.0 – 1.0 |
| `domain_focus` | Специфичность для домена Home & Kitchen | 0.0 – 1.0 |

Подробнее: [MAP_ELITES_FEATURES.md](MAP_ELITES_FEATURES.md)

### Структура возвращаемых метрик
Evaluator возвращает словарь со следующей структурой (требуется OpenEvolve):
```python
{
    "combined_score": float,  # На верхнем уровне (требуется OpenEvolve)
    # Feature dimensions для MAP-Elites
    "criteria_explicitness": float,  # Явность критериев (0-1)
    "domain_focus": float,           # Специфичность домена (0-1)
    # Legacy/additional features
    "prompt_length": float,
    "instruction_specificity": float,
    "sentiment_vocabulary": float,
    "metrics": {
        "R_global": float,
        "R_worst": float,
        "mae": float,
        "train_R_global": float,
        "generalization_gap": float,
        "generalization_gap_penalty": float,
    }
}
```

**Важно:** `combined_score` и feature dimensions должны быть на верхнем уровне словаря.

## Графики по поколениям
После окончания эволюции, из той же папки:

```
./analyze_improvements.ps1
```

Скрипт генерирует:
- `openevolve_output/learning_curves.png` — графики метрик по поколениям
- `openevolve_output/evolution_summary.txt` — summary эволюции
- `openevolve_output/visualizations/improvements_analysis.txt` — подробный анализ
- `openevolve_output/visualizations/improvements_summary.json` — JSON summary

## Финальный отчёт на test
После эволюции:

```
python final_report.py
```

Результат:
- `openevolve_output/final_report.json` — метрики на test

## Модель
- `gpt-oss-120b` (одна модель, эволюция промпта)


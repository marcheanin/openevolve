# Эксперимент 3: Ensemble + Majority Voting

## Цель
Исследовать, как ансамбль из трёх моделей с majority voting влияет на качество и устойчивость к пользователям, и как эволюция промпта улучшает этот ансамбль по сравнению с:
- одиночной моделью (Exp 1, 2),
- базовым ансамблем без эволюции (Exp 3a).

## Основные файлы
- `experiments/exp3_ensemble_voting/evaluator.py` — evaluator для OpenEvolve (ансамбль + voting)
- `experiments/exp3_ensemble_voting/config.yaml` — конфиг OpenEvolve + модели/датасет
- `experiments/exp3_ensemble_voting/initial_prompt.txt` — стартовый промпт для workers
- `experiments/exp3_ensemble_voting/run_baseline.py` — запуск baseline ансамбля (Exp 3a)
- `experiments/exp3_ensemble_voting/run_evolved.py` — запуск эволюции (Exp 3b)
- `experiments/exp3_ensemble_voting/analyze_improvements.ps1` — генерация графиков
- `experiments/exp3_ensemble_voting/final_report.py` — финальный прогон на test после эволюции

## Конфигурация ансамбля

### Модели workers (ансамбль из 3 моделей)
Из `config.yaml`:

- `yandexgpt` — YandexGPT 5
- `gemma3-27b` — Gemma 3 27B
- `gpt-oss-120b` — GPT-OSS-120B

Все workers используют общие настройки по умолчанию:
- `temperature: 0.1`
- `max_tokens: 64`
- `timeout: 60`
- `max_retries: 3`

Агрегация:
- `MajorityVoteAggregator` (простой majority voting по трём голосам).

## Запуск экспериментов

### Exp 3a: Ensemble baseline (без эволюции)
Из папки `wilds_experiment/experiments/exp3_ensemble_voting`:

```bash
python run_baseline.py
```

Что делает:
- Загружает `initial_prompt.txt`
- Запускает три worker-модели на сплите, указанном в `dataset.split` (`dataset.yaml`)
- Агрегирует предсказания через majority voting
- Считает метрики (включая Cohen's κ — mean_kappa)
- Сохраняет результаты в `results_baseline/`:
  - `metrics.json` — полные метрики
  - `predictions.json` — предсказания, gold, user_ids, worker_predictions
  - `summary.json` — краткое резюме метрик

### Exp 3b: Ensemble + OpenEvolve (эволюция промпта)
Из той же папки:

```bash
python run_evolved.py --iterations 100 --train-users 15
```

Где:
- `--iterations` — число итераций эволюции (поколений)
- `--train-users` — фиксированное число пользователей в train (через `OPENEVOLVE_MAX_TRAIN_USERS`)

Во время эволюции:
- Оценка идёт на `train` и `validation` сплитах (по логике `evaluator.py`)
- Каждый вызов evaluator:
  - Строит ансамбль из 3 workers
  - Агрегирует через majority voting
  - Считает метрики (включая Cohen's κ — mean_kappa)
  - Возвращает `combined_score` для OpenEvolve + feature dimensions (`sentiment_vocabulary_richness`, `mean_kappa`)

## Настройки датасета и пользователей

В `config.yaml`:

```yaml
dataset:
  config_path: "dataset.yaml"
  max_samples: null
  max_samples_train: null
  max_samples_val: null
  max_samples_test: null
  max_train_users: 10
  max_val_users: 15
```

- `max_train_users` — максимум пользователей в train сплите
- `max_val_users` — максимум пользователей в validation сплите

**Важно:**
- Значения `max_train_users` и `max_val_users` берутся из секции `dataset` `config.yaml` и при загрузке:
  - автоматически конвертируются в `int` (если заданы строкой),
  - при невалидном значении игнорируются с предупреждением.
- Дополнительно можно переопределить `max_train_users` через переменную окружения `OPENEVOLVE_MAX_TRAIN_USERS` (используется в `run_evolved.py`).

## Generalization gap penalty

Как и в Exp 2, используется штраф за переобучение:

- Считается `generalization_gap = max(0, train_R_global - val_R_global)`.
- Параметры берутся из секции `generalization` `config.yaml`:
  - `gap_threshold` (например, `0.10`)
  - `gap_penalty` (например, `0.50`)
- Если gap превышает порог, итоговый `combined_score` для валидации умножается на `(1 - gap_penalty * excess_gap)`, где `excess_gap = generalization_gap - gap_threshold`.

Таким образом, промпты, которые слишком хорошо работают на train, но хуже на val, получают штраф.

## Combined Score (ансамбль)

Для ансамбля используется унифицированная формула `compute_combined_score_unified(metrics, is_ensemble=True)`, которая:
- Балансирует:
  - глобальную точность `R_global`,
  - худший пер-пользовательский accuracy `R_worst`,
  - точность по MAE,
- Добавляет бонус за согласованность ансамбля (Cohen's Kappa, обрезанная до [0, 1]).

Базовая идея:
- `base_score = 0.4 * R_global + 0.3 * R_worst + 0.3 * (1 - MAE/4)`
- `kappa_score = max(0, mean_kappa)` (отрицательная каппа не снижает скор)
- `consistency_bonus = 0.1 * kappa_score`
- `combined_score = base_score + consistency_bonus`

Для fair-сравнения с одиночной моделью:
- та же базовая формула (0.4/0.3/0.3) используется и для single-model экспериментов (Exp 2),
- ансамбль дополнительно получает consistency-бонус за Cohen's Kappa (реальное согласие, а не просто низкий disagreement).

## Кэширование данных

Как и в Exp 2, evaluator использует многоуровневое кэширование:
- **Дисковый кэш** предобработанных данных (через общий `wilds_experiment/evaluator.py`):
  - один раз извлекаются и предобрабатываются данные для категории `Home_and_Kitchen`,
  - сохраняются в `.pkl`, доступны между процессами (multiprocessing).
- **In-memory кэш** splits внутри процесса:
  - `train/validation/test` сплиты создаются один раз,
  - переиспользуются для всех вызовов evaluator.

Это особенно важно для Exp 3, где:
- каждый вызов evaluator вызывает три LLM-а,
- эволюция требует сотни запусков evaluator,
- без кэша загрузка WILDS Amazon была бы узким местом.

## MAP-Elites Feature Dimensions (Ensemble)

Для ансамблевых экспериментов используются специализированные feature dimensions:

```yaml
# config.yaml
database:
  feature_dimensions: ["sentiment_vocabulary_richness", "mean_kappa"]
```

| Dimension | Описание | Диапазон |
|-----------|----------|----------|
| `sentiment_vocabulary_richness` | Богатство словаря эмоций + примеров (см. MAP_ELITES_FEATURES) | 0.0 – 1.0 |
| `mean_kappa` | Cohen's Kappa между workers (обрезанная до [0, 1]) | 0.0 – 1.0 |

**Почему `mean_kappa` (Cohen's Kappa):**
- Строже, чем disagreement rate: наказывает стратегию "всегда предсказывать большинство" (κ=0 при случайном угадывании).
- Шкала Landis & Koch: <0 плохо, 0–0.20 незначительное, 0.21–0.40 слабое, 0.41–0.60 умеренное, 0.61–0.80 существенное, 0.81–1.0 почти идеальное согласие.
- Создаёт сетку по реальному межаннотаторскому согласию, а не просто по доле совпадений.

Подробнее: [MAP_ELITES_FEATURES.md](MAP_ELITES_FEATURES.md)

## Структура возвращаемых метрик (для OpenEvolve)

Функция `evaluate()` в `exp3_ensemble_voting/evaluator.py` возвращает словарь:

```python
{
    "combined_score": float,         # На верхнем уровне (требуется OpenEvolve)
    # Feature dimensions для MAP-Elites
    "sentiment_vocabulary_richness": float,
    "mean_kappa": float,            # Cohen's Kappa, обрезанная до [0, 1]
    # Legacy/additional features
    "prompt_length": float,
    "domain_focus": float,
    "sentiment_vocabulary": float,
    "metrics": {
        "R_global": float,                  # val
        "R_worst": float,                   # val
        "mae": float,                       # val
        "mean_kappa": float,                # val (Cohen's κ между workers)
        "train_R_global": float,
        "train_R_worst": float,
        "train_mae": float,
        "train_mean_kappa": float,
        "train_combined_score": float,
        "val_combined_score": float,
        "generalization_gap": float,
        "generalization_gap_threshold": float,
        "generalization_gap_penalty": float,
    },
}
```

**Критично:** `combined_score` и feature dimensions должны быть на верхнем уровне — это то, что использует OpenEvolve для отбора и эволюции программ.

## Графики и анализ по поколениям

После окончания эволюции (Exp 3b), из папки `exp3_ensemble_voting` запусти:

```bash
./analyze_improvements.ps1
```

Скрипт:
- проверяет наличие `openevolve_output/evolution_trace.jsonl`,
- запускает:
  - `../../visualize_evolution.py` — строит кривые метрик по поколениям,
  - `../../analyze_improvements.py` — детальный анализ улучшений промптов.

Выходные файлы:
- `openevolve_output/learning_curves.png` — кривые метрик (включая ensemble-специфичные метрики)
- `openevolve_output/evolution_summary.txt` — summary эволюции
- `openevolve_output/visualizations/improvements_analysis.txt` — подробный отчёт
- `openevolve_output/visualizations/improvements_summary.json` — JSON summary
- `openevolve_output/visualizations/improvements/` — артефакты по конкретным улучшениям
- `openevolve_output/best/best_program.txt` — лучший эволюционированный промпт для ансамбля

## Финальный отчёт на test (после эволюции)

После того как эволюция завершена и выбран лучший промпт (`openevolve_output/best/best_program.txt`):

```bash
python final_report.py
```

По умолчанию:
- Использует `openevolve_output/best/best_program.txt` как промпт,
- Прогоняет ансамбль на `test` сплите,
- Считает ensemble-метрики и `combined_score` (через `compute_combined_score_ensemble` на test),
- Сохраняет отчёт в:
  - `openevolve_output/final_report.json`

Файл отчёта содержит:
- финальные метрики на test (`test_metrics`),
- информацию о промпте и конфигурации workers,
- количество использованных тестовых примеров.

## Сравнение с Exp 2

- Exp 2: одна модель `gpt-oss-120b`, без κ и disagreement.
- Exp 3:
  - три модели (`yandexgpt`, `gemma3-27b`, `gpt-oss-120b`) + majority voting,
  - дополнительные метрики:
    - `mean_kappa` — Cohen's Kappa между workers (согласованность с учётом случайности),
  - унифицированный `combined_score` с consistency-бонусом `0.1 * max(0, kappa)`.

Итог:
- Exp 3a даёт baseline ансамбля (на тех же данных Home and Kitchen),
- Exp 3b показывает, сколько добавляет эволюция промпта **поверх** самого факта ансамблирования.


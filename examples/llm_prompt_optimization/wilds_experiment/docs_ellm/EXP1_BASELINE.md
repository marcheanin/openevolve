# Эксперимент 1: Single Model Baseline

## Цель
Получить базовые метрики качества на одной модели без эволюции промпта.

## Датасет
- WILDS Amazon Reviews
- Категория: Home and Kitchen (`category_id: 24`)
- Сплит: `test`
- Фильтр пользователей: минимум 10 отзывов на пользователя

## Основные файлы
- `experiments/exp1_baseline/evaluator.py` — загрузка данных и оценка
- `experiments/exp1_baseline/config.yaml` — конфиг эксперимента
- `experiments/exp1_baseline/dataset.yaml` — настройки датасета
- `experiments/exp1_baseline/initial_prompt.txt` — фиксированный базовый промпт
- `experiments/exp1_baseline/run.py` — запуск baseline

## Запуск
Из папки `wilds_experiment/experiments/exp1_baseline`:

```
python run.py
```

## Выходные файлы
- `results/metrics.json` — полные метрики
- `results/summary.json` — краткие метрики
- `results/predictions.json` — предсказания, gold labels, user_ids

## Метрики
- `R_global` — accuracy по всем примерам
- `R_worst` — 10-й перцентиль per-user accuracy
- `mae` — mean absolute error
- `combined_score` — `R_global * (1 - 0.2 * MAE/4)`

## Замечания
- Используется одна модель (`gpt-oss-120b` по умолчанию).
- `max_samples` можно ограничить в `config.yaml` для быстрой проверки.


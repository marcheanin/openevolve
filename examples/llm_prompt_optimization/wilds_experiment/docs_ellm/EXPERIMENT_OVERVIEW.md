# eLLM Ensemble Experiments: Обзор

## Цель экспериментов

Исследование эволюции промптов и ансамблирования LLM для задачи классификации отзывов на WILDS Amazon (Home and Kitchen).

**Ключевая идея**: Сравнить различные подходы к улучшению качества LLM-аннотаторов:
1. Эволюция промпта для одной модели
2. Ансамбль моделей с majority voting
3. Ансамбль моделей с LLM-агрегатором

---

## Структура экспериментов

```
┌─────────────────────────────────────────────────────────────────────┐
│                        4 ЭКСПЕРИМЕНТА                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐                        │
│  │ Exp 1: Baseline │ →  │ Exp 2: Single   │                        │
│  │ (1 модель,      │    │ Model + OpenEvo │                        │
│  │  без эволюции)  │    │ (1 модель,      │                        │
│  └─────────────────┘    │  с эволюцией)   │                        │
│         ↓               └─────────────────┘                        │
│                                ↓                                    │
│  ┌──────────────────────────────────────────┐                      │
│  │ Exp 3: Ensemble + Voting                  │                      │
│  │ (3 модели, majority vote)                │                      │
│  │  • 3a: без эволюции (baseline ансамбля)  │                      │
│  │  • 3b: с эволюцией промпта               │                      │
│  └──────────────────────────────────────────┘                      │
│         ↓                                                           │
│  ┌──────────────────────────────────────────┐                      │
│  │ Exp 4: Ensemble + LLM Aggregator          │                      │
│  │ (3 модели + LLM анализирует их выходы)   │                      │
│  │  • Промпт агрегатора: статический        │                      │
│  │  • Эволюция: только промпт workers       │                      │
│  └──────────────────────────────────────────┘                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Эксперимент 1: Single Model Baseline

**Цель**: Установить baseline без какой-либо оптимизации.

| Параметр | Значение |
|----------|----------|
| Модель | gpt-oss-120b (одна) |
| Эволюция | НЕТ |
| Агрегация | НЕТ (одна модель) |
| Промпт | Фиксированный базовый |

**Что делаем**:
1. Запускаем одну модель на test split
2. Считаем ВСЕ метрики (R_global, R_worst, MAE)
3. Сохраняем как baseline для сравнения

Подробнее: `EXP1_BASELINE.md`

**Выход**:
- `exp1_baseline/results/metrics.json` - полные метрики
- `exp1_baseline/results/summary.json` - метрики в кратком виде
- `exp1_baseline/results/predictions.json` - предсказания + gold + user_ids

---

## Эксперимент 2: Single Model + OpenEvolve

**Цель**: Показать эффект эволюции промпта на одной модели.

| Параметр | Значение |
|----------|----------|
| Модель | gpt-oss-120b (одна) |
| Эволюция | ДА (через OpenEvolve) |
| Агрегация | НЕТ (одна модель) |
| Поколений | ~50-100 |

**Что делаем**:
1. Запускаем OpenEvolve: оценка на train + валидация на val в каждом поколении
2. Эволюционируем промпт
3. Строим кривые метрик по поколениям
4. После эволюции прогоняем лучший промпт на test
5. Сравниваем с Exp 1

Примечание: train ограничен фиксированным числом пользователей (`--train-users`).

**Выход**: 
- `exp2_single_evolved/openevolve_output/learning_curves.png` - графики метрик по поколениям (train/val)
- `exp2_single_evolved/openevolve_output/evolution_summary.txt` - summary эволюции
- `exp2_single_evolved/openevolve_output/best/best_program.txt` - лучший промпт
- `exp2_single_evolved/openevolve_output/final_report.json` - итоговый отчёт на test

Подробнее: `EXP2_SINGLE_EVOLVED.md`

---

## Эксперимент 3: Ensemble + Voting

**Цель**: Показать эффект ансамбля с majority voting.

| Параметр | Значение |
|----------|----------|
| Модели | gpt-oss-120b, YandexGPT 5, Gemma3-27B |
| Эволюция | 3a: НЕТ, 3b: ДА |
| Агрегация | Majority Voting |

### Эксперимент 3a: Без эволюции
- Запускаем 3 модели с базовым промптом
- Агрегируем через majority vote
- Считаем метрики (включая Cohen's κ — mean_kappa)
- Это baseline ансамбля

### Эксперимент 3b: С эволюцией
- Запускаем OpenEvolve
- Evaluator вызывает 3 модели и агрегирует
- Эволюционируем промпт для workers
- Строим кривые метрик

**Выход**:
- `exp3_ensemble_voting/baseline_results.json` - 3a
- `exp3_ensemble_voting/evolved_results.json` - 3b
- `exp3_ensemble_voting/evolution_history.json` - кривые

---

## Эксперимент 4: Ensemble + LLM Aggregator

**Цель**: Заменить majority voting на умную LLM-агрегацию.

| Параметр | Значение |
|----------|----------|
| Workers | gpt-oss-120b, YandexGPT 5, Gemma3-27B |
| Агрегатор | Qwen3-235B (отдельная модель-судья) |
| Эволюция | Только промпт workers |
| Промпт агрегатора | Статический (написан вручную) |

**Ключевое отличие от Exp 3**:
- Workers возвращают **полные выходы** (с reasoning)
- LLM-агрегатор **анализирует** эти выходы
- Агрегатор выдаёт финальный rating (1-5)

**Что делаем**:
1. Workers генерируют развёрнутые ответы
2. LLM-агрегатор получает все ответы и review
3. Агрегатор выдаёт финальный rating
4. Считаем метрики, строим кривые

**Выход**:
- `exp4_llm_aggregator/results.json`
- `exp4_llm_aggregator/evolution_history.json`
- `exp4_llm_aggregator/aggregator_prompt.txt`

---

## Метрики для всех экспериментов

| Метрика | Формула | Описание | Применимость |
|---------|---------|----------|--------------|
| **R_global** | `correct / total` | Общая accuracy | Все |
| **R_worst** | `10th_percentile(per_user_acc)` | Fairness метрика WILDS | Все |
| **MAE** | `mean(abs(pred - true))` | Mean Absolute Error | Все |
| **κ (kappa)** | Cohen's kappa (mean_kappa), обрезанная до [0, 1] в скоринге | IAA между workers | Exp 3, 4 |

---

## Конфигурация моделей (Yandex Cloud)

| Role | Model | URI |
|------|-------|-----|
| Worker 1 (Baseline) | gpt-oss-120b | `gpt://b1gemincl8p7b2uiv5nl/gpt-oss-120b/latest` |
| Worker 2 | YandexGPT 5 | `gpt://b1gemincl8p7b2uiv5nl/yandexgpt/latest` |
| Worker 3 | Gemma3-27B | `gpt://b1gemincl8p7b2uiv5nl/gemma-3-27b-it/latest` |
| LLM Aggregator | Qwen3-235B | `gpt://b1gemincl8p7b2uiv5nl/qwen3-235b-a22b-fp8/latest` |
| OpenEvolve Sampler | Qwen3-235B | `gpt://b1gemincl8p7b2uiv5nl/qwen3-235b-a22b-fp8/latest` |

---

## Ожидаемые результаты

| Эксперимент | R_global | R_worst | κ | Улучшение |
|-------------|----------|---------|---|-----------|
| 1. Baseline | ~70% | ~55% | N/A | - |
| 2. Single + Evo | ~75-77% | ~60-65% | N/A | +5-7% R_global |
| 3a. Ensemble baseline | ~75% | ~62% | ~0.45 | Ансамбль помогает |
| 3b. Ensemble + Evo | ~78% | ~68% | ~0.65 | +3% от эволюции |
| 4. LLM Aggregator | ~79-80% | ~70% | ~0.70 | Лучший результат |

---

## Структура файлов

```
wilds_experiment/
├── experiments/
│   ├── __init__.py
│   ├── metrics.py                  # R_global, R_worst, κ, D, MAE
│   ├── workers.py                  # LLMWorker для Yandex Cloud
│   ├── aggregators.py              # MajorityVote + LLMAggregator
│   │
│   ├── exp1_baseline/
│   │   ├── evaluator.py
│   │   ├── config.yaml
│   │   ├── run.py
│   │   └── results/
│   │
│   ├── exp2_single_evolved/
│   │   ├── evaluator.py
│   │   ├── config.yaml
│   │   ├── run.py
│   │   └── results/
│   │
│   ├── exp3_ensemble_voting/
│   │   ├── evaluator.py
│   │   ├── config.yaml
│   │   ├── run_baseline.py
│   │   ├── run_evolved.py
│   │   └── results/
│   │
│   └── exp4_llm_aggregator/
│       ├── evaluator.py
│       ├── aggregator_prompt.txt
│       ├── config.yaml
│       ├── run.py
│       └── results/
```

---

## Важные принципы

1. **Каждый эксперимент - отдельная директория** (не переписываем файлы)
2. **OpenEvolve для эволюции** (не собственный GA)
3. **Evaluator возвращает combined_score** для OpenEvolve
4. **Кривые по поколениям** для каждого эксперимента с эволюцией
5. **Полные метрики** в каждом эксперименте

---

## Связанные документы

- [OPENEVOLVE_INTEGRATION.md](OPENEVOLVE_INTEGRATION.md) - как работает OpenEvolve
- [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) - детали реализации
- [CODE_REFERENCE.md](CODE_REFERENCE.md) - примеры кода


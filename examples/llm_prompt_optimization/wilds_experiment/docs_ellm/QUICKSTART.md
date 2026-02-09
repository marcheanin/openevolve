# QUICKSTART: eLLM Ensemble Experiments

## TL;DR

**Цель**: Сравнить подходы к оптимизации LLM-аннотаторов на WILDS Amazon (Home and Kitchen).

**4 эксперимента**:
1. **Baseline**: 1 модель, без эволюции → R_global ~70%
2. **Single + OpenEvolve**: 1 модель, с эволюцией → R_global ~77%
3. **Ensemble + Voting**: 3 модели, majority vote → R_global ~78%
4. **Ensemble + LLM Aggregator**: 3 модели, LLM агрегатор → R_global ~80%

---

## Архитектура в 30 секунд

```
┌─────────────────────────────────────────────────────────┐
│                    OpenEvolve                            │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │ Database │ → │ Sampler  │ → │ LLM Gen  │          │
│  │(MAP-Elites)│   │          │    │ Mutations│          │
│  └──────────┘    └──────────┘    └──────────┘          │
│       ↑                               │                 │
│       │        ┌──────────────┐       │                 │
│       └────────│  EVALUATOR   │←──────┘                 │
│                │ (наш код!)   │                         │
│                └──────────────┘                         │
└─────────────────────────────────────────────────────────┘
```

**Наша задача**: написать только **Evaluator**, который оценивает промпт.

---

## Ключевые метрики

| Метрика | Формула | Что показывает |
|---------|---------|----------------|
| **R_global** | `correct / total` | Общая accuracy |
| **R_worst** | `10th_percentile(per_user_acc)` | Fairness (WILDS) |
| **MAE** | `mean(abs(pred - gold))` | Ошибка в звёздах |
| **κ** | Cohen's kappa | Согласие workers |
| **D** | `disagreement_count / N` | Diversity |

**Combined Score** для OpenEvolve:
- Single model: `R_global`
- Ensemble: `0.4*R_global + 0.4*R_worst - 0.2*(1-D)`

---

## Модели (Yandex Cloud)

| Role | Model |
|------|-------|
| Worker 1 (Baseline) | gpt-oss-120b |
| Worker 2 | YandexGPT 5 |
| Worker 3 | Gemma3-27B |
| LLM Aggregator | Qwen3-235B |
| OpenEvolve Sampler | Qwen3-235B |

---

## Структура файлов

```
experiments/
├── metrics.py          # compute_metrics()
├── workers.py          # LLMWorker class
├── aggregators.py      # MajorityVote, LLMAggregator
│
├── exp1_baseline/
│   └── run.py          # Запуск без OpenEvolve
│
├── exp2_single_evolved/
│   ├── evaluator.py    # Для OpenEvolve
│   └── config.yaml
│
├── exp3_ensemble_voting/
│   ├── evaluator.py
│   └── config.yaml
│
└── exp4_llm_aggregator/
    ├── evaluator.py
    ├── aggregator_prompt.txt
    └── config.yaml
```

---

## Минимальный Evaluator

```python
def evaluate(prompt_path: str) -> dict:
    """Главная функция для OpenEvolve."""
    
    with open(prompt_path, 'r') as f:
        prompt = f.read()
    
    # Получаем предсказания
    predictions = []
    for text in validation_texts:
        pred = worker.predict(text, prompt)
        predictions.append(pred)
    
    # Считаем метрики
    metrics = compute_metrics(predictions, gold_labels, user_ids)
    
    # ОБЯЗАТЕЛЬНО: combined_score
    metrics['combined_score'] = metrics['R_global']
    
    return metrics
```

---

## Запуск

```bash
# Exp 1: Baseline (без OpenEvolve)
python exp1_baseline/run.py

# Exp 2-4: С OpenEvolve
cd exp2_single_evolved
python -m openevolve \
    --initial-program initial_prompt.txt \
    --evaluator evaluator.py \
    --config config.yaml \
    --iterations 100
```

---

## Документы

| Файл | Содержание |
|------|------------|
| [EXPERIMENT_OVERVIEW.md](EXPERIMENT_OVERVIEW.md) | Полное описание 4 экспериментов |
| [OPENEVOLVE_INTEGRATION.md](OPENEVOLVE_INTEGRATION.md) | Как работает OpenEvolve |
| [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) | Детали реализации всех компонентов |
| [CODE_REFERENCE.md](CODE_REFERENCE.md) | Примеры кода и формулы |

---

## Checklist

- [ ] `metrics.py` - compute_metrics()
- [ ] `workers.py` - LLMWorker
- [ ] `aggregators.py` - MajorityVote, LLMAggregator
- [ ] Exp 1: baseline без OpenEvolve
- [ ] Exp 2: evaluator для single model + OpenEvolve
- [ ] Exp 3: evaluator для ensemble + voting
- [ ] Exp 4: evaluator + LLM aggregator
- [ ] Сравнительная таблица результатов


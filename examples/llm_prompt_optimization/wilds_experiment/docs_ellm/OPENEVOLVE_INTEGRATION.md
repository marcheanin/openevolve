# Интеграция с OpenEvolve

## Как работает OpenEvolve

OpenEvolve - это система эволюционной оптимизации программ/промптов. Она автоматически генерирует мутации и отбирает лучшие варианты.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         OpenEvolve                                   │
│                                                                     │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐          │
│   │  Program    │     │   Prompt    │     │    LLM      │          │
│   │  Database   │ ──→ │   Sampler   │ ──→ │   Sampler   │          │
│   │ (MAP-Elites)│     │             │     │ (генератор  │          │
│   └─────────────┘     └─────────────┘     │  мутаций)   │          │
│         ↑                                  └─────────────┘          │
│         │                                        │                  │
│         │              ┌─────────────┐           │                  │
│         └───────────── │  Evaluator  │ ←─────────┘                  │
│                        │ (наш код!)  │                              │
│                        └─────────────┘                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Ключевые компоненты

| Компонент | Что делает | Наша задача |
|-----------|------------|-------------|
| **Program Database** | Хранит популяцию программ (MAP-Elites) | Настроить в config.yaml |
| **Prompt Sampler** | Строит промпт для LLM с историей эволюции | Настроить system_message |
| **LLM Sampler** | Генерирует мутации промпта | Настроить модель в config |
| **Evaluator** | Оценивает программу, возвращает метрики | **РЕАЛИЗОВАТЬ** |

---

## Что такое Evaluator

Evaluator - это Python-функция (или модуль), которая:
1. Получает путь к файлу с программой/промптом
2. Выполняет оценку
3. Возвращает словарь с метриками

```python
def evaluate(prompt_path: str) -> Dict[str, Any]:
    """
    Главная функция evaluator.
    
    Args:
        prompt_path: Путь к файлу с текущим промптом
        
    Returns:
        Dict с метриками, ОБЯЗАТЕЛЬНО содержащий 'combined_score'
    """
    # 1. Читаем промпт
    with open(prompt_path, 'r') as f:
        prompt = f.read()
    
    # 2. Выполняем оценку (наша логика)
    metrics = run_evaluation(prompt)
    
    # 3. Возвращаем метрики
    return {
        'combined_score': metrics['accuracy'],  # ОБЯЗАТЕЛЬНО
        'accuracy': metrics['accuracy'],
        'mae': metrics['mae'],
        # ... другие метрики
    }
```

**ВАЖНО**: `combined_score` - это главная метрика, которую OpenEvolve оптимизирует (максимизирует).

---

## Структура конфигурации OpenEvolve

```yaml
# config.yaml для OpenEvolve

max_iterations: 100
checkpoint_interval: 10

# LLM для генерации мутаций
llm:
  api_base: "https://llm.api.cloud.yandex.net/v1"
  models:
    - name: "gpt://b1g.../qwen3-235b-a22b-fp8/latest"
      weight: 1.0
  temperature: 0.8
  max_tokens: 4096

# Настройки промпта для LLM Sampler
prompt:
  system_message: |
    You are an expert at creating prompts for sentiment classification.
    Your goal is to evolve prompts that maximize accuracy...

# База данных (MAP-Elites)
database:
  population_size: 50
  feature_dimensions: ["prompt_length", "reasoning_strategy"]
  feature_bins: 10

# Настройки evaluator
evaluator:
  timeout: 1800
  max_retries: 3
  cascade_evaluation: true
  cascade_thresholds: [0.5]
```

---

## Как OpenEvolve использует Evaluator

### Цикл эволюции

```python
# Псевдокод цикла OpenEvolve

for iteration in range(max_iterations):
    # 1. Prompt Sampler строит промпт для LLM
    evolution_prompt = prompt_sampler.build_prompt(
        current_program=best_program.code,
        program_metrics=best_program.metrics,
        top_programs=database.get_top_programs(),
    )
    
    # 2. LLM генерирует новую версию программы
    new_program_code = llm.generate(evolution_prompt)
    
    # 3. Evaluator оценивает новую программу
    # OpenEvolve сохраняет код во временный файл и вызывает evaluate()
    metrics = evaluator.evaluate(temp_file_path)
    
    # 4. Программа добавляется в базу если хорошая
    if metrics['combined_score'] > threshold:
        database.add(Program(code=new_program_code, metrics=metrics))
```

### Cascade Evaluation (опционально)

OpenEvolve поддерживает каскадную оценку для экономии ресурсов:

```python
# evaluator.py

def evaluate_stage1(prompt_path: str) -> Dict:
    """Быстрая оценка на малой выборке"""
    # 10 примеров
    return {'combined_score': 0.7, 'accuracy': 0.7}

def evaluate_stage2(prompt_path: str) -> Dict:
    """Полная оценка если stage1 прошёл threshold"""
    # 100 примеров
    return {'combined_score': 0.75, 'accuracy': 0.75, ...}

def evaluate(prompt_path: str) -> Dict:
    """Главная функция - вызывает stage1, потом stage2"""
    stage1 = evaluate_stage1(prompt_path)
    if stage1['combined_score'] < CASCADE_THRESHOLD:
        return stage1  # Не проходит - не тратим ресурсы
    return evaluate_stage2(prompt_path)
```

---

## Наши Evaluators для экспериментов

### Exp 1 & 2: Single Model Evaluator

```python
# exp1_baseline/evaluator.py и exp2_single_evolved/evaluator.py

def evaluate(prompt_path: str) -> Dict[str, Any]:
    with open(prompt_path, 'r') as f:
        prompt = f.read()
    
    # Одна модель
    worker = LLMWorker("qwen3-235b")
    
    # Получаем предсказания
    predictions = []
    for text, gold, user_id in validation_data:
        pred = worker.predict(text, prompt)
        predictions.append(pred)
    
    # Считаем метрики
    metrics = compute_metrics(
        predictions=predictions,
        gold_labels=gold_labels,
        user_ids=user_ids
    )
    
    return {
        'combined_score': metrics['R_global'],  # Или формула
        'R_global': metrics['R_global'],
        'R_worst': metrics['R_worst'],
        'mae': metrics['mae'],
    }
```

### Exp 3: Ensemble + Voting Evaluator

```python
# exp3_ensemble_voting/evaluator.py

def evaluate(prompt_path: str) -> Dict[str, Any]:
    with open(prompt_path, 'r') as f:
        prompt = f.read()
    
    # Три модели
    workers = [
        LLMWorker("qwen3-235b"),
        LLMWorker("gemma3-27b"),
        LLMWorker("gpt-oss-120b"),
    ]
    
    # Получаем предсказания от всех workers
    worker_predictions = []
    for worker in workers:
        preds = [worker.predict(text, prompt) for text, _, _ in validation_data]
        worker_predictions.append(preds)
    
    # Агрегируем через majority vote
    final_predictions = majority_vote_aggregate(worker_predictions)
    
    # Считаем метрики
    metrics = compute_ensemble_metrics(
        final_predictions=final_predictions,
        worker_predictions=worker_predictions,
        gold_labels=gold_labels,
        user_ids=user_ids
    )
    
    # Combined score для ансамбля
    combined_score = (
        0.4 * metrics['R_global'] +
        0.4 * metrics['R_worst'] -
        0.1 * max(0, metrics['mean_kappa'])
    )
    
    return {
        'combined_score': combined_score,
        'R_global': metrics['R_global'],
        'R_worst': metrics['R_worst'],
        'mean_kappa': metrics['mean_kappa'],
        'mae': metrics['mae'],
    }
```

### Exp 4: Ensemble + LLM Aggregator Evaluator

```python
# exp4_llm_aggregator/evaluator.py

def evaluate(prompt_path: str) -> Dict[str, Any]:
    with open(prompt_path, 'r') as f:
        worker_prompt = f.read()
    
    # Загружаем статический промпт агрегатора
    with open('aggregator_prompt.txt', 'r') as f:
        aggregator_prompt = f.read()
    
    # Workers
    workers = [
        LLMWorker("qwen3-235b"),
        LLMWorker("gemma3-27b"),
        LLMWorker("gpt-oss-120b"),
    ]
    
    # LLM агрегатор
    aggregator = LLMAggregator("qwen3-235b", aggregator_prompt)
    
    final_predictions = []
    worker_predictions = [[] for _ in workers]
    
    for text, gold, user_id in validation_data:
        # Workers выдают ПОЛНЫЕ ответы
        worker_outputs = []
        for i, worker in enumerate(workers):
            full_output, rating = worker.predict_with_reasoning(text, worker_prompt)
            worker_outputs.append(full_output)
            worker_predictions[i].append(rating)
        
        # LLM-агрегатор анализирует и выдаёт финальный ответ
        final_rating = aggregator.aggregate(worker_outputs, text)
        final_predictions.append(final_rating)
    
    # Метрики
    metrics = compute_ensemble_metrics(...)
    
    return {
        'combined_score': combined_score,
        ...
    }
```

---

## Запуск OpenEvolve

### Команда запуска

```bash
# Из директории эксперимента
cd wilds_experiment/experiments/exp2_single_evolved

# Запуск OpenEvolve
python -m openevolve \
    --initial-program initial_prompt.txt \
    --evaluator evaluator.py \
    --config config.yaml \
    --output-dir ./results
```

### Или через Python API

```python
from openevolve import run_evolution

result = run_evolution(
    initial_program="initial_prompt.txt",
    evaluator="evaluator.py",
    config="config.yaml",
    iterations=100,
    output_dir="./results"
)

print(f"Best score: {result.best_score}")
print(f"Best prompt: {result.best_code}")
```

---

## Мониторинг эволюции

### Evolution Trace

OpenEvolve логирует историю эволюции в JSONL файл:

```yaml
# config.yaml
evolution_trace:
  enabled: true
  format: "jsonl"
  include_code: true
  output_path: "results/evolution_trace.jsonl"
```

### Чтение истории

```python
import json

history = []
with open('results/evolution_trace.jsonl', 'r') as f:
    for line in f:
        entry = json.loads(line)
        history.append({
            'iteration': entry['iteration'],
            'score': entry['metrics'].get('combined_score', 0),
            'R_global': entry['metrics'].get('R_global', 0),
            'R_worst': entry['metrics'].get('R_worst', 0),
        })

# Построение кривых
import matplotlib.pyplot as plt

iterations = [h['iteration'] for h in history]
scores = [h['score'] for h in history]

plt.plot(iterations, scores)
plt.xlabel('Iteration')
plt.ylabel('Combined Score')
plt.title('Evolution Progress')
plt.savefig('evolution_curve.png')
```

---

## Что НЕ нужно реализовывать

| Компонент | Почему не нужен |
|-----------|-----------------|
| **Mutator** | OpenEvolve сам генерирует мутации через PromptSampler + LLM |
| **GA Loop** | OpenEvolve содержит полный эволюционный цикл |
| **Selection** | MAP-Elites в OpenEvolve автоматически отбирает лучших |
| **Crossover** | PromptSampler комбинирует лучшие программы |

**Наша задача только**: реализовать **Evaluator**, который оценивает промпт и возвращает метрики.

---

## Checklist интеграции

- [ ] Создать `evaluator.py` с функцией `evaluate(prompt_path) -> Dict`
- [ ] Вернуть `combined_score` как главную метрику
- [ ] Создать `config.yaml` с настройками OpenEvolve
- [ ] Создать `initial_prompt.txt` с начальным промптом
- [ ] Настроить `evolution_trace` для логирования истории
- [ ] Запустить через `python -m openevolve ...`


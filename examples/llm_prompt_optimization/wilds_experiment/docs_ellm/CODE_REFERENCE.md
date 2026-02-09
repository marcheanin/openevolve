# Code Reference: Примеры кода и формулы

## Быстрый справочник метрик

```python
# ============== МЕТРИКИ ==============

# R_global (Average Accuracy)
R_global = sum(pred == gold) / N

# R_worst (10th Percentile Fairness)
per_user_acc = {user: accuracy(preds[user], gold[user]) for user}
R_worst = percentile(per_user_acc.values(), 10)

# MAE (Mean Absolute Error)
MAE = mean(abs(pred - gold))

# Cohen's κ (Inter-Annotator Agreement)
κ = (p_observed - p_expected) / (1 - p_expected)

# Disagreement Rate
D = count(len(set(worker_votes)) > 1) / N

# ============== COMBINED SCORES ==============

# Single Model (Exp 1, 2)
combined_score = R_global * (1 - 0.2 * MAE/4)

# Ensemble (Exp 3, 4)
combined_score = 0.4 * R_global + 0.4 * R_worst - 0.2 * (1 - D)
```

---

## 1. Полная функция расчёта метрик

```python
import numpy as np
from sklearn.metrics import cohen_kappa_score
from typing import Dict, List, Optional, Any

def compute_metrics(
    predictions: np.ndarray,
    gold_labels: np.ndarray,
    user_ids: np.ndarray,
    worker_predictions: Optional[List[np.ndarray]] = None
) -> Dict[str, Any]:
    """
    Полный расчёт всех метрик.
    
    Args:
        predictions: shape (N,), финальные предсказания
        gold_labels: shape (N,), истинные метки
        user_ids: shape (N,), ID пользователей
        worker_predictions: List[shape (N,)], предсказания workers (опционально)
    
    Returns:
        Dict с метриками
    """
    N = len(predictions)
    
    # R_global
    correct = (predictions == gold_labels)
    R_global = float(np.mean(correct))
    
    # MAE
    mae = float(np.mean(np.abs(predictions - gold_labels)))
    
    # Per-user accuracy
    accuracy_per_user = {}
    for user_id in np.unique(user_ids):
        mask = (user_ids == user_id)
        user_acc = np.mean(correct[mask])
        accuracy_per_user[int(user_id)] = float(user_acc)
    
    # R_worst (10th percentile)
    all_accs = list(accuracy_per_user.values())
    R_worst = float(np.percentile(all_accs, 10)) if all_accs else 0.0
    
    result = {
        'R_global': R_global,
        'R_worst': R_worst,
        'mae': mae,
        'accuracy_per_user': accuracy_per_user,
        'num_users': len(np.unique(user_ids)),
        'num_examples': N,
    }
    
    # Ensemble метрики
    if worker_predictions and len(worker_predictions) > 1:
        # κ (Cohen's kappa)
        kappas = []
        M = len(worker_predictions)
        for i in range(M):
            for j in range(i + 1, M):
                k = cohen_kappa_score(worker_predictions[i], worker_predictions[j])
                kappas.append(k)
        result['mean_kappa'] = float(np.mean(kappas))
        
        # Disagreement rate
        disagree = 0
        for idx in range(N):
            votes = set(int(wp[idx]) for wp in worker_predictions)
            if len(votes) > 1:
                disagree += 1
        # mean_kappa используется для combined_score и MAP-Elites; disagreement_rate не используется в Exp 3
    
    return result
```

---

## 2. Majority Vote агрегация

```python
from collections import Counter
import numpy as np

def majority_vote(predictions: list) -> int:
    """Агрегация для одного примера."""
    return Counter(predictions).most_common(1)[0][0]

def aggregate_all(worker_preds: list) -> np.ndarray:
    """
    Агрегация для всех примеров.
    
    Args:
        worker_preds: List[np.ndarray], M массивов по N элементов
    
    Returns:
        np.ndarray shape (N,)
    """
    N = len(worker_preds[0])
    result = []
    for i in range(N):
        votes = [int(wp[i]) for wp in worker_preds]
        result.append(majority_vote(votes))
    return np.array(result)

# Пример использования:
# worker1 = np.array([4, 5, 3, 4, 2])
# worker2 = np.array([4, 5, 3, 5, 2])
# worker3 = np.array([4, 4, 3, 4, 3])
# 
# final = aggregate_all([worker1, worker2, worker3])
# # [4, 5, 3, 4, 2]  <- majority для каждого примера
```

---

## 3. Parsing ответа LLM

```python
import re

def parse_rating(response: str) -> int:
    """
    Извлекает rating 1-5 из ответа модели.
    
    Поддерживаемые форматы:
    - "4"
    - "Rating: 4"
    - "4 out of 5"
    - "4/5"
    - "**4**"
    - "My answer is 4"
    """
    patterns = [
        r'^([1-5])$',                           # Просто число
        r'(?:rating|score|answer)[:\s]+([1-5])', # "Rating: 4"
        r'\b([1-5])\s*(?:out of 5|/5|stars?)',   # "4/5", "4 stars"
        r'^\s*\**\s*([1-5])\s*\**\s*$',          # "**4**"
        r'final.*?([1-5])',                      # "final answer is 4"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            return int(match.group(1))
    
    # Fallback: последнее число 1-5 в тексте
    numbers = re.findall(r'\b([1-5])\b', response)
    if numbers:
        return int(numbers[-1])
    
    return 3  # Default

# Примеры:
# parse_rating("Rating: 4") -> 4
# parse_rating("I think this deserves 5 out of 5 stars") -> 5
# parse_rating("4") -> 4
# parse_rating("The answer is **3**") -> 3
```

---

## 4. Пример evaluator для OpenEvolve

```python
# evaluator.py

import numpy as np

def evaluate(prompt_path: str) -> dict:
    """
    Evaluator для OpenEvolve.
    
    ВАЖНО: OpenEvolve ожидает 'combined_score' в возвращаемом dict.
    """
    # 1. Читаем промпт
    with open(prompt_path, 'r') as f:
        prompt = f.read()
    
    # 2. Загружаем данные (кэшируем для скорости)
    data = load_cached_validation_data()
    
    # 3. Получаем предсказания
    predictions = []
    for item in data:
        pred = call_llm(item['text'], prompt)
        predictions.append(pred)
    
    # 4. Считаем метрики
    metrics = compute_metrics(
        predictions=np.array(predictions),
        gold_labels=np.array([d['label'] for d in data]),
        user_ids=np.array([d['user_id'] for d in data])
    )
    
    # 5. Добавляем combined_score (ОБЯЗАТЕЛЬНО!)
    metrics['combined_score'] = metrics['R_global']  # или другая формула
    
    return metrics
```

---

## 5. Cascade Evaluation для экономии ресурсов

```python
# evaluator.py с cascade

CASCADE_THRESHOLD = 0.5
STAGE1_SIZE = 20
STAGE2_SIZE = 200

_data = None

def get_data():
    global _data
    if _data is None:
        _data = load_cached_validation_data()
    return _data

def evaluate_stage1(prompt_path: str) -> dict:
    """Быстрая оценка на малой выборке."""
    with open(prompt_path, 'r') as f:
        prompt = f.read()
    
    data = get_data()[:STAGE1_SIZE]
    
    predictions = [call_llm(d['text'], prompt) for d in data]
    gold = [d['label'] for d in data]
    
    accuracy = sum(p == g for p, g in zip(predictions, gold)) / len(data)
    
    return {'combined_score': accuracy, 'stage': 1, 'samples': STAGE1_SIZE}

def evaluate_stage2(prompt_path: str) -> dict:
    """Полная оценка."""
    with open(prompt_path, 'r') as f:
        prompt = f.read()
    
    data = get_data()[:STAGE2_SIZE]
    
    predictions = np.array([call_llm(d['text'], prompt) for d in data])
    gold = np.array([d['label'] for d in data])
    user_ids = np.array([d['user_id'] for d in data])
    
    metrics = compute_metrics(predictions, gold, user_ids)
    metrics['combined_score'] = metrics['R_global']
    metrics['stage'] = 2
    metrics['samples'] = STAGE2_SIZE
    
    return metrics

def evaluate(prompt_path: str) -> dict:
    """Главная функция с cascade."""
    stage1 = evaluate_stage1(prompt_path)
    
    if stage1['combined_score'] < CASCADE_THRESHOLD:
        return stage1  # Не проходит - не тратим ресурсы
    
    return evaluate_stage2(prompt_path)
```

---

## 6. LLM-агрегатор: промпт и логика

### Промпт агрегатора

```text
You are an expert judge that analyzes multiple annotator opinions
to determine the final rating for a product review.

## Review:
{review}

## Annotator Analyses:
{worker_outputs}

## Your Task:
1. Read each annotator's analysis and their reasoning
2. Consider:
   - Do they agree or disagree?
   - Whose reasoning is most sound?
   - What aspects of the review did they focus on?
3. Make your final decision

## Important:
- Focus on the CONTENT of the review, not reviewer personality
- A critical but fair review can still be rated highly
- Consider both positive and negative points mentioned

## Output:
Provide ONLY the final rating as a single digit (1-5).

Rating:
```

### Код агрегатора

```python
class LLMAggregator:
    def __init__(self, model_name: str, prompt_template: str):
        self.model_name = model_name
        self.prompt_template = prompt_template
    
    def aggregate(self, worker_outputs: list, review_text: str) -> int:
        """
        Args:
            worker_outputs: List[str] - полные ответы workers
            review_text: str - исходный отзыв
        
        Returns:
            int: финальный rating 1-5
        """
        # Форматируем выходы workers
        outputs_str = ""
        for i, output in enumerate(worker_outputs, 1):
            outputs_str += f"\n### Annotator {i}:\n{output}\n"
        
        # Строим промпт
        prompt = self.prompt_template.format(
            review=review_text,
            worker_outputs=outputs_str
        )
        
        # Вызываем LLM
        response = call_llm_raw(prompt, self.model_name)
        
        # Парсим ответ
        return parse_rating(response)
```

---

## 7. Config.yaml для OpenEvolve

```yaml
# config.yaml

max_iterations: 100
checkpoint_interval: 10

# LLM для генерации мутаций
llm:
  api_base: "https://llm.api.cloud.yandex.net/v1"
  models:
    - name: "gpt://b1gemincl8p7b2uiv5nl/qwen3-235b-a22b-fp8/latest"
      weight: 1.0
  temperature: 0.8
  max_tokens: 4096

# Prompt для эволюции
prompt:
  system_message: |
    You are an expert at creating prompts for product review rating prediction.
    Your task is to evolve prompts that maximize classification accuracy
    while ensuring fairness across different user types.
    
    Consider:
    - Clear rating scale definitions
    - Examples of edge cases
    - User-agnostic language
    - Handling of mixed sentiment

# База данных (MAP-Elites)
database:
  population_size: 50
  feature_dimensions:
    - name: "prompt_length"
      min_val: 100
      max_val: 2000
    - name: "has_examples"
      type: "binary"
  feature_bins: 10

# Evaluator
evaluator:
  timeout: 1800
  max_retries: 3
  cascade_evaluation: true
  cascade_thresholds: [0.5]

# Evolution trace для логирования
evolution_trace:
  enabled: true
  format: "jsonl"
  include_code: true
  output_path: "results/evolution_trace.jsonl"
```

---

## 8. Визуализация результатов

```python
import json
import matplotlib.pyplot as plt

def plot_evolution_history(trace_path: str, output_path: str):
    """
    Строит кривые эволюции метрик.
    """
    # Читаем trace
    history = []
    with open(trace_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            history.append({
                'iteration': entry.get('iteration', len(history)),
                'score': entry['metrics'].get('combined_score', 0),
                'R_global': entry['metrics'].get('R_global', 0),
                'R_worst': entry['metrics'].get('R_worst', 0),
            })
    
    # Извлекаем данные
    iterations = [h['iteration'] for h in history]
    scores = [h['score'] for h in history]
    R_global = [h['R_global'] for h in history]
    R_worst = [h['R_worst'] for h in history]
    
    # Строим графики
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(iterations, scores, 'b-')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Combined Score')
    axes[0].set_title('Evolution Progress')
    axes[0].grid(True)
    
    axes[1].plot(iterations, R_global, 'g-', label='R_global')
    axes[1].plot(iterations, R_worst, 'r-', label='R_worst')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('R_global vs R_worst')
    axes[1].legend()
    axes[1].grid(True)
    
    # Best score over time
    best_so_far = []
    best = 0
    for s in scores:
        best = max(best, s)
        best_so_far.append(best)
    
    axes[2].plot(iterations, best_so_far, 'purple')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Best Score So Far')
    axes[2].set_title('Best Score Evolution')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved plot to {output_path}")

# Использование:
# plot_evolution_history('results/evolution_trace.jsonl', 'results/evolution_plot.png')
```

---

## 9. Сравнение экспериментов

```python
def compare_experiments(results_paths: dict) -> dict:
    """
    Сравнивает результаты разных экспериментов.
    
    Args:
        results_paths: Dict[exp_name → path_to_results.json]
    
    Returns:
        Dict с таблицей сравнения
    """
    comparison = {}
    
    for exp_name, path in results_paths.items():
        with open(path, 'r') as f:
            metrics = json.load(f)
        
        comparison[exp_name] = {
            'R_global': metrics.get('R_global', 0),
            'R_worst': metrics.get('R_worst', 0),
            'MAE': metrics.get('mae', 0),
            'κ': metrics.get('mean_kappa', 'N/A'),
        }
    
    # Форматируем таблицу
    print("\n" + "="*70)
    print("COMPARISON TABLE")
    print("="*70)
    print(f"{'Experiment':<25} {'R_global':>10} {'R_worst':>10} {'MAE':>8} {'κ':>8}")
    print("-"*70)
    
    for exp, m in comparison.items():
        kappa = f"{m['κ']:.3f}" if isinstance(m['κ'], float) else m['κ']
        print(f"{exp:<25} {m['R_global']:>10.1%} {m['R_worst']:>10.1%} {m['MAE']:>8.3f} {kappa:>8}")
    
    print("="*70)
    
    return comparison

# Использование:
# compare_experiments({
#     'Exp1: Baseline': 'exp1_baseline/results/metrics.json',
#     'Exp2: Single+Evo': 'exp2_single_evolved/results/metrics.json',
#     'Exp3a: Ensemble Base': 'exp3_ensemble_voting/results/baseline.json',
#     'Exp3b: Ensemble+Evo': 'exp3_ensemble_voting/results/evolved.json',
#     'Exp4: LLM Aggregator': 'exp4_llm_aggregator/results/metrics.json',
# })
```

---

## 10. Troubleshooting

### Проблема: `combined_score` всегда 0

```python
# НЕПРАВИЛЬНО:
def evaluate(prompt_path):
    metrics = compute_metrics(...)
    return metrics  # Нет combined_score!

# ПРАВИЛЬНО:
def evaluate(prompt_path):
    metrics = compute_metrics(...)
    metrics['combined_score'] = metrics['R_global']  # ДОБАВЛЯЕМ!
    return metrics
```

### Проблема: LLM возвращает невалидный rating

```python
# Добавить более robust parsing
def parse_rating(response: str) -> int:
    # ... паттерны ...
    
    # Если ничего не найдено - логируем
    import logging
    logging.warning(f"Could not parse rating from: {response[:100]}...")
    
    return 3  # Безопасный default
```

### Проблема: Timeout на evaluation

```yaml
# config.yaml
evaluator:
  timeout: 3600  # Увеличить timeout
  max_retries: 5  # Больше retry
```

### Проблема: Память переполняется

```python
# Не загружать всё в память сразу
def evaluate(prompt_path):
    # Используем generator
    for batch in batch_generator(data, batch_size=50):
        process_batch(batch)
```

---

## Quick Commands

```bash
# Установка зависимостей
pip install numpy scikit-learn openai matplotlib

# Запуск baseline
python exp1_baseline/run.py

# Запуск OpenEvolve
python -m openevolve \
    --initial-program initial_prompt.txt \
    --evaluator evaluator.py \
    --config config.yaml \
    --iterations 100

# Построение графиков
python -c "from utils import plot_evolution_history; plot_evolution_history('results/trace.jsonl', 'plot.png')"

# Сравнение экспериментов
python compare_all.py
```


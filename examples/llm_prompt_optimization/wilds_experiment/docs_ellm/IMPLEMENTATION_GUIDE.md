# Implementation Guide: eLLM Ensemble Experiments

## Обзор компонентов

```
experiments/
├── metrics.py          # Расчёт всех метрик
├── workers.py          # LLMWorker для Yandex Cloud
├── aggregators.py      # MajorityVote + LLMAggregator
├── feature_dimensions.py # Feature dimensions для MAP-Elites
├── data_loader.py      # Загрузка WILDS данных
│
├── exp1_baseline/      # Эксперимент 1
├── exp2_single_evolved/# Эксперимент 2
├── exp3_ensemble_voting/# Эксперимент 3
└── exp4_llm_aggregator/ # Эксперимент 4
```

---

## 1. Модуль metrics.py

### Функция compute_metrics

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
    Вычисляет все метрики для оценки качества.
    
    Args:
        predictions: Финальные предсказания, shape (N,)
        gold_labels: Истинные метки, shape (N,)
        user_ids: ID пользователей для каждого примера, shape (N,)
        worker_predictions: Опционально - предсказания каждого worker'а
                           List из M массивов, каждый shape (N,)
                           Нужно для расчёта κ (mean_kappa)
    
    Returns:
        Dict с метриками:
        - R_global: общая accuracy
        - R_worst: 10th percentile per-user accuracy (fairness)
        - mae: mean absolute error
        - accuracy_per_user: Dict[user_id → accuracy]
        
        Если worker_predictions предоставлены:
        - mean_kappa: средний Cohen's κ между парами workers (для скоринга и MAP-Elites обрезается до [0, 1])
    """
    N = len(predictions)
    
    # ========== R_global (Average Accuracy) ==========
    correct = (predictions == gold_labels).astype(int)
    R_global = float(np.mean(correct))
    
    # ========== MAE (Mean Absolute Error) ==========
    mae = float(np.mean(np.abs(predictions - gold_labels)))
    
    # ========== R_worst (10th Percentile Per-User) ==========
    accuracy_per_user = {}
    unique_users = np.unique(user_ids)
    
    for user_id in unique_users:
        user_mask = (user_ids == user_id)
        user_correct = correct[user_mask]
        if len(user_correct) > 0:
            accuracy_per_user[int(user_id)] = float(np.mean(user_correct))
    
    all_accuracies = list(accuracy_per_user.values())
    R_worst = float(np.percentile(all_accuracies, 10)) if all_accuracies else 0.0
    
    result = {
        'R_global': R_global,
        'R_worst': R_worst,
        'mae': mae,
        'accuracy_per_user': accuracy_per_user,
        'num_users': len(unique_users),
        'num_examples': N,
    }
    
    # ========== Ensemble метрики (если есть worker_predictions) ==========
    if worker_predictions is not None and len(worker_predictions) > 1:
        # Cohen's Kappa между парами workers
        kappas = []
        M = len(worker_predictions)
        for i in range(M):
            for j in range(i + 1, M):
                try:
                    kappa_ij = cohen_kappa_score(
                        worker_predictions[i],
                        worker_predictions[j]
                    )
                    kappas.append(kappa_ij)
                except Exception:
                    pass
        
        result['mean_kappa'] = float(np.mean(kappas)) if kappas else 0.0
    
    return result
```

### Функция compute_combined_score

```python
def compute_combined_score_unified(
    metrics: Dict[str, Any],
    is_ensemble: bool = False,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Унифицированная формула combined score для single model и ensemble.
    Позволяет корректно сравнивать результаты между экспериментами.
    
    Базовая формула (одинаковая для всех):
        score = 0.4 * R_global + 0.3 * R_worst + 0.3 * (1 - MAE/4)
    
    Для ensemble добавляется бонус за согласованность (Cohen's Kappa, обрезанная до [0, 1]):
        score += 0.1 * max(0, mean_kappa)
    
    Args:
        metrics: Словарь с метриками (R_global, R_worst, mae, mean_kappa для ensemble)
        is_ensemble: True если это ensemble эксперимент
        weights: Опциональные веса компонентов
    
    Returns:
        Combined score в диапазоне [0.0, 1.0]
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
    
    # Нормализуем MAE к диапазону 0-1 (инвертируем: меньше ошибок = лучше)
    mae_normalized = 1.0 - (mae / 4.0)
    
    # Базовая формула
    base_score = (
        weights["global"] * r_global +
        weights["worst"] * r_worst +
        weights["mae"] * mae_normalized
    )
    
    # Бонус за согласованность ансамбля (только для ensemble): Cohen's Kappa
    if is_ensemble and weights["consistency"] > 0:
        kappa_score = max(0.0, metrics.get("mean_kappa", 0.0))
        consistency_bonus = weights["consistency"] * kappa_score
        score = base_score + consistency_bonus
    else:
        score = base_score
    
    return float(max(0.0, min(1.0, score)))


# DEPRECATED: Используйте compute_combined_score_unified()
def compute_combined_score_single(metrics: Dict) -> float:
    """DEPRECATED: Use compute_combined_score_unified(metrics, is_ensemble=False)"""
    return compute_combined_score_unified(metrics, is_ensemble=False)


def compute_combined_score_ensemble(metrics: Dict) -> float:
    """DEPRECATED: Use compute_combined_score_unified(metrics, is_ensemble=True)"""
    return compute_combined_score_unified(metrics, is_ensemble=True)
```

---

## 2. Модуль workers.py

### Класс LLMWorker

```python
import os
import re
from typing import Tuple, Optional
from openai import OpenAI

class LLMWorker:
    """
    Worker для вызова LLM через Yandex Cloud API.
    """
    
    # Маппинг коротких имён на полные URI
    MODEL_MAPPING = {
        'qwen3-235b': 'gpt://b1gemincl8p7b2uiv5nl/qwen3-235b-a22b-fp8/latest',
        'gemma3-27b': 'gpt://b1gemincl8p7b2uiv5nl/gemma-3-27b-it/latest',
        'gpt-oss-120b': 'gpt://b1gemincl8p7b2uiv5nl/gpt-oss-120b/latest',
        'yandexgpt': 'gpt://b1gemincl8p7b2uiv5nl/yandexgpt/latest',
    }
    
    def __init__(self, model_name: str, api_base: str = None):
        """
        Args:
            model_name: Короткое имя ('qwen3-235b') или полный URI
            api_base: URL API (по умолчанию Yandex Cloud)
        """
        self.model_name = model_name
        
        # Определяем URI модели
        if model_name in self.MODEL_MAPPING:
            self.model_uri = self.MODEL_MAPPING[model_name]
        else:
            self.model_uri = model_name
        
        # Инициализируем клиент
        self.api_base = api_base or "https://llm.api.cloud.yandex.net/v1"
        self.client = OpenAI(base_url=self.api_base)
    
    def predict(self, text: str, instruction: str) -> int:
        """
        Получить предсказание (только rating 1-5).
        
        Args:
            text: Текст отзыва
            instruction: Промпт/инструкция
            
        Returns:
            rating: 1-5
        """
        prompt = instruction.replace('{review}', text)
        
        response = self.client.chat.completions.create(
            model=self.model_uri,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.0,
        )
        
        response_text = response.choices[0].message.content.strip()
        return self._parse_rating(response_text)
    
    def predict_with_reasoning(
        self, 
        text: str, 
        instruction: str
    ) -> Tuple[str, int]:
        """
        Получить полный ответ с reasoning и rating.
        Используется для LLM-агрегатора (Exp 4).
        
        Args:
            text: Текст отзыва
            instruction: Промпт/инструкция
            
        Returns:
            Tuple[full_output, rating]:
                - full_output: полный текст ответа (с reasoning)
                - rating: извлечённый rating 1-5
        """
        prompt = instruction.replace('{review}', text)
        
        response = self.client.chat.completions.create(
            model=self.model_uri,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,  # Больше токенов для reasoning
            temperature=0.0,
        )
        
        full_output = response.choices[0].message.content.strip()
        rating = self._parse_rating(full_output)
        
        return full_output, rating
    
    def _parse_rating(self, response: str) -> int:
        """Извлекает rating 1-5 из ответа модели."""
        # Паттерны для поиска
        patterns = [
            r'^([1-5])$',
            r'(?:rating|score|answer)[:\s]+([1-5])',
            r'\b([1-5])\s*(?:out of 5|/5|stars?)',
            r'^\s*\**\s*([1-5])\s*\**\s*$',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                return int(match.group(1))
        
        # Fallback: ищем любое число 1-5
        numbers = re.findall(r'\b([1-5])\b', response)
        if numbers:
            return int(numbers[-1])  # Берём последнее
        
        return 3  # Default если ничего не найдено
```

---

## 3. Модуль aggregators.py

### MajorityVoteAggregator

```python
from collections import Counter
from typing import List
import numpy as np

class MajorityVoteAggregator:
    """
    Агрегация через majority voting.
    Используется в Exp 3.
    """
    
    def aggregate_single(self, predictions: List[int]) -> int:
        """
        Агрегировать предсказания для одного примера.
        
        Args:
            predictions: Список предсказаний от workers [4, 4, 5]
            
        Returns:
            Финальное предсказание (наиболее частое)
        """
        vote_counts = Counter(predictions)
        return vote_counts.most_common(1)[0][0]
    
    def aggregate_all(
        self, 
        worker_predictions: List[np.ndarray]
    ) -> np.ndarray:
        """
        Агрегировать предсказания для всех примеров.
        
        Args:
            worker_predictions: List из M массивов, каждый shape (N,)
            
        Returns:
            Финальные предсказания, shape (N,)
        """
        N = len(worker_predictions[0])
        final_predictions = []
        
        for i in range(N):
            votes = [int(wp[i]) for wp in worker_predictions]
            final_pred = self.aggregate_single(votes)
            final_predictions.append(final_pred)
        
        return np.array(final_predictions)
```

### LLMAggregator

```python
class LLMAggregator:
    """
    Агрегация через LLM, который анализирует выходы workers.
    Используется в Exp 4.
    """
    
    def __init__(
        self, 
        model_name: str, 
        prompt_template: str,
        api_base: str = None
    ):
        """
        Args:
            model_name: Модель для агрегации ('qwen3-235b')
            prompt_template: Шаблон промпта агрегатора
            api_base: URL API
        """
        self.worker = LLMWorker(model_name, api_base)
        self.prompt_template = prompt_template
    
    def aggregate(
        self, 
        worker_outputs: List[str], 
        review_text: str
    ) -> int:
        """
        Агрегировать выходы workers через LLM.
        
        Args:
            worker_outputs: Полные выходы от каждого worker (с reasoning)
            review_text: Исходный текст отзыва
            
        Returns:
            Финальный rating 1-5
        """
        # Форматируем промпт
        workers_section = ""
        for i, output in enumerate(worker_outputs, 1):
            workers_section += f"\n=== Worker {i} Analysis ===\n{output}\n"
        
        prompt = self.prompt_template.format(
            review=review_text,
            worker_outputs=workers_section
        )
        
        # Получаем ответ агрегатора
        rating = self.worker.predict(review_text, prompt)
        
        return rating
```

### Шаблон промпта для LLM-агрегатора

```
# aggregator_prompt.txt

You are an expert aggregator that analyzes multiple annotator opinions 
and determines the final rating.

## Review to Rate:
{review}

## Annotator Analyses:
{worker_outputs}

## Your Task:
1. Read each annotator's analysis carefully
2. Consider their reasoning and conclusions
3. Identify areas of agreement and disagreement
4. Weigh the quality of each annotator's reasoning
5. Determine the most appropriate final rating

## Output Format:
Provide ONLY the final rating as a single number (1-5).

Your rating:
```

---

## 4. Структура Evaluator для каждого эксперимента

### Exp 1: Baseline Evaluator (без OpenEvolve)

```python
# exp1_baseline/evaluator.py

"""
Эксперимент 1: Baseline без эволюции.
Запускается напрямую, не через OpenEvolve.
"""

import sys
sys.path.append('..')

from metrics import compute_metrics, compute_combined_score_unified
from workers import LLMWorker
from data_loader import load_validation_data

def run_baseline(prompt_path: str, output_path: str):
    """Запуск baseline эксперимента."""
    
    # Загружаем промпт
    with open(prompt_path, 'r') as f:
        prompt = f.read()
    
    # Загружаем данные
    validation_data = load_validation_data()
    texts = [d['text'] for d in validation_data]
    gold_labels = [d['label'] for d in validation_data]
    user_ids = [d['user_id'] for d in validation_data]
    
    # Одна модель
    worker = LLMWorker('qwen3-235b')
    
    # Получаем предсказания
    print("Running predictions...")
    predictions = []
    for i, text in enumerate(texts):
        pred = worker.predict(text, prompt)
        predictions.append(pred)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(texts)}")
    
    # Считаем метрики
    metrics = compute_metrics(
        predictions=np.array(predictions),
        gold_labels=np.array(gold_labels),
        user_ids=np.array(user_ids)
    )
    
    metrics['combined_score'] = compute_combined_score_unified(metrics, is_ensemble=False)
    
    # Сохраняем результаты
    import json
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    print(f"R_global: {metrics['R_global']:.2%}")
    print(f"R_worst: {metrics['R_worst']:.2%}")
    print(f"MAE: {metrics['mae']:.3f}")
    
    return metrics


if __name__ == '__main__':
    run_baseline(
        prompt_path='initial_prompt.txt',
        output_path='results/baseline_metrics.json'
    )
```

### Exp 2: Single Model + OpenEvolve Evaluator

```python
# exp2_single_evolved/evaluator.py

"""
Эксперимент 2: Single Model с эволюцией через OpenEvolve.
Этот evaluator вызывается OpenEvolve.
"""

import sys
import os
from pathlib import Path
import yaml
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

# Импорты из общих модулей
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from metrics import compute_metrics, compute_combined_score_unified
from workers import LLMWorker

# Используем кэширование из базового evaluator
from evaluator import (
    load_preprocessed_data,
    save_preprocessed_data,
    create_splits_from_preprocessed,
    preprocess_category_data,
    load_wilds_dataset,
)

# Глобальные кэши для multiprocessing
_DATASET_SPLITS_CACHE = None
_PREPROCESSED_CACHE = None


def load_experiment_config() -> Dict[str, Any]:
    """Загружает config.yaml эксперимента."""
    config_path = Path(__file__).resolve().parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_cached_splits(dataset_cfg: Dict[str, Any]) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Получает кэшированные splits и preprocessed данные.
    Использует дисковый кэш для работы между процессами (multiprocessing).
    """
    global _DATASET_SPLITS_CACHE, _PREPROCESSED_CACHE
    
    # Проверяем кэш в памяти (для оптимизации внутри одного процесса)
    if _DATASET_SPLITS_CACHE is not None and _PREPROCESSED_CACHE is not None:
        return _DATASET_SPLITS_CACHE, _PREPROCESSED_CACHE
    
    # Загружаем preprocessed данные из дискового кэша (работает между процессами)
    _PREPROCESSED_CACHE = load_preprocessed_data(dataset_cfg)
    
    if _PREPROCESSED_CACHE is not None:
        # Создаём splits из предобработанных данных
        _DATASET_SPLITS_CACHE = create_splits_from_preprocessed(
            _PREPROCESSED_CACHE,
            train_ratio=dataset_cfg.get("train_ratio", 0.7),
            val_ratio=dataset_cfg.get("validation_ratio", 0.15),
            test_ratio=dataset_cfg.get("test_ratio", 0.15),
            seed=dataset_cfg.get("split_seed", 42),
        )
        return _DATASET_SPLITS_CACHE, _PREPROCESSED_CACHE
    
    # Кэш не найден - создаём его (только при первом запуске)
    dataset, _ = load_wilds_dataset(dataset_cfg)
    category_id = dataset_cfg.get("category_id", 24)
    _PREPROCESSED_CACHE = preprocess_category_data(dataset, category_id)
    save_preprocessed_data(dataset_cfg, _PREPROCESSED_CACHE)
    
    _DATASET_SPLITS_CACHE = create_splits_from_preprocessed(
        _PREPROCESSED_CACHE,
        train_ratio=dataset_cfg.get("train_ratio", 0.7),
        val_ratio=dataset_cfg.get("validation_ratio", 0.15),
        test_ratio=dataset_cfg.get("test_ratio", 0.15),
        seed=dataset_cfg.get("split_seed", 42),
    )
    
    return _DATASET_SPLITS_CACHE, _PREPROCESSED_CACHE


def evaluate(prompt_path: str | None = None) -> Dict[str, Any]:
    """
    Главная функция evaluator для OpenEvolve.
    
    Args:
        prompt_path: Путь к файлу с промптом (или None для использования из config)
        
    Returns:
        Dict с метриками:
        - combined_score: на верхнем уровне (требуется OpenEvolve)
        - prompt_length, reasoning_strategy: feature dimensions для MAP-Elites
        - metrics: вложенный словарь с детальными метриками
    """
    config = load_experiment_config()
    
    prompt_file = prompt_path or config.get("prompt_path", "initial_prompt.txt")
    prompt_path_abs = Path(__file__).resolve().parent / prompt_file
    with open(prompt_path_abs, "r", encoding="utf-8") as f:
        prompt_template = f.read()
    
    # Загружаем данные с кэшированием
    dataset_cfg = _load_dataset_config(config["dataset"]["config_path"])
    splits, preprocessed = _get_cached_splits(dataset_cfg)
    
    # Оценка на train и validation
    train_metrics, train_preds, train_gold, train_users = _evaluate_split(
        "train", prompt_template, config, None
    )
    val_metrics, val_preds, val_gold, val_users = _evaluate_split(
        "validation", prompt_template, config, None
    )
    
    # Combined score с использованием унифицированной формулы
    val_metrics["combined_score"] = compute_combined_score_unified(
        val_metrics, is_ensemble=False
    )
    train_metrics["combined_score"] = compute_combined_score_unified(
        train_metrics, is_ensemble=False
    )
    
    # Generalization gap penalty
    train_r_global = train_metrics["R_global"]
    val_r_global = val_metrics["R_global"]
    generalization_gap = max(0.0, float(train_r_global - val_r_global))
    gap_cfg = config.get("generalization", {})
    gap_threshold = float(gap_cfg.get("gap_threshold", 0.1))
    gap_weight = float(gap_cfg.get("gap_penalty", 0.5))
    excess_gap = max(0.0, generalization_gap - gap_threshold)
    gap_penalty = min(1.0, gap_weight * excess_gap)
    combined_score = float(val_metrics["combined_score"] * (1.0 - gap_penalty))
    
    # Feature dimensions для MAP-Elites (улучшенные метрики)
    # Используем новый модуль feature_dimensions.py
    from experiments.feature_dimensions import calculate_all_features
    features = calculate_all_features(prompt_template, metrics=None, is_ensemble=False)
    
    criteria_explicitness = features['criteria_explicitness']
    domain_focus = features['domain_focus']
    
    # OpenEvolve ожидает combined_score на верхнем уровне
    return {
        "combined_score": combined_score,  # Требуется OpenEvolve
        # Feature dimensions для MAP-Elites
        "criteria_explicitness": criteria_explicitness,
        "domain_focus": domain_focus,
        # Legacy features
        "prompt_length": features['prompt_length'],
        "metrics": {
            "R_global": val_metrics["R_global"],
            "R_worst": val_metrics["R_worst"],
            "mae": val_metrics["mae"],
            "train_R_global": train_metrics["R_global"],
            "generalization_gap": generalization_gap,
            "generalization_gap_penalty": gap_penalty,
        },
    }
```

### Exp 3: Ensemble + Voting Evaluator

```python
# exp3_ensemble_voting/evaluator.py

"""
Эксперимент 3: Ensemble с majority voting.
Этот evaluator вызывается OpenEvolve.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from metrics import compute_metrics, compute_combined_score_unified
from workers import LLMWorker
from aggregators import MajorityVoteAggregator
from data_loader import load_validation_data

_VALIDATION_DATA = None
_WORKERS = None
_AGGREGATOR = None

def get_validation_data():
    global _VALIDATION_DATA
    if _VALIDATION_DATA is None:
        _VALIDATION_DATA = load_validation_data()
    return _VALIDATION_DATA

def get_workers():
    global _WORKERS
    if _WORKERS is None:
        _WORKERS = [
            LLMWorker('qwen3-235b'),
            LLMWorker('gemma3-27b'),
            LLMWorker('gpt-oss-120b'),
        ]
    return _WORKERS

def get_aggregator():
    global _AGGREGATOR
    if _AGGREGATOR is None:
        _AGGREGATOR = MajorityVoteAggregator()
    return _AGGREGATOR


def evaluate(prompt_path: str) -> dict:
    """
    Evaluator для ансамбля с majority voting.
    """
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt = f.read()
    
    data = get_validation_data()
    texts = [d['text'] for d in data]
    gold_labels = np.array([d['label'] for d in data])
    user_ids = np.array([d['user_id'] for d in data])
    
    workers = get_workers()
    aggregator = get_aggregator()
    
    # Предсказания от всех workers
    worker_predictions = []
    for worker in workers:
        preds = [worker.predict(text, prompt) for text in texts]
        worker_predictions.append(np.array(preds))
    
    # Агрегация
    final_predictions = aggregator.aggregate_all(worker_predictions)
    
    # Метрики
    metrics = compute_metrics(
        predictions=final_predictions,
        gold_labels=gold_labels,
        user_ids=user_ids,
        worker_predictions=worker_predictions
    )
    
    metrics['combined_score'] = compute_combined_score_unified(metrics, is_ensemble=True)
    
    return metrics
```

### Exp 4: Ensemble + LLM Aggregator Evaluator

```python
# exp4_llm_aggregator/evaluator.py

"""
Эксперимент 4: Ensemble с LLM-агрегатором.
LLM-агрегатор анализирует полные выходы workers.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from metrics import compute_metrics, compute_combined_score_unified
from workers import LLMWorker
from aggregators import LLMAggregator
from data_loader import load_validation_data

_VALIDATION_DATA = None
_WORKERS = None
_AGGREGATOR = None

def get_validation_data():
    global _VALIDATION_DATA
    if _VALIDATION_DATA is None:
        _VALIDATION_DATA = load_validation_data()
    return _VALIDATION_DATA

def get_workers():
    global _WORKERS
    if _WORKERS is None:
        _WORKERS = [
            LLMWorker('qwen3-235b'),
            LLMWorker('gemma3-27b'),
            LLMWorker('gpt-oss-120b'),
        ]
    return _WORKERS

def get_aggregator():
    global _AGGREGATOR
    if _AGGREGATOR is None:
        # Загружаем статический промпт агрегатора
        aggregator_prompt_path = os.path.join(
            os.path.dirname(__file__), 
            'aggregator_prompt.txt'
        )
        with open(aggregator_prompt_path, 'r') as f:
            aggregator_prompt = f.read()
        
        _AGGREGATOR = LLMAggregator('qwen3-235b', aggregator_prompt)
    return _AGGREGATOR


def evaluate(prompt_path: str) -> dict:
    """
    Evaluator для ансамбля с LLM-агрегатором.
    
    Workers используют промпт из prompt_path.
    Агрегатор использует статический промпт.
    """
    with open(prompt_path, 'r', encoding='utf-8') as f:
        worker_prompt = f.read()
    
    data = get_validation_data()
    texts = [d['text'] for d in data]
    gold_labels = np.array([d['label'] for d in data])
    user_ids = np.array([d['user_id'] for d in data])
    
    workers = get_workers()
    aggregator = get_aggregator()
    
    final_predictions = []
    worker_predictions = [[] for _ in workers]
    
    for text in texts:
        # Workers выдают полные ответы
        worker_outputs = []
        for i, worker in enumerate(workers):
            full_output, rating = worker.predict_with_reasoning(text, worker_prompt)
            worker_outputs.append(full_output)
            worker_predictions[i].append(rating)
        
        # LLM-агрегатор принимает решение
        final_rating = aggregator.aggregate(worker_outputs, text)
        final_predictions.append(final_rating)
    
    # Конвертируем в numpy
    final_predictions = np.array(final_predictions)
    worker_predictions = [np.array(wp) for wp in worker_predictions]
    
    # Метрики
    metrics = compute_metrics(
        predictions=final_predictions,
        gold_labels=gold_labels,
        user_ids=user_ids,
        worker_predictions=worker_predictions
    )
    
    metrics['combined_score'] = compute_combined_score_unified(metrics, is_ensemble=True)
    
    return metrics
```

---

## 5. Датасет: WILDS Amazon Home and Kitchen

### Характеристики

| Параметр | Значение |
|----------|----------|
| Категория | Home_and_Kitchen |
| Всего отзывов | ~1,298 |
| Train | ~908 (70%) |
| Validation | ~195 (15%) |
| Test | ~195 (15%) |
| User-disjoint | Да |
| Классы | 1-5 звёзд |

### Загрузка данных

```python
# data_loader.py

import os
import pickle
from typing import List, Dict

CACHE_DIR = os.path.join(os.path.dirname(__file__), '.cache')
os.makedirs(CACHE_DIR, exist_ok=True)

def load_validation_data() -> List[Dict]:
    """
    Загружает validation данные.
    
    Returns:
        List[Dict] где каждый Dict содержит:
        - text: str (текст отзыва)
        - label: int (rating 1-5)
        - user_id: int
    """
    cache_path = os.path.join(CACHE_DIR, 'validation_data.pkl')
    
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    # Загрузка из WILDS
    # (используем существующую логику из evaluator.py)
    from wilds_experiment.evaluator import get_dataset_splits
    
    splits_data = get_dataset_splits()
    preprocessed = splits_data['preprocessed']
    val_split = splits_data['splits']['validation']
    
    validation_data = []
    for idx in val_split['indices']:
        validation_data.append({
            'text': preprocessed['texts'][idx],
            'label': preprocessed['labels'][idx],
            'user_id': preprocessed['user_ids'][idx],
        })
    
    # Кэшируем
    with open(cache_path, 'wb') as f:
        pickle.dump(validation_data, f)
    
    return validation_data
```

---

## 6. Запуск экспериментов

### Exp 1: Baseline (без OpenEvolve)

```bash
cd experiments/exp1_baseline
python run.py
```

### Exp 2-4: С OpenEvolve

```bash
cd experiments/exp2_single_evolved

python -m openevolve \
    --initial-program initial_prompt.txt \
    --evaluator evaluator.py \
    --config config.yaml \
    --iterations 100 \
    --output-dir results/
```

### Скрипт для всех экспериментов

```bash
#!/bin/bash
# run_all_experiments.sh

echo "=== Experiment 1: Baseline ==="
cd exp1_baseline && python run.py
cd ..

echo "=== Experiment 2: Single + OpenEvolve ==="
cd exp2_single_evolved
python -m openevolve --initial-program initial_prompt.txt \
    --evaluator evaluator.py --config config.yaml --iterations 100
cd ..

echo "=== Experiment 3a: Ensemble Baseline ==="
cd exp3_ensemble_voting && python run_baseline.py
cd ..

echo "=== Experiment 3b: Ensemble + OpenEvolve ==="
cd exp3_ensemble_voting
python -m openevolve --initial-program initial_prompt.txt \
    --evaluator evaluator.py --config config.yaml --iterations 100
cd ..

echo "=== Experiment 4: LLM Aggregator ==="
cd exp4_llm_aggregator
python -m openevolve --initial-program initial_prompt.txt \
    --evaluator evaluator.py --config config.yaml --iterations 100
cd ..

echo "=== All experiments complete ==="
```

---

## Checklist реализации

### Общие модули
- [ ] `metrics.py` - compute_metrics, compute_combined_score_unified
- [ ] `workers.py` - LLMWorker с Yandex Cloud API
- [ ] `aggregators.py` - MajorityVoteAggregator, LLMAggregator
- [ ] `data_loader.py` - load_validation_data

### Эксперимент 1
- [ ] `exp1_baseline/run.py`
- [ ] `exp1_baseline/initial_prompt.txt`
- [ ] Запуск и сохранение результатов

### Эксперимент 2
- [ ] `exp2_single_evolved/evaluator.py`
- [ ] `exp2_single_evolved/config.yaml`
- [ ] `exp2_single_evolved/initial_prompt.txt`
- [ ] Запуск OpenEvolve, кривые метрик

### Эксперимент 3
- [ ] `exp3_ensemble_voting/evaluator.py`
- [ ] `exp3_ensemble_voting/run_baseline.py`
- [ ] `exp3_ensemble_voting/config.yaml`
- [ ] Запуск 3a (baseline) и 3b (evolved)

### Эксперимент 4
- [ ] `exp4_llm_aggregator/evaluator.py`
- [ ] `exp4_llm_aggregator/aggregator_prompt.txt`
- [ ] `exp4_llm_aggregator/config.yaml`
- [ ] Запуск OpenEvolve с LLM-агрегатором


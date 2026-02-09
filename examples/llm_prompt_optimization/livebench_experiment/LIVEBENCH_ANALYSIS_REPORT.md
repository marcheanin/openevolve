# Анализ LiveBench для эволюции промптов

## 📋 Содержание
1. [Обзор структуры репозитория](#обзор-структуры-репозитория)
2. [Доступные датасеты и их релевантность](#доступные-датасеты-и-их-релевантность)
3. [Готовые скрипты оценки](#готовые-скрипты-оценки)
4. [Рекомендуемый датасет для эволюции промптов](#рекомендуемый-датасет-для-эволюции-промптов)
5. [Скорректированный план реализации](#скорректированный-план-реализации)
6. [Структура файлов эксперимента](#структура-файлов-эксперимента)

---

## 🗂️ Обзор структуры репозитория

```
LiveBench/
├── livebench/
│   ├── if_runner/                    # 🎯 ГЛАВНОЕ: Instruction Following
│   │   ├── ifbench/                  # IFBench (новый, сложный)
│   │   │   ├── evaluation_lib.py     # ✅ Готовый скрипт оценки
│   │   │   ├── instructions.py       # 58 новых ограничений
│   │   │   └── instructions_registry.py
│   │   └── instruction_following_eval/  # IFEval (оригинальный Google)
│   │       ├── evaluation_main.py    # ✅ Готовый скрипт оценки
│   │       ├── instructions.py       # Оригинальные ограничения
│   │       └── instructions_registry.py
│   ├── process_results/
│   │   └── instruction_following/
│   │       └── utils.py              # ✅ Функции подсчёта скора
│   ├── gen_api_answer.py             # Генерация ответов через API
│   ├── gen_ground_truth_judgment.py  # Оценка ответов
│   ├── run_livebench.py              # Главный скрипт запуска
│   └── download_questions.py         # Загрузка датасетов
└── README.md
```

---

## 📊 Доступные датасеты и их релевантность

### HuggingFace датасеты LiveBench

| Категория | HuggingFace путь | Релевантность | Описание |
|-----------|------------------|---------------|----------|
| **Instruction Following** | `livebench/instruction_following` | ⭐⭐⭐⭐⭐ | **ИДЕАЛЬНО** — прямая задача следования инструкциям |
| Reasoning | `livebench/reasoning` | ⭐⭐⭐ | Логические головоломки (web_of_lies, zebra_puzzle) |
| Math | `livebench/math` | ⭐⭐ | Математические соревнования |
| Coding | `livebench/coding` | ⭐ | Требует Docker, сложная интеграция |
| Language | `livebench/language` | ⭐⭐ | Typos, connections, plot_unscrambling |
| Data Analysis | `livebench/data_analysis` | ⭐⭐ | Работа с таблицами |

### Два формата Instruction Following в LiveBench

#### 1. **IFEval формат** (оригинальный Google)
- **Источник**: `livebench/if_runner/instruction_following_eval/`
- **Типы ограничений** (25 типов):
  - `keywords:existence` — наличие ключевых слов
  - `keywords:forbidden_words` — запрещённые слова
  - `length_constraints:number_words` — количество слов
  - `length_constraints:number_sentences` — количество предложений
  - `length_constraints:number_paragraphs` — количество параграфов
  - `detectable_format:json_format` — JSON формат
  - `detectable_format:number_bullet_lists` — списки
  - `change_case:english_capital` — заглавные буквы
  - `punctuation:no_comma` — без запятых
  - И другие...

#### 2. **IFBench формат** (новый, сложнее)
- **Источник**: `livebench/if_runner/ifbench/`
- **58 новых ограничений** (примеры):
  - `count:word_count_range` — диапазон количества слов
  - `count:unique_word_count` — уникальные слова
  - `ratio:sentence_type` — соотношение типов предложений
  - `words:alphabet` — слова по алфавиту
  - `words:palindrome` — палиндромы
  - `sentence:alliteration_increment` — аллитерация
  - `format:parentheses` — вложенные скобки
  - `format:emoji` — эмодзи требования
  - И другие...

---

## ✅ Готовые скрипты оценки

### 1. Основной скрипт оценки IF

**Файл**: `livebench/process_results/instruction_following/utils.py`

```python
def score_results(follow_all_instructions, follow_instruction_list, threshold=0.2):
    """
    Возвращает скор от 0 до 1:
    - score_1: 1 если все инструкции выполнены, иначе 0
    - score_2: доля выполненных инструкций
    - avg_score: среднее score_1 и score_2
    """
    score_1 = 1 if follow_all_instructions else 0
    score_2 = sum([1 if follow else 0 for follow in follow_instruction_list]) / len(follow_instruction_list)
    avg_score = (score_1 + score_2) / 2
    return avg_score

def ifbench_process_results(question, llm_answer, debug=False) -> float:
    """Оценка для IFBench формата."""
    # Использует evaluation_lib.test_instruction_following_strict()
    ...

def instruction_following_process_results(questions, model_answers, task, model_id, debug=False):
    """Оценка для IFEval формата."""
    # Использует evaluation_main.evaluator()
    ...
```

### 2. IFEval оценка (оригинальная Google)

**Файл**: `livebench/if_runner/instruction_following_eval/evaluation_main.py`

```python
def test_instruction_following_strict(inp, prompt_to_response):
    """Строгая проверка: все инструкции должны быть выполнены."""
    ...

def test_instruction_following_loose(inp, prompt_to_response):
    """Мягкая проверка: пробует разные варианты ответа."""
    ...

def evaluator(questions, model_answers, _OUTPUT_DIR, model_id):
    """Главная функция оценки."""
    ...
```

### 3. IFBench оценка (новая, сложнее)

**Файл**: `livebench/if_runner/ifbench/evaluation_lib.py`

```python
def test_instruction_following_strict(inp: InputExample, response: str):
    """Строгая проверка IFBench ограничений."""
    ...

def test_instruction_following_loose(inp: InputExample, response: str):
    """Мягкая проверка IFBench ограничений."""
    ...
```

---

## 🎯 Рекомендуемый датасет для эволюции промптов

### Выбор: **LiveBench Instruction Following**

| Критерий | IFEval (наш текущий) | LiveBench IF |
|----------|---------------------|--------------|
| **Готовые скрипты оценки** | ✅ Google | ✅ LiveBench (содержит оба!) |
| **Объективная оценка** | ✅ | ✅ |
| **Сложность задач** | Средняя | Выше (IFBench) |
| **Разнообразие** | 25 типов | 25 + 58 = 83 типа |
| **Защита от contamination** | ❌ Статичный | ✅ Обновляется |
| **Размер датасета** | 541 примеров | Варьируется по релизам |

### Преимущества LiveBench IF:
1. ✅ **Содержит оба формата**: IFEval + IFBench
2. ✅ **Готовые скрипты оценки** — не нужно писать свои
3. ✅ **Регулярные обновления** — новые задачи каждый месяц
4. ✅ **Объективные метрики** — без LLM-судьи
5. ✅ **Совместимость** — использует тот же API что и наш ifeval_experiment

---

## 📐 Скорректированный план реализации

### Фаза 1: Подготовка (1-2 часа)

#### 1.1. Установка зависимостей

```bash
cd livebench_experiment/LiveBench
pip install -e .
```

#### 1.2. Загрузка датасета

```bash
cd livebench
python download_questions.py
```

Датасет будет загружен в: `livebench/data/live_bench/instruction_following/`

#### 1.3. Структура вопросов

```json
{
  "question_id": "abc123...",
  "category": "instruction_following",
  "task": "ifeval_v2",  // или "ifbench"
  "turns": ["Your instruction prompt here..."],
  "instruction_id_list": ["keywords:existence", "length_constraints:number_words"],
  "kwargs": [{"keywords": ["example"]}, {"num_words": 100}],
  "ground_truth": null  // Оценка через check_following()
}
```

### Фаза 2: Создание evaluator.py (2-3 часа)

```python
"""
Evaluator for LiveBench Instruction Following prompt optimization.
Использует ОФИЦИАЛЬНЫЕ скрипты оценки LiveBench.
"""

import os
import sys
import random
from typing import Dict, Any, Tuple, List

# Добавляем путь к LiveBench
LIVEBENCH_PATH = os.path.join(os.path.dirname(__file__), "LiveBench", "livebench")
sys.path.insert(0, LIVEBENCH_PATH)

from if_runner.instruction_following_eval import evaluation_main
from if_runner.ifbench import evaluation_lib
from process_results.instruction_following.utils import score_results, ifbench_process_results

# Глобальные переменные
_DATASET_SPLITS = {}
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".evaluation_cache")


def load_livebench_if_dataset(config: Dict) -> Tuple[List, List, List]:
    """
    Загрузка и разделение LiveBench IF датасета.
    """
    from datasets import load_dataset
    
    cache_key = "livebench_if"
    if cache_key in _DATASET_SPLITS:
        return (_DATASET_SPLITS[cache_key]["train"],
                _DATASET_SPLITS[cache_key]["validation"],
                _DATASET_SPLITS[cache_key]["test"])
    
    # Загрузка с HuggingFace
    dataset = load_dataset("livebench/instruction_following", split="test")
    
    # Фильтрация по релизу (опционально)
    release = config.get("livebench_release", "2024-11-25")
    dataset = [q for q in dataset if q.get("livebench_release_date", "") <= release]
    
    # Разделение
    train_ratio = config.get("train_ratio", 0.7)
    val_ratio = config.get("validation_ratio", 0.15)
    seed = config.get("split_seed", 42)
    
    random.seed(seed)
    random.shuffle(dataset)
    
    n = len(dataset)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train = dataset[:train_end]
    val = dataset[train_end:val_end]
    test = dataset[val_end:]
    
    _DATASET_SPLITS[cache_key] = {"train": train, "validation": val, "test": test}
    print(f"LiveBench IF split: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    
    return train, val, test


def evaluate_single_example(
    prompt_template: str,
    example: Dict[str, Any],
    client,
    model_name: str
) -> Tuple[float, Dict]:
    """
    Оценка одного примера используя ОФИЦИАЛЬНЫЕ скрипты LiveBench.
    """
    # Форматирование промпта
    instruction = example["turns"][0]
    formatted_prompt = prompt_template.format(instruction=instruction)
    
    # Получение ответа модели
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": formatted_prompt}],
        temperature=0.1,
        max_tokens=4096
    )
    model_response = response.choices[0].message.content.strip()
    
    # Определяем формат задачи
    task = example.get("task", "")
    
    if "ifbench" in task.lower():
        # Используем IFBench оценку
        score = ifbench_process_results(example, model_response, debug=False)
    else:
        # Используем IFEval оценку
        inp = evaluation_main.InputExample(
            key=example.get("question_id", 0),
            instruction_id_list=example["instruction_id_list"],
            prompt=instruction,
            kwargs=example["kwargs"]
        )
        prompt_to_response = {instruction: model_response}
        result = evaluation_main.test_instruction_following_strict(inp, prompt_to_response)
        score = score_results(result.follow_all_instructions, result.follow_instruction_list)
    
    details = {
        "question_id": example.get("question_id"),
        "task": task,
        "score": score,
        "instruction_ids": example.get("instruction_id_list", [])
    }
    
    return score, details


def evaluate(program_path: str) -> Dict[str, Any]:
    """
    Главная функция оценки для OpenEvolve.
    """
    # Загрузка промпта
    with open(program_path, "r", encoding="utf-8") as f:
        prompt_template = f.read().strip()
    
    # Загрузка конфига и датасета
    config = load_config()
    train_dataset, _, _ = load_livebench_if_dataset(config)
    
    # Инициализация клиента
    client = get_llm_client(config)
    model_name = config.get("model_name")
    
    # Каскадная оценка
    use_cascade = config.get("evaluator", {}).get("cascade_evaluation", True)
    
    if use_cascade:
        # Stage 1
        stage1_samples = 10
        stage1_threshold = 0.6
        
        indices = random.sample(range(len(train_dataset)), min(stage1_samples, len(train_dataset)))
        scores = []
        for idx in indices:
            score, _ = evaluate_single_example(prompt_template, train_dataset[idx], client, model_name)
            scores.append(score)
        
        stage1_accuracy = sum(scores) / len(scores)
        
        if stage1_accuracy < stage1_threshold:
            return {
                "combined_score": stage1_accuracy * 0.5,
                "accuracy": stage1_accuracy,
                "stage": 1,
                "passed_stage1": False,
                **calculate_prompt_features(prompt_template)
            }
        
        # Stage 2
        stage2_samples = 40
        indices = random.sample(range(len(train_dataset)), min(stage2_samples, len(train_dataset)))
        scores = []
        for idx in indices:
            score, _ = evaluate_single_example(prompt_template, train_dataset[idx], client, model_name)
            scores.append(score)
        
        accuracy = sum(scores) / len(scores)
    else:
        # Без каскада
        indices = random.sample(range(len(train_dataset)), min(50, len(train_dataset)))
        scores = []
        for idx in indices:
            score, _ = evaluate_single_example(prompt_template, train_dataset[idx], client, model_name)
            scores.append(score)
        accuracy = sum(scores) / len(scores)
    
    # LLM Feedback (опционально)
    llm_feedback = get_llm_feedback(prompt_template) if config.get("use_llm_feedback") else 0.0
    
    # Combined Score
    combined_score = 0.7 * accuracy + 0.3 * llm_feedback
    
    return {
        "combined_score": combined_score,
        "accuracy": accuracy,
        "llm_feedback": llm_feedback,
        **calculate_prompt_features(prompt_template),
        "stage": 2 if use_cascade else 0,
        "passed_stage1": True
    }
```

### Фаза 3: Конфигурация и запуск (1 час)

#### 3.1. `livebench_prompt_dataset.yaml`

```yaml
# LiveBench Instruction Following dataset configuration
dataset_name: "livebench/instruction_following"
input_field: "turns"
target_field: "instruction_id_list"
split: "test"

# Dataset splitting
train_ratio: 0.7
validation_ratio: 0.15
test_ratio: 0.15
split_seed: 42

# LiveBench specific
livebench_release: "2024-11-25"
is_livebench: true
streaming: false
```

#### 3.2. `livebench_prompt.txt`

```
Follow the instruction below carefully and precisely. Pay attention to all requirements and constraints specified in the instruction.

Instruction: {instruction}

Response:
```

#### 3.3. `config.yaml`

```yaml
max_iterations: 100
checkpoint_interval: 10
log_level: "INFO"
diff_based_evolution: false
max_code_length: 10000
language: "text"

llm:
  api_base: "https://llm.api.cloud.yandex.net/v1"
  models:
    - name: "gpt://b1gemincl8p7b2uiv5nl/qwen3-235b-a22b-fp8/latest"
      weight: 1.0
  temperature: 0.8
  max_tokens: 4096
  timeout: 60
  retries: 3

database:
  population_size: 50
  archive_size: 500
  num_islands: 4
  feature_dimensions: ["prompt_length", "reasoning_strategy"]
  feature_bins: 10

evaluator:
  timeout: 1800
  max_retries: 3
  parallel_evaluations: 4
  cascade_evaluation: true
  cascade_thresholds: [0.6]
  use_llm_feedback: true
  llm_feedback_weight: 0.2

evolution_trace:
  enabled: true
  format: "jsonl"
  include_code: true
  include_prompts: true
```

### Фаза 4: Визуализация и анализ (1-2 часа)

Копируем и адаптируем из `ifeval_experiment`:
- `visualize_evolution.py`
- `analyze_improvements.py`

---

## 📁 Структура файлов эксперимента

```
livebench_experiment/
├── LiveBench/                          # Клонированный репозиторий (уже есть)
├── evaluator.py                        # 🆕 Создать
├── config.yaml                         # 🆕 Создать
├── livebench_prompt.txt                # 🆕 Создать
├── livebench_prompt_dataset.yaml       # 🆕 Создать
├── visualize_evolution.py              # 🆕 Скопировать и адаптировать
├── analyze_improvements.py             # 🆕 Скопировать и адаптировать
├── run_evolution.ps1                   # 🆕 Создать
├── requirements.txt                    # 🆕 Создать
├── README.md                           # 🆕 Создать
├── LIVEBENCH_ANALYSIS_REPORT.md        # ✅ Этот файл
└── openevolve_output/                  # Будет создана при запуске
    ├── best/
    ├── checkpoints/
    ├── logs/
    └── visualizations/
```

---

## 🔑 Ключевые отличия от ifeval_experiment

| Аспект | ifeval_experiment | livebench_experiment |
|--------|------------------|---------------------|
| **Датасет** | Google IFEval | LiveBench IF (IFEval + IFBench) |
| **Скрипты оценки** | Отдельный модуль | Встроены в LiveBench |
| **Типы ограничений** | 25 | 83 (25 + 58) |
| **Сложность** | Средняя | Выше |
| **Обновления** | Нет | Ежемесячные |
| **Формат вопросов** | `instruction_id_list` + `kwargs` | Аналогичный |

---

## ⚡ Быстрый старт

```bash
# 1. Установка LiveBench
cd livebench_experiment/LiveBench
pip install -e .

# 2. Загрузка вопросов
cd livebench
python download_questions.py

# 3. Запуск эволюции (после создания файлов)
cd ..
python ../../openevolve-run.py \
  --initial-program livebench_prompt.txt \
  --evaluator evaluator.py \
  --config config.yaml \
  --max-iterations 100
```

---

## 📚 Ссылки

- [LiveBench Paper](https://arxiv.org/abs/2406.19314)
- [LiveBench Leaderboard](https://livebench.ai/)
- [HuggingFace Datasets](https://huggingface.co/livebench)
- [IFBench Paper](https://arxiv.org/pdf/2507.02833)
- [Оригинальный IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval)

---

*Отчёт создан: $(date)*
*Версия: 1.0*


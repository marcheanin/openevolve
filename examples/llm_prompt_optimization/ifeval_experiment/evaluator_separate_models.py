"""
Evaluator for HuggingFace dataset-based prompt optimization with separate models for evolution and evaluation.

This evaluator supports using different models:
- Evolution model: Used by OpenEvolve to generate improved prompts (from config.yaml llm section)
- Evaluation model: Used to execute prompts on the dataset (from config.yaml evaluation section)
"""

import re
import traceback
import yaml
import os
import time
import random
import json
import pickle
from openai import OpenAI
from tqdm import tqdm
from datasets import load_dataset

# Глобальные переменные для разделения датасета (внутри процесса)
_DATASET_SPLITS = {}  # Кэш для train/validation/test разделений (внутри процесса)
_MAX_ITERATIONS = 100  # Максимальное количество итераций (берется из конфига при первом вызове)

# Файлы для межпроцессного кэша
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".evaluation_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

EVALUATION_COUNTER_FILE = os.path.join(CACHE_DIR, "evaluation_counter.json")
DATASET_SPLITS_CACHE_FILE = os.path.join(CACHE_DIR, "dataset_splits_cache.pkl")

# ========================================================================================
# ЧТЕНИЕ КОНФИГУРАЦИИ С ПОДДЕРЖКОЙ ОТДЕЛЬНЫХ МОДЕЛЕЙ
# ========================================================================================
# Определяем, какой конфиг использовать (приоритет: config_separate_models.yaml)
config_file = "config_separate_models.yaml"
if not os.path.exists(os.path.join(os.path.dirname(__file__), config_file)):
    config_file = "config.yaml"  # Fallback на обычный конфиг

config_path = os.path.join(os.path.dirname(__file__), config_file)
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Не логируем загрузку конфига при каждом импорте (слишком много вывода)
# print(f"Loaded config from: {config_file}")

# ========================================================================================
# НАСТРОЙКА МОДЕЛИ ДЛЯ ОЦЕНКИ (EVALUATION MODEL)
# ========================================================================================
evaluation_config = config.get("evaluation", {})

if evaluation_config:
    # Используем отдельную модель для оценки
    # Логируем только при первом импорте (проверяем по переменной окружения)
    if not os.environ.get("_EVAL_MODEL_INIT_LOGGED"):
        print("="*80)
        print("Using SEPARATE evaluation model configuration")
        print("="*80)
        os.environ["_EVAL_MODEL_INIT_LOGGED"] = "1"
    
    # API base для оценки (может отличаться от эволюции)
    eval_api_base = evaluation_config.get("api_base")
    if eval_api_base is None:
        # Если не указан, используем тот же, что для эволюции
        llm_config = config.get("llm", {})
        eval_api_base = llm_config.get("api_base", "http://localhost:1234/v1")
    
    # Модель для оценки
    TASK_MODEL_NAME = evaluation_config.get("model", "default-model")
    
    # Настройки для оценки
    eval_temperature = evaluation_config.get("temperature", 0.1)
    MAX_TOKENS = evaluation_config.get("max_tokens", 4096)
    eval_timeout = evaluation_config.get("timeout", 120)
    
    if not os.environ.get("_EVAL_MODEL_INIT_LOGGED"):
        print(f"Evaluation Model: {TASK_MODEL_NAME}")
        print(f"Evaluation API Base: {eval_api_base}")
        print(f"Evaluation Temperature: {eval_temperature}")
        print(f"Evaluation Max Tokens: {MAX_TOKENS}")
        print("="*80)
else:
    # Fallback: используем модель из llm секции (как раньше)
    # Не логируем каждый раз
    llm_config = config.get("llm", {})
    eval_api_base = llm_config.get("api_base", "http://localhost:1234/v1")
    
    models = llm_config.get("models", [])
    if models:
        TASK_MODEL_NAME = models[0].get("name", "default-model")
    else:
        TASK_MODEL_NAME = llm_config.get("primary_model", "default-model")
    
    eval_temperature = 0.1  # Low temperature для оценки
    MAX_TOKENS = llm_config.get("max_tokens", 4096)

# Получаем настройки для разных стадий оценки (опционально)
eval_stage1_model = evaluation_config.get("stage1_model") if evaluation_config else None
eval_stage2_model = evaluation_config.get("stage2_model") if evaluation_config else None

# Get evaluator settings
evaluator_config = config.get("evaluator", {})
MAX_RETRIES = evaluator_config.get("max_retries", 3)

# Initialize OpenAI client для оценки
test_model = OpenAI(base_url=eval_api_base)
# Не логируем при каждом импорте (слишком много вывода в multiprocessing)

# ========================================================================================
# ОПРЕДЕЛЕНИЕ КОНФИГА ДАТАСЕТА
# ========================================================================================
# Determine which dataset to use based on the OPENEVOLVE_PROMPT environment variable
import sys

# Глобальная переменная для пути к конфигу датасета
DATASET_CONFIG_PATH = None

def init_dataset_config_path():
    """Инициализирует DATASET_CONFIG_PATH на основе OPENEVOLVE_PROMPT."""
    global DATASET_CONFIG_PATH
    
    if DATASET_CONFIG_PATH is not None:
        return  # Уже инициализирован
    
    prompt_file = os.environ.get("OPENEVOLVE_PROMPT")
    if not prompt_file:
        # Default to a generic dataset config if not using the wrapper script
        evaluator_dir = os.path.dirname(os.path.abspath(__file__))
        DATASET_CONFIG_PATH = os.path.join(evaluator_dir, "dataset_settings.yaml")
        # Не логируем каждый раз
    else:
        basename = os.path.basename(prompt_file)
        dataset_filename = basename.replace("_prompt.txt", "_prompt_dataset.yaml").replace(
            ".txt", "_dataset.yaml"
        )
        evaluator_dir = os.path.dirname(os.path.abspath(__file__))
        DATASET_CONFIG_PATH = os.path.join(evaluator_dir, dataset_filename)
        # Не логируем каждый раз

# Инициализируем при импорте модуля
init_dataset_config_path()


# ========================================================================================
# ФУНКЦИИ ДЛЯ МЕЖПРОЦЕССНОГО КЭШИРОВАНИЯ (работают между процессами)
# ========================================================================================

def get_current_iteration_from_checkpoint():
    """
    Получить текущий номер итерации на основе самого свежего checkpoint.
    
    Новая упрощенная стратегия:
    1. Находим самый свежий checkpoint (по времени создания metadata.json)
    2. Если checkpoint старше 1 часа - это новый запуск, возвращаем 0
    3. Если checkpoint свежий (< 1 часа) - используем last_iteration из metadata.json
    4. Не используем счетчик вызовов - он ненадежен в multiprocessing среде
    
    Returns:
        int: Текущая итерация (0 если новый запуск или checkpoint не найден)
    """
    try:
        import time
        evaluator_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoints_dir = os.path.join(evaluator_dir, "openevolve_output", "checkpoints")
        
        if not os.path.exists(checkpoints_dir):
            return 0
        
        # Ищем все checkpoint'ы и находим самый свежий по времени
        latest_checkpoint_path = None
        latest_timestamp = 0
        latest_iteration = 0
        
        for item in os.listdir(checkpoints_dir):
            checkpoint_path = os.path.join(checkpoints_dir, item)
            if not os.path.isdir(checkpoint_path) or not item.startswith("checkpoint_"):
                continue
            
            metadata_path = os.path.join(checkpoint_path, "metadata.json")
            if not os.path.exists(metadata_path):
                continue
            
            try:
                # Получаем время создания checkpoint
                checkpoint_time = os.path.getmtime(metadata_path)
                
                # Читаем last_iteration из metadata.json
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    checkpoint_iteration = metadata.get("last_iteration", 0)
                    # Если last_iteration нет, пытаемся извлечь из имени директории
                    if checkpoint_iteration == 0:
                        try:
                            checkpoint_iteration = int(item.replace("checkpoint_", ""))
                        except ValueError:
                            continue
                
                # Ищем самый свежий checkpoint
                if checkpoint_time > latest_timestamp:
                    latest_timestamp = checkpoint_time
                    latest_checkpoint_path = checkpoint_path
                    latest_iteration = checkpoint_iteration
                    
            except Exception:
                continue
        
        # Если не нашли ни одного checkpoint - начинаем с 0
        if latest_checkpoint_path is None or latest_iteration == 0:
            return 0
        
        # Проверяем, насколько свежий checkpoint
        current_time = time.time()
        checkpoint_age = current_time - latest_timestamp
        
        # Если checkpoint старше 1 часа - считаем это новым запуском
        if checkpoint_age > 3600:  # 1 час
            return 0
        
        # Checkpoint свежий - используем его last_iteration как текущую итерацию
        # Не добавляем оценку прогресса, т.к. это ненадежно в multiprocessing
        return max(0, int(latest_iteration))
        
    except Exception:
        # В случае любой ошибки возвращаем 0
        return 0


def get_evaluation_counter_from_file():
    """
    Получить счетчик вызовов оценки из файла (работает между процессами).
    Если файл очень старый (> 1 час), сбрасываем счетчик (вероятно, новый запуск).
    
    Returns:
        int: Текущий счетчик вызовов
    """
    try:
        if os.path.exists(EVALUATION_COUNTER_FILE):
            # Проверяем время создания файла
            import time
            file_age = time.time() - os.path.getmtime(EVALUATION_COUNTER_FILE)
            
            # Если файл старше 1 часа, считаем это новым запуском - сбрасываем счетчик
            if file_age > 3600:  # 1 час
                # Сбрасываем файл (но не удаляем, чтобы избежать race condition)
                try:
                    with open(EVALUATION_COUNTER_FILE, "w", encoding="utf-8") as f:
                        json.dump({"counter": 0}, f)
                except Exception:
                    pass
                return 0
            
            with open(EVALUATION_COUNTER_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("counter", 0)
    except Exception:
        pass
    return 0


def increment_evaluation_counter_file():
    """
    Увеличить счетчик вызовов оценки в файле (работает между процессами).
    Использует простую реализацию с retry для thread-safety.
    
    Returns:
        int: Новое значение счетчика
    """
    max_retries = 10
    retry_delay = 0.1
    
    for attempt in range(max_retries):
        try:
            # Читаем текущее значение
            counter = get_evaluation_counter_from_file()
            
            # Пытаемся записать новое значение
            new_counter = counter + 1
            with open(EVALUATION_COUNTER_FILE, "w", encoding="utf-8") as f:
                json.dump({"counter": new_counter}, f)
            
            # Проверяем, что значение действительно записалось
            # (простая проверка на race condition)
            time.sleep(0.01)
            verify_counter = get_evaluation_counter_from_file()
            if verify_counter >= new_counter:
                return verify_counter
            
            # Если проверка не прошла, повторяем
            time.sleep(retry_delay)
            
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                # Последняя попытка: просто возвращаем увеличенное значение
                try:
                    counter = get_evaluation_counter_from_file()
                    new_counter = counter + 1
                    with open(EVALUATION_COUNTER_FILE, "w", encoding="utf-8") as f:
                        json.dump({"counter": new_counter}, f)
                    return new_counter
                except Exception:
                    return 1  # Fallback значение
    
    return 1  # Fallback


def get_cached_dataset_splits(cache_key):
    """
    Получить разделенный датасет из файлового кэша.
    
    Args:
        cache_key: Ключ кэша (dataset_name_split)
    
    Returns:
        dict с ключами "train", "validation", "test" или None
    """
    try:
        if os.path.exists(DATASET_SPLITS_CACHE_FILE):
            with open(DATASET_SPLITS_CACHE_FILE, "rb") as f:
                cache = pickle.load(f)
                return cache.get(cache_key)
    except Exception:
        pass
    return None


def save_dataset_splits_to_cache(cache_key, train_dataset, validation_dataset, test_dataset):
    """
    Сохранить разделенный датасет в файловый кэш.
    
    Args:
        cache_key: Ключ кэша (dataset_name_split)
        train_dataset: Тренировочный датасет
        validation_dataset: Валидационный датасет
        test_dataset: Тестовый датасет
    
    Note:
        Сохраняем только индексы, а не сами данные, т.к. датасеты нельзя сериализовать напрямую.
        Вместо этого сохраняем информацию о размерах для проверки.
    """
    try:
        # Загружаем существующий кэш
        cache = {}
        if os.path.exists(DATASET_SPLITS_CACHE_FILE):
            try:
                with open(DATASET_SPLITS_CACHE_FILE, "rb") as f:
                    cache = pickle.load(f)
            except Exception:
                cache = {}
        
        # Сохраняем метаданные (не сами датасеты)
        # Проверяем, что датасеты имеют __len__ перед вызовом len()
        train_size = len(train_dataset) if hasattr(train_dataset, "__len__") else 0
        validation_size = len(validation_dataset) if hasattr(validation_dataset, "__len__") else 0
        test_size = len(test_dataset) if hasattr(test_dataset, "__len__") else 0
        
        cache[cache_key] = {
            "train_size": train_size,
            "validation_size": validation_size,
            "test_size": test_size,
            "cached": True
        }
        
        # Сохраняем кэш
        with open(DATASET_SPLITS_CACHE_FILE, "wb") as f:
            pickle.dump(cache, f)
    except Exception:
        pass  # Если не получилось сохранить - не критично


def calculate_prompt_features(prompt):
    """
    Calculate features for MAP-Elites diversity.
    Returns prompt_length and reasoning_sophistication score.
    """
    prompt_length = len(prompt)

    # Calculate reasoning sophistication based on keywords
    reasoning_keywords = [
        "step",
        "think",
        "reason",
        "analyze",
        "consider",
        "first",
        "then",
        "finally",
        "because",
        "therefore",
        "example",
        "for instance",
    ]
    prompt_lower = prompt.lower()
    keyword_count = sum(1 for keyword in reasoning_keywords if keyword in prompt_lower)
    reasoning_sophistication = min(keyword_count / len(reasoning_keywords), 1.0)

    return prompt_length, reasoning_sophistication


def split_dataset_train_val_test(dataset, train_ratio=0.7, validation_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Разделяет датасет на train/validation/test выборки.
    
    Args:
        dataset: HuggingFace dataset
        train_ratio: Доля тренировочной выборки (default: 0.7)
        validation_ratio: Доля валидационной выборки (default: 0.15)
        test_ratio: Доля тестовой выборки (default: 0.15)
        seed: Seed для воспроизводимости (default: 42)
    
    Returns:
        Tuple (train_dataset, validation_dataset, test_dataset)
    """
    if not hasattr(dataset, "__len__"):
        raise ValueError("Cannot split streaming dataset. Set streaming=False in config.")
    
    total_size = len(dataset)
    
    # Перемешиваем датасет для случайности
    dataset_shuffled = dataset.shuffle(seed=seed)
    
    # Вычисляем размеры выборок
    train_size = int(train_ratio * total_size)
    validation_size = int(validation_ratio * total_size)
    test_size = total_size - train_size - validation_size
    
    # Разделяем
    train_dataset = dataset_shuffled.select(range(train_size))
    validation_dataset = dataset_shuffled.select(range(train_size, train_size + validation_size))
    test_dataset = dataset_shuffled.select(range(train_size + validation_size, total_size))
    
    return train_dataset, validation_dataset, test_dataset


def get_dataset_splits(dataset_name, split="train"):
    """
    Получить разделенные датасеты (train/validation/test) из кэша.
    
    Args:
        dataset_name: Имя датасета
        split: Исходный split (обычно "train")
    
    Returns:
        Dict с ключами "train", "validation", "test" или None если не найдено
    """
    cache_key = f"{dataset_name}_{split}"
    return _DATASET_SPLITS.get(cache_key)


def load_hf_dataset(config):
    """
    Load HuggingFace dataset based on configuration.
    For IFEval: automatically splits into train/validation/test (70/15/15).
    """
    dataset_name = config["dataset_name"]
    dataset_config = config.get("dataset_config", None)
    split = config.get("split", "test")
    trust_remote_code = config.get("trust_remote_code", True)
    is_ifeval = config.get("is_ifeval", False)
    
    # Для IFEval используем разделение на train/validation/test
    if is_ifeval:
        cache_key = f"{dataset_name}_{split}"
        
        # Сначала проверяем кэш в памяти (быстро, внутри процесса)
        if cache_key in _DATASET_SPLITS:
            return _DATASET_SPLITS[cache_key]["train"]
        
        # Затем проверяем файловый кэш (между процессами)
        cached_meta = get_cached_dataset_splits(cache_key)
        if cached_meta and cached_meta.get("cached"):
            # Если кэш есть, все равно нужно разделить заново (датасет не сериализуем)
            # Но можем пропустить вывод сообщения
            pass

    # Для IFEval и других датасетов без streaming (для возможности разделения)
    streaming = config.get("streaming", False) if is_ifeval else config.get("streaming", True)
    
    # Legacy handling for HotpotQA - always use non-streaming mode
    if dataset_name == "hotpot_qa" or config.get("is_hotpotqa", False):
        streaming = False

    try:
        # Try to load the specified split
        if dataset_config:
            dataset = load_dataset(
                dataset_name,
                dataset_config,
                split=split,
                trust_remote_code=trust_remote_code,
                streaming=streaming,
            )
        else:
            dataset = load_dataset(
                dataset_name, split=split, trust_remote_code=trust_remote_code, streaming=streaming
            )
    except Exception:
        # Fallback to train split if test is not available (без лишнего логирования)
        if dataset_config:
            dataset = load_dataset(
                dataset_name,
                dataset_config,
                split="train",
                trust_remote_code=trust_remote_code,
                streaming=streaming,
            )
        else:
            dataset = load_dataset(
                dataset_name,
                split="train",
                trust_remote_code=trust_remote_code,
                streaming=streaming,
            )

    # Для IFEval: разделяем на train/validation/test
    if is_ifeval and not streaming and hasattr(dataset, "__len__"):
        cache_key = f"{dataset_name}_{split}"
        
        # Проверяем, нужно ли разделять (проверяем файловый кэш)
        cached_meta = get_cached_dataset_splits(cache_key)
        should_split = True
        should_log = True
        
        if cached_meta and cached_meta.get("cached"):
            # Кэш есть, но нам все равно нужно разделить (датасет не сериализуем)
            # Но можем проверить размеры
            if hasattr(dataset, "__len__"):
                dataset_size = len(dataset)  # type: ignore
                if (cached_meta.get("train_size") == int(0.7 * dataset_size) and
                    cached_meta.get("validation_size") == int(0.15 * dataset_size)):
                    should_log = False  # Не логируем, если уже разделяли
        
        train_ratio = config.get("train_ratio", 0.7)
        validation_ratio = config.get("validation_ratio", 0.15)
        test_ratio = config.get("test_ratio", 0.15)
        split_seed = config.get("split_seed", 42)
        
        train_dataset, validation_dataset, test_dataset = split_dataset_train_val_test(
            dataset, train_ratio, validation_ratio, test_ratio, seed=split_seed
        )
        
        # Сохраняем в кэш в памяти (для этого процесса)
        _DATASET_SPLITS[cache_key] = {
            "train": train_dataset,
            "validation": validation_dataset,
            "test": test_dataset
        }
        
        # Сохраняем метаданные в файловый кэш (для других процессов)
        save_dataset_splits_to_cache(cache_key, train_dataset, validation_dataset, test_dataset)
        
        # Логируем только при первом разделении
        if should_log:
            print(f"IFEval dataset split: Train={len(train_dataset)}, Validation={len(validation_dataset)}, Test={len(test_dataset)}")
        
        # Возвращаем train для эволюции
        return train_dataset

    # Dataset info не логируем - слишком много вывода

    return dataset


def get_evaluation_model(stage=None):
    """
    Получить модель для оценки в зависимости от стадии.
    
    Args:
        stage: 1 для Stage 1, 2 для Stage 2, None для дефолтной
    
    Returns:
        Tuple (model_name, temperature)
    """
    global TASK_MODEL_NAME, eval_temperature, eval_stage1_model, eval_stage2_model
    
    if stage == 1 and eval_stage1_model:
        return eval_stage1_model, eval_temperature
    elif stage == 2 and eval_stage2_model:
        return eval_stage2_model, eval_temperature
    else:
        return TASK_MODEL_NAME, eval_temperature


def evaluate_prompt(prompt, dataset, config, num_samples):
    """Evaluate a prompt on a subset of the dataset."""
    input_field = config["input_field"]
    target_field = config["target_field"]

    # Check dataset type
    dataset_name = config.get("dataset_name", "").lower()
    is_emotion = "emotion" in dataset_name
    is_gsm8k = "gsm8k" in dataset_name
    is_hotpotqa = config.get("is_hotpotqa", False)
    is_ifeval = config.get("is_ifeval", False)
    is_hover = config.get("is_hover", False)

    # Sample from dataset - handle both streaming and non-streaming
    # ИСПОЛЬЗУЕМ СЛУЧАЙНУЮ ВЫБОРКУ вместо фиксированной
    if hasattr(dataset, "take"):
        # Streaming dataset - нельзя сделать случайную выборку, используем первые N
        samples = dataset.take(num_samples)
        sample_iter = tqdm(samples, desc=f"Evaluating {num_samples} samples", total=num_samples)
    else:
        # Non-streaming dataset - СЛУЧАЙНАЯ выборка
        dataset_size = len(dataset)
        num_samples_actual = min(num_samples, dataset_size)
        
        # Случайная выборка с фиксированным seed для воспроизводимости внутри одного вызова
        # Но seed меняется при каждом вызове для разнообразия
        random.seed()  # Используем системный seed для случайности
        indices = random.sample(range(dataset_size), num_samples_actual)
        samples = dataset.select(indices)
        sample_iter = tqdm(samples, desc=f"Evaluating {num_samples_actual} random samples")

    # Get model for evaluation once at the start (логируем при первом использовании)
    eval_model, eval_temp = get_evaluation_model()
    
    # Логируем модель для оценки при первом использовании в evaluate_prompt
    if not hasattr(evaluate_prompt, '_eval_model_logged'):
        print(f"[EVALUATION MODEL] Using model: {eval_model} (temperature={eval_temp})")
        evaluate_prompt._eval_model_logged = True

    correct = 0
    total = 0

    for example in sample_iter:
        input_text = example[input_field]
        expected = example[target_field]

        # Prepare the prompt with appropriate formatting
        if is_hotpotqa:
            # Format context from paragraphs
            context_items = example.get("context", {})
            context_text = ""
            if "title" in context_items and "sentences" in context_items:
                # Handle the specific structure of HotpotQA when that dataset is used
                for i, (title, sentences) in enumerate(
                    zip(context_items["title"], context_items["sentences"])
                ):
                    context_text += f"Paragraph {i+1} ({title}):\n"
                    context_text += " ".join(sentences) + "\n\n"
            formatted_prompt = prompt.format(context=context_text.strip(), question=input_text)
        elif is_ifeval:
            # IFEval uses 'prompt' field directly
            formatted_prompt = prompt.format(instruction=input_text)
        elif is_hover:
            # HoVer uses claim field
            formatted_prompt = prompt.format(claim=input_text)
        else:
            # Default formatting for other datasets
            formatted_prompt = prompt.format(input_text=input_text)

        # Prepare the message for the LLM
        messages = [{"role": "user", "content": formatted_prompt}]

        # Call the LLM with retry logic
        for attempt in range(MAX_RETRIES):
            try:
                # Use max_tokens from config
                response = test_model.chat.completions.create(  # type: ignore
                    model=eval_model,
                    messages=messages,  # type: ignore
                    temperature=eval_temp,  # Используем температуру из конфига оценки
                    max_tokens=MAX_TOKENS,
                )
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"Failed to get response after {MAX_RETRIES} attempts: {e}")
                    raise e
                time.sleep(1)

        # Handle potential None response
        if not response:
            print(f"Warning: No response object from LLM")
            total += 1  # Count as incorrect
            continue

        if not response.choices:
            print(f"Warning: No choices in response from LLM")
            total += 1  # Count as incorrect
            continue

        if not response.choices[0].message:
            print(f"Warning: No message in response choice")
            total += 1  # Count as incorrect
            continue

        output_text = response.choices[0].message.content
        if output_text is None:
            print(f"Warning: None content in LLM response")
            print(f"Full response: {response}")
            total += 1  # Count as incorrect
            continue

        output_text = output_text.strip()

        # Extract prediction from output
        # Для IFEval используем отдельную обработку ошибок
        if is_ifeval:
            # ОБЯЗАТЕЛЬНО использовать только официальный скрипт Google
            import sys
            import os
            
            # Определяем путь заранее
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            try:
                # СНАЧАЛА добавляем родительскую директорию в путь
                if parent_dir not in sys.path:
                    sys.path.insert(0, parent_dir)
                
                # ПОТОМ импортируем модуль
                from ifeval_official_evaluation import evaluate_ifeval_official_only, initialize_official_evaluation
                
                # Инициализация (выбросит ошибку если недоступен)
                if not hasattr(evaluate_ifeval_official_only, '_initialized'):
                    eval_path = os.environ.get("IFEVAL_EVAL_PATH")  # Может быть None
                    try:
                        initialize_official_evaluation(eval_path)  # type: ignore
                        evaluate_ifeval_official_only._initialized = True
                    except RuntimeError as e:
                        print(str(e))
                        raise RuntimeError(
                            "Официальный скрипт оценки IFEval от Google недоступен. "
                            "Эволюция не может продолжаться. Установите официальный скрипт."
                        )
                
                # Получаем instruction_id_list и kwargs из датасета
                instruction_id_list = example.get("instruction_id_list", [])
                kwargs = example.get("kwargs", [])
                
                if not instruction_id_list:
                    raise RuntimeError(
                        f"instruction_id_list отсутствует в примере датасета. "
                        f"Невозможно провести официальную оценку."
                    )
                
                # Выполняем оценку через официальный скрипт
                eval_path = os.environ.get("IFEVAL_EVAL_PATH")  # Может быть None
                passed, eval_details = evaluate_ifeval_official_only(
                    output_text, input_text, instruction_id_list, kwargs, eval_path  # type: ignore
                )
                
                if passed:
                    correct += 1

                total += 1
                continue
                
            except (ImportError, RuntimeError, ValueError) as e:
                # Критические ошибки пробрасываем дальше - не скрываем
                error_msg = str(e)
                print(f"КРИТИЧЕСКАЯ ОШИБКА IFEval: {error_msg}")
                if isinstance(e, ImportError):
                    print(f"Родительская директория: {parent_dir}")
                    eval_file = os.path.join(parent_dir, 'ifeval_official_evaluation.py')
                    print(f"Файл существует: {os.path.exists(eval_file)}")
                raise  # Пробрасываем дальше - не обрабатываем в общем блоке
        
        try:
            if is_gsm8k:
                # For GSM8K, extract the numeric answer after ####
                # First, extract the expected answer from the ground truth
                expected_answer = expected.split("####")[-1].strip()
                try:
                    expected_number = float(expected_answer.replace(",", ""))
                except Exception:
                    print(f"Warning: Could not parse expected answer: {expected_answer}")
                    total += 1
                    continue

                # Extract prediction from model output
                prediction = None
                if "####" in output_text:
                    predicted_answer = output_text.split("####")[-1].strip()
                    # Extract just the number, removing any extra text like $ signs
                    import re

                    numbers = re.findall(r"-?\$?[\d,]+\.?\d*", predicted_answer)
                    if numbers:
                        try:
                            # Remove $ and , from the number
                            number_str = numbers[0].replace("$", "").replace(",", "")
                            prediction = float(number_str)
                        except Exception:
                            pass

                # If we found a prediction, check if it matches
                if prediction is not None:
                    # Check if answers match (with small tolerance for floats)
                    if abs(prediction - expected_number) < 0.001:
                        correct += 1

                total += 1
                continue  # Skip the general case to avoid double counting

            elif is_hotpotqa:
                # For HotpotQA (legacy support), do exact match comparison (case-insensitive)
                output_lower = output_text.lower().strip()
                expected_lower = str(expected).lower().strip()

                # Remove common punctuation for better matching
                output_lower = output_lower.rstrip(".,!?;:")
                expected_lower = expected_lower.rstrip(".,!?;:")

                if output_lower == expected_lower:
                    correct += 1
                elif expected_lower in output_lower:
                    # Partial credit if answer is contained in response
                    correct += 1

                total += 1
                continue

            elif is_hover:
                # For HoVer, check if prediction matches SUPPORTED/NOT_SUPPORTED
                output_upper = output_text.upper()
                expected_upper = str(expected).upper()

                # Look for the verdict in the output
                if "SUPPORTED" in output_upper and "NOT" not in output_upper.replace(
                    "NOT SUPPORTED", ""
                ):
                    prediction = "SUPPORTED"
                elif "NOT SUPPORTED" in output_upper or "NOT_SUPPORTED" in output_upper:
                    prediction = "NOT_SUPPORTED"
                else:
                    prediction = None

                if prediction == expected_upper:
                    correct += 1

                total += 1
                continue

            elif is_emotion:
                # For emotion classification (0-5)
                numbers = re.findall(r"\b[0-5]\b", output_text)
                if numbers:
                    prediction = int(numbers[-1])  # Use the last number found
                else:
                    # Try to infer from emotion keywords
                    output_lower = output_text.lower()
                    emotion_map = {
                        "sadness": 0,
                        "sad": 0,
                        "joy": 1,
                        "happy": 1,
                        "happiness": 1,
                        "love": 2,
                        "anger": 3,
                        "angry": 3,
                        "fear": 4,
                        "afraid": 4,
                        "scared": 4,
                        "surprise": 5,
                        "surprised": 5,
                    }
                    prediction = -1
                    for emotion, label in emotion_map.items():
                        if emotion in output_lower:
                            prediction = label
                            break
            else:
                # For sentiment classification (0-1)
                numbers = re.findall(r"\b[01]\b", output_text)
                if numbers:
                    prediction = int(numbers[-1])  # Use the last number found
                else:
                    # Try to infer from keywords
                    output_lower = output_text.lower()
                    if "positive" in output_lower:
                        prediction = 1
                    elif "negative" in output_lower:
                        prediction = 0
                    else:
                        prediction = -1  # Invalid prediction

            if prediction == expected:
                correct += 1

            total += 1

        except Exception as e:
            error_msg = str(e)
            # Ограничиваем длину выводимого текста для читаемости
            output_preview = output_text[:100] + "..." if len(output_text) > 100 else output_text
            print(f"Error parsing response (first 100 chars: '{output_preview}'): {error_msg}")
            total += 1  # Count as incorrect

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total


def get_adaptive_sample_size(stage, current_iteration, max_iterations=100):
    """
    Адаптивно увеличиваем размер выборки по мере прогресса эволюции.
    Размер выборки рассчитывается пропорционально max_iterations.
    
    Args:
        stage: 1 для Stage 1, 2 для Stage 2
        current_iteration: Текущий номер итерации (читается из checkpoint)
        max_iterations: Максимальное количество итераций
    
    Returns:
        Размер выборки для данного этапа
    """
    # Используем реальный номер итерации из checkpoint
    # Нормализуем прогресс (0.0 - 1.0) относительно max_iterations
    progress = min(current_iteration / max_iterations, 1.0) if max_iterations > 0 else 0.0
    
    if stage == 1:
        # Stage 1: плавное линейное увеличение от 10 до 20 примеров
        # progress = 0.0 → 10, progress = 1.0 → 20
        min_samples = 10
        max_samples = 20
        return int(min_samples + (max_samples - min_samples) * progress)
    else:  # stage == 2
        # Stage 2: плавное линейное увеличение от 20 до 60 примеров
        # progress = 0.0 → 20, progress = 1.0 → 60
        min_samples = 20
        max_samples = 60
        return int(min_samples + (max_samples - min_samples) * progress)


def load_prompt_config(prompt_path):
    """Load the prompt from text file and dataset config from matching _dataset.yaml file."""
    global DATASET_CONFIG_PATH
    
    # Убеждаемся, что DATASET_CONFIG_PATH инициализирован
    if DATASET_CONFIG_PATH is None:
        init_dataset_config_path()
    
    # Проверяем, что путь установлен после инициализации
    if DATASET_CONFIG_PATH is None:
        raise RuntimeError("DATASET_CONFIG_PATH не был инициализирован")
    
    try:
        # Load prompt from text file
        with open(prompt_path, "r", encoding="utf-8", errors="replace") as f:
            prompt = f.read().strip()
    except (UnicodeDecodeError, FileNotFoundError) as e:
        raise RuntimeError(f"Failed to load prompt file '{prompt_path}': {str(e)}") from e

    # Load the configuration (already determined from environment variable)
    if not os.path.exists(DATASET_CONFIG_PATH):
        raise FileNotFoundError(f"Dataset configuration not found: {DATASET_CONFIG_PATH}")

    try:
        with open(DATASET_CONFIG_PATH, "r", encoding="utf-8", errors="replace") as f:
            dataset_config = yaml.safe_load(f)
    except (UnicodeDecodeError, yaml.YAMLError) as e:
        raise RuntimeError(f"Failed to load dataset config '{DATASET_CONFIG_PATH}': {str(e)}") from e

    return dataset_config, prompt


def evaluate_stage1(prompt_path):
    """
    Stage 1 evaluation: Quick evaluation with adaptive sample size

    Args:
        prompt_path: Path to the prompt file

    Returns:
        Dictionary with combined_score metric
    """
    global _MAX_ITERATIONS
    
    print("-" * 80)
    print("Starting Stage 1 evaluation...")
    print("-" * 80)

    try:
        # Загружаем max_iterations из конфига при первом вызове
        if _MAX_ITERATIONS == 100:
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config_temp = yaml.safe_load(f)
                    _MAX_ITERATIONS = config_temp.get("max_iterations", 100)
            except Exception:
                pass  # Используем значение по умолчанию
        
        # Увеличиваем счетчик вызовов в файле (только для логирования, не для расчета итерации)
        evaluation_counter = increment_evaluation_counter_file()
        
        # Load prompt configuration
        dataset_config, prompt = load_prompt_config(prompt_path)
        print(f"Loaded prompt configuration")

        # Load dataset (будет автоматически разделен для IFEval)
        dataset = load_hf_dataset(dataset_config)

        # Получаем текущую итерацию из самого свежего checkpoint
        current_iteration = get_current_iteration_from_checkpoint()
        
        # Адаптивный размер выборки (используем номер итерации из checkpoint)
        stage1_samples = get_adaptive_sample_size(1, current_iteration, _MAX_ITERATIONS)
        
        print(f"Stage 1: Evaluating {stage1_samples} random samples (adaptive, iteration {current_iteration}/{_MAX_ITERATIONS}, calls={evaluation_counter})...")

        # Run evaluation
        accuracy, correct, total = evaluate_prompt(prompt, dataset, dataset_config, stage1_samples)

        print(f"Stage 1 accuracy: {accuracy:.3f} ({correct}/{total})")
        print("-" * 80)

        # Calculate custom features
        prompt_length, reasoning_sophistication = calculate_prompt_features(prompt)
        print(
            f"Prompt features - Length: {prompt_length} chars, Reasoning sophistication: {reasoning_sophistication:.3f}"
        )

        return {
            "combined_score": accuracy,
            "prompt_length": prompt_length,
            "reasoning_strategy": reasoning_sophistication,
        }

    except Exception as e:
        error_msg = str(e)
        # Краткое сообщение об ошибке без полного traceback
        print(f"Stage 1 evaluation failed: {error_msg}")
        print("-" * 80)

        # Always return feature dimensions, even on failure
        try:
            # Try to calculate features from the failed prompt
            with open(prompt_path, "r", encoding="utf-8") as f:
                failed_prompt = f.read().strip()
            prompt_length, reasoning_sophistication = calculate_prompt_features(failed_prompt)
        except Exception:
            # Fallback values if prompt can't be read
            prompt_length, reasoning_sophistication = 0, 0.0

        return {
            "combined_score": 0.0,
            "prompt_length": prompt_length,
            "reasoning_strategy": reasoning_sophistication,
            "error": str(e),
        }


def evaluate_stage2(prompt_path):
    """
    Stage 2 evaluation: Comprehensive evaluation with adaptive sample size

    Args:
        prompt_path: Path to the prompt file

    Returns:
        Dictionary with combined_score metric
    """
    global _MAX_ITERATIONS
    
    print("-" * 80)
    print("Starting Stage 2 evaluation...")
    print("-" * 80)

    try:
        # Загружаем max_iterations из конфига при первом вызове
        if _MAX_ITERATIONS == 100:
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config_temp = yaml.safe_load(f)
                    _MAX_ITERATIONS = config_temp.get("max_iterations", 100)
            except Exception:
                pass  # Используем значение по умолчанию
        
        # Получаем счетчик из файла (работает между процессами)
        # Если Stage 1 уже увеличил счетчик, используем его значение
        evaluation_counter = get_evaluation_counter_from_file()
        if evaluation_counter == 0:
            # Если счетчик 0, увеличиваем (Stage 2 вызван отдельно)
            evaluation_counter = increment_evaluation_counter_file()
        
        # Load prompt configuration
        dataset_config, prompt = load_prompt_config(prompt_path)
        print(f"Loaded prompt configuration")

        # Load dataset (будет автоматически разделен для IFEval)
        dataset = load_hf_dataset(dataset_config)

        # Получаем реальный номер итерации из checkpoint
        current_iteration = get_current_iteration_from_checkpoint()
        
        # Адаптивный размер выборки (используем реальный номер итерации из checkpoint)
        stage2_samples = get_adaptive_sample_size(2, current_iteration, _MAX_ITERATIONS)
        
        print(f"Stage 2: Evaluating {stage2_samples} random samples (adaptive, iteration {current_iteration}/{_MAX_ITERATIONS}, calls={evaluation_counter})...")

        # Run evaluation
        accuracy, correct, total = evaluate_prompt(prompt, dataset, dataset_config, stage2_samples)

        print(f"Stage 2 accuracy: {accuracy:.3f} ({correct}/{total})")
        print("-" * 80)

        # Calculate custom features
        prompt_length, reasoning_sophistication = calculate_prompt_features(prompt)
        print(
            f"Prompt features - Length: {prompt_length} chars, Reasoning sophistication: {reasoning_sophistication:.3f}"
        )

        return {
            "combined_score": accuracy,
            "prompt_length": prompt_length,
            "reasoning_strategy": reasoning_sophistication,
        }

    except Exception as e:
        error_msg = str(e)
        # Краткое сообщение об ошибке без полного traceback
        print(f"Stage 2 evaluation failed: {error_msg}")
        print("-" * 80)

        # Always return feature dimensions, even on failure
        try:
            # Try to calculate features from the failed prompt
            with open(prompt_path, "r", encoding="utf-8") as f:
                failed_prompt = f.read().strip()
            prompt_length, reasoning_sophistication = calculate_prompt_features(failed_prompt)
        except Exception:
            # Fallback values if prompt can't be read
            prompt_length, reasoning_sophistication = 0, 0.0

        return {
            "combined_score": 0.0,
            "prompt_length": prompt_length,
            "reasoning_strategy": reasoning_sophistication,
            "error": str(e),
        }


def evaluate(prompt_path):
    """
    Main evaluation function - for backwards compatibility
    Calls evaluate_stage2 for full evaluation

    Args:
        prompt_path: Path to the prompt file

    Returns:
        Dictionary with combined_score metric
    """
    return evaluate_stage2(prompt_path)


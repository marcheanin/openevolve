"""
Evaluator for WILDS Amazon sentiment classification prompt optimization.
Supports user-disjoint split, class stratification, and cascade evaluation.
"""

import re
import traceback
import yaml
import os
import ssl
import time
import random
import json
import pickle
import numpy as np
from collections import Counter
from openai import OpenAI
from typing import Dict, List, Tuple, Any, Optional

# Импорты для автоматических few-shot примеров
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.neighbors import NearestNeighbors
    # Для BGE-M3 можно использовать FlagEmbedding, но пробуем через SentenceTransformer
    # try:
    #     from FlagEmbedding import BGEM3FlagModel
    #     BGEM3FlagModel_available = True
    # except ImportError:
    #     BGEM3FlagModel_available = False
except ImportError:
    print("Warning: sentence-transformers or sklearn not available. Automatic few-shot examples will be disabled.")
    SentenceTransformer = None
    NearestNeighbors = None

# Отключаем проверку SSL сертификатов (для загрузки WILDS)
ssl._create_default_https_context = ssl._create_unverified_context

# Глобальные переменные
_DATASET_SPLITS = {}  # Кэш для train/validation/test разделений
_MAX_ITERATIONS = 100

# Файлы для кэша
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".evaluation_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

EVALUATION_COUNTER_FILE = os.path.join(CACHE_DIR, "evaluation_counter.json")
DATASET_SPLITS_CACHE_FILE = os.path.join(CACHE_DIR, "dataset_splits_cache.pkl")
PREPROCESSED_DATA_CACHE_FILE = os.path.join(CACHE_DIR, "preprocessed_category_data.pkl")

# Файлы для кэша эмбеддингов
TRAIN_EMBEDDINGS_CACHE_FILE = os.path.join(CACHE_DIR, "train_embeddings.pkl")
TRAIN_EMBEDDINGS_INDEX_FILE = os.path.join(CACHE_DIR, "train_embeddings_index.pkl")
VAL_EMBEDDINGS_CACHE_FILE = os.path.join(CACHE_DIR, "val_embeddings.pkl")
TEST_EMBEDDINGS_CACHE_FILE = os.path.join(CACHE_DIR, "test_embeddings.pkl")

# Глобальные переменные для эмбеддингов
_EMBEDDING_MODEL = None
_TRAIN_EMBEDDINGS_DATA = None
_TRAIN_EMBEDDINGS_INDEX = None
_VAL_EMBEDDINGS_DATA = None
_TEST_EMBEDDINGS_DATA = None
_EMBEDDINGS_BY_TEXT = None  # Кэш эмбеддингов по тексту отзыва (для всех сплитов)

# Read config.yaml
with open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Get model settings from config
llm_config = config.get("llm", {})
api_base = llm_config.get("api_base", "http://localhost:1234/v1")

models = llm_config.get("models", [])
if models:
    TASK_MODEL_NAME = models[0].get("name", "default-model")
else:
    TASK_MODEL_NAME = llm_config.get("primary_model", "default-model")

# Get evaluator settings
evaluator_config = config.get("evaluator", {})
MAX_RETRIES = evaluator_config.get("max_retries", 3)
MAX_TOKENS = llm_config.get("max_tokens", 4096)

# Initialize OpenAI client
test_model = OpenAI(base_url=api_base)

# Глобальная переменная для пути к конфигу датасета
DATASET_CONFIG_PATH = None


def init_dataset_config_path():
    """Инициализирует DATASET_CONFIG_PATH на основе OPENEVOLVE_PROMPT."""
    global DATASET_CONFIG_PATH
    
    if DATASET_CONFIG_PATH is not None:
        return
    
    prompt_file = os.environ.get("OPENEVOLVE_PROMPT")
    if not prompt_file:
        evaluator_dir = os.path.dirname(os.path.abspath(__file__))
        DATASET_CONFIG_PATH = os.path.join(evaluator_dir, "wilds_prompt_dataset.yaml")
    else:
        basename = os.path.basename(prompt_file)
        dataset_filename = basename.replace("_prompt.txt", "_prompt_dataset.yaml").replace(
            ".txt", "_dataset.yaml"
        )
        evaluator_dir = os.path.dirname(os.path.abspath(__file__))
        DATASET_CONFIG_PATH = os.path.join(evaluator_dir, dataset_filename)


init_dataset_config_path()


def load_prompt_config() -> Dict:
    """Загружает конфигурацию датасета."""
    global DATASET_CONFIG_PATH
    init_dataset_config_path()
    
    if DATASET_CONFIG_PATH and os.path.exists(DATASET_CONFIG_PATH):
        with open(DATASET_CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    
    # Default config
    return {
        "dataset_name": "wilds_amazon",
        "category": "Office_Products",
        "category_id": 14,
        "stage1_samples": 10,
        "stage2_samples": 50,  # Увеличено с 20
        "validation_samples": 100,
        "train_ratio": 0.7,
        "validation_ratio": 0.15,
        "test_ratio": 0.15,
        "split_seed": 42,
        "user_disjoint": True,
        "stratify_by_class": True,
        # Validation strategy
        "validate_every_stage2": True,
        "train_weight_in_score": 0.6,  # 60% train
        "val_weight_in_score": 0.4,    # 40% validation
    }


def get_current_iteration_from_checkpoint() -> int:
    """Получить текущий номер итерации из checkpoint."""
    try:
        evaluator_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoints_dir = os.path.join(evaluator_dir, "openevolve_output", "checkpoints")
        
        if not os.path.exists(checkpoints_dir):
            return 0
        
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
                checkpoint_time = os.path.getmtime(metadata_path)
                
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    checkpoint_iteration = metadata.get("last_iteration", 0)
                    if checkpoint_iteration == 0:
                        try:
                            checkpoint_iteration = int(item.replace("checkpoint_", ""))
                        except ValueError:
                            continue
                
                if checkpoint_time > latest_timestamp:
                    latest_timestamp = checkpoint_time
                    latest_iteration = checkpoint_iteration
                    
            except Exception:
                continue
        
        if latest_iteration == 0:
            return 0
        
        current_time = time.time()
        checkpoint_age = current_time - latest_timestamp
        
        if checkpoint_age > 3600:  # 1 час
            return 0
        
        return max(0, int(latest_iteration))
        
    except Exception:
        return 0


def load_preprocessed_data(config: Dict) -> Optional[Dict]:
    """
    Загружает предобработанные данные из кэша.
    Это НАМНОГО быстрее, чем парсить CSV каждый раз.
    
    Returns:
        Dict с текстами, метками и метаданными или None если кэш не найден
    """
    category_id = config.get("category_id", 14)
    cache_file = PREPROCESSED_DATA_CACHE_FILE.replace(".pkl", f"_cat{category_id}.pkl")
    
    if os.path.exists(cache_file):
        try:
            print(f"Loading preprocessed data from cache...")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            print(f"✓ Loaded {len(data['texts'])} reviews from cache")
            return data
        except Exception as e:
            print(f"Cache load failed: {e}")
    
    return None


def save_preprocessed_data(config: Dict, data: Dict):
    """Сохраняет предобработанные данные в кэш."""
    category_id = config.get("category_id", 14)
    cache_file = PREPROCESSED_DATA_CACHE_FILE.replace(".pkl", f"_cat{category_id}.pkl")
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ Saved preprocessed data to cache ({len(data['texts'])} reviews)")
    except Exception as e:
        print(f"Could not save cache: {e}")


def preprocess_category_data(dataset, category_id: int) -> Dict:
    """
    Извлекает и предобрабатывает данные для конкретной категории.
    Это делается один раз, потом используется быстрый кэш.
    
    Returns:
        Dict с texts, labels, user_ids для категории
    """
    print(f"Preprocessing category {category_id}...")
    
    train_data = dataset.get_subset('train')
    
    # Индексы метаданных
    metadata_fields = dataset.metadata_fields
    user_idx = metadata_fields.index('user')
    category_idx = metadata_fields.index('category')
    
    metadata_array = train_data.metadata_array
    y_array = train_data.y_array
    
    # Фильтруем по категории
    category_mask = (metadata_array[:, category_idx] == category_id)
    category_indices = np.where(category_mask)[0]
    
    print(f"Found {len(category_indices)} examples in category")
    
    # Извлекаем данные
    texts = []
    labels = []
    user_ids = []
    original_indices = []
    
    for i, idx in enumerate(category_indices):
        if i % 500 == 0:
            print(f"  Processing {i}/{len(category_indices)}...")
        
        x, y, metadata = train_data[idx]
        texts.append(x)  # Текст отзыва
        labels.append(int(y) + 1)  # 0-4 -> 1-5
        user_ids.append(int(metadata_array[idx, user_idx]))
        original_indices.append(int(idx))
    
    print(f"✓ Preprocessed {len(texts)} reviews")
    
    return {
        'texts': texts,
        'labels': labels,
        'user_ids': user_ids,
        'original_indices': original_indices,
        'category_id': category_id,
    }


def create_splits_from_preprocessed(
    preprocessed: Dict,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, Dict]:
    """
    Создаёт user-disjoint split из предобработанных данных.
    """
    np.random.seed(seed)
    random.seed(seed)
    
    texts = preprocessed['texts']
    labels = preprocessed['labels']
    user_ids = preprocessed['user_ids']
    
    # Уникальные пользователи
    unique_users = list(set(user_ids))
    np.random.shuffle(unique_users)
    
    n_users = len(unique_users)
    n_train = int(n_users * train_ratio)
    n_val = int(n_users * val_ratio)
    
    train_users = set(unique_users[:n_train])
    val_users = set(unique_users[n_train:n_train + n_val])
    test_users = set(unique_users[n_train + n_val:])
    
    print(f"Split users: train={len(train_users)}, val={len(val_users)}, test={len(test_users)}")
    
    # Разделяем данные по пользователям
    train_indices = []
    val_indices = []
    test_indices = []
    
    for i, user_id in enumerate(user_ids):
        if user_id in train_users:
            train_indices.append(i)
        elif user_id in val_users:
            val_indices.append(i)
        else:
            test_indices.append(i)
    
    def create_split_data(indices):
        return {
            'indices': np.array(indices),
            'y': np.array([labels[i] for i in indices]),
        }
    
    print(f"Split sizes: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
    
    return {
        'train': create_split_data(train_indices),
        'validation': create_split_data(val_indices),
        'test': create_split_data(test_indices),
    }


def load_wilds_dataset(config: Dict) -> Tuple[Any, Dict]:
    """
    Загружает WILDS Amazon датасет и фильтрует по категории.
    
    Returns:
        Tuple[dataset, metadata_map]: датасет и маппинг метаданных
    """
    from wilds import get_dataset
    
    data_root = config.get("data_root", "./data")
    evaluator_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(evaluator_dir, data_root)
    
    print(f"Loading WILDS Amazon dataset from {data_path}...")
    print("(First load is slow due to CSV parsing, subsequent loads use cache)")
    dataset = get_dataset(dataset='amazon', download=False, root_dir=data_path)
    
    return dataset, dataset._metadata_map if hasattr(dataset, '_metadata_map') else {}


def create_user_disjoint_split(
    dataset,
    category_id: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, Dict]:
    """
    Создаёт user-disjoint split для указанной категории.
    Отзывы одного пользователя не попадают в разные сплиты.
    
    Returns:
        Dict с train/validation/test индексами и метаданными
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Получаем train subset
    train_data = dataset.get_subset('train')
    
    # Индексы полей в метаданных
    metadata_fields = dataset.metadata_fields
    user_idx = metadata_fields.index('user')
    category_idx = metadata_fields.index('category')
    y_idx = metadata_fields.index('y')
    
    # Фильтруем по категории
    metadata_array = train_data.metadata_array
    y_array = train_data.y_array
    
    category_mask = (metadata_array[:, category_idx] == category_id)
    category_indices = np.where(category_mask)[0]
    
    if len(category_indices) == 0:
        raise ValueError(f"No examples found for category_id={category_id}")
    
    print(f"Found {len(category_indices)} examples in category {category_id}")
    
    # Получаем уникальных пользователей в этой категории
    category_users = metadata_array[category_indices, user_idx].numpy()
    unique_users = np.unique(category_users)
    
    print(f"Found {len(unique_users)} unique users in category")
    
    # Разделяем ПОЛЬЗОВАТЕЛЕЙ на train/val/test
    np.random.shuffle(unique_users)
    
    n_users = len(unique_users)
    n_train = int(n_users * train_ratio)
    n_val = int(n_users * val_ratio)
    
    train_users = set(unique_users[:n_train])
    val_users = set(unique_users[n_train:n_train + n_val])
    test_users = set(unique_users[n_train + n_val:])
    
    print(f"Split users: train={len(train_users)}, val={len(val_users)}, test={len(test_users)}")
    
    # Разделяем примеры по пользователям
    train_indices = []
    val_indices = []
    test_indices = []
    
    for idx in category_indices:
        user_id = metadata_array[idx, user_idx].item()
        if user_id in train_users:
            train_indices.append(idx)
        elif user_id in val_users:
            val_indices.append(idx)
        elif user_id in test_users:
            test_indices.append(idx)
    
    # Создаём словарь с данными для каждого сплита
    def create_split_data(indices):
        return {
            'indices': np.array(indices),
            'y': y_array[indices].numpy() if hasattr(y_array[indices], 'numpy') else np.array([y_array[i] for i in indices]),
            'users': np.array([metadata_array[i, user_idx].item() for i in indices]),
        }
    
    splits = {
        'train': create_split_data(train_indices),
        'validation': create_split_data(val_indices),
        'test': create_split_data(test_indices),
    }
    
    # Статистика по классам
    for split_name, split_data in splits.items():
        class_counts = Counter(split_data['y'])
        print(f"  {split_name}: {len(split_data['indices'])} examples, classes: {dict(sorted(class_counts.items()))}")
    
    return splits


def stratified_sample(
    split_data: Dict,
    n_samples: int,
    seed: int = None
) -> List[int]:
    """
    Стратифицированная выборка по классам (случайная).
    Старается выбрать равное количество примеров из каждого класса.
    
    Args:
        split_data: словарь с 'indices' и 'y'
        n_samples: сколько примеров выбрать
        seed: random seed
        
    Returns:
        List[int]: индексы выбранных примеров (в оригинальном датасете)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    indices = split_data['indices']
    labels = split_data['y']
    
    # Группируем по классам
    class_indices = {}
    for i, (idx, label) in enumerate(zip(indices, labels)):
        label_int = int(label)
        if label_int not in class_indices:
            class_indices[label_int] = []
        class_indices[label_int].append(idx)
    
    # Определяем сколько примеров брать из каждого класса
    num_classes = len(class_indices)
    samples_per_class = max(1, n_samples // num_classes)
    
    sampled_indices = []
    
    for class_label in sorted(class_indices.keys()):
        class_idx = class_indices[class_label]
        n_available = len(class_idx)
        n_to_sample = min(samples_per_class, n_available)
        
        if n_available > 0:
            selected = np.random.choice(class_idx, n_to_sample, replace=False)
            sampled_indices.extend(selected)
    
    # Если недобрали — добираем случайно из оставшихся
    if len(sampled_indices) < n_samples:
        remaining = list(set(indices) - set(sampled_indices))
        n_remaining = min(n_samples - len(sampled_indices), len(remaining))
        if n_remaining > 0:
            additional = np.random.choice(remaining, n_remaining, replace=False)
            sampled_indices.extend(additional)
    
    np.random.shuffle(sampled_indices)
    return sampled_indices[:n_samples]


def rotation_sample(
    split_data: Dict,
    n_samples: int,
    iteration: int,
    base_seed: int = 42
) -> List[int]:
    """
    Ротационная выборка примеров - детерминированный обход данных.
    На каждой итерации используются РАЗНЫЕ примеры без повторов.
    После прохода всех данных - начинается новый цикл.
    
    Преимущества:
    - Каждый пример будет использован равное количество раз
    - Нет случайных пропусков важных примеров
    - Детерминированность: одинаковые результаты при одинаковых входных данных
    - Стратификация сохраняется внутри каждой ротации
    
    Args:
        split_data: словарь с 'indices' и 'y'
        n_samples: сколько примеров выбрать
        iteration: номер текущей итерации
        base_seed: базовый seed для начальной сортировки
        
    Returns:
        List[int]: индексы выбранных примеров
    """
    indices = list(split_data['indices'])
    labels = list(split_data['y'])
    n_total = len(indices)
    
    if n_total == 0:
        return []
    
    # Создаём детерминированный порядок обхода (один раз на основе base_seed)
    np.random.seed(base_seed)
    
    # Группируем по классам для стратификации
    class_indices = {}
    for idx, label in zip(indices, labels):
        label_int = int(label)
        if label_int not in class_indices:
            class_indices[label_int] = []
        class_indices[label_int].append(idx)
    
    # Перемешиваем внутри каждого класса детерминированно
    for class_label in class_indices:
        np.random.shuffle(class_indices[class_label])
    
    # Создаём стратифицированный порядок: берём по одному из каждого класса по очереди
    sorted_classes = sorted(class_indices.keys())
    stratified_order = []
    max_len = max(len(v) for v in class_indices.values())
    
    for i in range(max_len):
        for class_label in sorted_classes:
            if i < len(class_indices[class_label]):
                stratified_order.append(class_indices[class_label][i])
    
    n_stratified = len(stratified_order)
    
    # Вычисляем offset для текущей итерации
    # Ротация: каждая итерация сдвигается на n_samples
    rotation_offset = (iteration * n_samples) % n_stratified
    
    # Выбираем n_samples с учётом циклического переноса
    sampled_indices = []
    for i in range(n_samples):
        idx = (rotation_offset + i) % n_stratified
        sampled_indices.append(stratified_order[idx])
    
    return sampled_indices


def get_dataset_splits() -> Dict:
    """
    Получает или создаёт разбиение датасета.
    Использует оптимизированный кэш предобработанных данных.
    """
    global _DATASET_SPLITS
    
    # Проверяем кэш в памяти
    if _DATASET_SPLITS and 'preprocessed' in _DATASET_SPLITS:
        return _DATASET_SPLITS
    
    config = load_prompt_config()
    category_id = config.get("category_id", 14)
    
    # Проверяем предобработанный кэш
    preprocessed = load_preprocessed_data(config)
    
    if preprocessed is not None:
        # Создаём разбиение из предобработанных данных
        splits = create_splits_from_preprocessed(
            preprocessed,
            train_ratio=config.get("train_ratio", 0.7),
            val_ratio=config.get("validation_ratio", 0.15),
            test_ratio=config.get("test_ratio", 0.15),
            seed=config.get("split_seed", 42)
        )
        
        _DATASET_SPLITS = {
            'preprocessed': preprocessed,
            'splits': splits,
            'config': config,
        }
        return _DATASET_SPLITS
    
    # Первый запуск - загружаем полный датасет и создаём кэш
    print("=" * 60)
    print("First run: building preprocessed cache (this takes ~2-3 min)...")
    print("Subsequent runs will be MUCH faster!")
    print("=" * 60)
    
    dataset, metadata_map = load_wilds_dataset(config)
    
    # Предобрабатываем и кэшируем данные для категории
    preprocessed = preprocess_category_data(dataset, category_id)
    save_preprocessed_data(config, preprocessed)
    
    # Создаём разбиение
    splits = create_splits_from_preprocessed(
        preprocessed,
        train_ratio=config.get("train_ratio", 0.7),
        val_ratio=config.get("validation_ratio", 0.15),
        test_ratio=config.get("test_ratio", 0.15),
        seed=config.get("split_seed", 42)
    )
    
    _DATASET_SPLITS = {
        'preprocessed': preprocessed,
        'splits': splits,
        'config': config,
    }
    
    return _DATASET_SPLITS


def parse_star_rating(response: str) -> Optional[int]:
    """
    Извлекает звёздный рейтинг (1-5) из ответа модели.
    
    Returns:
        int (1-5) или None если не удалось распарсить
    """
    response = response.strip()
    
    # Паттерны для поиска числа
    patterns = [
        r'^([1-5])$',                           # Просто число
        r'^([1-5])\s*(?:star|stars|⭐)?',       # "5 stars" или "5⭐"
        r'(?:rating|score|answer)[:\s]+([1-5])',  # "rating: 5"
        r'\b([1-5])\s*(?:out of 5|/5)',         # "4 out of 5" или "4/5"
        r'^\s*\**\s*([1-5])\s*\**\s*$',         # "*5*" с возможными звёздочками
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return int(match.group(1))
    
    # Fallback: ищем любое число 1-5 в тексте
    numbers = re.findall(r'\b([1-5])\b', response)
    if numbers:
        # Берём последнее найденное число (обычно это ответ)
        return int(numbers[-1])
    
    return None


# ============================================================================
# AUTOMATIC FEW-SHOT EXAMPLES VIA EMBEDDINGS
# ============================================================================

def get_embedding_model():
    """Ленивая загрузка модели эмбеддингов."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None and SentenceTransformer is not None:
        try:
            # Используем BGE-M3 модель
            print("Загрузка модели эмбеддингов: BAAI/bge-m3...")
            _EMBEDDING_MODEL = SentenceTransformer('BAAI/bge-m3')
            print("✓ Модель эмбеддингов BGE-M3 загружена")
            
            # Предыдущая модель (закомментирована):
            # print("Загрузка модели эмбеддингов: all-mpnet-base-v2...")
            # _EMBEDDING_MODEL = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            # print("✓ Модель эмбеддингов загружена")
        except Exception as e:
            print(f"Ошибка загрузки модели эмбеддингов: {e}")
            _EMBEDDING_MODEL = False  # Помечаем как недоступную
    return _EMBEDDING_MODEL if _EMBEDDING_MODEL is not False else None


def precompute_embeddings_for_split(split_name: str) -> Dict[str, Any]:
    """
    Предрассчитывает эмбеддинги для указанного сплита (train, validation, test).
    
    Args:
        split_name: Имя сплита ('train', 'validation', 'test')
    
    Returns:
        Dict с embeddings, texts, labels, indices
    """
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers not available")
    
    # Загружаем данные
    splits_data = get_dataset_splits()
    splits = splits_data['splits']
    preprocessed = splits_data['preprocessed']
    
    if split_name not in splits:
        raise ValueError(f"Unknown split: {split_name}. Available: {list(splits.keys())}")
    
    split = splits[split_name]
    split_indices = split['indices']
    
    texts = preprocessed['texts']
    labels = preprocessed['labels']
    
    # Извлекаем данные для сплита
    split_texts = [texts[i] for i in split_indices]
    split_labels = [labels[i] for i in split_indices]
    
    # Загружаем модель
    model = get_embedding_model()
    if model is None:
        raise RuntimeError("Embedding model not available")
    
    # Вычисляем эмбеддинги
    print(f"Вычисление эмбеддингов для {len(split_texts)} {split_name} примеров...")
    embeddings = model.encode(
        split_texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    print(f"✓ Вычислено эмбеддингов для {split_name}: {embeddings.shape}")
    
    return {
        'embeddings': embeddings,
        'texts': split_texts,
        'labels': split_labels,
        'indices': split_indices.tolist() if hasattr(split_indices, 'tolist') else list(split_indices),
    }


def precompute_train_embeddings() -> Dict[str, Any]:
    """
    Предрассчитывает эмбеддинги для всех train примеров.
    
    Returns:
        Dict с embeddings, texts, labels, indices
    """
    return precompute_embeddings_for_split('train')


def precompute_val_embeddings() -> Dict[str, Any]:
    """
    Предрассчитывает эмбеддинги для всех validation примеров.
    
    Returns:
        Dict с embeddings, texts, labels, indices
    """
    return precompute_embeddings_for_split('validation')


def precompute_test_embeddings() -> Dict[str, Any]:
    """
    Предрассчитывает эмбеддинги для всех test примеров.
    
    Returns:
        Dict с embeddings, texts, labels, indices
    """
    return precompute_embeddings_for_split('test')


def precompute_all_embeddings():
    """
    Предрассчитывает эмбеддинги для всех сплитов (train, validation, test).
    """
    print("=" * 80)
    print("Предрасчет эмбеддингов для всех сплитов")
    print("=" * 80)
    
    # Train
    train_data = precompute_train_embeddings()
    save_train_embeddings(train_data)
    
    # Validation
    val_data = precompute_val_embeddings()
    try:
        with open(VAL_EMBEDDINGS_CACHE_FILE, "wb") as f:
            pickle.dump(val_data, f)
        print(f"✓ Validation эмбеддинги сохранены: {VAL_EMBEDDINGS_CACHE_FILE}")
    except Exception as e:
        print(f"Ошибка сохранения validation эмбеддингов: {e}")
    
    # Test
    test_data = precompute_test_embeddings()
    try:
        with open(TEST_EMBEDDINGS_CACHE_FILE, "wb") as f:
            pickle.dump(test_data, f)
        print(f"✓ Test эмбеддинги сохранены: {TEST_EMBEDDINGS_CACHE_FILE}")
    except Exception as e:
        print(f"Ошибка сохранения test эмбеддингов: {e}")
    
    print("=" * 80)
    print("✓ Предрасчет завершён!")
    print("=" * 80)


def load_train_embeddings() -> Optional[Dict[str, Any]]:
    """Загружает предрассчитанные train эмбеддинги из кэша."""
    if os.path.exists(TRAIN_EMBEDDINGS_CACHE_FILE):
        try:
            with open(TRAIN_EMBEDDINGS_CACHE_FILE, "rb") as f:
                data = pickle.load(f)
            print(f"✓ Загружены train эмбеддинги из кэша: {len(data['texts'])} примеров")
            return data
        except Exception as e:
            print(f"Ошибка загрузки train эмбеддингов из кэша: {e}")
    return None


def load_val_embeddings() -> Optional[Dict[str, Any]]:
    """Загружает предрассчитанные validation эмбеддинги из кэша."""
    if os.path.exists(VAL_EMBEDDINGS_CACHE_FILE):
        try:
            with open(VAL_EMBEDDINGS_CACHE_FILE, "rb") as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            print(f"Ошибка загрузки validation эмбеддингов из кэша: {e}")
    return None


def load_test_embeddings() -> Optional[Dict[str, Any]]:
    """Загружает предрассчитанные test эмбеддинги из кэша."""
    if os.path.exists(TEST_EMBEDDINGS_CACHE_FILE):
        try:
            with open(TEST_EMBEDDINGS_CACHE_FILE, "rb") as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            print(f"Ошибка загрузки test эмбеддингов из кэша: {e}")
    return None


def load_all_embeddings() -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]:
    """
    Загружает все предрассчитанные эмбеддинги (train, val, test).
    
    Returns:
        Tuple[(train_data, val_data, test_data)]
    """
    train_data = load_train_embeddings()
    val_data = load_val_embeddings()
    test_data = load_test_embeddings()
    
    if val_data:
        print(f"✓ Загружены validation эмбеддинги: {len(val_data['texts'])} примеров")
    if test_data:
        print(f"✓ Загружены test эмбеддинги: {len(test_data['texts'])} примеров")
    
    return train_data, val_data, test_data


def build_embeddings_cache_by_text() -> Dict[str, np.ndarray]:
    """
    Создаёт кэш эмбеддингов по тексту отзыва для быстрого доступа.
    Загружает все предрассчитанные эмбеддинги (train, val, test) и создаёт словарь.
    
    Returns:
        Dict[text -> embedding] для всех сплитов
    """
    global _EMBEDDINGS_BY_TEXT
    
    if _EMBEDDINGS_BY_TEXT is not None:
        return _EMBEDDINGS_BY_TEXT
    
    train_data, val_data, test_data = load_all_embeddings()
    
    embeddings_by_text = {}
    
    # Добавляем train эмбеддинги
    if train_data:
        for text, embedding in zip(train_data['texts'], train_data['embeddings']):
            embeddings_by_text[text] = embedding
    
    # Добавляем val эмбеддинги
    if val_data:
        for text, embedding in zip(val_data['texts'], val_data['embeddings']):
            embeddings_by_text[text] = embedding
    
    # Добавляем test эмбеддинги
    if test_data:
        for text, embedding in zip(test_data['texts'], test_data['embeddings']):
            embeddings_by_text[text] = embedding
    
    _EMBEDDINGS_BY_TEXT = embeddings_by_text
    
    print(f"✓ Создан кэш эмбеддингов: {len(embeddings_by_text)} отзывов")
    
    return embeddings_by_text


def save_train_embeddings(data: Dict[str, Any]):
    """Сохраняет эмбеддинги в кэш."""
    try:
        with open(TRAIN_EMBEDDINGS_CACHE_FILE, "wb") as f:
            pickle.dump(data, f)
        print(f"✓ Эмбеддинги сохранены в кэш: {TRAIN_EMBEDDINGS_CACHE_FILE}")
    except Exception as e:
        print(f"Ошибка сохранения эмбеддингов: {e}")


def build_similarity_index(embeddings: np.ndarray) -> Any:
    """
    Создаёт индекс для быстрого поиска похожих примеров.
    
    Args:
        embeddings: Массив эмбеддингов (n_samples, embedding_dim)
    
    Returns:
        Обученный NearestNeighbors индекс
    """
    if NearestNeighbors is None:
        raise ImportError("sklearn not available")
    
    print("Создание индекса для поиска похожих примеров...")
    # Используем cosine similarity
    index = NearestNeighbors(
        n_neighbors=10,  # Максимум примеров
        metric='cosine',
        algorithm='brute'  # Для cosine similarity
    )
    index.fit(embeddings)
    print("✓ Индекс создан")
    return index


def load_or_create_embeddings_index() -> Tuple[Optional[Dict[str, Any]], Optional[Any]]:
    """
    Загружает или создаёт эмбеддинги и индекс.
    Также предрассчитывает эмбеддинги для val и test, если их нет.
    
    Returns:
        Tuple[(embeddings_data, index) или (None, None) если недоступно]
    """
    global _TRAIN_EMBEDDINGS_DATA, _TRAIN_EMBEDDINGS_INDEX
    
    if _TRAIN_EMBEDDINGS_DATA is not None and _TRAIN_EMBEDDINGS_INDEX is not None:
        # Проверяем, есть ли val и test эмбеддинги, если нет - предрассчитываем
        if not os.path.exists(VAL_EMBEDDINGS_CACHE_FILE) or not os.path.exists(TEST_EMBEDDINGS_CACHE_FILE):
            try:
                print("Предрассчитываем эмбеддинги для val и test...")
                val_data = precompute_val_embeddings()
                test_data = precompute_test_embeddings()
                with open(VAL_EMBEDDINGS_CACHE_FILE, "wb") as f:
                    pickle.dump(val_data, f)
                with open(TEST_EMBEDDINGS_CACHE_FILE, "wb") as f:
                    pickle.dump(test_data, f)
                print("✓ Val и test эмбеддинги предрассчитаны и сохранены")
            except Exception as e:
                print(f"Не удалось предрассчитать val/test эмбеддинги: {e}")
        
        return _TRAIN_EMBEDDINGS_DATA, _TRAIN_EMBEDDINGS_INDEX
    
    # Пытаемся загрузить из кэша
    embeddings_data = load_train_embeddings()
    
    if embeddings_data is None:
        # Предрассчитываем train
        try:
            embeddings_data = precompute_train_embeddings()
            save_train_embeddings(embeddings_data)
        except Exception as e:
            print(f"Не удалось предрассчитать train эмбеддинги: {e}")
            return None, None
    
    # Предрассчитываем val и test, если их нет
    if not os.path.exists(VAL_EMBEDDINGS_CACHE_FILE):
        try:
            val_data = precompute_val_embeddings()
            with open(VAL_EMBEDDINGS_CACHE_FILE, "wb") as f:
                pickle.dump(val_data, f)
            print(f"✓ Validation эмбеддинги сохранены: {VAL_EMBEDDINGS_CACHE_FILE}")
        except Exception as e:
            print(f"Не удалось предрассчитать validation эмбеддинги: {e}")
    
    if not os.path.exists(TEST_EMBEDDINGS_CACHE_FILE):
        try:
            test_data = precompute_test_embeddings()
            with open(TEST_EMBEDDINGS_CACHE_FILE, "wb") as f:
                pickle.dump(test_data, f)
            print(f"✓ Test эмбеддинги сохранены: {TEST_EMBEDDINGS_CACHE_FILE}")
        except Exception as e:
            print(f"Не удалось предрассчитать test эмбеддинги: {e}")
    
    # Создаём или загружаем индекс
    if os.path.exists(TRAIN_EMBEDDINGS_INDEX_FILE):
        try:
            with open(TRAIN_EMBEDDINGS_INDEX_FILE, "rb") as f:
                index = pickle.load(f)
            print("✓ Индекс загружен из кэша")
        except Exception as e:
            print(f"Ошибка загрузки индекса: {e}")
            index = build_similarity_index(embeddings_data['embeddings'])
            try:
                with open(TRAIN_EMBEDDINGS_INDEX_FILE, "wb") as f:
                    pickle.dump(index, f)
            except:
                pass
    else:
        index = build_similarity_index(embeddings_data['embeddings'])
        try:
            with open(TRAIN_EMBEDDINGS_INDEX_FILE, "wb") as f:
                pickle.dump(index, f)
        except:
            pass
    
    _TRAIN_EMBEDDINGS_DATA = embeddings_data
    _TRAIN_EMBEDDINGS_INDEX = index
    
    return embeddings_data, index


def find_similar_examples(
    review_text: str,
    similarity_threshold: float = 0.85,
    max_examples: int = 10,
    min_examples: int = 3
) -> List[Dict[str, Any]]:
    """
    Находит похожие примеры из train датасета для заданного отзыва.
    Использует предрассчитанные эмбеддинги, если доступны.
    Гарантирует возврат минимум min_examples примеров, даже если similarity < threshold.
    
    Args:
        review_text: Текст отзыва
        similarity_threshold: Минимальная cosine similarity (0.85 = очень похожие)
        max_examples: Максимум примеров для возврата
        min_examples: Минимум примеров для возврата (даже если similarity < threshold)
    
    Returns:
        Список словарей с 'text', 'label', 'similarity'
    """
    embeddings_data, index = load_or_create_embeddings_index()
    
    if embeddings_data is None or index is None:
        return []  # Недоступно
    
    # Пытаемся использовать предрассчитанный эмбеддинг
    embeddings_cache = build_embeddings_cache_by_text()
    
    if review_text in embeddings_cache:
        # Используем предрассчитанный эмбеддинг
        review_embedding = embeddings_cache[review_text].reshape(1, -1)
    else:
        # Вычисляем эмбеддинг на лету (если отзыв не в кэше)
        model = get_embedding_model()
        if model is None:
            return []
        review_embedding = model.encode([review_text], convert_to_numpy=True)
    
    # Ищем ближайших соседей (берём больше, чтобы гарантировать min_examples)
    n_neighbors = max(max_examples + 5, min_examples + 10, len(embeddings_data['texts']))
    n_neighbors = min(n_neighbors, len(embeddings_data['texts']))
    
    distances, indices = index.kneighbors(
        review_embedding,
        n_neighbors=n_neighbors,
        return_distance=True
    )
    
    # Собираем все примеры с их similarity
    all_candidates = []
    seen_texts = set()
    
    for dist, idx in zip(distances[0], indices[0]):
        similarity = 1.0 - dist  # Преобразуем distance в similarity
        text = embeddings_data['texts'][idx]
        label = embeddings_data['labels'][idx]
        
        # Исключаем дубликаты и сам отзыв
        if text not in seen_texts and text != review_text:
            seen_texts.add(text)
            all_candidates.append({
                'text': text,
                'label': label,
                'similarity': float(similarity)
            })
    
    # Сортируем по similarity (от большего к меньшему)
    all_candidates.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Сначала берём примеры выше порога
    similar_examples = [ex for ex in all_candidates if ex['similarity'] >= similarity_threshold]
    
    # Если не хватает до min_examples, добавляем лучшие из оставшихся
    if len(similar_examples) < min_examples:
        # Берём лучшие примеры, даже если они ниже порога
        remaining = [ex for ex in all_candidates if ex['similarity'] < similarity_threshold]
        needed = min_examples - len(similar_examples)
        # Берём столько, сколько нужно, но не больше доступных
        similar_examples.extend(remaining[:needed])
    
    # Гарантируем минимум min_examples (если есть достаточно кандидатов)
    if len(all_candidates) >= min_examples and len(similar_examples) < min_examples:
        # Если всё ещё не хватает, берём лучшие из всех кандидатов
        similar_examples = all_candidates[:min_examples]
    
    # Ограничиваем максимумом
    similar_examples = similar_examples[:max_examples]
    
    # Финальная проверка: убеждаемся, что сам отзыв не попал в результаты
    # (дополнительная защита на случай edge cases)
    similar_examples = [ex for ex in similar_examples if ex['text'] != review_text]
    
    return similar_examples


def format_few_shot_examples(examples: List[Dict[str, Any]]) -> str:
    """
    Форматирует few-shot примеры для вставки в промпт.
    Включает информацию о similarity каждого примера.
    
    Args:
        examples: Список примеров с 'text', 'label', 'similarity'
    
    Returns:
        Отформатированная строка с примерами
    """
    if not examples:
        return ""
    
    formatted = "Few-shot examples (similar reviews from training data):\n\n"
    
    for i, example in enumerate(examples, 1):
        similarity = example.get('similarity', 0.0)
        formatted += f"Example {i} (similarity: {similarity:.3f}):\n"
        formatted += f"Review: \"{example['text']}\"\n"
        formatted += f"Rating: {example['label']} stars\n\n"
    
    return formatted


def query_model(prompt: str, review: str, max_retries: int = 3, use_automatic_few_shot: bool = True) -> Tuple[str, bool]:
    """
    Отправляет запрос к модели.
    
    Args:
        prompt: Текст промпта (может содержать {review})
        review: Текст отзыва для оценки
        max_retries: Максимум попыток при ошибке
        use_automatic_few_shot: Использовать ли автоматические few-shot примеры
    
    Returns:
        Tuple[response_text, success]
    """
    # Добавляем автоматические few-shot примеры, если включено
    if use_automatic_few_shot:
        similar_examples = find_similar_examples(
            review,
            similarity_threshold=0.85,
            max_examples=10,
            min_examples=3  # Гарантируем минимум 3 примера
        )
        
        if similar_examples:
            few_shot_section = format_few_shot_examples(similar_examples)
            # Вставляем few-shot примеры перед {review}
            # Важно: few-shot должны быть после инструкций, но перед review
            if "{review}" in prompt:
                # Разделяем промпт на части до и после {review}
                parts = prompt.split("{review}", 1)
                # Вставляем few-shot между инструкциями и review
                formatted_prompt = parts[0] + "\n\n" + few_shot_section + "\n" + "{review}" + parts[1]
                # Заменяем {review} на реальный текст отзыва
                formatted_prompt = formatted_prompt.replace("{review}", review)
            else:
                # Если нет {review}, добавляем few-shot перед промптом и добавляем отзыв в конец
                formatted_prompt = few_shot_section + "\n\n" + prompt + "\n\nReview: " + review
        else:
            formatted_prompt = prompt.replace("{review}", review)
    else:
        formatted_prompt = prompt.replace("{review}", review)
    
    for attempt in range(max_retries):
        try:
            response = test_model.chat.completions.create(
                model=TASK_MODEL_NAME,
                messages=[
                    {"role": "user", "content": formatted_prompt}
                ],
                max_tokens=50,  # Короткий ответ для classification
                temperature=0.0,  # Детерминистичный вывод
            )
            
            return response.choices[0].message.content.strip(), True
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"Error querying model: {e}")
                return "", False
    
    return "", False


def evaluate_on_samples(
    prompt: str,
    sample_indices: List[int],
    preprocessed: Dict,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Оценивает промпт на заданных примерах.
    
    Args:
        prompt: Текст промпта
        sample_indices: Индексы в preprocessed данных
        preprocessed: Dict с texts, labels из предобработанного кэша
        verbose: Выводить детальную информацию
    
    Returns:
        Dict с метриками: accuracy, per_class_accuracy, predictions, etc.
    """
    texts = preprocessed['texts']
    labels = preprocessed['labels']
    
    correct = 0
    total = 0
    predictions = []
    ground_truth = []
    errors = []
    
    for idx in sample_indices:
        try:
            review = texts[idx]
            true_rating = labels[idx]  # Уже 1-5 после предобработки
            
            response, success = query_model(prompt, review)
            
            if not success:
                errors.append(idx)
                continue
            
            predicted_rating = parse_star_rating(response)
            
            if predicted_rating is not None:
                predictions.append(predicted_rating)
                ground_truth.append(true_rating)
                
                if predicted_rating == true_rating:
                    correct += 1
                total += 1
                
                if verbose:
                    print(f"  [{idx}] True: {true_rating}, Pred: {predicted_rating}, {'✓' if predicted_rating == true_rating else '✗'}")
            else:
                errors.append(idx)
                if verbose:
                    print(f"  [{idx}] Could not parse response: {response[:50]}...")
                    
        except Exception as e:
            errors.append(idx)
            if verbose:
                print(f"  [{idx}] Error: {e}")
    
    # Вычисляем метрики
    accuracy = correct / total if total > 0 else 0.0
    
    # Per-class accuracy
    per_class_correct = {i: 0 for i in range(1, 6)}
    per_class_total = {i: 0 for i in range(1, 6)}
    
    for pred, true in zip(predictions, ground_truth):
        per_class_total[true] += 1
        if pred == true:
            per_class_correct[true] += 1
    
    per_class_accuracy = {
        f"class_{i}_accuracy": per_class_correct[i] / per_class_total[i] if per_class_total[i] > 0 else 0.0
        for i in range(1, 6)
    }
    
    # Mean Absolute Error
    mae = np.mean([abs(p - t) for p, t in zip(predictions, ground_truth)]) if predictions else 0.0
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "mae": mae,
        "predictions": predictions,
        "ground_truth": ground_truth,
        "errors": len(errors),
        **per_class_accuracy,
    }


def get_adaptive_sample_size(iteration: int, max_iterations: int, config: Dict) -> Tuple[int, int]:
    """
    Возвращает размеры выборок для текущей итерации.
    В начале — меньше примеров для быстрой фильтрации.
    К концу — больше примеров для точной оценки.
    
    Returns:
        Tuple[stage1_samples, stage2_samples]
    """
    stage1_base = config.get("stage1_samples", 10)
    stage2_base = config.get("stage2_samples", 20)
    
    # Прогресс эволюции (0.0 - 1.0)
    progress = iteration / max(max_iterations, 1)
    
    # Увеличиваем размер выборки к концу (x1.0 -> x1.5)
    multiplier = 1.0 + 0.5 * progress
    
    stage1 = int(stage1_base * multiplier)
    stage2 = int(stage2_base * multiplier)
    
    return stage1, stage2


def _evaluate_stage1_internal(prompt: str, splits_data: Dict, config: Dict, seed: int) -> Dict[str, Any]:
    """
    Stage 1 internal: Быстрая оценка на малой выборке.
    Использует РОТАЦИЮ примеров вместо случайной выборки.
    """
    iteration = get_current_iteration_from_checkpoint()
    stage1_samples, _ = get_adaptive_sample_size(iteration, _MAX_ITERATIONS, config)
    
    train_split = splits_data['splits']['train']
    
    # Используем ротацию вместо случайной выборки
    sample_indices = rotation_sample(
        train_split, 
        stage1_samples, 
        iteration=iteration,
        base_seed=config.get("split_seed", 42)
    )
    
    results = evaluate_on_samples(
        prompt,
        sample_indices,
        splits_data['preprocessed'],
        verbose=False
    )
    
    results['stage'] = 1
    results['samples_used'] = len(sample_indices)
    results['rotation_iteration'] = iteration
    
    return results


def _evaluate_stage2_internal(prompt: str, splits_data: Dict, config: Dict, seed: int) -> Dict[str, Any]:
    """
    Stage 2 internal: Расширенная оценка на большей выборке.
    Использует РОТАЦИЮ примеров с offset от stage1.
    """
    iteration = get_current_iteration_from_checkpoint()
    _, stage2_samples = get_adaptive_sample_size(iteration, _MAX_ITERATIONS, config)
    
    train_split = splits_data['splits']['train']
    
    # Используем ротацию с offset для stage2 (чтобы не пересекаться с stage1)
    # Умножаем итерацию на 2 чтобы stage2 брал другие примеры
    sample_indices = rotation_sample(
        train_split, 
        stage2_samples, 
        iteration=iteration * 2 + 1,  # Offset для отличия от stage1
        base_seed=config.get("split_seed", 42)
    )
    
    results = evaluate_on_samples(
        prompt,
        sample_indices,
        splits_data['preprocessed'],
        verbose=False
    )
    
    results['stage'] = 2
    results['samples_used'] = len(sample_indices)
    results['rotation_iteration'] = iteration
    
    return results


def _evaluate_validation_internal(prompt: str, splits_data: Dict, config: Dict) -> Dict[str, Any]:
    """
    Валидация на 100 примерах из validation split.
    Используется стратифицированная выборка (не ротация) для стабильности.
    """
    val_samples = config.get("validation_samples", 100)
    
    val_split = splits_data['splits']['validation']
    
    # Для валидации используем фиксированную стратифицированную выборку
    # (одни и те же примеры каждый раз для сравнимости)
    sample_indices = stratified_sample(val_split, val_samples, seed=999)
    
    results = evaluate_on_samples(
        prompt,
        sample_indices,
        splits_data['preprocessed'],
        verbose=False
    )
    
    results['stage'] = 'validation'
    results['samples_used'] = len(sample_indices)
    
    return results


def evaluate_validation(prompt: str, splits_data: Dict, config: Dict) -> Dict[str, Any]:
    """
    Оценка на валидационном сплите (legacy wrapper).
    """
    return _evaluate_validation_internal(prompt, splits_data, config)


def evaluate_stage1(prompt_path: str) -> Dict[str, Any]:
    """
    Stage 1 evaluation: Quick evaluation with adaptive sample size.
    Called directly by OpenEvolve when cascade_evaluation is enabled.
    
    Args:
        prompt_path: Path to the prompt file
        
    Returns:
        Dictionary with combined_score, accuracy, and features for MAP-Elites
    """
    # Read prompt from file
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()
    
    # Load data
    splits_data = get_dataset_splits()
    config = splits_data['config']
    
    # Get seed for reproducibility
    iteration = get_current_iteration_from_checkpoint()
    seed = config.get("split_seed", 42) + iteration
    
    # Run stage 1
    results = _evaluate_stage1_internal(prompt, splits_data, config, seed)
    
    # Calculate combined_score for stage 1 (with penalty for being stage1 only)
    accuracy = results.get('accuracy', 0.0)
    mae = results.get('mae', 0.0)
    
    # Stage 1 score: accuracy with penalty (lower weight since it's preliminary)
    mae_penalty = mae / 4.0
    combined_score = accuracy * 0.8 * (1 - 0.2 * mae_penalty)  # 0.8 multiplier for stage1
    
    # Add features
    features = calculate_features(prompt, results)
    
    # Return only what OpenEvolve needs
    return {
        "combined_score": combined_score,
        "accuracy": accuracy,
        "mae": mae,
        **features,  # prompt_length, reasoning_strategy
    }


def _get_best_score_cache_path() -> str:
    """Возвращает путь к файлу кэша лучшего score."""
    return os.path.join(CACHE_DIR, "best_score_cache.json")


def _get_cached_best_score() -> Tuple[float, float]:
    """
    Получает лучший score из кэша.
    Returns: (best_train_score, best_val_accuracy)
    """
    cache_path = _get_best_score_cache_path()
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
                return data.get('best_train_score', 0.0), data.get('best_val_accuracy', 0.0)
        except:
            pass
    return 0.0, 0.0


def _update_best_score_cache(train_score: float, val_accuracy: float, prompt: str, iteration: int):
    """Обновляет кэш лучшего score."""
    cache_path = _get_best_score_cache_path()
    data = {
        'best_train_score': train_score,
        'best_val_accuracy': val_accuracy,
        'iteration': iteration,
        'prompt_snippet': prompt[:200] + '...' if len(prompt) > 200 else prompt,
        'timestamp': time.time()
    }
    try:
        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=2)
    except:
        pass


def _log_validation_result(iteration: int, train_score: float, val_results: Dict, prompt: str):
    """Логирует результат валидации в отдельный файл."""
    log_path = os.path.join(CACHE_DIR, "validation_log.jsonl")
    entry = {
        'iteration': iteration,
        'train_score': train_score,
        'val_accuracy': val_results.get('accuracy', 0.0),
        'val_mae': val_results.get('mae', 0.0),
        'val_samples': val_results.get('samples_used', 0),
        'timestamp': time.time(),
        'prompt_length': len(prompt)
    }
    try:
        with open(log_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    except:
        pass


def evaluate_stage2(prompt_path: str) -> Dict[str, Any]:
    """
    Stage 2 evaluation: Comprehensive evaluation with adaptive sample size.
    Called directly by OpenEvolve when cascade_evaluation is enabled.
    
    Логика (v2 - с валидацией на каждом stage2):
    1. Оценка на train выборке с ротацией (50 примеров)
    2. Валидация на 100 примерах на КАЖДОМ stage2
    3. combined_score = 0.6 * train_acc + 0.4 * val_acc (с MAE penalty)
    
    Args:
        prompt_path: Path to the prompt file
        
    Returns:
        Dictionary with combined_score, accuracy, val_accuracy, and features for MAP-Elites
    """
    # Read prompt from file
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()
    
    # Load data
    splits_data = get_dataset_splits()
    config = splits_data['config']
    
    # Get seed for reproducibility
    iteration = get_current_iteration_from_checkpoint()
    seed = config.get("split_seed", 42) + iteration
    
    # Run stage 2 on train split (with rotation)
    results = _evaluate_stage2_internal(prompt, splits_data, config, seed)
    train_accuracy = results.get('accuracy', 0.0)
    train_mae = results.get('mae', 0.0)
    
    # ВСЕГДА запускаем валидацию на каждом stage2
    print(f"  Stage 2 train: accuracy={train_accuracy:.2%}, MAE={train_mae:.2f}")
    print(f"  Running validation on 100 examples...")
    
    val_results = _evaluate_validation_internal(prompt, splits_data, config)
    val_accuracy = val_results.get('accuracy', 0.0)
    val_mae = val_results.get('mae', 0.0)
    
    print(f"  Validation: accuracy={val_accuracy:.2%}, MAE={val_mae:.2f}")
    
    # Combined score formula: 60% train + 40% val, with MAE penalty
    # Используем среднее MAE для penalty
    avg_mae = (train_mae + val_mae) / 2
    mae_penalty = avg_mae / 4.0  # Normalize by max MAE = 4
    
    # Веса: 60% train, 40% validation
    train_weight = config.get("train_weight_in_score", 0.6)
    val_weight = config.get("val_weight_in_score", 0.4)
    
    blended_accuracy = train_weight * train_accuracy + val_weight * val_accuracy
    combined_score = blended_accuracy * (1 - 0.2 * mae_penalty)
    
    # Add features
    features = calculate_features(prompt, results)
    
    print(f"  Combined: {combined_score:.4f} (train={train_accuracy:.2%} * {train_weight} + val={val_accuracy:.2%} * {val_weight})")
    
    # Log validation result
    _log_validation_result(iteration, combined_score, val_results, prompt)
    
    # Update best score cache if improved
    best_train_score, best_val_accuracy = _get_cached_best_score()
    if combined_score > best_train_score:
        print(f"  New best combined score: {combined_score:.4f} > {best_train_score:.4f}")
        if val_accuracy > best_val_accuracy:
            print(f"  New best validation accuracy: {val_accuracy:.2%} > {best_val_accuracy:.2%}")
        _update_best_score_cache(combined_score, val_accuracy, prompt, iteration)
    
    # Return metrics - ВСЕГДА включаем val_accuracy
    return {
        "combined_score": combined_score,
        "accuracy": train_accuracy,  # train accuracy для сравнимости
        "val_accuracy": val_accuracy,
        "mae": train_mae,
        "val_mae": val_mae,
        **features,  # prompt_length, reasoning_strategy
    }


def calculate_features(prompt: str, metrics: Dict) -> Dict[str, float]:
    """
    Вычисляет feature dimensions для MAP-Elites.
    """
    # Feature 1: Длина промпта (логарифмическая шкала)
    # Более плавное распределение, не сатурируется быстро:
    #   100 chars  → 0.33
    #   500 chars  → 0.56
    #   1000 chars → 0.67
    #   2000 chars → 0.77
    #   5000 chars → 0.93
    #   10000 chars → 1.0
    prompt_length = len(prompt)
    if prompt_length > 0:
        normalized_length = np.log10(prompt_length + 1) / np.log10(10000)
    else:
        normalized_length = 0.0
    
    # Feature 2: Reasoning strategy (по ключевым словам)
    prompt_lower = prompt.lower()
    
    reasoning_score = 0.0
    
    # Step-by-step indicators
    if any(kw in prompt_lower for kw in ['step by step', 'first', 'then', 'finally', 'analyze']):
        reasoning_score += 0.3
    
    # Few-shot indicators
    if any(kw in prompt_lower for kw in ['example', 'for instance', 'such as']):
        reasoning_score += 0.3
    
    # Explicit guidelines
    if any(kw in prompt_lower for kw in ['guideline', 'consider', 'look for', 'pay attention']):
        reasoning_score += 0.2
    
    # Domain-specific keywords
    if any(kw in prompt_lower for kw in ['office', 'product', 'quality', 'value', 'recommend']):
        reasoning_score += 0.2
    
    return {
        "prompt_length": min(1.0, normalized_length),
        "reasoning_strategy": min(1.0, reasoning_score),
    }


def evaluate(prompt_path: str) -> Dict[str, Any]:
    """
    Main evaluation function called by OpenEvolve.
    Implements cascade evaluation: Stage 1 -> Stage 2.
    
    Args:
        prompt_path: Path to the prompt file
        
    Returns:
        Dict with metrics and features for MAP-Elites
    """
    try:
        # Read prompt from file
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read()
        
        # Load data
        splits_data = get_dataset_splits()
        config = splits_data['config']
        
        # Get seed for reproducibility
        iteration = get_current_iteration_from_checkpoint()
        seed = config.get("split_seed", 42) + iteration
        
        # Cascade evaluation settings
        cascade_enabled = evaluator_config.get("cascade_evaluation", True)
        cascade_thresholds = evaluator_config.get("cascade_thresholds", [0.5])
        
        # Stage 1
        stage1_results = _evaluate_stage1_internal(prompt, splits_data, config, seed)
        stage1_accuracy = stage1_results['accuracy']
        
        print(f"Stage 1: accuracy={stage1_accuracy:.2%} ({stage1_results['correct']}/{stage1_results['total']})")
        
        # Check threshold for Stage 2
        if cascade_enabled and stage1_accuracy < cascade_thresholds[0]:
            # Failed Stage 1 - return low score
            features = calculate_features(prompt, stage1_results)
            
            return {
                "combined_score": stage1_accuracy * 0.5,  # Penalty for failing cascade
                "accuracy": stage1_accuracy,
                "stage": 1,
                "passed_cascade": False,
                **features,
                **{k: v for k, v in stage1_results.items() if k.startswith("class_")},
            }
        
        # Stage 2
        stage2_results = _evaluate_stage2_internal(prompt, splits_data, config, seed)
        stage2_accuracy = stage2_results['accuracy']
        
        print(f"Stage 2: accuracy={stage2_accuracy:.2%} ({stage2_results['correct']}/{stage2_results['total']})")
        
        # Combined score
        # Weighted average of Stage 1 and Stage 2
        combined_accuracy = 0.3 * stage1_accuracy + 0.7 * stage2_accuracy
        
        # MAE penalty (for large errors)
        mae_penalty = stage2_results.get('mae', 0) / 4.0  # Normalize by max MAE = 4
        
        combined_score = combined_accuracy * (1 - 0.2 * mae_penalty)
        
        features = calculate_features(prompt, stage2_results)
        
        return {
            "combined_score": combined_score,
            "accuracy": stage2_accuracy,
            "stage1_accuracy": stage1_accuracy,
            "stage2_accuracy": stage2_accuracy,
            "mae": stage2_results.get('mae', 0),
            "stage": 2,
            "passed_cascade": True,
            "samples_evaluated": stage1_results['samples_used'] + stage2_results['samples_used'],
            **features,
            **{k: v for k, v in stage2_results.items() if k.startswith("class_")},
        }
        
    except Exception as e:
        print(f"Evaluation error: {e}")
        traceback.print_exc()
        
        # Try to get prompt length for features even on error
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt = f.read()
            prompt_len = len(prompt) / 2000
        except:
            prompt_len = 0.0
        
        return {
            "combined_score": 0.0,
            "accuracy": 0.0,
            "error": str(e),
            "prompt_length": prompt_len,
            "reasoning_strategy": 0.0,
        }


# For testing
if __name__ == "__main__":
    # Get path to test prompt
    evaluator_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(evaluator_dir, "wilds_prompt.txt")
    
    print("Testing evaluator with default prompt...")
    print(f"Prompt file: {prompt_path}")
    print("=" * 60)
    
    # Call evaluate with prompt path (as OpenEvolve does)
    results = evaluate(prompt_path)
    
    print("\n" + "=" * 60)
    print("Results:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


"""
Baseline pipeline для оценки отзывов Amazon с использованием эмбеддингов.

Архитектура:
1. Загрузка данных (WILDS Amazon, Office_Products)
2. Генерация эмбеддингов (sentence-transformers)
3. Обучение модели классификации/регрессии
4. Оценка на validation и test сплитах

Рекомендуемые эмбеддеры:
- all-MiniLM-L6-v2: быстрый, хорош для sentiment (384 dim)
- all-mpnet-base-v2: лучше качество, медленнее (768 dim)
- paraphrase-multilingual-MiniLM-L12-v2: мультиязычный (384 dim)
"""

import os
import pickle
import numpy as np
import yaml
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import ssl
import warnings
warnings.filterwarnings('ignore')

# Отключаем SSL проверку для WILDS
ssl._create_default_https_context = ssl._create_unverified_context

# Импорты для ML
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    import joblib
except ImportError:
    print("Установите зависимости: pip install sentence-transformers scikit-learn joblib")
    raise

# Импорты для XGBoost
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: xgboost не установлен. Установите: pip install xgboost")

# Импорты для RoBERTa sentiment модели
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers не установлен. Установите: pip install transformers torch")

# Импорт функций из evaluator для загрузки данных
import sys
sys.path.insert(0, os.path.dirname(__file__))
from evaluator import (
    load_wilds_dataset,
    preprocess_category_data,
    create_splits_from_preprocessed,
    save_preprocessed_data,
    load_preprocessed_data,
    CACHE_DIR
)


# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

# Путь к конфигу датасета
DATASET_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "wilds_prompt_dataset.yaml")

# Модели эмбеддингов (рекомендуемые)
EMBEDDING_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",  # Быстрый, 384 dim
    "mpnet": "sentence-transformers/all-mpnet-base-v2",  # Лучше качество, 768 dim
    "multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # Мультиязычный, 384 dim
    "roberta_sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest",  # Pre-trained на sentiment, 768 dim
}

# Модели классификации/регрессии
CLASSIFIER_MODELS = {
    "logistic": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "mlp": MLPClassifier,
}

# Добавляем XGBoost если доступен
if XGBOOST_AVAILABLE:
    CLASSIFIER_MODELS["xgb"] = XGBClassifier

REGRESSOR_MODELS = {
    "random_forest": RandomForestRegressor,
    "mlp": MLPRegressor,
}

# Добавляем XGBoost регрессор если доступен
if XGBOOST_AVAILABLE:
    REGRESSOR_MODELS["xgb"] = XGBRegressor


# ============================================================================
# ЗАГРУЗКА ДАННЫХ
# ============================================================================

def load_dataset_splits(config_path: Optional[str] = None) -> Tuple[Dict, Dict]:
    """
    Загружает датасет и создаёт сплиты (использует кэш из evaluator).
    
    Returns:
        (splits_data, config): сплиты и конфигурация
    """
    if config_path is None:
        config_path = DATASET_CONFIG_PATH
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Загружаем датасет
    dataset, metadata_map = load_wilds_dataset(config)
    
    # Предобрабатываем данные (использует кэш)
    preprocessed = load_preprocessed_data(config)
    if preprocessed is None:
        category_id = config.get("category_id", 14)
        preprocessed = preprocess_category_data(dataset, category_id)
        save_preprocessed_data(config, preprocessed)
    
    # Создаём сплиты
    train_ratio = config.get("train_ratio", 0.7)
    val_ratio = config.get("validation_ratio", 0.15)
    test_ratio = config.get("test_ratio", 0.15)
    split_seed = config.get("split_seed", 42)
    
    splits = create_splits_from_preprocessed(
        preprocessed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=split_seed
    )
    
    # Добавляем тексты в сплиты
    texts = preprocessed['texts']
    for split_name in ['train', 'validation', 'test']:
        indices = splits[split_name]['indices']
        splits[split_name]['texts'] = [texts[i] for i in indices]
        splits[split_name]['labels'] = splits[split_name]['y']
    
    return {
        'splits': splits,
        'preprocessed': preprocessed,
        'config': config
    }, config


# ============================================================================
# ГЕНЕРАЦИЯ ЭМБЕДДИНГОВ
# ============================================================================

def generate_embeddings(
    texts: List[str],
    model_name: str = "minilm",
    batch_size: int = 32,
    show_progress: bool = True
) -> np.ndarray:
    """
    Генерирует эмбеддинги для списка текстов.
    
    Args:
        texts: Список текстов отзывов
        model_name: Ключ из EMBEDDING_MODELS ("minilm", "mpnet", "multilingual", "roberta_sentiment")
        batch_size: Размер батча для обработки
        show_progress: Показывать прогресс
    
    Returns:
        Массив эмбеддингов (n_samples, embedding_dim)
    """
    if model_name not in EMBEDDING_MODELS:
        raise ValueError(f"Неизвестная модель: {model_name}. Доступны: {list(EMBEDDING_MODELS.keys())}")
    
    model_path = EMBEDDING_MODELS[model_name]
    
    # Специальная обработка для RoBERTa sentiment модели
    if model_name == "roberta_sentiment":
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Для roberta_sentiment требуется transformers. Установите: pip install transformers torch")
        
        print(f"Загрузка эмбеддера: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        model.eval()
        
        # Используем GPU если доступен
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        print(f"Генерация эмбеддингов для {len(texts)} текстов...")
        embeddings_list = []
        
        # Обработка батчами
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Токенизация
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            
            # Получение эмбеддингов (mean pooling последнего hidden state)
            with torch.no_grad():
                outputs = model(**encoded)
                # Берем mean pooling по токенам (исключая padding)
                last_hidden = outputs.last_hidden_state
                attention_mask = encoded["attention_mask"]
                
                # Mean pooling с учетом attention mask
                mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                sum_embeddings = torch.sum(last_hidden * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                batch_embeddings = sum_embeddings / sum_mask
                
                embeddings_list.append(batch_embeddings.cpu().numpy())
            
            if show_progress and (i + batch_size) % (batch_size * 10) == 0:
                print(f"  Обработано {min(i + batch_size, len(texts))}/{len(texts)} текстов...")
        
        embeddings = np.vstack(embeddings_list)
        print(f"✓ Сгенерировано эмбеддингов: {embeddings.shape}")
        return embeddings
    
    # Обычная обработка для sentence-transformers моделей
    print(f"Загрузка эмбеддера: {model_path}")
    model = SentenceTransformer(model_path)
    
    print(f"Генерация эмбеддингов для {len(texts)} текстов...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True
    )
    
    print(f"✓ Сгенерировано эмбеддингов: {embeddings.shape}")
    return embeddings


# ============================================================================
# ОБУЧЕНИЕ МОДЕЛИ
# ============================================================================

def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "logistic",
    use_class_weights: bool = False,
    **model_kwargs
):
    """
    Обучает классификатор на эмбеддингах.
    
    Args:
        X_train: Эмбеддинги обучающей выборки
        y_train: Метки (1-5)
        model_type: Тип модели ("logistic", "random_forest", "mlp")
        use_class_weights: Использовать ли class weights для борьбы с дисбалансом
        **model_kwargs: Параметры модели
    
    Returns:
        Обученная модель
    """
    if model_type not in CLASSIFIER_MODELS:
        raise ValueError(f"Неизвестный тип модели: {model_type}")
    
    ModelClass = CLASSIFIER_MODELS[model_type]
    
    # Параметры по умолчанию
    default_params = {
        "logistic": {
            "max_iter": 1000,
            "random_state": 42,
            "n_jobs": -1,
        },
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 10,  # Уменьшено для предотвращения переобучения
            "min_samples_split": 10,  # Минимум примеров для разделения узла
            "min_samples_leaf": 5,  # Минимум примеров в листе
            "max_features": "sqrt",  # Ограничение признаков для каждого дерева
            "random_state": 42,
            "n_jobs": -1,
        },
        "mlp": {
            "hidden_layer_sizes": (128, 64),
            "max_iter": 500,
            "random_state": 42,
            "early_stopping": True,
            "validation_fraction": 0.1,
        },
        "xgb": {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0.1,
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "mlogloss",
        } if XGBOOST_AVAILABLE else {},
    }
    
    params = default_params.get(model_type, {})
    
    # Добавляем class weights если нужно
    if use_class_weights:
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        class_weights = compute_class_weight(
            'balanced',
            classes=classes,
            y=y_train
        )
        weight_dict = dict(zip(classes, class_weights))
        
        if model_type == "logistic":
            params["class_weight"] = weight_dict
        elif model_type == "random_forest":
            params["class_weight"] = weight_dict
        elif model_type == "mlp":
            # MLP не поддерживает class_weight напрямую, используем sample_weight
            # Это будет обработано в fit()
            pass
        elif model_type == "xgb":
            # XGBoost использует scale_pos_weight или sample_weight
            # Для многоклассовой задачи используем sample_weight
            pass  # Будет обработано в fit()
        
        print(f"  Используются class weights: {weight_dict}")
    
    params.update(model_kwargs)
    
    print(f"Обучение {model_type} классификатора...")
    model = ModelClass(**params)
    
    # XGBoost требует классы от 0 до n-1, а у нас от 1 до 5
    # Преобразуем метки для XGBoost
    if model_type == "xgb":
        y_train_xgb = y_train - 1  # Преобразуем 1-5 в 0-4
    else:
        y_train_xgb = y_train
    
    # Для MLP и XGBoost с class weights используем sample_weight
    if use_class_weights and (model_type == "mlp" or model_type == "xgb"):
        sample_weights = np.array([weight_dict[y] for y in y_train])
        model.fit(X_train, y_train_xgb, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train_xgb)
    
    # Сохраняем информацию о преобразовании для XGBoost
    if model_type == "xgb":
        model._label_offset = 1  # Добавим обратно при предсказании
    
    print(f"✓ Модель обучена")
    return model


def train_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "random_forest",
    **model_kwargs
):
    """
    Обучает регрессор на эмбеддингах.
    
    Args:
        X_train: Эмбеддинги обучающей выборки
        y_train: Метки (1-5, как float)
        model_type: Тип модели ("random_forest", "mlp")
        **model_kwargs: Параметры модели
    
    Returns:
        Обученная модель
    """
    if model_type not in REGRESSOR_MODELS:
        raise ValueError(f"Неизвестный тип модели: {model_type}")
    
    ModelClass = REGRESSOR_MODELS[model_type]
    
    default_params = {
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 20,
            "random_state": 42,
            "n_jobs": -1,
        },
        "mlp": {
            "hidden_layer_sizes": (128, 64),
            "max_iter": 500,
            "random_state": 42,
            "early_stopping": True,
            "validation_fraction": 0.1,
        }
    }
    
    params = default_params.get(model_type, {})
    params.update(model_kwargs)
    
    print(f"Обучение {model_type} регрессора...")
    model = ModelClass(**params)
    model.fit(X_train, y_train)
    
    print(f"✓ Модель обучена")
    return model


# ============================================================================
# ОЦЕНКА
# ============================================================================

def evaluate_model(
    model,
    X: np.ndarray,
    y_true: np.ndarray,
    task: str = "classification",  # "classification" или "regression"
    verbose: bool = True
) -> Dict[str, float]:
    """
    Оценивает модель на данных.
    
    Returns:
        Словарь с метриками: accuracy, mae, и другие
    """
    if task == "classification":
        y_pred = model.predict(X)
        # XGBoost возвращает классы 0-4, нужно преобразовать в 1-5
        if hasattr(model, '_label_offset'):
            y_pred = y_pred + model._label_offset
            y_pred = np.clip(y_pred, 1, 5)  # На всякий случай ограничиваем
    else:  # regression
        y_pred_float = model.predict(X)
        y_pred = np.round(np.clip(y_pred_float, 1, 5)).astype(int)
    
    accuracy = accuracy_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    results = {
        "accuracy": accuracy,
        "mae": mae,
    }
    
    if verbose:
        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  MAE: {mae:.2f}")
        
        if task == "classification":
            print("\n  Classification Report:")
            print(classification_report(y_true, y_pred, target_names=[f"{i}⭐" for i in range(1, 6)]))
    
    return results


# ============================================================================
# ОСНОВНОЙ ПАЙПЛАЙН
# ============================================================================

def run_baseline_pipeline(
    embedding_model: str = "minilm",
    classifier_model: str = "logistic",
    task: str = "classification",  # "classification" или "regression"
    use_class_weights: bool = False,
    save_model: bool = True,
    output_dir: str = "baseline_output"
):
    """
    Запускает полный пайплайн бейзлайна.
    
    Args:
        embedding_model: Модель эмбеддингов ("minilm", "mpnet", "multilingual")
        classifier_model: Модель классификации ("logistic", "random_forest", "mlp")
        task: Тип задачи ("classification" или "regression")
        save_model: Сохранять ли обученную модель
        output_dir: Директория для сохранения результатов
    """
    print("=" * 80)
    print("BASELINE PIPELINE: Embedding-based Sentiment Classification")
    print("=" * 80)
    
    # 1. Загрузка данных
    print("\n[1/5] Загрузка данных...")
    splits_data, config = load_dataset_splits()
    splits = splits_data['splits']
    
    train_texts = splits['train']['texts']
    train_labels = splits['train']['labels']
    val_texts = splits['validation']['texts']
    val_labels = splits['validation']['labels']
    test_texts = splits['test']['texts']
    test_labels = splits['test']['labels']
    
    print(f"  Train: {len(train_texts)} примеров")
    print(f"  Validation: {len(val_texts)} примеров")
    print(f"  Test: {len(test_texts)} примеров")
    
    # 2. Генерация эмбеддингов
    print("\n[2/5] Генерация эмбеддингов...")
    train_embeddings = generate_embeddings(train_texts, model_name=embedding_model)
    val_embeddings = generate_embeddings(val_texts, model_name=embedding_model, show_progress=False)
    test_embeddings = generate_embeddings(test_texts, model_name=embedding_model, show_progress=False)
    
    # 3. Обучение модели
    print("\n[3/5] Обучение модели...")
    if task == "classification":
        model = train_classifier(
            train_embeddings, 
            train_labels, 
            model_type=classifier_model,
            use_class_weights=use_class_weights
        )
    else:
        model = train_regressor(train_embeddings, train_labels.astype(float), model_type=classifier_model)
    
    # 4. Оценка
    print("\n[4/5] Оценка модели...")
    
    print("\n  Train Set:")
    train_results = evaluate_model(model, train_embeddings, train_labels, task=task, verbose=False)
    print(f"    Accuracy: {train_results['accuracy']:.2%}, MAE: {train_results['mae']:.2f}")
    
    print("\n  Validation Set:")
    val_results = evaluate_model(model, val_embeddings, val_labels, task=task, verbose=True)
    
    print("\n  Test Set:")
    test_results = evaluate_model(model, test_embeddings, test_labels, task=task, verbose=True)
    
    # 5. Сохранение результатов
    if save_model:
        print("\n[5/5] Сохранение результатов...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Сохраняем модель
        model_path = os.path.join(output_dir, f"model_{embedding_model}_{classifier_model}.pkl")
        joblib.dump(model, model_path)
        print(f"  Модель сохранена: {model_path}")
        
        # Сохраняем эмбеддинги (опционально, для анализа)
        embeddings_path = os.path.join(output_dir, "embeddings.pkl")
        with open(embeddings_path, "wb") as f:
            pickle.dump({
                "train": train_embeddings,
                "val": val_embeddings,
                "test": test_embeddings,
            }, f)
        print(f"  Эмбеддинги сохранены: {embeddings_path}")
        
        # Сохраняем результаты
        results = {
            "embedding_model": embedding_model,
            "classifier_model": classifier_model,
            "task": task,
            "train": train_results,
            "validation": val_results,
            "test": test_results,
            "config": config,
        }
        
        results_path = os.path.join(output_dir, "results.json")
        import json
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Результаты сохранены: {results_path}")
    
    print("\n" + "=" * 80)
    print("✓ Пайплайн завершён!")
    print("=" * 80)
    
    return {
        "model": model,
        "train_results": train_results,
        "val_results": val_results,
        "test_results": test_results,
    }


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Baseline pipeline для оценки отзывов Amazon")
    parser.add_argument(
        "--embedding",
        type=str,
        default="minilm",
        choices=list(EMBEDDING_MODELS.keys()),
        help=f"Модель эмбеддингов ({', '.join(EMBEDDING_MODELS.keys())})"
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="logistic",
        choices=list(CLASSIFIER_MODELS.keys()),
        help="Модель классификации (logistic, random_forest, mlp)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="classification",
        choices=["classification", "regression"],
        help="Тип задачи (classification или regression)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="baseline_output",
        help="Директория для сохранения результатов"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Не сохранять модель и результаты"
    )
    parser.add_argument(
        "--class-weights",
        action="store_true",
        help="Использовать class weights для борьбы с дисбалансом классов"
    )
    
    args = parser.parse_args()
    
    run_baseline_pipeline(
        embedding_model=args.embedding,
        classifier_model=args.classifier,
        task=args.task,
        use_class_weights=args.class_weights,
        save_model=not args.no_save,
        output_dir=args.output_dir
    )


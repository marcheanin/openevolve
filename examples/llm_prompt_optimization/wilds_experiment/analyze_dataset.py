"""
Скрипт для анализа датасета WILDS Amazon.
Загружает датасет и выводит статистику по категориям товаров.
"""

import os
import ssl
import urllib.request
import numpy as np
from collections import Counter

# Отключаем проверку SSL сертификатов (для обхода ошибки expired certificate)
ssl._create_default_https_context = ssl._create_unverified_context

# Также патчим urllib для wilds
try:
    import certifi
    os.environ['SSL_CERT_FILE'] = certifi.where()
except ImportError:
    pass

def analyze_wilds_amazon():
    """Анализирует датасет WILDS Amazon и выводит статистику."""
    
    print("=" * 80)
    print("WILDS Amazon Dataset Analysis")
    print("=" * 80)
    
    try:
        from wilds import get_dataset
        
        print("\n[1] Загрузка датасета из локальной директории...")
        dataset = get_dataset(dataset='amazon', download=False, root_dir='./data')
        print("✓ Датасет загружен успешно!")
        
        # Общая информация
        print("\n" + "=" * 80)
        print("[2] ОБЩАЯ ИНФОРМАЦИЯ")
        print("=" * 80)
        
        print(f"\nMetadata fields: {dataset.metadata_fields}")
        print(f"Number of classes (y): {dataset.n_classes}")
        
        # Получаем все сплиты
        splits = ['train', 'val', 'test', 'id_val', 'id_test']
        split_sizes = {}
        
        for split_name in splits:
            try:
                subset = dataset.get_subset(split_name)
                split_sizes[split_name] = len(subset)
                print(f"  {split_name}: {len(subset):,} примеров")
            except:
                pass
        
        # Анализ train сплита
        print("\n" + "=" * 80)
        print("[3] АНАЛИЗ TRAIN СПЛИТА")
        print("=" * 80)
        
        train_data = dataset.get_subset('train')
        
        # Получаем все метаданные
        metadata_array = train_data.metadata_array
        y_array = train_data.y_array
        
        print(f"\nРазмер train: {len(train_data):,}")
        print(f"Metadata shape: {metadata_array.shape}")
        print(f"Labels shape: {y_array.shape}")
        
        # Статистика по меткам классов
        print("\n--- Распределение по классам (звёздам) ---")
        label_counts = Counter(y_array.numpy() if hasattr(y_array, 'numpy') else y_array)
        for label in sorted(label_counts.keys()):
            count = label_counts[label]
            pct = count / len(y_array) * 100
            stars = label + 1  # 0-4 -> 1-5
            print(f"  {stars} звёзд: {count:,} ({pct:.1f}%)")
        
        # Статистика по пользователям (reviewers)
        print("\n--- Статистика по пользователям ---")
        reviewer_ids = metadata_array[:, 0]  # Первый столбец - reviewer
        unique_reviewers = np.unique(reviewer_ids)
        print(f"  Уникальных пользователей в train: {len(unique_reviewers):,}")
        
        reviews_per_user = Counter(reviewer_ids.numpy() if hasattr(reviewer_ids, 'numpy') else reviewer_ids)
        reviews_counts = list(reviews_per_user.values())
        print(f"  Мин. отзывов на пользователя: {min(reviews_counts)}")
        print(f"  Макс. отзывов на пользователя: {max(reviews_counts)}")
        print(f"  Среднее отзывов на пользователя: {np.mean(reviews_counts):.1f}")
        
        # Проверяем, есть ли категории в метаданных
        print("\n--- Проверка наличия категорий ---")
        if len(dataset.metadata_fields) > 1:
            print(f"  Поля метаданных: {dataset.metadata_fields}")
            
            # Пробуем найти категорию
            for i, field in enumerate(dataset.metadata_fields):
                if 'category' in field.lower() or 'product' in field.lower():
                    print(f"  Найдено поле '{field}' в позиции {i}")
                    field_values = metadata_array[:, i]
                    unique_values = np.unique(field_values)
                    print(f"  Уникальных значений: {len(unique_values)}")
                    
                    if len(unique_values) <= 50:
                        print(f"  Значения: {unique_values[:20]}...")  # Первые 20
        
        # Анализ OOD сплитов
        print("\n" + "=" * 80)
        print("[4] АНАЛИЗ OOD (OUT-OF-DISTRIBUTION) СПЛИТОВ")
        print("=" * 80)
        
        val_data = dataset.get_subset('val')
        test_data = dataset.get_subset('test')
        
        val_reviewers = np.unique(val_data.metadata_array[:, 0])
        test_reviewers = np.unique(test_data.metadata_array[:, 0])
        train_reviewers = unique_reviewers
        
        print(f"\n  Train пользователи: {len(train_reviewers):,}")
        print(f"  Val (OOD) пользователи: {len(val_reviewers):,}")
        print(f"  Test (OOD) пользователи: {len(test_reviewers):,}")
        
        # Проверяем пересечение
        train_set = set(train_reviewers.numpy() if hasattr(train_reviewers, 'numpy') else train_reviewers)
        val_set = set(val_reviewers.numpy() if hasattr(val_reviewers, 'numpy') else val_reviewers)
        test_set = set(test_reviewers.numpy() if hasattr(test_reviewers, 'numpy') else test_reviewers)
        
        train_val_overlap = len(train_set & val_set)
        train_test_overlap = len(train_set & test_set)
        val_test_overlap = len(val_set & test_set)
        
        print(f"\n  Пересечение train-val: {train_val_overlap} пользователей")
        print(f"  Пересечение train-test: {train_test_overlap} пользователей")
        print(f"  Пересечение val-test: {val_test_overlap} пользователей")
        
        if train_val_overlap == 0 and train_test_overlap == 0:
            print("\n  ✓ ПОДТВЕРЖДЕНО: Разделение по пользователям (user-disjoint split)")
            print("    Train и Test/Val содержат РАЗНЫХ пользователей!")
        else:
            print("\n  ⚠ Есть пересечение пользователей между сплитами")
        
        # Анализ категорий
        print("\n" + "=" * 80)
        print("[5] ДЕТАЛЬНЫЙ АНАЛИЗ КАТЕГОРИЙ")
        print("=" * 80)
        
        category_idx = dataset.metadata_fields.index('category')
        category_column = metadata_array[:, category_idx]
        
        # Получаем маппинг категорий
        if hasattr(dataset, '_metadata_map') and 'category' in dataset._metadata_map:
            category_names = dataset._metadata_map['category']
            print(f"\nНазвания категорий ({len(category_names)} шт.):")
            
            # Считаем количество отзывов в каждой категории
            category_counts = Counter(category_column.numpy() if hasattr(category_column, 'numpy') else category_column)
            
            print("\n{:<5} {:<40} {:>12} {:>8}".format("ID", "Категория", "Отзывов", "%"))
            print("-" * 70)
            
            sorted_cats = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
            for cat_id, count in sorted_cats:
                cat_name = category_names[cat_id] if cat_id < len(category_names) else f"Unknown_{cat_id}"
                pct = count / len(category_column) * 100
                print(f"{cat_id:<5} {cat_name:<40} {count:>12,} {pct:>7.1f}%")
        else:
            # Fallback: просто считаем по ID
            category_counts = Counter(category_column.numpy() if hasattr(category_column, 'numpy') else category_column)
            print("\nКатегории по ID (названия недоступны):")
            for cat_id, count in sorted(category_counts.items()):
                pct = count / len(category_column) * 100
                print(f"  Category {cat_id}: {count:,} ({pct:.1f}%)")
        
        # Проверяем распределение категорий в OOD сплитах
        print("\n--- Категории в OOD сплитах ---")
        val_categories = Counter(val_data.metadata_array[:, category_idx].numpy())
        test_categories = Counter(test_data.metadata_array[:, category_idx].numpy())
        
        train_cat_set = set(category_counts.keys())
        val_cat_set = set(val_categories.keys())
        test_cat_set = set(test_categories.keys())
        
        print(f"  Категорий в train: {len(train_cat_set)}")
        print(f"  Категорий в val (OOD): {len(val_cat_set)}")
        print(f"  Категорий в test (OOD): {len(test_cat_set)}")
        print(f"  Общие категории train-val: {len(train_cat_set & val_cat_set)}")
        print(f"  Общие категории train-test: {len(train_cat_set & test_cat_set)}")
        
        if train_cat_set == val_cat_set == test_cat_set:
            print("\n  ✓ ПОДТВЕРЖДЕНО: Все сплиты содержат ОДИНАКОВЫЕ категории")
            print("    Сдвиг происходит по ПОЛЬЗОВАТЕЛЯМ, а не по категориям!")
        
        # Примеры данных
        print("\n" + "=" * 80)
        print("[6] ПРИМЕРЫ ДАННЫХ ПО КАТЕГОРИЯМ")
        print("=" * 80)
        
        # Показываем по одному примеру из разных категорий
        shown_categories = set()
        example_count = 0
        
        for i in range(len(train_data)):
            if example_count >= 5:
                break
            x, y, metadata = train_data[i]
            cat_id = metadata[category_idx].item()
            
            if cat_id not in shown_categories:
                shown_categories.add(cat_id)
                example_count += 1
                
                cat_name = "Unknown"
                if hasattr(dataset, '_metadata_map') and 'category' in dataset._metadata_map:
                    cat_name = dataset._metadata_map['category'][cat_id]
                
                print(f"\n--- Категория: {cat_name} (ID={cat_id}) ---")
                print(f"  Отзыв: {x[:200]}...")
                print(f"  Рейтинг: {y + 1} звёзд")
        
        print("\n" + "=" * 80)
        print("АНАЛИЗ ЗАВЕРШЁН")
        print("=" * 80)
        
    except ImportError as e:
        print(f"Ошибка импорта: {e}")
        print("Установите wilds: pip install wilds")
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    analyze_wilds_amazon()


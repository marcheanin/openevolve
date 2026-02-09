#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Анализ эффективности MAP-Elites сетки для exp2_single_evolved.
Оценивает покрытие пространства, разнообразие решений и качество в разных нишах.
"""

import json
import sys
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import numpy as np

# Установка UTF-8 для вывода
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

def load_evolution_trace(trace_path: str) -> List[Dict[str, Any]]:
    """Загрузить evolution trace из JSONL файла."""
    traces = []
    if not Path(trace_path).exists():
        print(f"Error: Evolution trace not found at {trace_path}")
        return traces
    
    with open(trace_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    trace = json.loads(line)
                    traces.append(trace)
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse line: {e}")
                    continue
    
    return traces

def extract_feature_data(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Извлечь данные о feature dimensions и scores."""
    # Собираем все уникальные комбинации (criteria_explicitness, domain_focus)
    # MAP-Elites хранит лучший промпт для каждой ниши
    niche_best = {}  # (criteria_bin, domain_bin) -> best score
    niche_data = defaultdict(list)  # (criteria_bin, domain_bin) -> list of scores
    
    # Все значения для анализа распределения
    all_criteria = []
    all_domain = []
    all_scores = []
    
    # Конфигурация сетки (из config.yaml)
    feature_bins = 10  # 10x10 сетка = 100 ниш
    
    for trace in traces:
        child_metrics = trace.get("child_metrics", {})
        if not child_metrics:
            continue
        
        criteria = child_metrics.get("criteria_explicitness")
        domain = child_metrics.get("domain_focus")
        score = child_metrics.get("combined_score", 0.0)
        
        if criteria is None or domain is None:
            continue
        
        all_criteria.append(criteria)
        all_domain.append(domain)
        all_scores.append(score)
        
        # Определяем bin для MAP-Elites (0-9 для каждой оси)
        criteria_bin = min(int(criteria * feature_bins), feature_bins - 1)
        domain_bin = min(int(domain * feature_bins), feature_bins - 1)
        niche_key = (criteria_bin, domain_bin)
        
        # Обновляем лучший score для этой ниши
        if niche_key not in niche_best or score > niche_best[niche_key]:
            niche_best[niche_key] = score
        
        niche_data[niche_key].append(score)
    
    return {
        "niche_best": niche_best,
        "niche_data": dict(niche_data),
        "all_criteria": all_criteria,
        "all_domain": all_domain,
        "all_scores": all_scores,
        "feature_bins": feature_bins,
    }

def analyze_coverage(data: Dict[str, Any]) -> Dict[str, Any]:
    """Анализ покрытия пространства."""
    niche_best = data["niche_best"]
    feature_bins = data["feature_bins"]
    total_niches = feature_bins * feature_bins
    
    filled_niches = len(niche_best)
    coverage_percent = (filled_niches / total_niches) * 100
    
    # Распределение по осям
    criteria_bins_filled = set()
    domain_bins_filled = set()
    
    for criteria_bin, domain_bin in niche_best.keys():
        criteria_bins_filled.add(criteria_bin)
        domain_bins_filled.add(domain_bin)
    
    criteria_coverage = (len(criteria_bins_filled) / feature_bins) * 100
    domain_coverage = (len(domain_bins_filled) / feature_bins) * 100
    
    return {
        "total_niches": total_niches,
        "filled_niches": filled_niches,
        "coverage_percent": coverage_percent,
        "criteria_bins_filled": len(criteria_bins_filled),
        "domain_bins_filled": len(domain_bins_filled),
        "criteria_coverage_percent": criteria_coverage,
        "domain_coverage_percent": domain_coverage,
    }

def analyze_quality_distribution(data: Dict[str, Any]) -> Dict[str, Any]:
    """Анализ распределения качества по нишам."""
    niche_best = data["niche_best"]
    all_scores = data["all_scores"]
    
    if not niche_best:
        return {}
    
    niche_scores = list(niche_best.values())
    
    return {
        "best_score_overall": max(niche_scores) if niche_scores else 0.0,
        "worst_score_overall": min(niche_scores) if niche_scores else 0.0,
        "mean_score": np.mean(niche_scores) if niche_scores else 0.0,
        "median_score": np.median(niche_scores) if niche_scores else 0.0,
        "std_score": np.std(niche_scores) if niche_scores else 0.0,
        "mean_all_evaluations": np.mean(all_scores) if all_scores else 0.0,
        "num_evaluations": len(all_scores),
    }

def analyze_feature_distribution(data: Dict[str, Any]) -> Dict[str, Any]:
    """Анализ распределения по feature dimensions."""
    all_criteria = data["all_criteria"]
    all_domain = data["all_domain"]
    
    if not all_criteria or not all_domain:
        return {}
    
    return {
        "criteria": {
            "min": min(all_criteria),
            "max": max(all_criteria),
            "mean": np.mean(all_criteria),
            "median": np.median(all_criteria),
            "std": np.std(all_criteria),
        },
        "domain": {
            "min": min(all_domain),
            "max": max(all_domain),
            "mean": np.mean(all_domain),
            "median": np.median(all_domain),
            "std": np.std(all_domain),
        },
    }

def find_best_niches(data: Dict[str, Any], top_n: int = 5) -> List[Tuple[Tuple[int, int], float]]:
    """Найти лучшие ниши по combined_score."""
    niche_best = data["niche_best"]
    
    # Сортируем по score
    sorted_niches = sorted(niche_best.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_niches[:top_n]

def generate_heatmap_data(data: Dict[str, Any]) -> np.ndarray:
    """Создать данные для heatmap (10x10 матрица)."""
    niche_best = data["niche_best"]
    feature_bins = data["feature_bins"]
    
    # Создаем матрицу, заполняем NaN для пустых ниш
    heatmap = np.full((feature_bins, feature_bins), np.nan)
    
    for (criteria_bin, domain_bin), score in niche_best.items():
        heatmap[criteria_bin, domain_bin] = score
    
    return heatmap

def print_analysis_report(data: Dict[str, Any], coverage: Dict[str, Any], 
                         quality: Dict[str, Any], features: Dict[str, Any],
                         best_niches: List[Tuple[Tuple[int, int], float]]):
    """Вывести отчет об анализе."""
    print("=" * 80)
    print("MAP-ELITES СЕТКА: АНАЛИЗ ЭФФЕКТИВНОСТИ")
    print("=" * 80)
    print()
    
    print("📊 ПОКРЫТИЕ ПРОСТРАНСТВА")
    print("-" * 80)
    print(f"Всего ниш в сетке: {coverage['total_niches']} (10x10)")
    print(f"Заполнено ниш: {coverage['filled_niches']}")
    print(f"Покрытие: {coverage['coverage_percent']:.1f}%")
    print()
    print(f"Покрытие по критериям (criteria_explicitness):")
    print(f"  Заполнено bins: {coverage['criteria_bins_filled']}/{coverage['total_niches']//10}")
    print(f"  Покрытие: {coverage['criteria_coverage_percent']:.1f}%")
    print()
    print(f"Покрытие по домену (domain_focus):")
    print(f"  Заполнено bins: {coverage['domain_bins_filled']}/{coverage['total_niches']//10}")
    print(f"  Покрытие: {coverage['domain_coverage_percent']:.1f}%")
    print()
    
    print("🎯 КАЧЕСТВО РЕШЕНИЙ")
    print("-" * 80)
    print(f"Лучший combined_score: {quality['best_score_overall']:.4f}")
    print(f"Худший combined_score: {quality['worst_score_overall']:.4f}")
    print(f"Средний combined_score (по нишам): {quality['mean_score']:.4f}")
    print(f"Медианный combined_score: {quality['median_score']:.4f}")
    print(f"Стд. отклонение: {quality['std_score']:.4f}")
    print()
    print(f"Всего оценок: {quality['num_evaluations']}")
    print(f"Средний score всех оценок: {quality['mean_all_evaluations']:.4f}")
    print()
    
    print("📈 РАСПРЕДЕЛЕНИЕ ПО FEATURE DIMENSIONS")
    print("-" * 80)
    if features:
        print("criteria_explicitness:")
        print(f"  Диапазон: [{features['criteria']['min']:.3f}, {features['criteria']['max']:.3f}]")
        print(f"  Среднее: {features['criteria']['mean']:.3f}")
        print(f"  Медиана: {features['criteria']['median']:.3f}")
        print(f"  Стд. отклонение: {features['criteria']['std']:.3f}")
        print()
        print("domain_focus:")
        print(f"  Диапазон: [{features['domain']['min']:.3f}, {features['domain']['max']:.3f}]")
        print(f"  Среднее: {features['domain']['mean']:.3f}")
        print(f"  Медиана: {features['domain']['median']:.3f}")
        print(f"  Стд. отклонение: {features['domain']['std']:.3f}")
    print()
    
    print("🏆 ТОП-5 ЛУЧШИХ НИШ")
    print("-" * 80)
    for i, ((criteria_bin, domain_bin), score) in enumerate(best_niches, 1):
        criteria_val = criteria_bin / 10.0
        domain_val = domain_bin / 10.0
        print(f"{i}. Ниша ({criteria_bin}, {domain_bin}) "
              f"[criteria={criteria_val:.1f}, domain={domain_val:.1f}]: "
              f"score={score:.4f}")
    print()
    
    print("=" * 80)
    print("ОЦЕНКА ЭФФЕКТИВНОСТИ")
    print("=" * 80)
    print()
    
    # Оценка эффективности
    coverage_score = coverage['coverage_percent'] / 100.0
    quality_score = quality['best_score_overall']  # Нормализуем к 0-1 (score уже в этом диапазоне)
    diversity_score = features['criteria']['std'] * features['domain']['std'] if features else 0.0
    
    print(f"Покрытие пространства: {coverage_score*100:.1f}% "
          f"{'✅ Отлично' if coverage_score > 0.5 else '⚠️ Низкое' if coverage_score < 0.2 else '✅ Хорошо'}")
    print(f"Качество решений: {quality['best_score_overall']:.4f} "
          f"{'✅ Отлично' if quality['best_score_overall'] > 0.6 else '⚠️ Низкое' if quality['best_score_overall'] < 0.4 else '✅ Хорошо'}")
    print(f"Разнообразие: std(criteria)={features['criteria']['std']:.3f}, "
          f"std(domain)={features['domain']['std']:.3f} "
          f"{'✅ Хорошо' if features['criteria']['std'] > 0.2 and features['domain']['std'] > 0.2 else '⚠️ Низкое'}")
    print()
    
    # Итоговая оценка
    if coverage_score > 0.3 and quality['best_score_overall'] > 0.5:
        print("✅ СЕТКА ЭФФЕКТИВНА: Хорошее покрытие и качество решений")
    elif coverage_score > 0.5:
        print("⚠️ СЕТКА ЧАСТИЧНО ЭФФЕКТИВНА: Хорошее покрытие, но качество можно улучшить")
    elif quality['best_score_overall'] > 0.5:
        print("⚠️ СЕТКА ЧАСТИЧНО ЭФФЕКТИВНА: Хорошее качество, но низкое покрытие")
    else:
        print("❌ СЕТКА НЕЭФФЕКТИВНА: Низкое покрытие и/или качество")
    print()

def main():
    """Главная функция."""
    exp_dir = Path(__file__).parent
    trace_path = exp_dir / "openevolve_output" / "evolution_trace.jsonl"
    
    if not trace_path.exists():
        print(f"Error: Evolution trace not found at {trace_path}")
        sys.exit(1)
    
    print("Загрузка данных...")
    traces = load_evolution_trace(str(trace_path))
    
    if not traces:
        print("No traces found. Cannot analyze.")
        sys.exit(1)
    
    print(f"Загружено {len(traces)} записей из evolution trace")
    print()
    
    # Извлечение данных
    print("Извлечение данных о feature dimensions...")
    data = extract_feature_data(traces)
    
    if not data["niche_best"]:
        print("No feature dimension data found in traces.")
        sys.exit(1)
    
    # Анализ
    print("Анализ покрытия пространства...")
    coverage = analyze_coverage(data)
    
    print("Анализ качества решений...")
    quality = analyze_quality_distribution(data)
    
    print("Анализ распределения по feature dimensions...")
    features = analyze_feature_distribution(data)
    
    print("Поиск лучших ниш...")
    best_niches = find_best_niches(data, top_n=5)
    
    # Вывод отчета
    print_analysis_report(data, coverage, quality, features, best_niches)
    
    # Сохранение результатов
    output_path = exp_dir / "openevolve_output" / "map_elites_analysis.json"
    results = {
        "coverage": coverage,
        "quality": quality,
        "features": features,
        "best_niches": [
            {
                "niche": list(niche),
                "criteria_bin": niche[0],
                "domain_bin": niche[1],
                "criteria_value": niche[0] / 10.0,
                "domain_value": niche[1] / 10.0,
                "score": score
            }
            for niche, score in best_niches
        ],
        "total_niches_filled": len(data["niche_best"]),
        "total_evaluations": len(data["all_scores"]),
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Результаты сохранены в: {output_path}")

if __name__ == "__main__":
    main()

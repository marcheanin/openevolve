"""
Скрипт для сравнения старых и новых (унифицированных) метрик combined_score.
Показывает, как изменится оценка при переходе на унифицированную формулу.
"""

import json
from pathlib import Path
from typing import Dict, Any

from metrics import (
    compute_combined_score_single,
    compute_combined_score_ensemble,
    compute_combined_score_unified,
)


def load_metrics_from_file(path: Path) -> Dict[str, Any]:
    """Загружает метрики из JSON файла."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Извлекаем метрики из разных форматов файлов
    if "test_metrics" in data:
        return data["test_metrics"]
    elif "metrics" in data:
        return data["metrics"]
    else:
        return data


def compare_scores(metrics: Dict[str, Any], experiment_name: str, is_ensemble: bool = False):
    """Сравнивает старые и новые combined scores."""
    print(f"\n{'=' * 80}")
    print(f"Experiment: {experiment_name}")
    print(f"{'=' * 80}")
    
    # Извлекаем метрики
    r_global = metrics.get("R_global", 0.0)
    r_worst = metrics.get("R_worst", 0.0)
    mae = metrics.get("mae", 0.0)
    mean_kappa = metrics.get("mean_kappa", None)
    
    print(f"Metrics:")
    print(f"  R_global:        {r_global:.4f} ({r_global*100:.2f}%)")
    print(f"  R_worst:         {r_worst:.4f} ({r_worst*100:.2f}%)")
    print(f"  MAE:             {mae:.4f}")
    if mean_kappa is not None:
        print(f"  Mean Kappa (κ):  {mean_kappa:.4f}")
    
    # Старая формула
    if is_ensemble:
        old_score = compute_combined_score_ensemble(metrics)
        old_formula = "0.4*R_global + 0.4*R_worst - 0.2*(1 - D)"
    else:
        old_score = compute_combined_score_single(metrics)
        old_formula = "R_global * (1 - 0.2 * MAE/4)"
    
    # Новая унифицированная формула
    new_score = compute_combined_score_unified(metrics, is_ensemble=is_ensemble)
    
    print(f"\nCombined Score Comparison:")
    print(f"  Old formula:     {old_formula}")
    print(f"  Old score:       {old_score:.4f}")
    print(f"  New (unified):   {new_score:.4f}")
    print(f"  Difference:      {new_score - old_score:+.4f} ({((new_score - old_score) / old_score * 100):+.2f}%)")
    
    # Показываем компоненты новой формулы
    print(f"\nNew formula breakdown:")
    base = 0.4 * r_global + 0.3 * r_worst + 0.3 * (1.0 - mae / 4.0)
    print(f"  Base (40%*R_global + 30%*R_worst + 30%*(1-MAE/4)): {base:.4f}")
    if is_ensemble and disagreement is not None:
        consistency = 0.1 * (1.0 - disagreement)
        print(f"  Consistency bonus (10%*(1-D)):                  {consistency:.4f}")
        print(f"  Total:                                           {new_score:.4f}")
    else:
        print(f"  Total (single model, no bonus):                 {new_score:.4f}")


def main():
    """Главная функция."""
    experiments_root = Path(__file__).resolve().parent
    
    experiments = [
        {
            "name": "Exp 1: Single Model Baseline",
            "path": experiments_root / "exp1_baseline" / "results" / "summary.json",
            "is_ensemble": False,
        },
        {
            "name": "Exp 2: Single Model + Evolution",
            "path": experiments_root / "exp2_single_evolved" / "openevolve_output" / "final_report.json",
            "is_ensemble": False,
        },
        {
            "name": "Exp 3a: Ensemble Baseline",
            "path": experiments_root / "exp3_ensemble_voting" / "results_baseline" / "metrics.json",
            "is_ensemble": True,
        },
        {
            "name": "Exp 3b: Ensemble + Evolution",
            "path": experiments_root / "exp3_ensemble_voting" / "openevolve_output" / "final_report.json",
            "is_ensemble": True,
        },
    ]
    
    print("=" * 80)
    print("COMPARISON: Old vs New (Unified) Combined Score Formulas")
    print("=" * 80)
    
    results = []
    
    for exp in experiments:
        if not exp["path"].exists():
            print(f"\n[WARNING] File not found: {exp['path']}")
            continue
        
        try:
            metrics = load_metrics_from_file(exp["path"])
            compare_scores(metrics, exp["name"], exp["is_ensemble"])
            
            # Сохраняем для итоговой таблицы
            old_score = (
                compute_combined_score_ensemble(metrics)
                if exp["is_ensemble"]
                else compute_combined_score_single(metrics)
            )
            new_score = compute_combined_score_unified(metrics, is_ensemble=exp["is_ensemble"])
            
            results.append({
                "name": exp["name"],
                "old_score": old_score,
                "new_score": new_score,
                "difference": new_score - old_score,
                "is_ensemble": exp["is_ensemble"],
            })
        except Exception as e:
            print(f"\n[ERROR] Error processing {exp['name']}: {e}")
    
    # Итоговая таблица сравнения
    print(f"\n{'=' * 80}")
    print("SUMMARY TABLE")
    print(f"{'=' * 80}")
    print(f"{'Experiment':<40} {'Old Score':<12} {'New Score':<12} {'Diff':<12} {'Type':<10}")
    print("-" * 80)
    
    for r in results:
        exp_type = "Ensemble" if r["is_ensemble"] else "Single"
        print(
            f"{r['name']:<40} "
            f"{r['old_score']:>11.4f}  "
            f"{r['new_score']:>11.4f}  "
            f"{r['difference']:>+11.4f}  "
            f"{exp_type:<10}"
        )
    
    # Анализ сравнимости
    print(f"\n{'=' * 80}")
    print("ANALYSIS")
    print(f"{'=' * 80}")
    
    single_scores = [r["new_score"] for r in results if not r["is_ensemble"]]
    ensemble_scores = [r["new_score"] for r in results if r["is_ensemble"]]
    
    if single_scores and ensemble_scores:
        print(f"\n[OK] With unified formula, scores are now comparable:")
        print(f"   Single model scores:  {min(single_scores):.4f} - {max(single_scores):.4f}")
        print(f"   Ensemble scores:       {min(ensemble_scores):.4f} - {max(ensemble_scores):.4f}")
        print(f"\n   Best overall: {max(results, key=lambda x: x['new_score'])['name']}")
        print(f"   Score: {max(r['new_score'] for r in results):.4f}")
    
    print(f"\n{'=' * 80}")
    print("RECOMMENDATION")
    print(f"{'=' * 80}")
    print("""
The unified formula provides:
  [OK] Comparable scores between single model and ensemble
  [OK] Balanced consideration of all metrics (R_global, R_worst, MAE)
  [OK] Additional consistency bonus for ensemble (fairness)
  [OK] Same formula structure for both cases (easier to understand)

To migrate:
  1. Update evaluators to use compute_combined_score_unified()
  2. Recalculate combined_score for all experiments
  3. Update final_report.py to use unified scores
    """)


if __name__ == "__main__":
    main()

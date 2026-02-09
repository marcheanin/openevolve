"""
Генерация финального отчета со всеми экспериментами.
Собирает метрики из exp1, exp2, exp3 (baseline и evolved).
Использует унифицированную метрику для корректного сравнения.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import yaml

# Добавляем путь для импорта metrics
EXPERIMENTS_ROOT = Path(__file__).resolve().parent
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from metrics import compute_combined_score_unified  # noqa: E402


def load_json(path: Path) -> Dict[str, Any]:
    """Загружает JSON файл."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_percent(value: float) -> str:
    """Форматирует значение как процент."""
    return f"{value * 100:.2f}%"


def format_decimal(value: float, decimals: int = 3) -> str:
    """Форматирует значение как десятичное число."""
    return f"{value:.{decimals}f}"


def collect_experiment_data() -> List[Dict[str, Any]]:
    """Собирает данные из всех экспериментов."""
    experiments_root = Path(__file__).resolve().parent
    
    experiments = []
    
    # Experiment 1: Baseline (single model, no evolution)
    exp1_path = experiments_root / "exp1_baseline" / "results" / "summary.json"
    if exp1_path.exists():
        exp1_data = load_json(exp1_path)
        experiments.append({
            "name": "Exp 1: Single Model Baseline",
            "setup": "1 model (gpt-oss-120b), no evolution",
            "split": exp1_data.get("split", "test"),
            "R_global": exp1_data.get("R_global", 0.0),
            "R_worst": exp1_data.get("R_worst", 0.0),
            "mae": exp1_data.get("mae", 0.0),
            "combined_score_old": exp1_data.get("combined_score", 0.0),
            "num_users": exp1_data.get("num_users", 0),
            "num_examples": exp1_data.get("num_examples", 0),
            "mean_kappa": None,
            "disagreement_rate": None,
        })
        # Пересчитываем с унифицированной метрикой
        exp1_metrics = {
            "R_global": exp1_data.get("R_global", 0.0),
            "R_worst": exp1_data.get("R_worst", 0.0),
            "mae": exp1_data.get("mae", 0.0),
        }
        experiments[-1]["combined_score_unified"] = compute_combined_score_unified(
            exp1_metrics, is_ensemble=False
        )
    
    # Experiment 2: Single Model + Evolution
    exp2_path = experiments_root / "exp2_single_evolved" / "openevolve_output" / "final_report.json"
    if exp2_path.exists():
        exp2_data = load_json(exp2_path)
        test_metrics = exp2_data.get("test_metrics", {})
        experiments.append({
            "name": "Exp 2: Single Model + Evolution",
            "setup": "1 model (gpt-oss-120b), with OpenEvolve",
            "split": test_metrics.get("split", "test"),
            "R_global": test_metrics.get("R_global", 0.0),
            "R_worst": test_metrics.get("R_worst", 0.0),
            "mae": test_metrics.get("mae", 0.0),
            "combined_score_old": test_metrics.get("combined_score", 0.0),
            "num_users": test_metrics.get("num_users", 0),
            "num_examples": test_metrics.get("num_examples", 0),
            "mean_kappa": None,
            "disagreement_rate": None,
        })
        # Пересчитываем с унифицированной метрикой
        exp2_metrics = {
            "R_global": test_metrics.get("R_global", 0.0),
            "R_worst": test_metrics.get("R_worst", 0.0),
            "mae": test_metrics.get("mae", 0.0),
        }
        experiments[-1]["combined_score_unified"] = compute_combined_score_unified(
            exp2_metrics, is_ensemble=False
        )
    
    # Experiment 3a: Ensemble Baseline (no evolution)
    # Проверяем несколько возможных путей для baseline
    exp3_baseline_paths = [
        experiments_root / "exp3_ensemble_voting" / "results" / "summary.json",  # Новый baseline
        experiments_root / "exp3_ensemble_voting" / "results_baseline" / "metrics.json",  # Старый путь
        experiments_root / "exp3_ensemble_voting" / "results_baseline_first" / "summary.json",  # Еще один вариант
    ]
    
    exp3_baseline_data = None
    for path in exp3_baseline_paths:
        if path.exists():
            exp3_baseline_data = load_json(path)
            break
    
    if exp3_baseline_data:
        experiments.append({
            "name": "Exp 3a: Ensemble Baseline",
            "setup": "3 models (yandexgpt, gemma3-27b, gpt-oss-120b), majority vote, no evolution",
            "split": exp3_baseline_data.get("split", "test"),
            "R_global": exp3_baseline_data.get("R_global", 0.0),
            "R_worst": exp3_baseline_data.get("R_worst", 0.0),
            "mae": exp3_baseline_data.get("mae", 0.0),
            "combined_score_old": exp3_baseline_data.get("combined_score", 0.0),
            "num_users": exp3_baseline_data.get("num_users", 0),
            "num_examples": exp3_baseline_data.get("num_examples", 0),
            "mean_kappa": exp3_baseline_data.get("mean_kappa"),
            "disagreement_rate": exp3_baseline_data.get("disagreement_rate"),
        })
        # Пересчитываем с унифицированной метрикой
        exp3a_metrics = {
            "R_global": exp3_baseline_data.get("R_global", 0.0),
            "R_worst": exp3_baseline_data.get("R_worst", 0.0),
            "mae": exp3_baseline_data.get("mae", 0.0),
            "disagreement_rate": exp3_baseline_data.get("disagreement_rate", 0.0),
        }
        experiments[-1]["combined_score_unified"] = compute_combined_score_unified(
            exp3a_metrics, is_ensemble=True
        )
    
    # Experiment 3b: Ensemble + Evolution
    exp3_evolved_path = experiments_root / "exp3_ensemble_voting" / "openevolve_output" / "final_report.json"
    if exp3_evolved_path.exists():
        exp3_evolved_data = load_json(exp3_evolved_path)
        test_metrics = exp3_evolved_data.get("test_metrics", {})
        experiments.append({
            "name": "Exp 3b: Ensemble + Evolution",
            "setup": "3 models (yandexgpt, gemma3-27b, gpt-oss-120b), majority vote, with OpenEvolve",
            "split": test_metrics.get("split", "test"),
            "R_global": test_metrics.get("R_global", 0.0),
            "R_worst": test_metrics.get("R_worst", 0.0),
            "mae": test_metrics.get("mae", 0.0),
            "combined_score_old": test_metrics.get("combined_score", 0.0),
            "num_users": test_metrics.get("num_users", 0),
            "num_examples": test_metrics.get("num_examples", 0),
            "mean_kappa": test_metrics.get("mean_kappa"),
            "disagreement_rate": test_metrics.get("disagreement_rate"),
        })
        # Пересчитываем с унифицированной метрикой
        exp3b_metrics = {
            "R_global": test_metrics.get("R_global", 0.0),
            "R_worst": test_metrics.get("R_worst", 0.0),
            "mae": test_metrics.get("mae", 0.0),
            "disagreement_rate": test_metrics.get("disagreement_rate", 0.0),
        }
        experiments[-1]["combined_score_unified"] = compute_combined_score_unified(
            exp3b_metrics, is_ensemble=True
        )
    
    return experiments


def generate_markdown_table(experiments: List[Dict[str, Any]]) -> str:
    """Генерирует Markdown таблицу с результатами."""
    lines = [
        "# Final Experiment Report: WILDS Amazon Sentiment Classification",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary Table (Unified Metrics)",
        "",
        "| Experiment | Setup | R_global | R_worst | MAE | Combined (Unified) | Old Combined | Kappa (κ) | Users | Examples |",
        "|------------|-------|----------|---------|-----|-------------------|--------------|-----------|-------|----------|",
    ]
    
    for exp in experiments:
        name = exp["name"]
        setup = exp["setup"]
        r_global = format_percent(exp["R_global"])
        r_worst = format_percent(exp["R_worst"])
        mae = format_decimal(exp["mae"], 3)
        combined_unified = format_decimal(exp.get("combined_score_unified", 0.0), 3)
        combined_old = format_decimal(exp.get("combined_score_old", 0.0), 3)
        kappa = format_decimal(exp["mean_kappa"], 3) if exp["mean_kappa"] is not None else "N/A"
        num_users = exp["num_users"]
        num_examples = exp["num_examples"]
        
        lines.append(
            f"| {name} | {setup} | {r_global} | {r_worst} | {mae} | {combined_unified} | {combined_old} | {kappa} | {num_users} | {num_examples} |"
        )
    
    lines.extend([
        "",
        "### Unified Formula Explanation",
        "",
        "The **Combined (Unified)** score uses a unified formula for fair comparison:",
        "",
        "**Base formula (for all experiments):**",
        "```",
        "base_score = 0.4 * R_global + 0.3 * R_worst + 0.3 * (1 - MAE/4)",
        "```",
        "",
        "**For ensemble experiments, adds consistency bonus (Cohen's Kappa):**",
        "```",
        "kappa_score = max(0, mean_kappa)",
        "consistency_bonus = 0.1 * kappa_score",
        "combined_score = base_score + consistency_bonus",
        "```",
        "",
        "κ (kappa) uses Landis & Koch scale: <0 Poor, 0–0.20 Slight, 0.21–0.40 Fair, 0.41–0.60 Moderate, 0.61–0.80 Substantial, 0.81–1.0 Almost perfect.",
        "",
        "This ensures:",
        "- Comparable scores between single model and ensemble",
        "- Balanced consideration of accuracy (R_global), fairness (R_worst), and precision (MAE)",
        "- Ensemble bonus rewards real inter-annotator agreement (kappa), not just low disagreement",
    ])
    
    return "\n".join(lines)


def generate_detailed_report(experiments: List[Dict[str, Any]]) -> str:
    """Генерирует детальный текстовый отчет."""
    lines = [
        "=" * 80,
        "FINAL EXPERIMENT REPORT: WILDS Amazon Sentiment Classification",
        "=" * 80,
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "OVERVIEW",
        "-" * 80,
        "",
        "This report summarizes results from three experimental setups:",
        "  1. Single model baseline (no evolution)",
        "  2. Single model with prompt evolution",
        "  3. Ensemble voting (baseline and evolved)",
        "",
        "=" * 80,
        "RESULTS SUMMARY",
        "=" * 80,
        "",
    ]
    
    for i, exp in enumerate(experiments, 1):
        lines.extend([
            f"Experiment {i}: {exp['name']}",
            "-" * 80,
            f"Setup: {exp['setup']}",
            f"Split: {exp['split']}",
            "",
            "Metrics:",
            f"  R_global (Global Accuracy):     {format_percent(exp['R_global'])}",
            f"  R_worst (10th percentile):      {format_percent(exp['R_worst'])}",
            f"  MAE (Mean Absolute Error):       {format_decimal(exp['mae'], 3)}",
            f"  Combined Score (Unified):        {format_decimal(exp.get('combined_score_unified', 0.0), 3)}",
            f"  Combined Score (Old):            {format_decimal(exp.get('combined_score_old', 0.0), 3)}",
        ])
        
        if exp["mean_kappa"] is not None:
            lines.append(f"  Mean Kappa (κ):                 {format_decimal(exp['mean_kappa'], 3)} (Cohen's kappa, Landis & Koch)")
        
        lines.extend([
            "",
            f"Dataset:",
            f"  Users:                          {exp['num_users']}",
            f"  Examples:                       {exp['num_examples']}",
            "",
        ])
    
    # Analysis section
    lines.extend([
        "=" * 80,
        "ANALYSIS",
        "=" * 80,
        "",
    ])
    
    if len(experiments) >= 2:
        exp1 = experiments[0]
        exp2 = experiments[1]
        
        r_global_improvement = exp2["R_global"] - exp1["R_global"]
        r_worst_improvement = exp2["R_worst"] - exp1["R_worst"]
        mae_improvement = exp1["mae"] - exp2["mae"]  # Lower is better
        
        lines.extend([
            "Single Model Evolution (Exp 2 vs Exp 1):",
            f"  R_global improvement:            {format_percent(r_global_improvement)} ({format_percent(exp1['R_global'])} → {format_percent(exp2['R_global'])})",
            f"  R_worst improvement:             {format_percent(r_worst_improvement)} ({format_percent(exp1['R_worst'])} → {format_percent(exp2['R_worst'])})",
            f"  MAE improvement:                 {format_decimal(mae_improvement, 3)} ({format_decimal(exp1['mae'], 3)} → {format_decimal(exp2['mae'], 3)})",
            "",
        ])
    
    if len(experiments) >= 4:
        exp1 = experiments[0]
        exp3a = experiments[2]
        exp3b = experiments[3]
        
        ensemble_baseline_improvement = exp3a["R_global"] - exp1["R_global"]
        ensemble_evolved_improvement = exp3b["R_global"] - exp1["R_global"]
        ensemble_evolution_improvement = exp3b["R_global"] - exp3a["R_global"]
        
        lines.extend([
            "Ensemble vs Single Model Baseline:",
            f"  Ensemble baseline improvement:   {format_percent(ensemble_baseline_improvement)} ({format_percent(exp1['R_global'])} → {format_percent(exp3a['R_global'])})",
            f"  Ensemble evolved improvement:    {format_percent(ensemble_evolved_improvement)} ({format_percent(exp1['R_global'])} → {format_percent(exp3b['R_global'])})",
            "",
            "Ensemble Evolution (Exp 3b vs Exp 3a):",
            f"  R_global improvement:            {format_percent(ensemble_evolution_improvement)} ({format_percent(exp3a['R_global'])} → {format_percent(exp3b['R_global'])})",
            "",
        ])
    
    lines.extend([
        "=" * 80,
        "NOTES",
        "=" * 80,
        "",
        "- R_global: Overall accuracy across all test examples",
        "- R_worst: 10th percentile of per-user accuracy (worst-case performance)",
        "- MAE: Mean Absolute Error (lower is better, range 0-4)",
        "- Combined Score (Unified): Weighted metric using unified formula:",
        "  0.4*R_global + 0.3*R_worst + 0.3*(1-MAE/4) [+ 0.1*max(0, kappa) for ensemble]",
        "- Combined Score (Old): Original formula (deprecated, kept for reference)",
        "- κ (kappa): Cohen's kappa for inter-annotator agreement (ensemble only); Landis & Koch scale",
        "",
    ])
    
    return "\n".join(lines)


def generate_json_report(experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Генерирует JSON отчет."""
    return {
        "timestamp": datetime.now().isoformat(),
        "experiments": experiments,
        "summary": {
            "total_experiments": len(experiments),
            "best_r_global": max(exp["R_global"] for exp in experiments) if experiments else 0.0,
            "best_r_worst": max(exp["R_worst"] for exp in experiments) if experiments else 0.0,
            "best_mae": min(exp["mae"] for exp in experiments) if experiments else 0.0,
            "best_combined_unified": max(exp.get("combined_score_unified", 0.0) for exp in experiments) if experiments else 0.0,
        },
    }


def main():
    """Главная функция."""
    experiments_root = Path(__file__).resolve().parent
    output_dir = experiments_root / "final_report"
    output_dir.mkdir(exist_ok=True)
    
    print("Collecting experiment data...")
    experiments = collect_experiment_data()
    
    if not experiments:
        print("ERROR: No experiment data found!")
        return
    
    print(f"Found {len(experiments)} experiments")
    
    # Generate Markdown table
    print("Generating Markdown table...")
    md_table = generate_markdown_table(experiments)
    md_path = output_dir / "summary_table.md"
    md_path.write_text(md_table, encoding="utf-8")
    print(f"  Saved: {md_path}")
    
    # Generate detailed report
    print("Generating detailed report...")
    detailed_report = generate_detailed_report(experiments)
    report_path = output_dir / "detailed_report.txt"
    report_path.write_text(detailed_report, encoding="utf-8")
    print(f"  Saved: {report_path}")
    
    # Generate JSON report
    print("Generating JSON report...")
    json_report = generate_json_report(experiments)
    json_path = output_dir / "final_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {json_path}")
    
    # Print summary to console (avoid Unicode issues in Windows console)
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    try:
        print(md_table)
    except UnicodeEncodeError:
        # Fallback: print ASCII-safe version
        print("Markdown table generated (see summary_table.md for full version)")
        for exp in experiments:
            print(f"  {exp['name']}: R_global={format_percent(exp['R_global'])}, "
                  f"R_worst={format_percent(exp['R_worst'])}, MAE={format_decimal(exp['mae'], 3)}")
    print("\n" + "=" * 80)
    print(f"Full reports saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

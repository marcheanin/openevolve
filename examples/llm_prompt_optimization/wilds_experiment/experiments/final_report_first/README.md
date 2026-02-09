# Final Experiment Report

Этот каталог содержит финальный отчет со всеми экспериментами по оптимизации промптов для WILDS Amazon sentiment classification.

## Файлы отчета

- **`summary_table.md`** - Краткая таблица в формате Markdown со всеми метриками
- **`detailed_report.txt`** - Детальный текстовый отчет с анализом результатов
- **`final_report.json`** - Полный отчет в формате JSON для программной обработки

## Как обновить отчет

Если вы запустили новые эксперименты или обновили результаты, запустите:

```bash
cd openevolve/examples/llm_prompt_optimization/wilds_experiment/experiments
python generate_final_report.py
```

Скрипт автоматически найдет результаты из:
- `exp1_baseline/results/summary.json`
- `exp2_single_evolved/openevolve_output/final_report.json`
- `exp3_ensemble_voting/results_baseline/metrics.json`
- `exp3_ensemble_voting/openevolve_output/final_report.json`

## Метрики в отчете

- **R_global**: Общая точность (accuracy) на тестовом наборе
- **R_worst**: 10-й перцентиль точности по пользователям (worst-case performance)
- **MAE**: Средняя абсолютная ошибка (Mean Absolute Error, диапазон 0-4)
- **Combined Score**: Взвешенная метрика, используемая для эволюции
- **Kappa (κ)**: Коэффициент согласия Коэна между воркерами ансамбля (только для ensemble)
- **Disagreement Rate**: Доля примеров, где воркеры ансамбля не согласны (только для ensemble)

## Структура экспериментов

1. **Exp 1: Single Model Baseline** - Базовый эксперимент с одной моделью без эволюции
2. **Exp 2: Single Model + Evolution** - Одна модель с эволюцией промпта через OpenEvolve
3. **Exp 3a: Ensemble Baseline** - Ансамбль из 3 моделей с majority voting, без эволюции
4. **Exp 3b: Ensemble + Evolution** - Ансамбль из 3 моделей с majority voting, с эволюцией промпта

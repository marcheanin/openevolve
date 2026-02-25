# Smoke run — быстрая проверка пайплайна

Минимальный прогон для проверки изменений без полного эксперимента.

## Параметры smoke

- **2 AL-цикла** (вместо 4–5)
- **4 итерации эволюции** на цикл (вместо 15–20)
- Вывод в отдельную папку **`results_smoke/`**, чтобы не смешивать с полными прогонами

Ожидаемое время: порядка 15–40 минут в зависимости от API.

## Как запустить

Из каталога `wilds_active_learn_approach`:

```powershell
# Windows
.\run_smoke.ps1
```

```bash
# Linux/macOS
chmod +x run_smoke.sh && ./run_smoke.sh
```

Или вручную:

```bash
python active_loop.py --smoke
python analyze_run.py --results-dir results_smoke
```

Перед запуском задайте ключ API: переменная окружения `OPENROUTER_API_KEY` (или `OPENAI_API_KEY`), либо `.env` в этой папке.

## Тайминги (только в smoke)

В smoke-режиме для каждого этапа логируется время выполнения:
- `load_dataset` — загрузка датасета и DataManager
- `init_full_eval` — первая оценка на всём train-пуле
- `baseline_test` — оценка стартового промпта на test
- `cycle_N_build_batch` — построение активного батча и error analysis
- `cycle_N_evolution` — запуск OpenEvolve (N итераций мутаций)
- `cycle_N_reeval_batch` — переоценка батча после эволюции
- `cycle_N_validation` — оценка на validation
- `cycle_N_consolidation` — консолидация (LLM + оценка консолидированного промпта)
- `final_test` — финальная оценка на test

После прогона выводится блок **«[smoke] Timing summary»** со списком этапов и итогом TOTAL. Те же события пишутся в `debug_trace.jsonl` с `event: "smoke_timing"`.

## Что смотреть по результату

1. **Вывод в консоль**  
   - Нет падений и traceback.  
   - В конце блок «Final test evaluation» и «Analysis of smoke run».

2. **Папка `results_smoke/`**  
   - `debug_trace.jsonl` — события пайплайна (init, cycle_start, evolution_done, consolidation_decision, final_test).  
   - `active_loop_log.json` — метрики по циклам (val_R_global, evo_best_score, consolidated и т.д.).  
   - `al_iter_0/`, `al_iter_1/` — для каждого цикла: `evolution_trace.jsonl`, `best_prompt.txt`, при успешной консолидации `consolidated_prompt.txt`.  
   - `baseline_test_metrics.json`, `final_test_metrics.json`, `final_prompt.txt`.

3. **Отчёт анализа** (после `analyze_run.py --results-dir results_smoke`)  
   - Секция «Pipeline Health Check»: все пункты [OK], артефакты в трассах есть.  
   - В «Consolidation» — решения accepted/rejected и скоры.  
   - В «Summary» — дельта R_global / combined_score и итоговый вердикт.

Чтобы я мог однозначно судить по результату smoke, пришли:

- Текстовый вывод последних 80–100 строк консоли (включая «Analysis of smoke run» и «Summary»), **или**
- Содержимое `results_smoke/active_loop_log.json` и вывод `python analyze_run.py --results-dir results_smoke` (скопировать в чат).

Опционально: одна запись из `results_smoke/debug_trace.jsonl` с `event: "consolidation_decision"` и одна с `event: "evolution_done"` — чтобы проверить, что консолидация и эволюция отрабатывают как задумано.

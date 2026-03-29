# Архитектура и файлы (актуальная привязка к коду)

Дополняет [PROJECT_REPORT.md](PROJECT_REPORT.md) сжатым списком модулей и констант из текущего дерева.

## Двухуровневая эволюция

1. **Уровень 1 — OpenEvolve** (`openevolve` + локальный `evaluator.py`): мутации только `<DynamicRules>` и `<FewShotExamples>`; родитель + топ из MAP-Elites + артефакты ошибок.
2. **Уровень 2 — консолидация** (`active_loop.py`): LLM может менять все секции, включая `<BaseGuidelines>`. Принятие по **consolidation gate** на **validation**, не на батче.

## Ключевые файлы

| Файл | Роль |
|------|------|
| `active_loop.py` | Оркестратор AL: батч → эволюция → реклассификация → val → консолидация → расширение пула; лог `active_loop_log.json`, `debug_trace.jsonl`. |
| `evaluator.py` | Оценка кандидата на батче/сплитах; combined fitness, артефакты (ошибки, RULE COVERAGE, few-shot balance, confusion pairs). |
| `data_manager.py` | Пул Seen/Unseen, Hard/Anchor, `build_active_batch`, refresh/expansion. |
| `error_analyzer.py` | Агрегация ошибок для артефактов мутатора. |
| `config.yaml` | LLM эволютор, воркеры, MAP-Elites, `active_learning`, датасет, `prompt_path`, `output_dir`. |
| `dataset.yaml` | WILDS: категория, сплиты, лимиты пользователей. |
| `initial_prompt.txt` | Стартовый XML-промпт. |
| `visualize.py` | Графики из `active_loop_log.json` (+ baseline/final test JSON при наличии). |
| `analyze_run.py` | Сводный анализ прогона (если используется). |

## MAP-Elites (из `config.yaml`)

- `feature_dimensions`: `prompt_length`, `Acc_Hard`
- `feature_bins`: 5×5
- `population_size` / `archive_size` / `num_islands`

## Фитнес на эволюции (батч)

Согласовано с `config.yaml` → `prompt.system_message` и реализацией в `evaluator.py`:

- `combined_score` / fitness-логика: **0.5 × Acc_Hard + 0.3 × Acc_Anchor + 0.2 × kappa_Hard − length_penalty**
- **Length penalty**: в коде константа `PROMPT_LEN_LIMIT = 2000` (токены); штраф нарастает за превышение (см. `evaluator.py` около расчёта `excess`).

## Валидация и отбор промпта

- На val считаются `R_global`, `R_worst`, MAE, kappa, Acc_Hard/Anchor на val и **`val_combined_score`** (веса как в логе; см. код метрик в `wilds_experiment/experiments`).
- Глобально трекается **`best_val_prompt`** → `best_val_prompt.txt` в каталоге результатов.

## Active Learning (`config.yaml` → `active_learning`)

| Параметр | Смысл |
|----------|--------|
| `uncertainty_threshold` | Порог disagreement для класса Hard (0.0 = любая несогласованность). |
| `batch_size` | Размер активного батча (фиксированный, например 80). |
| `hard_ratio` | Доля Hard в батче (остальное Anchor). |
| `expansion_trigger` | Если число Hard ≤ порога — расширение из Unseen. |
| `soft_expansion_patience` / `soft_expansion_min_delta` | Стагнация val → мягкое расширение. |
| `refresh_per_cycle` | Сколько Unseen→Seen промотировать за цикл. |
| `seen_reeval_interval` | Периодическая переоценка Seen (0 = выкл.). |
| `consolidation_gate_delta` | Макс. допустимое падение val combined у консолидации относительно evo val. |

## Модели

- **Эволютор / консолидация**: DeepSeek R1 (`deepseek/deepseek-r1`), OpenRouter, высокая temperature.
- **Воркеры ансамбля**: например DeepSeek Chat V3, Gemma3-27B, GPT-4o-mini; majority vote.

## Артефакты прогона (типичная структура `results_*`)

- `active_loop_log.json` — по циклу: val-метрики, `evo_best_score`, флаги `consolidated`, `expanded`, при наличии — **test_*** только для мониторинга.
- `baseline_test_metrics.json`, `final_test_metrics.json`
- `al_iter_k/start_prompt.txt`, `best_prompt.txt`, `consolidated_prompt.txt`, `openevolve_output/`

Подробные таблицы экспериментов v2/v3 — в [PROJECT_REPORT.md](PROJECT_REPORT.md); сравнения недавних прогонов — в [EXPERIMENTS_AND_LESSONS.md](EXPERIMENTS_AND_LESSONS.md).

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
| `min_hard_batch_ratio` | Предупреждение, если доля Hard в батче ниже порога (когда пул позволяет). |
| `expansion_trigger` | Если число Hard ≤ порога — расширение из Unseen. |
| `soft_expansion_patience` / `soft_expansion_min_delta` | Стагнация val → мягкое расширение. |
| `soft_expansion_skip_near_best` | Не запускать soft expansion, если val на пике (±epsilon от лучшего). |
| `refresh_per_cycle` | Сколько Unseen→Seen промотировать за цикл. |
| `seen_reeval_interval` | Периодическая переоценка Seen (0 = выкл.). |
| `consolidation_gate_delta` | Макс. допустимое падение val у консолидации vs evo val этого цикла. |
| `consolidation_gate_vs_best_delta` | Макс. допустимое падение val у консолидации vs глобальный лучший val. |
| `al_early_stopping_patience` | Остановка AL, если нет нового лучшего val за N циклов (0 = выкл.). |
| `al_early_stopping_min_cycles` | Мин. число циклов до срабатывания AL early stop. |

## Модели

- **Эволютор / консолидация**: DeepSeek R1 (`deepseek/deepseek-r1`), OpenRouter, `temperature: 0.8`.
- **Воркеры ансамбля**: DeepSeek Chat V3, Gemma3-27B, GPT-4o-mini; `temperature: 0.1` (рекомендовано → 0.0, см. [V9_V10_EXPERIMENTS.md](V9_V10_EXPERIMENTS.md)); majority vote.

## Конфиг OpenEvolve

| Параметр | Значение | Смысл |
|----------|----------|-------|
| `max_iterations` | 100 (soft; early stop обычно срабатывает раньше) | Макс. итераций эволюции за цикл. |
| `max_code_length` | 7500 | Макс. символов промпта. |
| `early_stopping_patience` | 5 | Итераций без улучшения → стоп эволюции цикла. |
| `early_stopping_metric` | `combined_score` | Метрика для отслеживания early stop. |
| `convergence_threshold` | 0.001 | Мин. улучшение для сброса счётчика patience. |

## Артефакты прогона (типичная структура `results_*`)

- `active_loop_log.json` — по циклу: val-метрики, `evo_best_score`, флаги `consolidated`, `expanded`, `best_val_cycle`, `global_best_val_score`, при наличии — **test_*** только для мониторинга.
- `baseline_test_metrics.json`, `final_test_metrics.json`
- `best_val_meta.json` — цикл и скор лучшего промпта по val.
- `final_prompt.txt` — промпт с лучшим val (используется для финального теста).
- `last_cycle_prompt.txt` — промпт последнего цикла (для сравнения).
- `al_iter_k/start_prompt.txt`, `best_prompt.txt`, `consolidated_prompt.txt`, `openevolve_output/`
- `debug_trace.jsonl` — детальные события (init, cycle_start, evolution_done, consolidation_decision, refresh, и т.д.).
- `token_usage_report.md` — расход токенов по моделям.

Подробные таблицы экспериментов v2/v3 — в [PROJECT_REPORT.md](PROJECT_REPORT.md); сравнения недавних прогонов — в [EXPERIMENTS_AND_LESSONS.md](EXPERIMENTS_AND_LESSONS.md); детально v9/v10 — в [V9_V10_EXPERIMENTS.md](V9_V10_EXPERIMENTS.md).

# Uncapped-train AL run on WILDS Amazon (all categories)

Active Learning эксперимент с большим train‑пулом и расширенными val/test.

Старый plain‑абляционный прогон (375‑ревью пул и val/test по 225) лежит в
`results_all_categories_plain_evolve_fixedsubsample/`. Для **сопоставления с
этим uncapped AL** нужен новый plain‑прогон с теми же загрузчиками данных, что и
в `config_all_categories_uncapped_train.yaml` (pool 5000, val/users 30,
test/users 40) — см. раздел ниже и флаг `--train-size/--val-size/--test-size`.

## TL;DR

```powershell
# Windows / локально
.\run_uncap_al.ps1            # полный прогон (~6-9 ч на одной машине)
.\run_uncap_al.ps1 -Smoke     # 2 цикла x 4 итерации, ~30 мин (sanity check)
```

```bash
# Linux / кластер
./run_uncap_al.sh             # полный прогон
./run_uncap_al.sh smoke       # smoke
./run_uncap_al.sh resume      # продолжить упавший прогон
```

Перед запуском: `OPENROUTER_API_KEY` (или `OPENAI_API_KEY`) в окружении или в
`.env` рядом с `active_loop.py`.

## Что меняется vs `config_all_categories.yaml`

| Параметр | Было | Стало | Зачем |
|---|---|---|---|
| `dataset.max_train_users` | 25 | **0** (снят) | AL acquisition должен иметь реальный выбор |
| `dataset.al_candidate_pool_size` | — | **5000** | Стратифицированный кандидатский пул из всего train (8× от старого 375) |
| `dataset.max_val_users` | 15 | **30** | val ≈ 450, R_worst считается стабильнее |
| `dataset.max_test_users` | (=val) | **40** | test ≈ 600 для per‑cycle лога |
| `active_learning.expansion_trigger` | 15 | **25** | расширяться раньше на большом пуле |
| `active_learning.refresh_per_cycle` | 10 | **40** | за цикл подкачивать в 4× больше Unseen |
| `active_learning.al_early_stopping_patience` | 2 | **3** | не выключаться слишком рано |

Размер активного батча и воркеры (`gpt-4o-mini`, `gemini-2.5-flash`,
`claude-3.5-haiku`) — без изменений: сравнение fair vs zaфиксированный plain.

Seed prompt и init‑стратегия: тот же `initial_prompt_all_categories.txt`.

## Что получится в `results_all_categories_uncapped_train/`

- `al_pool_manifest.json` — состав кандидатского пула 5000 (pool_indices в WILDS train, user_ids, гистограмма по юзерам). Записывается один раз на старте, нужен для воспроизводимости.
- `active_loop_log.json` — per‑cycle метрики (val/test combined, R_global/R_worst, mae, mean_kappa, n_seen/n_unseen, refresh/expansion события). Структурно совместим с `plot_ablation_compare.py`.
- `al_iter_<i>/best_prompt.txt`, `al_iter_<i>/openevolve_output/` — артефакты каждого AL‑цикла.
- `best_val_prompt.txt` + `best_val_meta.json` — prompt с лучшим val_combined_score за весь прогон. Именно его финально оценивают на test.
- `baseline_test_metrics.json` — стартовый prompt на capped test (~600).
- `final_test_metrics.json` — `best_val_prompt` на capped test (~600).
- **`full_uncapped_test_metrics.json`** — `best_val_prompt` на ПОЛНОМ WILDS test (все юзеры, все отзывы). Главный отчётный замер.
- `token_usage.json`, `token_usage_report.md` — учёт расхода токенов.
- `debug_trace.jsonl` — детальный JSONL‑лог.

## Бюджет (one seed=42, single node, OpenRouter)

| Этап | LLM‑вызовы | Wall‑time |
|---|---|---|
| AL init (full‑pool eval, 5000 × 3) | 15 000 | ~50 мин |
| Per AL cycle (evolve+val+test+refresh+consolidation) | ~7 500 | ~40 мин |
| 8 циклов | ~60 000 | ~5.5 ч |
| Final full uncapped test (~25–30 k × 3) | ~80 000 | ~3 ч |
| **Итого** | ~155 000 | **~9 ч** |

Smoke (2×4) — около 25–35 мин на ту же конфигурацию.

## Plain‑абляция под ту же конфигурацию данных (pool 5000, val≈450, test≈600)

`fixed_splits.py` строит **тот же** предел по пользователям/кандидатскому пулу,
что `data_manager` при `max_train_users: 0` и `al_candidate_pool_size: 5000`, и
вырезает **фиксированные** stratified‑подмножества (train 80 совпадает с
активным батчевым размером AL; val/test — таргеты 450/600 или фактический
размер пула ≤ этих чисел после `stratified_downsample_pick`).

Запуск (Windows, из `wilds_active_learn_approach`; те же ключи OpenRouter).
`--n-al 6` совпадает с фактическим числом циклов uncapped‑AL при early‑stop (до 8
можно поставить, если нужно тотальный бюджет шагов 8×15).

```powershell
python plain_evolution_fixedsubsample.py `
  --config config_all_categories_uncapped_train.yaml `
  --results-dir results_plain_uncapped_data_match `
  --manifest-name fixed_splits_uncapped_match_v1.json `
  --train-size 80 `
  --val-size 450 `
  --test-size 600 `
  --n-al 6 `
  --n-evolve 15
```

Отложенный старт **через 3 часа** (сначала пауза, затем тот же прогон; окно
PowerShell не закрывать, иначе таймер прервётся). В Windows PowerShell 5.1 у
`Start-Sleep` часто есть только **`-Seconds`** (нет `-Hours`); ниже используется
`10800` секунд (= 3 ч).

```powershell
cd c:\Users\march\things\mipt\AlphaEvolveProject\openevolve\examples\llm_prompt_optimization\wilds_active_learn_approach; Start-Sleep -Seconds 10800; python plain_evolution_fixedsubsample.py `
  --config config_all_categories_uncapped_train.yaml `
  --results-dir results_plain_uncapped_data_match `
  --manifest-name fixed_splits_uncapped_match_v1.json `
  --train-size 80 `
  --val-size 450 `
  --test-size 600 `
  --n-al 6 `
  --n-evolve 15
```

Чтобы таймер жил в **отдельном** окне (можно свернуть основной терминал):

```powershell
Start-Process powershell -ArgumentList @(
  '-NoExit','-Command',
  'cd ''c:\Users\march\things\mipt\AlphaEvolveProject\openevolve\examples\llm_prompt_optimization\wilds_active_learn_approach''; Start-Sleep -Seconds 10800; python plain_evolution_fixedsubsample.py --config config_all_categories_uncapped_train.yaml --results-dir results_plain_uncapped_data_match --manifest-name fixed_splits_uncapped_match_v1.json --train-size 80 --val-size 450 --test-size 600 --n-al 6 --n-evolve 15'
)
```

Сохранять старый plain не трогая: всегда **другая** директория `--results-dir` и при
необходимости свой `--manifest-name`. Повторный прогон с тем же manifest: добавить
`--reuse-fixed-splits` (если JSON уже есть).

Графики: `python visualize.py --results-dir results_plain_uncapped_data_match`.

## Оверлей со старым plain (несогласованные выборки; только исторический контекст)

```powershell
python plot_ablation_compare.py `
    --run-a results_all_categories_uncapped_train `
    --run-b results_all_categories_plain_evolve_fixedsubsample `
    --label-a "AL (uncapped train, pool=5000)" `
    --label-b "Plain (fixed 80, pool=375)" `
    --out plots\ablation_uncap_vs_plain `
    --extra-subdir uncap_vs_plain
```

## Оверлей с matched plain (рекомендуется)

```powershell
python plot_ablation_compare.py `
    --run-a results_all_categories_uncapped_train `
    --run-b results_plain_uncapped_data_match `
    --label-a "AL (uncapped, pool=5000)" `
    --label-b "Plain (same pool caps, fixed 80/450/600)" `
    --out plots\ablation_uncap_vs_plain_matched `
    --extra-subdir matched
```

На что смотреть в первую очередь:
- `val_combined_score` / `test_combined_score` per cycle — кривая обучения и стабильность.
- `R_worst` per cycle — основная цель active‑batch фокуса.
- `cycles‑to‑plateau` — ожидаем, что AL раньше выходит на плато.
- `final_test_metrics.json` (capped) и `full_uncapped_test_metrics.json` — финальный замер. На полном test разница 0.005 уже статистически различима (n≈25k).

## Resume / падение API

Если упало (например, 401), в `results_dir` будет `resume_state.json`. Просто:

```bash
./run_uncap_al.sh resume
```

Прогон возобновится с последнего завершённого цикла, full test тоже отработает.

## Что менять, если хочется второй seed позже

Единственный seed зашит в коде (`seed=42 + al_iter` для batch, RNG для пула — также 42).
Чтобы прогнать второй seed без правок ядра, скопируй конфиг в
`config_all_categories_uncapped_train_seed1337.yaml`, поменяй `dataset.split_seed`
в `dataset_all_categories.yaml` (создай отдельный `dataset_all_categories_seed1337.yaml`,
сошлись на него из конфига) и запусти с другим `--results-dir`. Для текущей задачи
это не нужно — сначала прогоняем один seed.

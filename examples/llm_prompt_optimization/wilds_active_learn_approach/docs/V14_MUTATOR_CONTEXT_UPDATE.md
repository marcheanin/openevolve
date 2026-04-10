# V14 Update: Richer Mutator Context + Hint-based Synthetic Few-shot

Этот документ фиксирует изменения, сделанные по последнему плану (v14), где цель была:

- дать мутатору больше полезного контекста;
- убрать "амнезию" few-shot между AL-циклами;
- сохранить свободу мутатора при работе с синтетикой;
- повысить устойчивость эволюции к плато.

## Что изменено

## 1) Режим синтетики: `replace` -> `inject_as_hint`

Файл: `../config.yaml`

- `active_learning.synthetic_fewshot.mode` переключен на `inject_as_hint`.
- `mutator_lock_fewshot` оставлен `false` (few-shot разрешено менять).
- `n_examples`: `10 -> 8`.
- `n_boundary_pairs`: `3 -> 4`.

Идея: не перезаписывать `<FewShotExamples>` перед каждым циклом, а передавать синтетический draft как подсказку в system message. Это позволяет сохранить удачные few-shot из прошлых циклов.

## 2) Обогащен `error_context` для мутатора

Файл: `../error_analyzer.py`

`ErrorAnalyzer.format_for_evolution(...)` расширен. Теперь может принимать:

- `predictions` (предсказания прошлого цикла),
- `worker_predictions` (голоса воркеров),
- `batch_indices`,
- `batch_stats`.

В контекст добавляются:

- сводка по батчу и пулу (Hard/Anchor/Seen/Unseen),
- confusion matrix по top-парам `gold -> pred` (из предыдущей оценки батча),
- representative примеры по confusion-парам с `Gold/Pred/Workers`,
- плюс прежний блок кластеризованных hard-примеров (fallback/дополнение).

## 3) Межцикловая память о качестве батча

Файл: `../active_loop.py`

Добавлено сохранение и перенос между циклами:

- `last_cycle_predictions`,
- `last_cycle_worker_preds`,
- `last_cycle_batch_indices`,
- `last_cycle_seed_val_score`.

Эти данные используются в следующем цикле для формирования более информативного `error_context`.

## 4) Добавлен `Cycle Context` в system message

Файл: `../active_loop.py`

Перед `Error Pattern Summary` теперь добавляется отдельный блок:

- номер цикла,
- состав текущего батча (Hard/Anchor),
- состояние пула (Seen/Unseen/Hard/Anchor),
- лучший `seed_val` глобально,
- `seed_val` предыдущего цикла,
- напоминание про length penalty.

Идея: дать мутатору situational awareness перед правками.

## 5) Синтетический генератор: structured confusion + contrastive focus

Файл: `../synthetic_fewshot_generator.py`

Изменения:

- `generate(...)` и `_build_messages(...)` теперь поддерживают `confusion_pairs`.
- В `active_loop.py` добавлен helper `_top_confusion_pairs_from_batch(...)`, который передает в генератор top confusion-пары из предыдущего батча.
- Если structured пары не переданы, остается fallback-парсинг из `error_context` (обратная совместимость).
- Промпт генератора усилен: фокус на top 3-4 границах и contrastive примерах.

## 6) Увеличен объем артефактов ошибок на итерацию

Файл: `../evaluator.py`

В `_format_error_artifacts(...)`:

- `max_errors`: `7 -> 10`,
- `max_borderline`: `3 -> 5`,
- `max_text_len`: `250 -> 350`.

Идея: мутатор видит больше реальных ошибок и меньше теряет из-за агрессивной обрезки текста.

## 7) Результат по архитектуре потока

После v14 поток внутри AL-цикла выглядит так:

1. Batch build -> Error analysis.
2. Synthetic draft генерируется в режиме hint.
3. Система собирает `augmented_system`:
   - базовые инструкции,
   - optional synthetic hint,
   - cycle context,
   - enriched error context.
4. OpenEvolve запускает итерации мутации.
5. После re-eval сохраняются предсказания/голоса для следующего цикла.

## Пример команды запуска v14 (fresh)

```powershell
python .\active_loop.py --prompt ".\initial_prompt.txt" --results-dir ".\results_v14_hint" --n-al 7 --n-evolve 20 --no-evolve-early-stop
```

## Примечание

В ходе последующих запусков модель мутатора была переключена на Gemini (`google/gemini-2.5-pro`) в `config.yaml`. Это отдельный операционный шаг поверх v14-изменений контекста/синтетики.


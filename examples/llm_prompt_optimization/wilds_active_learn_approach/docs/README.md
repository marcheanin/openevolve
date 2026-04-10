# Документация: WILDS Active Prompt Evolution

Центральный каталог контекста для работы с пайплайном **OpenEvolve + Active Learning** на Amazon Home & Kitchen (рейтинги 1–5).

## Оглавление

| Документ | Назначение |
|----------|------------|
| [pipeline_desc.txt](pipeline_desc.txt) | Краткое техническое описание v4-пайплайна (батч, консолидация, пул, фитнес). |
| [PROJECT_REPORT.md](PROJECT_REPORT.md) | Полный отчёт: идея, стек, архитектура, эксперименты v2/v3, метрики, выводы, направления развития. |
| [ARCHITECTURE_AND_FILES.md](ARCHITECTURE_AND_FILES.md) | Структура репозитория, модули, конфиг, метрики — сжато и с привязкой к файлам. |
| [EXPERIMENTS_AND_LESSONS.md](EXPERIMENTS_AND_LESSONS.md) | Хронология прогонов (v4_tes, v5, v6), ветка baserules-consolidation, длина промпта, утечки test, что пробовали. |
| [V8_PIPELINE_UPDATE.md](V8_PIPELINE_UPDATE.md) | Итоги **v8**, выводы по метрикам, список правок пайплайна после v8 (best_val final test, gates, batch, AL early stop). |
| [V9_V10_EXPERIMENTS.md](V9_V10_EXPERIMENTS.md) | Итоги **v9** и **v10**: эффект свободного system_message, сравнение всех прогонов, val-test gap, рекомендации по temperature и next steps. |
| [EXPERIMENT_V12_SYNTHETIC_FEWSHOT.md](EXPERIMENT_V12_SYNTHETIC_FEWSHOT.md) | План и запуск нового прогона **v12** с полностью синтетическими FewShotExamples (Synthetic FewShot-only). |
| [V14_MUTATOR_CONTEXT_UPDATE.md](V14_MUTATOR_CONTEXT_UPDATE.md) | Что изменено в **v14**: режим `inject_as_hint`, обогащенный контекст мутатора, carry-over batch-сигналов, больше error artifacts, обновления генератора синтетики. |
| [TEST_VAL_AND_AL_POLICY.md](TEST_VAL_AND_AL_POLICY.md) | Политика train/val/test: что влияет на обучение, что только для графиков. |
| [token_usage_reference.md](token_usage_reference.md) | Сводка по токенам (пример: `results_v4_tes`). |
| [SYNTHETIC_FEWSHOT_DESIGN.md](SYNTHETIC_FEWSHOT_DESIGN.md) | Дизайн-документ: синтетические граничные FewShotExamples (по мотивам AutoPrompt / Intent-based Prompt Calibration, ICLR 2024). Проблема, архитектура, план внедрения. |
| [error_context_sample.txt](error_context_sample.txt) | Снимок фрагмента контекста ошибок (статический пример). При запуске `active_loop.py` актуальный контекст перезаписывается в **`../error_context.txt`** в корне подхода. |

## Быстрый старт

- Конфигурация: `../config.yaml` (корень подхода).
- Стартовый промпт: `../initial_prompt.txt`.
- Точка входа: `../active_loop.py`.
- Графики: `../visualize.py` (читает `results/.../active_loop_log.json`).

## Примечание о путях

Файлы `PROJECT_REPORT.md` и `pipeline_desc.txt` перенесены сюда из корня `wilds_active_learn_approach/`; внутренние ссылки в старом отчёте могут указывать на прежние относительные пути — используйте структуру из [ARCHITECTURE_AND_FILES.md](ARCHITECTURE_AND_FILES.md).

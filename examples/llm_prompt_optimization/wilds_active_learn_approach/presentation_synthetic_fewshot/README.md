# Презентация: синтетические boundary few-shot

Материалы доклада по внедрению идеи из статьи **Intent-based Prompt Calibration** (Levi, Brosh, Friedmann), [arXiv:2402.03099](https://arxiv.org/abs/2402.03099).

## Файлы

| Файл | Описание |
|---|---|
| `presentation_synthetic_fewshot_demo_report.pptx` | Актуальная презентация, 10 слайдов (16:9), **язык: русский** |
| `presentation_synthetic_fewshot.pptx` | Предыдущая версия; могла быть заблокирована PowerPoint во время пересборки |
| `talk_script.md` | Текст доклада (~6–7 минут) |
| `build_presentation.py` | Сборка .pptx (`python build_presentation.py`) |
| `build_charts.py` | PNG в `assets/` (`python build_charts.py`) |
| `slides_preview/slide1.png` … `slide10.png` | Экспорт слайдов из PowerPoint для просмотра |

## Пересборка

```powershell
cd presentation_synthetic_fewshot
python build_charts.py
python build_presentation.py
```

## Слайд 3 (обзор статьи)

В презентации кратко отражены **эмпирические итоги IPC** из arXiv:2402.03099: постановки классификации и генерации, сравнение с OPRO/PE, роль синтетики и Analyzer, численные примеры по ranker для двух генеративных задач (табл. 2 в PDF).

## Расширенное демо (слайды 6–10)

- **Слайд 6:** отчёт о входе генератора: manifest, режим `inject_as_hint`, количество hard-примеров, целевые boundary-пары и контекст из `_build_messages()`.
- **Слайд 7:** почти полный synthetic pair `5★ / 4★` про coffee maker из `al_iter_5` + локальный эффект цикла.
- **Слайд 8:** почти полный synthetic pair `3★ / 4★` про knife set из `al_iter_5` + прирост `Acc_Hard`.
- **Слайд 9:** почти полный synthetic pair `2★ / 1★` про electric kettle из `al_iter_5`.
- **Слайд 10:** общий прирост метрик + локальный cycle-level effect `al_iter_5`.

Источники примеров:

- `results_all_categories_evolve_subsample/al_iter_5/synthetic_fewshot_examples.txt`
- `results_all_categories_evolve_subsample/al_iter_*/synthetic_fewshot_manifest.json`

## Нарратив по своим экспериментам

- Одна категория: R_worst **+6.6 п.п.** (v11 → v14_gemini_fresh).
- Все 15 категорий: R_worst **+13.4 п.п.** (uncapped_train → evolve_subsample); сравнение не полностью контролируемое по размеру train pool — это проговаривается в докладе.
- Абляция 8×20: устойчивый плюс по **Acc_Hard**; R_worst на малом тесте шумный.

## Зависимости

`python-pptx`, `matplotlib` (для графиков).

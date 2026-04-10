# Эксперимент v12: Synthetic FewShot-only

Этот эксперимент включает новый этап генерации **полностью синтетических**
`<FewShotExamples>` перед каждой эволюцией в AL-цикле.

## Что включено

- Новый модуль: `synthetic_fewshot_generator.py`
- Интеграция в `active_loop.py`:
  - перед эволюцией генерируется новый блок `<FewShotExamples>`;
  - блок полностью заменяет текущий few-shot;
  - в system message добавляется lock: мутатор не должен менять `<FewShotExamples>`;
  - артефакты цикла:
    - `al_iter_k/synthetic_fewshot_examples.txt`
    - `al_iter_k/synthetic_fewshot_manifest.json`
- Конфиг: `active_learning.synthetic_fewshot` в `config.yaml`

## Рекомендуемый запуск

Из каталога `wilds_active_learn_approach`:

```powershell
python active_loop.py --n-al 7 --n-evolve 20 --results-dir results_v12_synth
```

Если нужно продолжить после сбоя:

```powershell
python active_loop.py --resume-from-dir results_v12_synth --n-al 7 --n-evolve 20
```

## Что проверить после прогона

1. В `results_v12_synth/al_iter_0/` появился `synthetic_fewshot_examples.txt`.
2. В `active_loop_log.json` есть события `synthetic_fewshot` в `debug_trace.jsonl`.
3. `best_prompt.txt` каждого цикла содержит обновлённый `<FewShotExamples>`.
4. Сравнить с baseline:
   - `test_Acc_Hard`
   - `test_combined_score`
   - `val-test gap` по `combined_score`

## Ключевая гипотеза

Synthetic FewShot-only должен повысить качество на hard-boundary случаях
(в первую очередь `Acc_Hard`) за счёт контрастных граничных примеров, которые
реальный датасет не даёт в достаточно «чистом» виде.


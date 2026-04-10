# results_v12_synth

Папка под новый прогон с синтетическими FewShotExamples.

Запуск:

```powershell
python active_loop.py --n-al 7 --n-evolve 20 --results-dir results_v12_synth
```

Или:

```powershell
./run_v12_synthetic_fewshot.ps1
```

После запуска здесь появятся:
- `active_loop_log.json`
- `debug_trace.jsonl`
- `baseline_test_metrics.json`
- `final_test_metrics.json`
- `al_iter_k/...` артефакты циклов, включая synthetic few-shot файлы.


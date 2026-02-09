# WILDS Amazon Prompt Evolution Experiment

Эволюция промптов для задачи sentiment classification на датасете WILDS Amazon.

## 📚 Документация

- **[WILDS_EXPERIMENT_SUMMARY.md](WILDS_EXPERIMENT_SUMMARY.md)** — полная документация эксперимента
- **[Flow.md](Flow.md)** — теоретическое обоснование и мотивация

## 🚀 Быстрый старт

```powershell
# 1. Убедитесь, что датасет загружен (./data/amazon_v2.1/)
python analyze_dataset.py

# 2. Запустите baseline
python baseline_embedding_pipeline.py --embedding roberta_sentiment --classifier xgb

# 3. Запустите эволюцию
.\run_evolution.ps1 -iterations 100

# 4. Проанализируйте результаты
.\analyze_improvements.ps1
```

## 📊 Результаты

| Метод | Val Accuracy | Test Accuracy |
|-------|--------------|---------------|
| **Baseline (RoBERTa + XGB)** | 70.3% | 66.4% |
| **LLM + эволюция** | ~74% | ~70-72% |
| **LLM + auto few-shot** | ~75-77% | ~72-75% |

## 📁 Структура

```
wilds_experiment/
├── data/                          # Данные WILDS Amazon
├── experiments_wilds/             # Результаты экспериментов
├── baseline_output/               # Результаты baseline
├── config.yaml                    # Конфигурация OpenEvolve
├── wilds_prompt.txt               # Начальный промпт
├── evaluator.py                   # Основной evaluator
├── baseline_embedding_pipeline.py # Baseline пайплайн
├── experiments/                   # eLLM ensemble experiments (Exp1-4)
└── WILDS_EXPERIMENT_SUMMARY.md    # Полная документация
```

## 🧪 eLLM Ensemble Experiments

Эксперимент 1 (baseline) доступен в `experiments/exp1_baseline/`:

```powershell
cd experiments/exp1_baseline
python run.py
```

## 📦 Зависимости

```bash
pip install -r requirements.txt
```

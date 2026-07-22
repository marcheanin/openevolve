# ACLR Submission Bundle (PRIME / All Categories)

This folder contains a clean runnable bundle for the all-categories PRIME experiment,
using OpenEvolve as an importable Python library (`openevolve`).

## Included
- Training/evolution loop (`active_loop.py`)
- Evaluation and workers (`evaluator.py`, `workers.py`)
- Active-learning data manager (`data_manager.py`)
- Synthetic few-shot generator (`synthetic_fewshot_generator.py`)
- Error analyzer (`error_analyzer.py`)
- Full uncapped test runner (`run_full_test.py`)
- All-categories config and dataset config
- Example initial prompt (`initial_prompt_all_categories.txt`)
- One-command runner (`run_experiment.py`)

## Install
```bash
pip install -r requirements.txt
pip install openevolve
```

## Run all-categories experiment
```bash
python run_experiment.py --n-al 8 --n-evolve 15 --results-dir results_all_categories_prime_submission --run-full-test
```

## Run full test only (after evolution)
```bash
python run_experiment.py --skip-evolve --run-full-test --results-dir results_all_categories_prime_submission
```

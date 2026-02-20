# Active Prompt Evolution for WILDS Amazon

Active Learning approach for prompt optimization on WILDS Amazon Reviews (Home and Kitchen).

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   pip install -e ../../..  # OpenEvolve from openevolve/
   ```

2. Set `OPENROUTER_API_KEY` (or `OPENAI_API_KEY` for OpenRouter).

3. Ensure WILDS data is available (runs from `wilds_experiment/data` or downloads on first run).

## Pipeline

### 1. APE Initial Prompt
```bash
python prepare_initial_prompt.py
```
Paste the generated `ape_meta_prompt.txt` into ChatGPT, save the best variant to `initial_prompt.txt`.

### 2. Baseline
```bash
python run_baseline.py
```
Evaluates `initial_prompt.txt` on train+val, saves to `results/baseline/`.

### 3. Active Learning Loop
```bash
python active_loop.py --n-al 4 --n-evolve 15
```
Default: 4 AL cycles × 15 evolve iterations (~60 total). Adjust with `--n-al` / `--n-evolve`.

### 4. Report and Plots
```bash
python visualize.py
python generate_final_report.py
```

Or run all steps:
```bash
python run_pipeline.py --all
```

## Config

- `config.yaml` - OpenRouter models (DeepSeek, Gemma, GPT-4o-mini for ensemble; DeepSeek R1 for evolution)
- `dataset.yaml` - Home_and_Kitchen, category_id 24

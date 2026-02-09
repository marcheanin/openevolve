# LiveBench Instruction Following Prompt Evolution

This experiment uses OpenEvolve to evolve prompts for the LiveBench Instruction Following benchmark.

## Overview

LiveBench IF is a challenging benchmark that tests LLM's ability to follow precise instructions with format constraints. This experiment:

- Uses **official LiveBench evaluation scripts** (IFEval + IFBench)
- Supports **train/validation/test split** with random sampling
- Features **adaptive sample sizes** that increase with evolution progress
- Allows **separate models** for evolution and evaluation

## Quick Start

### 1. Install Dependencies

```bash
# Install LiveBench
cd LiveBench
pip install -e .
cd ..

# Install other dependencies
pip install -r requirements.txt
```

### 2. Run Evolution

```powershell
# Basic run (100 iterations)
.\run_evolution.ps1

# Custom iterations
.\run_evolution.ps1 -iterations 50

# With separate models for evolution and evaluation
.\run_evolution_separate_models.ps1 -iterations 100
```

### 3. Visualize Results

```bash
python visualize_evolution.py
python analyze_improvements.py
```

## Files

| File | Description |
|------|-------------|
| `evaluator.py` | Main evaluator using LiveBench official scripts |
| `evaluator_separate_models.py` | Evaluator with separate evolution/evaluation models |
| `config.yaml` | Configuration for single-model setup |
| `config_separate_models.yaml` | Configuration for separate models |
| `livebench_prompt.txt` | Initial prompt template |
| `livebench_prompt_dataset.yaml` | Dataset configuration |
| `visualize_evolution.py` | Generate learning curves and metrics plots |
| `analyze_improvements.py` | Analyze prompt improvements |
| `run_evolution.ps1` | PowerShell script to run evolution |

## Dataset Split

The LiveBench IF dataset is automatically split:

- **Train (70%)**: Used for evolution
- **Validation (15%)**: For monitoring overfitting (optional)
- **Test (15%)**: For final evaluation

Split is randomized but reproducible (seed=42).

## Adaptive Sample Sizes

Sample sizes increase as evolution progresses:

| Stage | Start | End |
|-------|-------|-----|
| Stage 1 (quick filter) | 10 | 20 |
| Stage 2 (comprehensive) | 20 | 60 |

## Evaluation Methods

The evaluator uses official LiveBench scripts:

1. **IFEval format** (original Google): 25 constraint types
   - `keywords:existence`, `length_constraints:number_words`, etc.

2. **IFBench format** (newer, harder): 58 constraint types
   - `count:word_count_range`, `ratio:sentence_type`, etc.

## Configuration

### Single Model (`config.yaml`)

Same model for evolution and evaluation.

### Separate Models (`config_separate_models.yaml`)

```yaml
# Evolution model (can be cheaper/faster)
llm:
  models:
    - name: "gpt://folder/qwen3-235b-a22b-fp8/latest"

# Evaluation model (high quality)
evaluation:
  model: "gpt://folder/yandexgpt/rc"
```

## Results

After running evolution:

```
openevolve_output/
├── best/
│   ├── best_program.txt      # Best evolved prompt
│   └── best_program_info.json
├── checkpoints/
├── evolution_trace.jsonl
└── visualizations/
    ├── learning_curve.png
    ├── metrics_evolution.png
    ├── evolution_report.txt
    └── improvements/
        ├── improvements_summary.json
        └── improvement_*.md
```

## Metrics

- **combined_score**: Accuracy on instruction following (0-1)
- **prompt_length**: Character count of prompt
- **reasoning_strategy**: Sophistication score (0-1)

## References

- [LiveBench Paper](https://arxiv.org/abs/2406.19314)
- [LiveBench Leaderboard](https://livebench.ai/)
- [IFBench Paper](https://arxiv.org/pdf/2507.02833)
- [HuggingFace Dataset](https://huggingface.co/datasets/livebench/instruction_following)


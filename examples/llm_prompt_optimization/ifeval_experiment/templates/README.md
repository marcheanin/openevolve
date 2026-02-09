# Templates Directory for IFEval Experiment

This directory contains prompt templates for the IFEval prompt optimization experiment.

## Files

### `evaluation.txt`
Template for LLM feedback evaluation. Used by `_llm_evaluate()` method to get qualitative feedback on prompt quality.

**Key features:**
- Evaluates prompts on 4 dimensions: Clarity, Specificity, Robustness, Format_specification
- Adapted specifically for IFEval (Instruction Following Evaluation) task
- Returns JSON format with scores and reasoning

### `evaluator_system_message.txt`
System message for the prompt evaluator. Provides context for the LLM evaluating prompts.

### `full_rewrite_user.txt`
Template for full prompt rewrites during evolution. Used when `diff_based_evolution: false`.

**Key features:**
- Adapted for IFEval task with emphasis on constraint adherence
- Requires preserving the `{instruction}` placeholder
- Focuses on improving instruction-following accuracy

## Usage

These templates are automatically loaded by `TemplateManager` when:
1. `template_dir: "templates"` is set in config.yaml
2. The templates directory exists in the experiment folder

The templates override default templates from `openevolve/openevolve/prompts/defaults/`.

## Template Loading Order

1. Default templates (from `openevolve/openevolve/prompts/defaults/`)
2. Custom templates (from this directory) - **these override defaults**

This means if `evaluation.txt` exists here, it will be used instead of the default evaluation template.


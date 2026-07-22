# Full test: per-worker vs ensemble (AL iter 5 prompt)

Source: `predict_progress/predict_progress_complete.json` (103 599 / 103 599 LLM cells, `reason: complete`).

Same evolved PRIME prompt (`al_iter_5`), full uncapped test split (34 533 reviews).

| Model | R_global | R_worst | MAE | mean_κ | combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Ensemble (majority vote)** | **75.41%** | 55.91% | **0.256** | **0.882** | **0.838** |
| openai/gpt-4o-mini | 74.40% | **56.80%** | 0.267 | — | 0.748 |
| google/gemini-2.5-flash | 75.01% | 52.99% | 0.262 | — | 0.739 |
| anthropic/claude-3.5-haiku | 74.24% | 53.25% | 0.269 | — | 0.736 |

Saved ensemble metrics (`full_uncapped_test_metrics.json`): R_global 75.41%, R_worst 55.91%, MAE 0.256, mean_κ 0.882, combined 0.838.

## Ensemble vs single GPT-4o-mini (same prompt)

| Metric | Ensemble | GPT-4o-mini only | Δ (ensemble − single) |
| --- | ---: | ---: | ---: |
| R_worst | 55.91% | 56.80% | −0.89 pp |
| R_global | 75.41% | 74.40% | +1.01 pp |
| MAE | 0.256 | 0.267 | −0.010 |

Reference: zero-shot single GPT on initial prompt (`results_single_gpt4o_mini`) — R_worst 53.18%, R_global 71.86%.

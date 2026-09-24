# Results — `E2_cheap_ensemble_global_tail/seed42_20260802_011259`

Live Amazon WILDS **user-OOD** with cheap ensemble + `global_tail_mix`.  
Wall ~3.5 h (clean sequential run). Exit 0.

> Prior parallel attempt `seed42_20260801_205604` is **CONTAMINATED** (shared `results/_oe/seed_c*`); ignore it.

## Method

| Knob | Value |
|------|--------|
| Fitness | `0.5·R_global + 0.5·R_tail(D_select) − length` (`tail_quantile=0.2`) |
| Gate | `anchor_gate=reject`, consolidation **off** |
| OE | 15×3, mutator `deepseek/deepseek-v4-pro` |
| Workers | `gpt-4o-mini`, `gemini-2.5-flash-lite`, `qwen3-32b` |
| Clusters | `pred_profile` **refit** (no Phase1 pin) |
| Seed prompt | short GRAPE `prompts/initial_prompt.txt` |

## Headline

**Regression vs seed on official OOD test — significant.**

| Metric | Seed | Final | Δ | boot CI95 | p |
|--------|-----:|------:|--:|-----------|--:|
| R_global | 0.690 | 0.663 | **−0.027** | [−0.041, −0.014] | **0.000** |
| R_tail (q=0.2) | 0.380 | 0.370 | −0.010 | [−0.039, +0.018] | 0.54 |
| R_worst (p10) | 0.375 | 0.375 | 0.000 | — | — |
| CVaR_q40 | 0.609 | 0.584 | −0.025 | — | — |
| MAE | 0.338 | 0.369 | worse | — | — |

McNemar: seed-only-correct **107** / final-only **55**, p=0.0001.

So putting `R_tail` into the OE scalar **did not** lift the test tail; it bought a D_select champion that **hurt** mean OOD accuracy.

## In-loop behavior

| Cycle | best D_select fitness | prompt vs seed |
|------:|----------------------:|----------------|
| 1 | 0.5599 | **changed** (len 1730→2610) |
| 2 | 0.5599 | frozen (same champion) |
| 3 | 0.5599 | frozen |

`best_selection_cycle` unset / unused under this skeleton. Gate did not prevent the OOD drop (anchor is in-domain heldout, not test).

## Cost (OpenRouter, from `summary.json` token_usage)

Dominant workers (~6.1–7.3M tokens each):

| Model | tokens | ~$ |
|-------|-------:|---:|
| gpt-4o-mini | 6.13M | ~0.94 |
| gemini-2.5-flash-lite | 6.42M | ~0.68 |
| qwen3-32b | 7.33M | ~0.81 |
| **Workers sum** | | **~$2.4** |

Also logged ~0.23M each on kimi / qwen-235b / deepseek-v4 (small; likely stray tracker / seed-cache edge — not the main ensemble). Mutator budget mostly under deepseek line (~0.23M). **Run ≈ $2.5–3.0** — under the $4–5 estimate.

## Interpretation

1. Cheap ensemble **does** open headroom vs Phase1 (seed test R_global **0.690** vs Phase1 **0.722**).
2. Soft-mixing `R_tail` into fitness is **not** a free robustness win on this shift — same demotion family as CVaR, milder but still significant on mean.
3. Keep Phase1 skeleton (`global` + reject) as the non-harm baseline; treat `global_tail_mix` as a **negative** Amazon user-OOD arm unless category-shift says otherwise.

Artifacts: `results/.../analysis/seed_vs_final.json`, `analysis/paired.txt`.

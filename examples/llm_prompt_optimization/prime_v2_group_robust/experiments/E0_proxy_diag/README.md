# E0 — Proxy diagnostics (go/no-go before E1)

## What this experiment is

E0 answers one question **before** spending budget on evolutionary search:

> Do our label-free groups act as a usable surrogate for the official
> worst-group metric (R_worst = 10th percentile of per-user accuracy)?

Cluster assignment does **not** change model predictions. Therefore we run the
starting prompt on official WILDS val **once**, cache predictions, and only
recompute grouping for a grid of K and variants (including `shuffle` as a
falsification control). Cost ≈ one val inference pass.

## Decision rule (locked — see `experiments/OBSERVATIONS.md`)

| Outcome | Meaning | Next step |
|---------|---------|-----------|
| **go** | permutation Kruskal–Wallis p < 0.05 and H beats shuffle | Lock K + feature variant → E1 |
| **weak_go** | p < 0.10 | Inspect; optional soft-SLT / feature repair |
| **no_go_proxy** | null on large-cap | Repair features or study framing — **do not** start E1 |

LOO Spearman is **diagnostic only** (biased under the null). Prefer large-cap
(≥100–150 val users) before trusting a go.

## Models (live)

| Role | Model |
|------|-------|
| Worker 1 | `deepseek/deepseek-v4-pro` |
| Worker 2 | `moonshotai/kimi-k2.5` |
| Worker 3 | `qwen/qwen3-235b-a22b-2507` |
| Mutator (unused in E0) | `z-ai/glm-5` |

API key: loaded from `.env` (`OPENAI_API_KEY` / aliased to `OPENROUTER_API_KEY`).

## How to run

```bash
cd openevolve/examples/llm_prompt_optimization/prime_v2_group_robust

# Smoke (mock)
python scripts/run_e0_proxy_diag.py --config experiments/E0_proxy_diag/config_smoke.yaml

# Live small-cap
python scripts/run_e0_proxy_diag.py --config experiments/E0_proxy_diag/config.yaml

# Live large-cap confirmation
python scripts/run_e0_proxy_diag.py --config experiments/E0_proxy_diag/config_large.yaml

# Offline recompute (KW + pred_profile) from cached preds
python scripts/recompute_e0_kw_pred.py --run-dir results/E0_proxy_diag_large/<run>
```

## Artifacts

Under `results/E0_proxy_diag*/seed<seed>_<timestamp>/`:

| File | Contents |
|------|----------|
| `FINDINGS.md` | Per-run conclusions (write after each live run) |
| `e0_report.md` / `.json` | Decision + grid |
| `val_predictions.npy` | Ensemble preds |
| `val_worker_predictions.npy` | Worker votes W×N (required for disagreement features) |
| `cluster_sample_K*.md` | Manual inspection sample |
| `recompute_kw_pred/` | Offline KW + pred_profile re-analysis |

Cross-run lessons: `experiments/OBSERVATIONS.md`.

Exit codes: `0` for go/weak_go, `2` for no_go_proxy.

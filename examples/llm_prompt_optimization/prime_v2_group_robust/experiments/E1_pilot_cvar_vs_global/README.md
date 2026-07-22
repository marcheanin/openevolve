# E1 Pilot: global fitness vs CVaR-cluster fitness

## Hypothesis
Optimizing CVaR over style clusters during evolution improves OOD `R_worst` and `CVaR_cluster` on official WILDS test vs global-accuracy fitness (v1-style objective).

## Live WILDS + OpenRouter (3 AL cycles)

**Требования:** `OPENROUTER_API_KEY`, WILDS Amazon (скачается в `./data` при первом запуске).

```bash
cd prime_v2_group_robust
$env:OPENROUTER_API_KEY = "sk-or-..."   # PowerShell

# Проверка конфига без запуска
python scripts/run_e1_wilds_live.py --dry-run

# Полный прогон (~30-90 мин, ~$5-15 в зависимости от батча)
python scripts/run_e1_wilds_live.py
```

Конфиг: `config_wilds_live_3cycle.yaml` — 3 AL-цикла, 3 inner OpenEvolve-итерации на цикл,
30 train users, batch 36, CVaR-fitness.

### Что использует OpenRouter

| Компонент | API | Модели |
|-----------|-----|--------|
| Ensemble inference (fitness) | `ensemble.api_base` | gpt-4o-mini, gemini-2.5-flash, claude-3.5-haiku |
| OpenEvolve mutator (inner loop) | `configs/openevolve_e1_live.yaml` | gemini-2.5-flash |
| OpenEvolve evaluator callbacks | те же 3 worker-модели | через `prime.evolution.evaluator_entry` |

**Mock** включается только если: `force_mock: true` **или** нет `OPENROUTER_API_KEY`.
`smoke: true` сам по себе mock **не** включает (исправлено).

## Smoke validation (all pipeline stages)

Runs every stage with verbose logs and writes `smoke_checklist.json`:

```bash
cd openevolve/examples/llm_prompt_optimization/prime_v2_group_robust
python scripts/run_e1_smoke_validate.py
# or:
python -m prime.cli --config experiments/E1_pilot_cvar_vs_global/config_smoke_validate.yaml
```

Stages checked: config → run_context → data → clustering → pool → inference →
acquisition → fitness → error_artifacts → evolution → consolidation → val_selection →
pool_expand → final_test → summary.

Artifacts: `results/E1_smoke_validate/.../smoke_trace.jsonl`, `smoke_checklist.json`.

## Commands (pilot arms)
```bash
cd openevolve/examples/llm_prompt_optimization/prime_v2_group_robust

# Smoke (no API key: mock ensemble; with API key: real LLM on capped data)
python -m prime.cli --config experiments/E1_pilot_cvar_vs_global/config_global.yaml --smoke
python -m prime.cli --config experiments/E1_pilot_cvar_vs_global/config_cvar.yaml --smoke

# Full pilot (requires OPENROUTER_API_KEY, WILDS data)
python -m prime.cli --config experiments/E1_pilot_cvar_vs_global/config_global.yaml
python -m prime.cli --config experiments/E1_pilot_cvar_vs_global/config_cvar.yaml
```

## Go / no-go
- **Go (method framing):** CVaR arm beats global on `R_worst` or `CVaR_cluster` on val/test with same seed/budget.
- **No-go:** No improvement → pivot to study framing (component ladder E2).

## Artifacts
- `results/E1_global_fitness/...` and `results/E1_cvar_fitness/...`
- Compare `summary.json` → `final_test.R_worst`, `final_test.CVaR_cluster`

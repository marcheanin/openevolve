# PROJECT_HISTORY_SOURCES

Копии всех документов, на которые ссылается [`../PROJECT_HISTORY.md`](../PROJECT_HISTORY.md).
Оригиналы **не переносились** — здесь только duplicates для удобного чтения офлайн / шаринга.

**Создано:** 2026-08-11 · **Файлов:** 35 (+ этот INDEX)

---

## Корень проекта

| Файл | Роль в истории |
|------|----------------|
| `PROJECT_HISTORY.md` | Сводная история (главный документ) |
| `README.md` | v1, рецензии, ландшафт, первый план |
| `SPEC.md` | Нормативная спецификация GRAPE / PRIME v2 |
| `APPROACH.md` | Описание GRAPE (v2-эра) |
| `APPROACH_V3.md` | Критическая ревизия архитектуры |
| `APPROACH_SLT.md` | Soft Latent Types (модуль групп) |
| `IMPLEMENTATION.md` | Фазы реализации 0–4, DoD |
| `STRATEGY_UNIVERSAL_OOD.md` | Стратегия multi-benchmark |
| `ROADMAP_PHASE2.md` | CivilComments E4, решения после Amazon |
| `ROADMAP_PHASE3.md` | E5 redesign, S9 matrix |

## docs/

| Файл | Роль |
|------|------|
| `V1_V2_RECONCILIATION.md` | Декомпозиция прироста v1 vs потолок v2 |

## configs/

| Файл | Роль |
|------|------|
| `REJECTED_MODELS.md` | Отклонённые воркеры (qwen3.7-flash и др.) |

## experiments/

| Файл | Роль |
|------|------|
| `OBSERVATIONS.md` | Живой журнал уроков M*, C*, O*, P* |

### E0 — диагностика прокси / Phase 0 offline

| Файл | Роль |
|------|------|
| `E0_phase0_offline/PHASE0_REPORT.md` | Offline bounds, закрытие Amazon headline |
| `E0_phase0_offline/FINDINGS.md` | Краткие выводы Phase 0 |
| `E0_phase0_offline/README.md` | Описание эксперимента |

### E1 — Amazon go/no-go

| Файл | Роль |
|------|------|
| `E1_pred_profile_cvar_vs_global/FINDINGS_lowvar.md` | Sig regression, D_select overfit |
| `E1_pred_profile_cvar_vs_global/RESULTS_pareto_v2_pair.md` | Pareto pair null |
| `E1_constraint_global/RESULTS.md` | Mechanics pass (global + reject gate) |
| `E1_v1_weighted_mini/RESULTS.md` | v1 fitness не просыпается |
| `E1_v1_prompt_transfer/RESULTS.md` | Перенос контента v1 (+2.2 pp) |

### E2 — Amazon варианты сдвига

| Файл | Роль |
|------|------|
| `E2_cheap_ensemble_global_tail/RESULTS.md` | Tail-mix регрессия |
| `E2_category_shift_books/RESULTS.md` | Category shift слабый + |

### E4 — CivilComments (ensemble-era)

| Файл | Роль |
|------|------|
| `E4_civilcomments/README.md` | Обзор E4 |
| `E4_civilcomments/deep_analysis_20260806.md` | M27 stale score, rotation noise |
| `E4_civilcomments/phase2b_diagnostics/RESULTS.md` | GO gate, gap 35 pp |

### E5 — CivilComments (single scorer + GBA)

| Файл | Роль |
|------|------|
| `E5_civilcomments/README.md` | Обзор E5 |
| `E5_civilcomments/PREREGISTRATION.md` | Предрегистрация протокола |
| `E5_civilcomments/RESULTS.md` | S9 таблицы |
| `E5_civilcomments/S9_STATUS.md` | Статус матрицы бейзлайнов |
| `E5_civilcomments/NEXT_STEPS.md` | Следующие шаги |
| `E5_civilcomments/gpo_gate.json` | Gate реимплементации GPO |
| `E5_civilcomments/E5V2_WHAT_MOVED_EVOLUTION.md` | +3 pp, attribution v2 |
| `E5_civilcomments/E5V3_WHAT_MOVED.md` | v3: selection debt |
| `E5_civilcomments/E5V3_SELECTION_DEBT.md` | Долг по `_pick_heir` |

---

## Как обновлять

При изменении `PROJECT_HISTORY.md` или добавлении новых источников — перекопировать:

```powershell
cd openevolve\examples\llm_prompt_optimization\prime_v2_group_robust
# (повторить скрипт копирования из корня репозитория или попросить агента)
```

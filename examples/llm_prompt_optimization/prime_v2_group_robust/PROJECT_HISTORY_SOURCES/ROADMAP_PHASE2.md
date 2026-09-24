# Roadmap Phase 2 — решения от 2026-08-05

Фиксация решений после разбора итогов Amazon-кампании (OBSERVATIONS M15–M23), внешнего
LLM-аудита и сверки v1↔v2 (`docs/V1_V2_RECONCILIATION.md`). Документ — операционный план;
стратегический контекст в `STRATEGY_UNIVERSAL_OOD.md`, нормативная архитектура в `SPEC.md`.

## 0. Принятые решения

| Вопрос | Решение |
|---|---|
| Следующий субстрат | **CivilComments-WILDS первым**, category-shift (M23) вторым |
| Фрейминг статьи | **«Условия применимости»**: метод-фрейминг с Amazon как объяснённым negative case; study-фрейминг (SPEC C4 / E9) — подготовленный fallback |
| Ансамбль | **gemma-3-12b-it + qwen3.7-flash + gemma-4-26b-a4b** (smoke passed: spread 2.8пп, agree 76%, rule 100%). Отвергнуты смоуком: gpt-oss-20b (mandatory reasoning + lag), ministral-8b (acc 0.53). Резерв при agreement>90%: qwen3.5-9b, nemotron-3-nano |
| Мутатор | **deepseek-v4-pro** ($0.435/$0.87, ~$0.2/прогон) — подтверждён; бюджетный fallback deepseek-v4-flash; консолидация (если включим) — gemini-3.1-pro (O19) |
| Бюджет Фаз 2–3 | ~$20–30 (≈$1/прогон воркеров + мутатор) + flagship-валидация в конце |
| Агрегация воркеров (C12) | **Отложена** — см. §4 |
| Prompt routing / codebook (PCO-стиль) | Не строить, пока offline-bound на новом субстрате не покажет LOO > +5 пп (на Amazon E0a: LOO −3.6 пп) |
| Amazon user-shift | Закрыт. Не переоткрывать (M18–M21 + reconciliation). Используется как negative case и источник кэшей |

## 1. Центральная гипотеза Фазы 2 (зафиксирована ДО эксперимента)

Group-robust оптимизация текстового промпта работает ⇔ худшая группа удовлетворяет трём условиям:

1. **Распознаваемость** — принадлежность к группе видна из текста на inference (label-free триггер);
2. **Выразимость** — механизм ошибки формулируется как инструкция;
3. **Неконфликтность** — корректирующее правило не двигает границу решения мажоритарной группы.

Amazon user-shift нарушает все три (C10 label-mix, C9 неразделимость 4↔5, C11/C13 конфликт с 5★).
CivilComments a priori удовлетворяет всем трём (identity-термы в тексте; spurious correlation
«упоминание identity → toxic»; правило не конфликтует с детекцией реальной токсичности).
E4 — прямая проверка этой гипотезы. Формулировка условий — часть вклада статьи независимо от знака E4.

## 2. Фазы

### Фаза 2a — инфраструктура (≈неделя, ~0 API)
- [x] Загрузчик CivilComments-WILDS (официальные сплиты) + oracle identity-группы
      (режим `oracle` в кластерном модуле рядом с `style`/`pred_profile`)
      — `prime/data/civilcomments_loader.py` (HF mirror → pickle cache),
        `clusters.geometry: oracle`, `dataset.label_space: binary`
- [x] Seed-промпт бинарной классификации токсичности (короткий, по уроку M21: rich seed — не подарок)
      — `prompts/initial_prompt_civilcomments.txt`
- [x] Фикс M17: ротация/ресемпл D_select между циклами (`d_select_rotate`,
      group×label стратификация, O26 re-score чемпионов) — `eval_sets.rotate_d_select`,
      `controller._rotate_d_select_for_cycle`; тесты `tests/test_phase2a_fixes.py`
- [x] Фикс O22: механическая verbatim-инъекция few-shot из артефактов
      (`inject_verbatim_fewshot` до/после OE)
- [x] Смоук ≤20B-ансамбля: `scripts/smoke_phase2_ensemble.py` +
      `configs/ensemble_phase2_small.yaml` (**gemma-3-12b / qwen3.7-flash / gemma-4-26b**);
      gates passed 2026-08-05 (parse 0%, spread 2.8пп, agree 76%, rule 100%).
      Отвергнуты: gpt-oss-20b, ministral-8b
- [x] Стартовые замеры + анти-доминирование: в smoke-скрипте
- [x] Апгрейд фидбека мутатору: (а) O13 lint+strip cluster-addressed rules;
      (б) CONTRASTIVE PAIRS в `format_error_artifacts` (OBS O29 / M25)

### Фаза 2b — диагностика перед бюджетом (~$2)
- [x] Per-group профиль seed-ансамбля на официальном val; worst-group gap
      (`scripts/phase2b_diagnostics.py`, equal-group n=180): gap **35 pp**
      (male 0.90 → black 0.55); LGBTQ/muslim 0.60
- [x] Noise floor: 3 повторных скоринга фиксированного промпта (протокол M15):
      R_global SD **0.0056**; R_worst_group стабилен 0.550
- [x] Power analysis: D_select/test ≈ **404 / 673** при целевом MDE 0.05 (M13/bootstrap)
- [x] **Go/no-go: GO** (OBS M26). Identity error mass ~89%; class-mix R²≈0 (не C10);
      prompt headroom есть (20 all-wrong). Далее Фаза 2c.

### Фаза 2c — live E4 (~$6–8)
- [x] Плечо A (контроль): скелет Phase 1 — `global` fitness + anchor reject gate (M19)
      — lean 3×3 done; global regresses worst-group vs seed (deep_analysis)
- [x] Плечо B lean 3×3 + B 4×20: `min_group_lex` oracle — **механика работает в C1**;
      C2+ блокированы stale seed metrics → **M27 fix landed**; rotation noise → **M28**
- [ ] Плечо B post-fix: `config_arm_b_min_group_2x20.yaml` (2×20, D_select=720,
      `d_select_rotate=false`) + same-day noise — **ready, not launched**
- [ ] Плечо C (после B 2×20): inferred-группы — label-free история
- [ ] 1 сид capped → при сигнале 3 сида на headline
- [x] Offline bounds CivilComments (`phase2c_offline_bounds`) — portfolio LOO ~0 → routing deferred
- Fixes: M27 seed-metric sync; M28 lex rotation policy; M29 same-day paired protocol
  (`scripts/e4_test_noise.py`). Write-up: `experiments/E4_civilcomments/deep_analysis_20260806.md`

### Фаза 3 — baselines и второй субстрат (~$7–9)
- [ ] EvoPrompt / APO / OPRO (код v1) + GEPA при равном бюджете вызовов, worst-group на CivilComments
- [ ] Category-shift Books→прочее: 3 сида, lex-floor вместо mix (проверка слабого позитива M23)

### Точка выбора фрейминга — после Фазы 2c
E4 положительный → метод-фрейминг («условия применимости», Amazon = объяснённый negative case).
E4 нулевой → study-фрейминг: E9-сетка {APO, OPRO, EvoPrompt, GEPA} × {2 сдвига} × {avg, worst-group,
ID→OOD gap}; материал M15/M17/C8/C10/C13 уже собран.

## 3. GPO-ветка (дёшево, после 2c)
Unlabeled-target адаптация (Li et al., EMNLP 2023, GPO): включить неразмеченные target-тексты
в фидбек мутатору / pseudo-labeling. Одно плечо, не headline. Обязательная цитата в статье
в любом фрейминге (их мотивационный эксперимент = независимое подтверждение study-тезиса).

## 4. C12 — обучаемая агрегация воркеров (ОТЛОЖЕНО, вернуться после Фазы 3)

**Суть:** на Amazon oracle-over-workers = +9.1 пп над median/majority (C12); 30% ошибок ансамбля
исправимы правилом голосования, 70% единогласны (доступны только промпту). Рычаги комплементарны.

**Почему отложено:** (а) требует полноценной абляции 2×2 {seed, evolved} × {median, learned-agg},
иначе эффект агрегации маскируется под prompt-эффект — v1 уже неявно чинил голосование через κ
(reconciliation §5.3), и это загрязняло его headline; (б) сейчас критический путь — E4;
(в) смена ансамбля на ≤20B меняет и структуру разногласий — учить агрегацию надо на финальном стеке.

**План ветки, когда вернёмся:**
1. Лестница: статические веса воркеров → class-conditional веса → context-conditional
   (кластер/длина). Логрег/бустинг на кэшированных предсказаниях, ~0 API.
2. Дисциплина: обучение на D_select, выбор на val, один замер на тесте с bootstrap.
3. Абляция 2×2 на обоих субстратах; на CivilComments — взвешенное голосование с per-group порогом
   (прямой рычаг worst-group).
4. Позиционирование в статье: «замороженная LLM-система = промпт + правило голосования», обе части
   адаптируются без весов; либо явная ветка, либо честное out-of-scope с цифрой потолка.

**Триггер возврата:** завершена Фаза 3 (или E4 нулевой и нужен второй рычаг для метод-фрейминга).

## 5. Что решено НЕ делать
- Prompt routing / PCO-codebook до положительного offline-bound (E0a: LOO −3.6 пп на Amazon)
- Promptbreeder-эволюция mutation-промптов: узкое место — сигнал, не разнообразие (O17/O20)
- Замена OpenEvolve на GEPA как оптимизатора (GEPA — baseline в E3/E9, не замена)
- Abstention как headline (легитимно как секция анализа risk-coverage, не как задача)
- Любые новые live-прогоны на Amazon user-shift

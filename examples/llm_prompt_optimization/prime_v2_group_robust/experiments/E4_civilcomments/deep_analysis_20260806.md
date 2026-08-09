# E4 deep analysis — 2026-08-06 (A lean, B lean, B 4×20 vs seed)

Runs analyzed:
- seed baseline (same test, n=800): `baseline_initial_prompt_test/` (day-1) + same-day shot in B4×20 `evals/test_noise/ensemble_seed.npy`
- A lean 3×3: `results/E4_civilcomments_arm_a_global/seed42_20260805_104411`
- B lean 3×3: `results/E4_civilcomments_arm_b_min_group/seed42_20260805_135157`
- **B 4×20**: `results/E4_civilcomments_arm_b_min_group_4x20/seed42_20260805_214134` (+3-repeat noise, `evals/test_noise/`)

## 1. Headline (test n=800)

| prompt | R_global | R_worst_group | R_macro | Acc tox |
|---|---:|---:|---:|---:|
| seed day-1 | 0.8638 | 0.636 | 0.663 | 0.410 |
| seed same-day (B4×20) | 0.8700 | 0.636 | — | 0.386 |
| A lean (global) | 0.8662 | **0.591** | 0.664 | 0.410 |
| B lean (lex) | 0.8712 | 0.682 | 0.672 | 0.422 |
| **B 4×20 (lex)** | 0.8738 | **0.682** | 0.647 | 0.361 |

Noise (3 repeats of B4×20 final on test): R_global SD 0.0012, **R_worst_group SD 0.0000**, R_macro SD 0.006; flips ~1%.

### Paired same-day (честное сравнение)

| pair | net flips | McNemar p (one-sided) |
|---|---:|---:|
| B4×20 vs seed same-day | **+3** | 0.30 |
| seed day-1 vs seed day-2 (дрейф API) | +5 | 0.11 |

**Междневный дрейф seed сопоставим с "эффектом метода" по global.** Любые сравнения — только парные и same-day.

### Per-group same-day (seed → B4×20)

muslim +0.071 (+2 ex), white +0.045 (+1 ex), none +0.004; black −0.053 (−1 ex), christian −0.015 (−1 ex); tox recall 0.386→0.361. Это перераспределение ±1–2 примера на клетках n=11–28, кроме muslim/white — направленно в worst-группы.

## 2. B 4×20: как реально шла эволюция

### C1 — единственный содержательный цикл
OE hill-climb на D_select (fresh seed, 20 iters):

| iter | lex fitness | R_worst_group(D_sel) | R_macro(D_sel) |
|---:|---:|---:|---:|
| 0 (seed+fewshot) | 0.604 | 0.596 | 0.708 |
| 4 | 0.654 | 0.646 | 0.732 |
| **10 (champion)** | **0.674** | **0.667** | **0.746** |
| 11–20 | плато / клоны чемпиона | | |

Реальный монотонный прогресс за ~10 итераций, дальше плато. Дифф чемпиона против seed — интерпретируемый и точно в worst-группу (white/race-дискурс):
- правила: contempt/mockery/veiled hostility→1; civil critique of ideas/institutions→0; describing group characteristics w/o derogatory language→0; quoting hate to condemn→0;
- few-shot заменены на пограничные примеры про white privilege / affirmative action (gold 1 и 0).

### C2–C4 — мертвый груз (~7ч, ~60 итераций)
val key `[0.6875, 0.6835]` не менялся все 4 цикла; heir после C1 не сменился ни разу.

**Верифицированный stale-score leak.** Честный lex-fitness чемпиона по сетам (пересчитан offline из pred_cache):

| цикл | D_select hash | честный fitness | что стояло в фронте |
|---:|---|---:|---:|
| 2 | 4f31eb70 | **0.593** | 0.674 (stale) |
| 3 | 32088f1c (= C1!) | 0.674 | 0.674 |
| 4 | ab4f0949 | **0.551** | 0.674 (stale) |

O26-рескор при ротации запускается (`champions_rescored=5–6`), но чемпион возвращается в фронт/OE seed-checkpoint со скором лучшего для себя сета. Дети OE, честно оскоренные на текущем сете (≤0.66), проиграть stale 0.674 не могут → эволюция после C1 структурно заблокирована.

**Ротация — доминирующий шум.** Один и тот же промпт: 0.551–0.674 (12pp) между ротациями. Ожидаемо: worst-группа в D_select ~40–47 примеров → SD ≈ 8pp. Сигнал между кандидатами внутри цикла ≪ межротационного шума. Плюс ротация циклится по ~3 партициям (C3 воспроизвела сет C1).

**Gate:** C1 accept (tolerated_noise, drop 0.03); C3 — единственный кандидат лучше чемпиона (0.682 на сете C1) **отклонён** (drop 0.08 на D_anchor, p=0.019) — отказ обоснованный: реальная регрессия на решённых примерах.

## 3. Что реально сработало / не сработало

Сработало:
1. **`min_group_lex` даёт поисковый градиент** (C1: worst-group на D_select +7pp за 10 итераций); `global` (arm A) выбирает промпты с регрессом worst-группы — неверная цель подтверждена.
2. **Anchor gate**: все 3 отказа за кампанию обоснованы (майорити-коллапс 0.883 в A; настоящая регрессия 0.08 в 4×20 C3).
3. **Мутатор-фидбек** (contrastive pairs, verbatim few-shot inject): правки семантически точные, по нужной границе, без прокси-меток групп в тексте.
4. **Noise-harness**: same-day скоринг воспроизводим (flips ~1%); выявил, что междневный дрейф ≈ размер эффекта.

Не сработало / сломано:
1. Stale-score leak (выше) — циклы 2+ бесполезны, пока не починен.
2. Ротация D_select в текущем виде вредит: шум ≫ сигнал для lex-цели, эффективных партиций ~3.
3. Мульти-цикл при замороженном heir — трата бюджета (в 4×20 полезен только C1).
4. Тест n=800 (worst-группы n=11–22) не разрешает эффект +4.5pp: это ±1 пример.

## 4. Вердикт по результатам

- **Механика метода работает**: поиск двигает worst-group на D_select, gate отсекает деградацию, финал не регрессирует.
- **Внешнего статистически чистого выигрыша пока нет**: global — в пределах дрейфа; worst-group +0.045 — 1 пример в группе n=22 (стабильный по повторам, направленно совпадающий с muslim +2, но не claim).
- A (global) — единственная арка с регрессом worst-группы: направленный аргумент за lex.

## 5. Что нужно для финализации архитектуры

1. **Фикс stale-score**: при построении фронта в конце цикла все carried/OE-seed кандидаты должны иметь скор текущего D_select (рескор фронта уже есть — донести до seed-checkpoint и до сравнения `best_evo_score`).
2. **Ротация**: либо фиксированный D_select на прогон (анти-оверфит обеспечивают gate + val-селекция), либо усреднение champion-скора по 2 сетам. Для lex-цели worst-группе нужно n_g ≥ 80–100 в D_select.
3. **Бюджет формы**: 1–2 цикла × 20–30 итераций (C1 сошёлся за ~10; мульти-цикл оправдан только после фикса 1–2).
4. **Мощность headline**: capped test с гарантией n_g ≥ 50 у малых групп (~2–3k примеров) + 3 сида + paired bootstrap; все сравнения same-day.

## 6. План

1. (~0 API) **DONE 2026-08-06:** Фикс stale-score (M27) + guard ротации (M28) + юнит-тесты
   `tests/test_m27_stale_seed_metrics.py`; конфиг `config_arm_b_min_group_2x20.yaml`.
2. Перегон B: 2×20 с фиксами (`scripts/run_e4_b_2x20.py`) — проверить, что C2 способен сменить heir.
3. Arm C (style, тот же бюджет) — inferred-vs-oracle (SPEC Р15).
4. При воспроизводстве +worst-group: 3 сида + расширенный тест → headline-числа.
5. Фаза 3: EvoPrompt/GEPA/APO equal-budget на CivilComments.
6. **DONE:** OBSERVATIONS M27–M29.

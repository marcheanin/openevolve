# Глоссарий метрик и протоколов

Все RQ ссылаются сюда. Если метрика менялась между постановками, это указано.

## Amazon (порядковая оценка 1–5)

| Имя | Определение | Где headline |
|-----|-------------|--------------|
| `R_global` | доля примеров с `pred = gold` | v1 full test, E1, E2 |
| `R_worst` | 10-й перцентиль per-user accuracy | v1 headline; **залипает на 0.5** при 8 отзывах на юзера (M9) |
| `MAE` | среднее \|pred − gold\| на шкале 1–5 | v1 combined |
| `mean_kappa` / κ | среднее quadratic weighted Cohen's κ по парам воркеров | v1 fitness и selection; циркулярно |
| `CVaR_cluster` | среднее accuracy худшей трети кластеров (`q=0.33`) | E1 fitness `cvar_lex` |
| `R_tail` | accuracy на худшем квантиле пользователей | отбор E1; худший дискриминатор (M14) |
| `cvar25` (E6) | CVaR@25% от macro-per-class внутри pred_profile-кластера | E6 primary |
| `op_shift` | сдвиг operating point (как охотно модель ставит высокий рейтинг) | E6 диагностика |
| `combined` (v1) | `0.4·R_global + 0.3·R_worst + 0.3·(1−MAE/4) + 0.1·κ` | отбор чекпоинта v1 |

**Группы Amazon.** Не user_id. Сначала пробовали стиль текста (`full_T`: эмбеддинг + длина/пунктуация) — Kruskal–Wallis null. Потом `pred_profile`: k-means на смеси предсказанных рейтингов пользователя (mean, std, entropy). K=6.

**Сплиты.** v1: кастомный 70/15/15 user-disjoint из train-пользователей, seed 42, **не** официальный WILDS OOD. v2: официальный WILDS OOD.

## CivilComments (бинарная токсичность)

| Имя | Определение | Где headline |
|-----|-------------|--------------|
| `GBA_g` | `½(TPR_g + TNR_g)` внутри identity-группы g | E5; пол all-zeros = 0.5 |
| `R_worst_gba` / hard-min GBA | `min_g GBA_g` по 8 overlapping identities (`none` исключён) | S9 primary, потом снят |
| `CVaR@25%` | среднее двух худших GBA (8 групп) | S10 primary; power audit F2 |
| `mean GBA` | среднее восьми GBA | диагностика, отбор |
| `softmin_τ` | log-sum-exp по группам; τ→0 → hard min | PRIME shipped τ=0.1, shrink w=40 |
| `worst-class acc` | min(точность класса 0, точность класса 1), **без групп** | R16 / S10 selection |
| `R_worst_group` | min per-group accuracy | E4; **максимизируется молчанием** (M31) |
| `toxic_recall` / `specificity` | TPR / TNR глобально | operating point |
| `op_shift` / `|recall−spec|` | насколько промпт смещён к over-flagging | F3 |

**Группы CivilComments.** Oracle identity WILDS: male, female, LGBTQ, christian, muslim, other_religions, black, white; `none` исключается из GBA по умолчанию.

**Фиксированные множества E5** (отпечатки в `experiments/E5_civilcomments/fixed_sets/`):

| Set | n | fingerprint | роль |
|-----|--:|-------------|------|
| `test_fixed` | 1800 | `5cfb7ebde3c5` | S9 report |
| `test_fixed_large` | 5251 | `1ed1114b72e6` | S10 headline, суперсет 1800 |
| `d_dev` uniform | 900 | `1c10514d553c` | 9 групп × 50 pos + 50 neg |
| `d_dev_targeted` | 900 | `594125cf6a52` | группы 8, 3, 5 × 150/cell |

## Два типа шума

- **Шум измерения.** Два фиксированных промпта, один тест. Вопрос: различимы ли
  они. Инструмент — парный бутстрап по примерам, CI95, McNemar. Растёт только с
  числом примеров.
- **Шум метода.** Повторный запуск оптимизатора (другой сид, другой прогон).
  Вопрос: воспроизведётся ли прирост. Инструмент — разброс между прогонами и
  сидами. Числом примеров не лечится.

Одно и то же число может пройти первый барьер и провалить второй: v1 headline
+4.11 pp на n=34 533 — не случайность скоринга, но второй прогон того же метода
дал +0.81. И наоборот, прирост может провалить уже первый барьер: PRIME на
CivilComments +2.5 pp при CI95 [−1.5, +7.0].

## Статистика

- **Парный бутстрап:** 10 000 (или 1500/2000 в аудитах) итераций, resample внутри `group×label` ячеек, оба промпта на одних и тех же строках.
- **p двусторонний:** `p = 2·min(P(Δ≥0), P(Δ≤0))`, чтобы байт-идентичный seed давал p=1, а не 0.
- **Holm–Bonferroni** по контрастам метод-против-seed внутри сида.
- **Spearman ρ:** ранговая корреляция dev-статистики с test CVaR@25% по пулу кандидатов.
- **Selection regret:** `best_in_pool(test) − picked_by_rule(test)`.
- **McNemar:** число исправленных vs сломанных примеров при парном сравнении промптов.

## Scorer'ы по постановкам

| Постановка | Scorer |
|------------|--------|
| v1 headline | ансамбль gpt-4o-mini + gemini-2.5-flash + claude-3.5-haiku, majority |
| E1 Amazon | deepseek-v4-pro / kimi-k2.5 / qwen3-235b |
| E2 Amazon | cheap ensemble |
| E4 CivilComments | gemma-3-12b + qwen3.7-flash + gemma-4-26b — **аннулировано M31** |
| E5 / S9 / R15 | **один** `google/gemma-3-12b-it`, fail-closed |
| E6 Amazon | `openai/gpt-4o-mini` |

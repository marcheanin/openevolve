# RQ4 — Конфаунд: групповая работа или сдвиг строгости

## Вопрос

Отличим ли выигрыш оптимизаторов на худшей группе от простого перемещения точки
срабатывания — того, насколько охотно модель ставит положительный класс?

## Короткий ответ

В основном неотличим. Две трети разброса worst-group качества объясняются
средним качеством (R²=0.687). Бесплатный контроль из одной заменённой строки
seed-промпта занимает 4-е место из 35 и ни одна ячейка матрицы методов не бьёт
его при Holm<0.05. На Amazon однострочная правка `deflate_positive` повторяет
GPO ровно (оба 0.458 cvar25 при seed 0.432).

Аргумент пока **корреляционный**. Контрфакт «выровнять operating point и
сравнить остаток» не сделан.

---

## Что проверяли

### F3. Регрессия worst-group на среднее и на imbalance

**Данные.** Все 34 оценённых промпта S9 (seed + 11 методов × 3 сида),
majority-of-3 на `test_fixed` n=1800. Без нового скоринга.
Скрипт: `scripts/e5_metric_variance_audit.py`.
Отчёт: `experiments/E5_civilcomments/POWER_AUDIT.md`.
JSON: `results/E5_s9_matrix/stable_session/metric_variance_audit.json`.

| Связь | Значение |
|-------|---------:|
| corr(CVaR@25%, mean GBA) | **+0.83** |
| R² CVaR@25% ~ mean GBA | **0.687** |
| добавление \|recall − spec\| | **+0.000** |
| corr(CVaR@25%, \|recall − spec\|) | **−0.67** |

Низ матрицы — systematic over-flagger:

| Промпт | recall | spec | \|r−s\| |
|--------|-------:|-----:|--------:|
| apo@42/43, ape@42, random_al@42 | ≈0.85 | ≈0.53 | 0.30–0.32 |
| evoprompt_de@44 (верх) | | | 0.003 |
| prime@44 | | | 0.017 |
| gpo@42 | | | 0.084 |

Mean operating point S9 (`RESULTS.md` §E.3):

| method | toxic_recall | specificity |
|--------|-------------:|------------:|
| apo | 0.827 | 0.548 |
| ape | 0.788 | 0.607 |
| gpo | 0.756 | 0.662 |
| prime | 0.701 | 0.715 |
| seed | 0.748 | 0.655 |

Интерпретация F3: worst-group score почти целиком функция среднего качества, а
среднее — функция порога токсичности. Оптимизаторы двигают operating point;
проигрывающие over-flag. Никто не закрывает *групповой разрыв* сверх того, что
предсказывает его среднее.

### R15 / F4. Контроль из 12 однострочных правок

**Пул.** 12 промптов = seed с одной заменой tie-break правила:

- 8 явных порогов «label 1 only if P(toxic) ≥ X%», X = 90, 80, …, 20
- 4 natural-language варианта (lenient_max, strict, …)

Нет поиска, нет групповых меток, нет optimizer LLM, нет train данных.
Пул: `experiments/E5_civilcomments/pools/strictness_sweep/`.
Скрипты: `scripts/build_r15_strictness_pool.py`,
`scripts/run_e5_selection_control.py`, `scripts/compare_r15_vs_s9.py`.
Отчёт: `experiments/E5_civilcomments/R15_CALIBRATION_CONTROL.md`.

Отбор: тот же D_dev rule, что у методов (для headline сравнения —
group-free worst-class). Тест: `test_fixed_large` n=5251 как суперсет S9 1800,
чтобы сравнивать paired на тех же строках.

| | CVaR@25% @1800 | @5251 | @clean 3451 |
|--|---------------:|------:|------------:|
| **R15 dev-pick `nl_lenient_max`** | **0.6475** | 0.6608 | 0.6613 |
| лучший в свипе на тесте `thr_p90` | 0.6500 | 0.6617 | 0.6650 |
| seed | 0.6300 | 0.6542 | 0.6613 |
| GPO@42 (лучшая S9 ячейка) | 0.6675 | — | — |
| PRIME@42 | 0.6600 | — | — |
| oracle (APO, fully labeled target) | 0.6200 | — | — |

Против всех 34 S9 ячеек, paired bootstrap, Holm:

- **0 из 34** бьют R15 при Holm < 0.05.
- R15 ранг **4 из 35**. Номинально выше: gpo@42 (+0.020), prime@42 (+0.013),
  random_al@44 (+0.005) — ни один разрешим.
- R15 vs seed +0.0175; vs oracle +0.0275.

Честное чтение: не «R15 победил», а «12-точечный однострочный свип без групп
попадает в шумовую полосу всей матрицы оптимизаторов», включая методы с бюджетом
меток 240 и сотнями вызовов optimizer-LLM.

### F6. Числовые пороги — no-op для этого scorer

Инструкция `gemma-3-12b-it` «label 1 only if P(toxic) ≥ X%» **не двигает**
operating point монотонно.

Recall при X = 90→20: **0.732, 0.746, 0.746, 0.753, 0.764, 0.749, 0.746, 0.740**
— полоса 0.03 без тренда.

Natural-language строгость двигает: recall 0.721 (`nl_lenient_max`) → 0.849
(`nl_strict`); spec 0.703 → 0.539.

Следствия: любой метод, который «подкручивает порог вероятности» текстом, на
этом scorer — no-op. Свип R15 покрывает только середину кривой; экстремумы
аутсайдеров S9 (recall ≈ 0.85) воспроизводятся только NL-strict вариантами.
Это **ослабляет** контроль относительно замысла — и делает 4-е место ещё
удивительнее.

F6 exploratory: один scorer. На втором не проверено.

### Репликация Amazon E6

Субстрат: Books→non-Books, scorer `gpt-4o-mini`, pred_profile K=6, primary
cvar25 macro-within-cluster. Harshness pool: `scripts/build_e6_harshness_pool.py`.
Отчёт: `experiments/E6_amazon_category_controls/RESULTS.md`.

| prompt | cvar25 | R_global | op_shift |
|--------|-------:|---------:|---------:|
| seed | 0.432 | 0.579 | 0.055 |
| **GPO** | **0.458** | 0.576 | 0.110 |
| harshness `deflate_positive` | **0.458** | 0.581 | 0.045 |
| E2 category heir | 0.451 | 0.584 | 0.083 |

Однострочная правка **повторяет** лучший оптимизатор на primary. GPO улучшает
робастную метрику и **не** улучшает R_global — ещё один пример, что среднее
плохой прокси для заявки.

---

## Метрики

- Primary: CVaR@25% на одинаковых строках (R15 vs S9 paired).
- Диагностика operating point: recall, specificity, \|recall−spec\|.
- Регрессия: Pearson / R² CVaR ~ mean GBA.
- На Amazon: cvar25, op_shift, MAE.

---

## Графики и таблицы

| Что | Где |
|-----|-----|
| F3 корреляции | POWER_AUDIT.md F3 |
| R15 vs S9 | R15_CALIBRATION_CONTROL.md F4 |
| Recall sweep F6 | тот же файл, §F6 |
| S9 operating point | RESULTS.md §E.3 |
| E6 harshness vs GPO | E6 RESULTS.md Phase B |
| Канвас | `canvases/rq-04-operating-point.canvas.tsx` |

Рисунки для статьи: (1) scatter CVaR vs mean GBA, 34 точки, цветом \|r−s\|;
(2) recall/spec для R15-пула vs S9; (3) E6 bar seed / GPO / deflate.

---

## Вывод

На двух датасетах, двух scorer'ах, двух семействах метрик однострочное движение
порога неотличимо от «оптимизации под худшую группу». Это обязательный baseline
для любой будущей матрицы (зафиксировано в PREREGISTRATION_S10 как R15).

Не заявляем, что *вся* работа оптимизаторов = порог. Заявляем, что без контроля
порога отличить нельзя, и имеющийся контроль уже попадает в ту же полосу.

## Чего не хватает

1. **P0. Контрфакт выровненного operating point.** Для каждого промпта
   оптимизатора подобрать в свипе строгости точку с тем же recall (или тем же
   \|r−s\|) и сравнить CVaR при равной точке. Считается на кэше S9 + доскоринг
   нескольких точек. Превращает корреляцию в каузацию.
2. F6 на втором scorer (`gpt-4o-mini` уже есть на E6 — дешёво повторить numeric
   threshold sweep).
3. Расширить NL-свип к экстремумам, которые R15 не покрыл из-за F6.

# RQ3 — Цель поиска: помогает ли робастный objective внутри оптимизации

## Вопрос

Улучшает ли встраивание worst-group цели (CVaR, min по группам, soft-min) в саму
оптимизацию промпта качество на худшей группе **вне** обучающего распределения?

## Короткий ответ

На Amazon user-shift — нет, и механизм отказа установлен. Единственный
статистически значимый результат всего проекта — **регрессия** под CVaR:
тест R_global −0.053, p<0.001, при этом на отборочном множестве рост реальный
(+0.128). Худшая группа на D_select и на тесте — разные группы. Скалярная цель
прячет кривую размена 4★/5★.

На CivilComments чистой абляции цели поиска при исправленном измерении **нет**.
Рукава E4 (global vs min_group_lex) существовали, но все выводы E4 аннулированы
M31. Это главная дырка работы.

---

## Что проверяли

### 1. Live CVaR vs global на Amazon (E1)

**Дизайн.** Одна разница — `fitness.mode`: `cvar_lex` vs `global`. Официальный
WILDS, роли данных v3 (fitness на D_select), 1 сид.

**Lowvar CVaR** `E1_cvar_lowvar/seed42_20260730_114314` (M17) — первый
stat-sig результат проекта:

| Метрика | Тест | CI95 | p |
|---------|------|------|--:|
| R_global | **−0.053** | [−0.074, −0.032] | <0.001 |
| CVaR | −0.069 | | 0.014 |
| McNemar | 216 сломано / 115 исправлено | | |

Каждый кластер хуже (−0.035 … −0.109) — равномерный ущерб, не trade-off хвоста.

При этом **на D_select** рост 0.4966 → 0.6238 = **+0.128** при проверенном
шумовом поле 0.009 (три повторных скоринга того же промпта). Gain реален на
360 примерах отбора и чисто переобучен за их пределами.

In-run сигналы, которые поймали регрессию: anchor monitor C1 (0 improved / 15
worsened, p≈0); damage report C2. Val selection key **не** поймал
(0.5116 → 0.5182).

Отчёт: OBSERVATIONS M17; канвас `E1-cvar-run-postmortem.canvas.tsx`.

**Контроль global + reject gate** `E1_constraint_global/seed42_20260801_002853`
(M19):

| | Δ R_global | CI95 | McNemar |
|--|-----------:|------|---------|
| vs seed | **+0.004** н.з. | [−0.008, +0.017] | 63/55 |

Предсказания сдвинулись **вверх** (115↑ / 22↓) — противоположность lowvar.
Механика «не вредить» работает; робастная цель внутри поиска — нет.

### 2. Офлайн-перебор 30 кандидатов (Phase 0 / M18)

Скрипт `scripts/phase0_offline_bounds.py`. 30 кэшированных кандидатов ×
D_select 360, без новых API.

- По сырому `R_global` лучший кандидат — **initial prompt**.
- Balanced-CVaR fitness рос, raw accuracy seed не бил in-sample.
- Cluster-specialist portfolio LOO **−3.6 pp** vs best single (oracle fit
  только +2.2 pp).
- Demotion strength corr с R_global = −0.95, с acc_5 = −1.0.
- Good/weak Pareto зоны **пусты** (C13: 0/30 поднимают 4★, не убивая 5★).
- Family-DRO Spearman с R_global = 0.12; его топ всё равно проигрывает raw
  (0.667 vs 0.714).

### 3. Хвост не переносится: D_select argmin ≠ test argmin (C8)

Rank Spearman per-cluster accuracy D_select↔test = **+0.70** в обоих рукавах,
но **argmin разный**: худший на D_select = c3, на тесте = c0.

cvar_lex честно поднял c3 (test +0.050) и заплатил c0 (−0.087): хвост
сплющился, CVaR на тесте стал хуже. Все три воркера деградировали
(deepseek 0.691→0.669, kimi 0.695→0.668, qwen 0.653→0.640) — не артефакт
агрегации.

Причина: ~8 users/cluster на D_select недостаточно, чтобы идентифицировать
*какая* группа хвост. Это прямой предок RQ5/RQ6.

### 4. Скаляр прячет порог (C11)

Initial → cvar-final, per-class accuracy:

| Класс | initial | final | Δ |
|------:|--------:|------:|--:|
| 4★ | 0.378 | 0.539 | **+0.161** |
| 5★ | 0.899 | 0.779 | **−0.120** |
| 3★ | | | +0.026 |
| 2★ | | | −0.071 |
| R_global | | | −0.021 |

Эволюция нашла правильное направление (4↔5 collapse, C5: 28% примеров) и
перелетела порог, потому что скалярный фитнес не говорит, где остановиться.
Доля 5★ = 56% примеров.

95–100% ошибок, показанных мутатору, — соседние классы, направления
сбалансированы (C9). Мутатор отвечает однонаправленными крышками
(«Series/sequel → 4, not 5»), которые ломают правильные 5★ на OOD.

### 5. Мягкая смесь среднего и хвоста (E2)

`fitness: global_tail_mix`.

| Сдвиг | Run | Δ R_global | p | R_tail |
|-------|-----|-----------:|--:|--------|
| user-OOD | `seed42_20260802_011259` | **−0.027** | 0.000 | −0.010 н.з. |
| category Books→non-Books | `seed42_20260802_052137` | **+0.009** | 0.045 | +0.010 н.з. |

Tail-mix не универсален. На user-OOD вредит, на category-shift слабый плюс.
Отчёты: `experiments/E2_cheap_ensemble_global_tail/RESULTS.md`,
`experiments/E2_category_shift_books/RESULTS.md`. M22, M23.

### 6. CivilComments E4 — нечитаемо

Arm A (`global`) vs Arm B (`min_group_lex`). На D_select B C1 давал градиент
0.604 → 0.674 за ~10 итераций (`deep_analysis_20260806.md`). На тесте n=800
R_worst_group: seed 0.636, A 0.591, B 0.682 — но same-day rescore обнулил
дельты, B 2×20 дал −20 flips, и M31 признал все метрики мёртвыми.

**Нельзя цитировать E4 как абляцию objective.** Можно цитировать только как
историю поломки измерения (RQ7).

---

## Метрики этой RQ

- Primary на Amazon: paired `R_global` и `CVaR_cluster` на официальном OOD.
- Диагностика: per-cluster Δ, per-class confusion, McNemar, D_select vs test
  Spearman/argmin.
- Fitness формулы: `prime/fitness/objective.py` (`cvar_lex`, `min_group_lex`,
  `soft_min_lex`, `global`, `global_tail_mix`).

---

## Графики и таблицы

| Что | Где |
|-----|-----|
| M17 paired bootstrap | OBSERVATIONS M17 |
| CVaR postmortem canvas | `canvases/E1-cvar-run-postmortem.canvas.tsx` |
| Pareto pair | `canvases/E1-pareto-v2-pair-postmortem.canvas.tsx` |
| Phase 0 offline | `scripts/phase0_offline_bounds.py`; M18 |
| Per-class trade-off | OBSERVATIONS C11 |
| Argmin mismatch | OBSERVATIONS C8 |
| E2 tables | CONTEXT_FOR_CHATBOT/RESULTS.md §D |
| E4 (аннулировано) | `experiments/E4_civilcomments/compare_all_test/RESULTS.md` |

Рисунок для статьи: scatter D_select CVaR vs test CVaR с подписанным lowvar
прогоном (большой плюс на x, большой минус на y); рядом confusion 4★/5★.

---

## Вывод

На Amazon встраивание worst-group цели в поиск (а) переобучается на D_select,
(б) оптимизирует не ту группу, (в) двигает порог без стоп-сигнала. Контроль
«обычная точность + gate на не-ухудшение» не регрессирует и не выигрывает.
Это отрицательный результат с механизмом, не «не получилось».

Формулировка в статье должна быть сужена: **на Amazon user-shift при
pred_profile-группах**. Расширение на CivilComments требует нового эксперимента.

## Чего не хватает (главная дырка проекта)

**Абляция цели поиска на CivilComments при уже исправленном стеке E5:**
два рукава, одинаковые D_select / D_dev_targeted / test_fixed / single scorer /
fail-closed, разница только `soft_min_lex` vs `global`. Дорого (живые прогоны),
но без этого RQ3 — однодатасетный.

Пока этого нет — в abstract и contributions RQ3 не поднимать до универсального
утверждения.

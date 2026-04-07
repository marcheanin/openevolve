# Метрики и формулы (wilds_active_learn_approach)

Ниже собраны метрики, которые реально используются в пайплайне, с формулами и кратким смыслом.

## 1) Базовые метрики качества

### `R_global`

Доля правильно классифицированных примеров:

$$
R_{\text{global}} = \frac{1}{N}\sum_{i=1}^{N}\mathbf{1}\!\left(\hat y_i = y_i\right)
$$


### `R_worst`

Сначала считаем accuracy по каждому пользователю `u`:

$$
acc_u = \frac{1}{N_u}\sum_{i:\,u_i=u}\mathbf{1}\!\left(\hat y_i = y_i\right)
$$

Далее берется 10-й перцентиль:

$$
R_{\text{worst}} = P_{10}\!\left(\{acc_u\}\right)
$$


### `mae`

Mean Absolute Error на шкале 1..5:

$$
\mathrm{MAE} = \frac{1}{N}\sum_{i=1}^{N}\left|\hat y_i - y_i\right|
$$


## 2) Метрики согласованности ансамбля

### `mean_kappa`

Среднее quadratic weighted Cohen's kappa по всем парам воркеров:

$$
\mathrm{mean\_kappa}
= \frac{1}{\binom{W}{2}}\sum_{a<b}\kappa^{(\mathrm{quadratic})}(w_a,w_b)
$$


### `mean_kappa_unweighted`

Среднее обычной (невзвешенной) Cohen's kappa:

$$
\mathrm{mean\_kappa\_unweighted}
= \frac{1}{\binom{W}{2}}\sum_{a<b}\kappa(w_a,w_b)
$$


### `disagreement_rate`

Доля примеров, где хотя бы два воркера дали разные ответы:

$$
\mathrm{disagreement\_rate}
= \frac{1}{N}\sum_{i=1}^{N}\mathbf{1}\!\left(\left|\{\hat y_i^{(w)}\}\right|>1\right)
$$


## 3) Unified combined score (val/test)

Используется для full-evaluation (validation/test).

Базовая часть:

$$
\mathrm{base}
= 0.4\,R_{\text{global}}
+ 0.3\,R_{\text{worst}}
+ 0.3\left(1-\frac{\mathrm{MAE}}{4}\right)
$$

Для ансамбля добавляется бонус:

$$
\mathrm{score}
= \mathrm{base} + 0.1\cdot\max\!\left(0,\mathrm{mean\_kappa}\right)
$$

Итог ограничивается диапазоном:

$$
\mathrm{combined\_score}
= \min\!\left(1,\max(0,\mathrm{score})\right)
$$


## 4) Fitness для эволюции на active batch (`evaluate_fast`)

$$
\mathrm{fitness}
= 0.5\,\mathrm{Acc\_Hard}
+ 0.3\,\mathrm{Acc\_Anchor}
+ 0.2\cdot\max\!\left(0,\kappa_{\text{Hard}}\right)
- \mathrm{length\_penalty}
$$


Где:

- `Acc_Hard` — accuracy на Hard-примерах.
- `Acc_Anchor` — accuracy на Anchor-примерах.
- `kappa_Hard` — mean quadratic kappa между воркерами только на Hard.

### Штраф за длину (`length_penalty`)

Оценка токенов:

$$
\mathrm{tokens} \approx \left\lfloor\frac{\mathrm{len(prompt)}}{4}\right\rfloor
$$


Если `tokens <= 2000`, штраф 0.

Если `tokens > 2000`:

$$
\mathrm{length\_penalty}
= 0.02\cdot\frac{\mathrm{tokens}-2000}{100}
\quad\text{for }\mathrm{tokens}>2000
$$


## 5) Hard/Anchor и связанные метрики

### `disagreement_score` (ordinal-aware, [0,1])

$$
\mathrm{disagreement\_score}
= \frac{1}{\binom{W}{2}\left(r_{\max}-r_{\min}\right)}
\sum_{a<b}\left|\hat y^{(a)}-\hat y^{(b)}\right|
$$

В проекте: `r_min=1`, `r_max=5`.

### Правило классификации Hard/Anchor

Пример считается **Hard**, если:

$$
(\hat y \neq y)\ \lor\ (\mathrm{disagreement\_score}>\mathrm{uncertainty\_threshold})
$$


Иначе — Anchor.

### `Acc_Hard`

$$
\mathrm{Acc\_Hard}=\frac{\#\mathrm{correct\_hard}}{\#\mathrm{hard}}
$$


### `Acc_Anchor`

$$
\mathrm{Acc\_Anchor}=\frac{\#\mathrm{correct\_anchor}}{\#\mathrm{anchor}}
$$


Также логируются счетчики: `n_hard`, `n_anchor`, `n_total`.

## 6) Дополнительные feature-метрики

### `prompt_length` (feature dimension)

В feature dimensions:

$$
\mathrm{prompt\_length}=\min\!\left(1,\frac{\mathrm{len(prompt)}}{2000}\right)
$$


В `evaluate()` для active-batch-метрик встречается также версия:

$$
\mathrm{prompt\_length}=\min\!\left(1,\frac{\mathrm{tokens}}{3000}\right),
\quad \mathrm{tokens}\approx\frac{\mathrm{len(prompt)}}{4}
$$


### `sentiment_vocabulary_richness`

$$
\mathrm{sentiment\_vocabulary\_richness}
= 0.6\cdot\mathrm{sentiment\_vocabulary}
+ 0.4\cdot\mathrm{example\_richness}
$$


С ограничением в `[0,1]`.

---

## Где что используется

- **Эволюция (inner loop):** `fitness` из active-batch (`Acc_Hard`, `Acc_Anchor`, `kappa_Hard`, `length_penalty`).
- **Сравнение на val/test:** `combined_score` (unified), плюс `R_global`, `R_worst`, `mae`, `mean_kappa`.
- **Диагностика AL:** `test_Acc_Hard`, `test_Acc_Anchor`, `val_Acc_Hard`, `val_Acc_Anchor`, `n_hard`, `n_anchor`.


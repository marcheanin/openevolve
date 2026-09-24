# Soft Latent Types (SLT) — подход к группам для group-robust prompt evolution

Опорный дизайн группировки для PRIME v2. Не привязан к Amazon-WILDS как единственной
задаче: WILDS — первый стенд. Ниже: основания, полный алгоритм, почему должен работать,
общая применимость к OOD-аннотации, связь с текущим кодом.

---

## 0. Одна фраза

**Мы не кластеризуем user_id и не оптимизируем «худших знакомых авторов».**  
Мы оцениваем **мягкую принадлежность каждой единицы сдвига** (user / domain / site / …)
к небольшому набору **переносимых латентных типов поведения**, и ведём эволюцию промпта
как **Prompt-space CVaR/DRO по этим типам**, с интерпретируемыми правилами под каждый тип.

Имя механизма: **Soft Latent Types (SLT)**.

---

## 1. На чём основан (не «взяли k-means с потолка»)

Подход — стык четырёх линий, каждая решает свой кусок:

| Линия | Что берём | Что не копируем слепо |
|-------|-----------|------------------------|
| **GroupDRO / CVaR-DRO** (Sagawa et al.; CVaR-варианты) | Оптимизировать хвост групп, не среднее | Градиентный reweight весов модели |
| **Inferred / probabilistic groups** (PG-DRO AAAI'23; AGRO; gradient-space clustering DistShift'23) | Группы часто неизвестны → их *оценивают*; soft membership честнее hard labels | Нужны logits/gradients модели; у API-LLM этого нет |
| **Error-slice / failure mode discovery** | Ошибки структурированы: типы провалов, а не случайный шум | Обычно post-hoc анализ, не loop оптимизации промпта |
| **Prompt evolution + reflection** (GEPA, APO, Dynamic Cheatsheet) | Текстовый фидбек → правила в промпте | У них нет group/OOD-оси |

**Ключевой сдвиг относительно «просто кластеризовать юзеров»:**  
в weight-space DRO группы — это индексы для reweight loss.  
В **prompt-space** группы ещё и **семантика для мутатора**: тип должен быть описуем
на естественном языке («краткий сарказм», «длинный mixed-tone»), иначе CVaR улучшает
число, но не даёт *что* править в инструкции.

Поэтому SLT проектируется сразу под два выхода:
1. **числовой** — soft weights для CVaR / acquisition;
2. **текстовый** — краткие дескрипторы типов для error report и style-conditional rules.

---

## 2. Общая постановка (не только WILDS)

### 2.1 Единица сдвига (Shift Unit)

В разных OOD-задачах «что сдвигается» разное. Абстракция:

| Задача | Shift unit \(u\) | Что переносится на test |
|--------|------------------|-------------------------|
| Amazon-WILDS | reviewer (user) | стиль письма + калибровка оценок |
| CivilComments-WILDS | комментарий + identity-атрибуты | (oracle groups есть) или латентные стили токсичности |
| Медицинская разметка | больница / врач / корпус | жаргон, шаблоны заметок |
| Новости / модерация | источник / время | регистр, темы |
| Category shift | product category / домен | лексика домена |

**SLT всегда работает на уровне unit \(u\)**, а примеры наследуют типы своего unit.

Формально: есть пул размеченных (частично) примеров \((x_i, y_i, u_i)\),  
test приходит с **новыми** \(u\) (или новым доменом), метки на test — только для оценки.

### 2.2 Что оптимизируем

Промпт \(P\) + замороженный ансамбль \(f_1..f_M\).  
Цель: высокий **хвост качества по типам**, а не только mean accuracy:

\[
\max_P \; \mathrm{CVaR}_{q}\big(\{\mathrm{Acc}_\tau(P)\}_{\tau=1}^{K}\big)
\quad\text{(и параллельно не уронить global Acc)}
\]

где \(\mathrm{Acc}_\tau\) — accuracy, взвешенная soft-принадлежностью к типу \(\tau\).

Официальный протокол бенчмарка (для Amazon — 10th percentile **по user_id**) остаётся
внешней метрикой. Типы — **внутренний surrogate**, который должен *коррелировать* с
официальным хвостом (это проверяется пилотом E1).

---

## 3. Как устроены Soft Latent Types

### 3.1 Два блока признаков (принципиально разделены)

Проблема наивного «всё в один k-means»:  
признаки из **текста** переносятся на OOD unit без меток;  
признаки из **gold-рейтингов** (mean star) на test — это почти «группировать по ответу»,
что методологически грязно и на чисто unlabeled target недоступно.

**Блок T — Transferable (доступен всегда, в т.ч. на OOD без меток):**
- mean embedding отзывов / текстов unit (sentence-transformer или encoder задачи);
- length stats (mean/std log-length);
- опционально: доменные bag-of-cues (категория, язык, timestamp bucket) если есть в metadata;
- опционально: lexical markers (вопросность, восклицания, доля 1-го лица) — дешёвые и интерпретируемые.

**Блок C — Calibration (только там, где есть labels на source):**
- mean label, label std, доля экстремумов (1 и 5), confusion-prone soft signals;
- используется для **обогащения типов на source** и для интерпретации,  
  **не** как обязательный вход для assignment на unlabeled target.

### 3.2 Soft membership, не hard cluster id

Вместо «user ∈ cluster 3» — распределение \(\pi_u(\tau) = P(\tau \mid u)\), \(\sum_\tau \pi_u(\tau)=1\).

Почему soft (PG-DRO intuition):
- один автор может быть и «кратким», и «саркастичным»;
- hard assignment на маленьком \(K\) даёт пустые/крошечные кластеры → шум CVaR;
- soft позволяет CVaR считать как weighted accuracy:

\[
\mathrm{Acc}_\tau = \frac{\sum_i \pi_{u_i}(\tau)\, \mathbf{1}[\hat y_i=y_i]}{\sum_i \pi_{u_i}(\tau)}.
\]

Практическая реализация soft без тяжёлого grouper-network (для API-бюджета):
1. Fit **K прототипов** (центроидов) в стандартизованном пространстве блока T (+ опц. C на train).
2. \(\pi_u(\tau) \propto \exp(-\|z_u - c_\tau\|^2 / T_\mathrm{temp})\) (softmax по расстоянию).  
   Temperature \(T_\mathrm{temp}\) — один явный гиперпараметр (сетка в E6), не магическая смесь весов embedding vs stats.

Hard id для логов = \(\arg\max_\tau \pi_u(\tau)\), но **все метрики/квоты — через \(\pi\)**.

### 3.3 Иерархия типов (общая применимость)

Для задач шире Amazon:

```
Level Dom:  домен / категория / сайт          (если metadata есть)
Level Beh:  поведенческий стиль unit           (блок T)
Level Fail: (опционально, после 1-го прогона)  residual / disagreement slices
```

- Если oracle groups есть (CivilComments identity) — **Dom/oracle заменяет или дополняет** Beh;
  SLT не спорит с oracle, а даёт fallback когда oracle нет.
- Fail-уровень: после прогона ансамбля уточняем типы по паттерну ошибок
  (какие типы дают 3→4, какие — сарказм). Это близко к error-slice discovery, но без
  градиентов: признаки = (embedding unit, disagreement, confusion bin).

Итого на Amazon стартуем с **Beh (блок T) + мягкое использование C только на train**;  
Fail-уровень — вторая итерация после E1, не обязателен для пилота.

### 3.4 Текстовые дескрипторы типов (зачем prompt-space)

После fit прототипов для каждого \(\tau\):
- top-N nearest units на train;
- LLM-куратор (или шаблон) пишет 1–2 предложения: *«короткие отзывы с гиперболами и
  смешанной оценкой; часто путают 3 и 4»*.

Дескрипторы:
- попадают в mutator artifacts («ошибки типа τ₃: …»);
- становятся заготовкой для style-conditional rules в промпте;
- делают метод **интерпретируемым** — преимущество перед невидимыми reweight в LoRA/GroupDRO.

---

## 4. Полный цикл работы (как собирается с эволюцией промпта)

```
[0] Выбрать shift unit u (user / domain / …)

[1] SOURCE FIT (один раз на train)
    - построить z_u из блока T (+ C если labels есть)
    - StandardScaler на train
    - K прототипов (k-means / spherical k-means на нормированных z)
    - π_u(τ) = softmax(-dist²/T)
    - дескрипторы типов
    - сохранить: scaler, centroids, descriptors  → артефакт, переносимый на target

[2] TARGET ASSIGN (val/test / новый домен)
    - ТОЛЬКО блок T (+ metadata Dom)
    - тот же scaler + те же centroids → π_u(τ)
    - labels target НЕ нужны для assignment

[3] Каждый AL / evolution cycle
    a. Прогнать ансамбль на Seen / batch
    b. Soft Acc_τ и CVaR_q({Acc_τ})
    c. Acquisition:
         score_i = lex(err_i, d_i)          # информативность примера
         weight_i ∝ Σ_τ π_{u_i}(τ)·(1-Acc_τ) # больше веса хвосту типов
         взять top по score_i * weight_i, diversity по тексту внутри топа
    d. Fitness кандидата P:
         α·CVaR_soft + β·Acc_global + γ·κ − length
    e. Mutator feedback:
         per-type error slices + дескриптор типа → style-conditional rules
    f. Carryover: пул промптов; опц. держать специалистов по типам (Pareto по τ)

[4] Selection между циклами: первично CVaR_soft на val (через π), затем global

[5] External report: официальный OOD metric (R_worst users / group acc) + CVaR_soft
```

Это и есть целостная система: **одни и те же π** питают acquisition, fitness, feedback, selection.

---

## 5. Почему должно «выстрелить» (аргументы, не обещания)

### 5.1 Согласовано с природой сдвига
OOD в аннотации почти никогда не «случайные id». Это **повторяемые режимы письма и
калибровки**. Если промпт чинит режим, он чинит *класс* будущих авторов/доменов —
в отличие от запоминания конкретных train-user.

### 5.2 Surrogate для хвоста, который можно оптимизировать в батче
R_worst по юзерам на батче из 80 примеров — статистически мёртв (2–4 отзыва на юзера).  
Soft Acc по K=6–10 типам с десятками weighted-примеров — оцениваемый сигнал для эволюции.
Если типы хоть умеренно коррелируют с user-хвостом — CVaR тянет официальный R_worst
косвенно (гипотеза E1).

### 5.3 Уникальный рычаг prompt-space
Weight-DRO только перевзвешивает loss.  
SLT даёт мутатору **имя провала** → правило в тексте → то же правило применяется к
OOD unit того же типа без дообучения. Это и есть ставка на «почему LLM-подход к OOD
аннотации может быть лучше слепого FT на mean CE».

### 5.4 Soft > hard для маленького K
Меньше пустых кластеров, меньше скачков CVaR, честнее overlapping failure modes —
уроки PG-DRO, перенесённые в prompt-loop без нейросетевого grouper.

### 5.5 Разделение T/C чинит главную методологическую дыру текущего кода
OOD assignment без gold → честный перенос; calibration не «утекает» в определение
тестовых групп.

### 5.6 Общая применимость одним рецептом
Меняется только определение unit и блок Dom metadata. Алгоритм π → CVaR → acquisition →
style rules один и тот же для user-shift, domain-shift, site-shift.  
WILDS Amazon — инстанс с `u = user`. CivilComments — инстанс с oracle Dom + опц. Beh.
Медицина — `u = hospital`. Это и есть «не только WILDS».

---

## 6. Чем это лучше наивного «k-means по юзерам» (в т.ч. текущего v2-кода)

| Наивный / текущий код | SLT |
|-----------------------|-----|
| Hard cluster id | Soft \(\pi_u(\tau)\) |
| Один вектор: emb ‖ mean_rating ‖ … без scaler | Блок T / блок C разделены; StandardScaler; temperature |
| Test assignment с gold в профиле | Target assignment **только T** |
| `group_aware` ≈ сортировка по cluster_id | weight ∝ \(\sum_\tau \pi(\tau)(1-\mathrm{Acc}_\tau)\) |
| Quotas «≥1 кластер в батче» | Непрерывное перевзвешивание хвоста |
| Кластер как номер | Кластер + **текстовый дескриптор** для мутатора |
| Amazon-specific wording | Shift unit abstraction |

Текущий код v2 — **скелет** (профили, centroids, CVaR, attach ids). SLT — **спецификация,
в которую этот скелет нужно дотянуть**, а не выбросить.

---

## 7. Риски и как их закрывать (честно)

1. **Типы не коррелируют с R_worst** → E1 no-go; тогда либо refining Fail-уровнем, либо
   study-фрейминг (систематика prompt-opt under shift) без сильного method-claim.
2. **K и temperature магические** → сетка E6; выбирать по val CVaR / корреляции с R_worst,
   не по test.
3. **Дескрипторы галлюцинируют** → дескрипторы только из nearest neighbors + шаблон;
   LLM-редактор опционален и абляируется.
4. **Доминирование embedding** → StandardScaler + ограничение dim (PCA 32–64 на emb)
   или отдельный вес блока stats после нормировки.
5. **Oracle vs inferred** → на CivilComments всегда репортить оба; SLT не должен
   проигрывать oracle-GroupDRO-style selection на той же задаче слишком сильно.

---

## 8. Минимальный план внедрения (от текущего кода к SLT)

1. `profiles.py`: разделить `features_transferable(u)` и `features_calibration(u)`.
2. `clustering.py`: StandardScaler; fit на T (+C_train); `assign` только T; softmax soft-\(\pi\).
3. `metrics.py`: `Acc_τ` и CVaR через weights \(\pi\), не только hard ids.
4. `scoring.py` / `batch_builder.py`: заменить заглушку `group_aware` на
   `score * Σ π(τ)(1-Acc_τ)`.
5. `artifacts.py`: подмешивать дескрипторы типов.
6. Пилот E1: global fitness vs soft-CVaR (SLT) на официальном WILDS — go/no-go.

---

## 9. Как это формулировать в статье (черновик)

> Under annotation distribution shift, the relevant structure is not the identity of
> training units but **transferable latent types** of writing and labeling behavior.
> We estimate **soft membership** of each shift unit in a small set of types using
> label-free transferable features, optimize prompt evolution with a **CVaR objective
> over type-weighted accuracies**, and expose type descriptors to the mutator so that
> instructions become type-conditional. The same recipe applies whenever a shift unit
> can be defined (user, domain, site, …); we instantiate it first on Amazon-WILDS
> user shift and validate transfer of the recipe on a second shift benchmark.

---

## 10. Итог

**SLT** — не «ещё один k-means», а:
- правильная абстракция OOD для аннотации (shift unit + transferable types);
- soft Prompt-DRO вместо group-agnostic evolution;
- интерпретируемый мост в текстовые правила;
- один рецепт на класс задач, с WILDS Amazon как первым инстансом.

Именно это сочетание (переносимость + soft хвост + правила в промпте) даёт шанс
выиграть там, где generic EvoPrompt/APO поднимают mean и не двигают хвост.

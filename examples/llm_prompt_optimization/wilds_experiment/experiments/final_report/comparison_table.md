# Сравнительная таблица экспериментов

## Основные метрики

| Эксперимент | Настройка | R_global | R_worst | MAE | Combined Score (old) | Combined Score (unified) | Users | Examples |
|-------------|-----------|----------|---------|-----|----------------------|--------------------------|-------|----------|
| **Exp 1: Baseline** | 1 модель (gpt-oss-120b), без эволюции | 0.5102 | 0.3103 | 0.7043 | 0.4922 | 0.5443 | 102 | 1566 |
| **Exp 2: Single + Evolution** | 1 модель (gpt-oss-120b), с OpenEvolve | 0.5461 | 0.3455 | 0.6402 | 0.5286 | 0.5741 | 25 | 542 |
| **Exp 3a: Ensemble Baseline** | 3 модели (yandexgpt, gemma3-27b, gpt-oss-120b), majority vote, без эволюции | 0.6513 | 0.4401 | 0.3616 | 0.3336 | **0.7169** | 25 | 542 |
| **Exp 3b: Ensemble + Evolution (NEW)** | 3 модели, majority vote, с OpenEvolve (новая MAP-Elites) | **0.6827** | **0.4769** | **0.3376** | 0.3734 | **0.7360** | 25 | 542 |

## Метрики согласованности ансамбля (только для Exp 3)

| Эксперимент | Mean Kappa | Disagreement Rate |
|-------------|------------|-------------------|
| **Exp 3a: Ensemble Baseline** | 0.5280 | 0.4852 |
| **Exp 3b: Ensemble + Evolution (NEW)** | 0.4701 | 0.5480 |

## Анализ улучшений

### Exp 1 → Exp 2 (Single Model: Baseline → Evolution)
- **R_global**: +0.0359 (+7.0%) ✅
- **R_worst**: +0.0352 (+11.3%) ✅
- **MAE**: -0.0641 (-9.1%) ✅
- **Вывод**: Эволюция улучшила все метрики для одной модели

### Exp 3a Baseline → Exp 3b Evolution (NEW) (итоговое сравнение)
- **R_global**: +0.0314 (+4.8%) ✅
- **R_worst**: +0.0368 (+8.4%) ✅
- **MAE**: -0.0240 (-6.6%) ✅
- **Combined Score (unified)**: +0.0191 (+2.7%) ✅
- **Mean Kappa**: -0.0579 (-11.0%) ⚠️
- **Disagreement Rate**: +0.0628 (+12.9%) ⚠️
- **Вывод**: Эволюция значительно улучшила точность и общий combined score. Согласованность моделей немного снизилась, но это компенсируется улучшением точности

## Итоговые выводы

1. **Эволюция эффективна**: И Exp 2, и Exp 3 показывают улучшения после эволюции
2. **Ансамбль лучше одиночной модели**: Exp 3 (ensemble) значительно превосходит Exp 1 и Exp 2 по всем метрикам
3. **Лучший результат**: Exp 3b: Ensemble + Evolution (NEW) - R_global = 0.6827, Combined Score (unified) = 0.7360
4. **Улучшение от эволюции**: Exp 3b показывает +4.8% по R_global и +2.7% по Combined Score (unified) по сравнению с baseline

## Примечания о формулах

- **R_global**: средняя точность по всем примерам  
  `R_global = (кол-во правильных предсказаний) / N`.

- **R_worst**: 10‑й перцентиль точности по пользователям  
  1) для каждого пользователя считаем его accuracy:  
     `acc(user) = (правильные_для_user) / (все_для_user)`  
  2) берём 10‑й перцентиль значений `acc(user)` по всем пользователям.

- **MAE** (Mean Absolute Error): средняя абсолютная ошибка рейтинга (1–5)  
  `MAE = (1/N) * sum_i |y_hat_i - y_i|`.

- **Mean Kappa** (для ensemble): средний попарный коэффициент каппа Коэна между воркерами  
  усредняются все `kappa(worker_i, worker_j)` по всем парам моделей.

- **Disagreement Rate** (для ensemble): доля примеров, где воркеры дают разные ответы  
  `disagreement = (кол-во примеров, где есть >1 уникального ответа среди воркеров) / N`.

- **Combined Score (old)** (deprecated):  
  `combined_old = 0.4 * R_global + 0.4 * R_worst - 0.2 * (1 - disagreement)`.

- **Combined Score (unified)** (используется в отчёте):
  - базовая часть (общая для single и ensemble):  
    `base = 0.4 * R_global + 0.3 * R_worst + 0.3 * (1 - MAE/4)`
  - бонус за согласованность (только для ensemble):  
    `bonus = 0.1 * (1 - disagreement)`  
  - итоговая метрика:  
    `combined_unified = base + bonus`.

- **Проценты улучшения в таблице**:  
  - абсолютный прирост: `delta = new - old`  
  - относительное изменение (в скобках): `delta_% = (new - old) / old * 100%`.

## Примечания о MAP-Elites

- **Feature dimensions (Exp 2, Exp 3)**:  
  - `sentiment_vocabulary_richness` ∈ [0, 1] — комбинированная метрика:  
    `SVR = 0.6 * sentiment_vocabulary + 0.4 * example_richness`  
    (обе компоненты нормированы в [0, 1]).  
  - `domain_focus` ∈ [0, 1] — мера фокуса промпта на целевой домен (Exp 2).  
  - `mean_kappa` (Cohen's κ, обрезанная до [0, 1]) — межаннотаторское согласие воркеров (Exp 3).

- **Дискретизация признаков в ячейки** (`feature_bins = 10`):  
  для каждого признака `f` из диапазона [0, 1] индекс бина считается как:  
  `bin(f) = min( floor(f * feature_bins), feature_bins - 1 )`.  
  В Exp 3 получаем 2D‑решётку `10 x 10 = 100` ячеек по осям  
  `sentiment_vocabulary_richness` × `mean_kappa`.

- **Архив MAP‑Elites** (`archive_size = 500`):  
  хранится не более одной элиты на ячейку (по максимальному `combined_unified` в данной ячейке).  
  Параметры `elite_selection_ratio`, `exploration_ratio`, `exploitation_ratio` задают, какая доля новых кандидатов берётся из:
  - текущих элит (эксплуатация),
  - случайно выбранных решений (эксплорация),
  - топ‑элит по `combined_unified` (интенсивная доработка лучших).

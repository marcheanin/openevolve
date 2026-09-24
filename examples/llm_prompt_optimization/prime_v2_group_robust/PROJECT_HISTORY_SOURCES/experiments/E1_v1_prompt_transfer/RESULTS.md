# Results — v1 prompt transfer on Phase 1 OOD test

**Question:** does the historically successful v1 prompt (seed and/or evolved)
beat our GRAPE short seed on the *same* official WILDS OOD test + current workers?

**Substrate:** `E1_constraint_global/seed42_20260801_002853`  
test = 240 users / 1920 examples, workers = deepseek-v4-pro / kimi-k2.5 / qwen3-235b  
Artifacts: `results/.../evals/{grape_seed,v1_seed_all_categories,v1_final_evolve_subsample}/`

| Tag | Prompt |
|-----|--------|
| `grape_seed` | GRAPE `prompts/initial_prompt.txt` (~1.7k) — Phase 1 seed ensemble |
| `v1_seed` | `initial_prompt_all_categories.txt` (~3.4k) |
| `v1_final` | `results_all_categories_evolve_subsample/final_prompt.txt` (~6.3k) |

---

## Headline

| Prompt | R_global | CVaR_q40 | vs grape_seed ΔR | paired p |
|--------|--------:|---------:|-----------------:|---------:|
| grape_seed | **0.7219** | 0.6488 | — | — |
| v1_seed (rich) | 0.7104 | 0.6236 | **−0.0115** | 0.076 |
| v1_final (evolved) | **0.7328** | 0.6728 | **+0.0109** | 0.235 |

Noise floor: bootstrap SD(R_global)≈0.014 → min detectable ~**0.038** at 80% power.

### Did v1 evolution help *relative to its own seed*?

| | Δ R_global | CI95 | p |
|--|----------:|-----:|--:|
| v1_seed → v1_final | **+0.0224** | [+0.004, +0.040] | **0.015** |
| McNemar | 144 fixed / 101 broke | | 0.007 |

**Yes — on this OOD substrate the evolved v1 prompt significantly beats the rich v1 seed.**  
That means the *content* found by v1 evolution is real and transfers. It does **not**
significantly beat the short GRAPE seed (+0.011, p=0.24, inside noise).

---

## Reading (anti-tilt)

1. **Rich seed is not free headroom.** On current workers + official OOD, the short GRAPE seed *beats* the rich all-categories seed (−1.2 pp). Starting from v1's seed would have been a handicap, not a gift.

2. **v1 evolution was not an illusion** — it lifts its own seed by +2.2 pp on our OOD (p=0.015). The April climb was not only soft measurement on a custom split.

3. **Ceiling vs GRAPE seed is tiny.** Evolved v1 lands at 0.733 vs grape 0.722 — a point estimate in the right direction, but below the detectable effect. Same ballpark as Phase 1 final (0.726). Prompt-only deltas of ~0.01 are the wrong target on this stack.

4. **Caveat:** v1_final eval had 80/1920 kimi connection failures → default rating 3 (~1.4% of calls). Deepseek alone on that prompt is 0.722 (= grape ensemble). Direction of the story is unchanged; exact +0.011 may move slightly on a clean re-eval.

---

## Decision

| Hypothesis | Verdict |
|------------|---------|
| «Мы сломали рабочий v1 метод» | **Нет** — его финальный промпт жив и бьёт свой seed на OOD |
| «Нужно просто подставить v1-seed и всё поедет» | **Нет** — rich seed хуже short seed |
| «На Amazon ещё есть большой prompt-only прирост» | **Очень сомнительно** — даже лучший исторический промпт не выходит за noise vs текущего seed |

**Next:** Phase 2 на сдвиге с настоящими группами (CivilComments / category), или рычаг агрегации (C12, ~+0.09 oracle). Не тратить бюджет на «полный v1 replay» ради ещё +0.01 на Amazon.

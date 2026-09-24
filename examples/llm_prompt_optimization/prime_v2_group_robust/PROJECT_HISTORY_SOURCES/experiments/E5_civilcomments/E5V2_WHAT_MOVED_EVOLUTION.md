# E5v2 — What finally moved evolution forward

**Run:** `results/E5_civilcomments_prime_main_v2/seed42_20260809_080918`  
**Contrast:** `E5_civilcomments_prime_main/seed42_20260808_213053` (v1: OE found gains, shipped seed)  
**Date:** 2026-08-09  
**Related:** M31–M36, analysis/ under the run dir, canvas `e5-v2-run-comparison`

---

## Verdict

Evolution moved forward because **selection stopped vetoing honest D_select gains**.  
Search was already capable in E5v1 (C2 D_select fitness ↑); the broken piece was **cross-cycle Top-1 on full WILDS val**. Switching Top-1 + F8 to **D_dev softmin (GBA)** let C1 OE heir ship, then C4 refine it. On fair `test_fixed` (n=1800, same fingerprint): **raw worst-GBA 0.615 → 0.645 (+3.0 pp)** vs seed.

Necessary substrates (without which search signal is fake): single gemma-3, fail-closed, GBA/`soft_min_lex`, fixed balanced D_select. Those alone did **not** ship a better prompt in v1.

---

## Component attribution (ranked)

| Rank | Component | Role this run | Evidence |
|---:|---|---|---|
| **1** | **`selection_split: d_dev`** | **Decisive.** Promotes OE heir when D_dev softmin ≥ champion. | v1: val softmin seed 0.703 > OE 0.700 → kept seed. v2 C1: D_dev softmin 0.6879 > seed 0.6863 → shipped OE. Same-day rescore earlier showed C2 heir **+4 pp** worst-GBA on test_fixed while val said no. |
| **2** | **GBA + `soft_min_lex` (F3/F4)** | Makes D_select / D_dev / test_fixed *alive* (silence = 0.5). | Fitness identity: `fitness = softmin + 0.01·gba_mean` exact on C1 best. ALL-ZERO floor 0.5 on test_fixed. Rejects `-1e9` for empty GBA / degenerate. |
| **3** | **Single scorer + fail-closed (F1/F5)** | Removes M31 dead-worker / silent-0 hacking. | `invalid_rate=0` on shipped evals; OE rejects fire (`fitness=-1e9`). Scorer = gemma-3 only. |
| **4** | **F8 D_dev gate** | Confirms promote without blocking true lifts. | C1 accept `within_tolerance`, drop **−0.0016** (candidate better). C4 drop **−0.0005**. 0/4 rejects. `generalization_gap≈0.035` logged. |
| **5** | **Fixed D_select, no rotate (M28)** | Stable search target; OE can beat seed by ~1.5 pp fitness. | `content_hash=60577460b4ad8614`, 720 = 8×2×45. C1 best_evo 0.7229 vs seed entry 0.7082. |
| **6** | **`test_fixed` headline (F9)** | Honest OOD measure (not evolution driver). | fp `5cfb7ebde3c5`; comparable to seed/C2 same-day. |
| **7** | **`len_penalty_start=300`** | Minor enabler (C1=255 words unpaid; C4=312 lightly paid). | Not the main story vs selection. |
| **8** | **4×18 cycles** | Marginal: C4 tiny D_dev lift; final prompt ≠ C1. | C2–C3 carried; C4 heir=oe, softmin 0.6884 vs 0.6879. |
| **—** | **F6 self-consistency AL** | **Not a driver.** | `mean_disagreement_hard=0` in batch diag; SC often unanimous. |
| **—** | **Anchor gate** | Passive, not decisive. | Accepts with `tolerated_noise`, drop 0.03 (M12-class). |

---

## Timeline (v2)

| Cycle | What happened | Selection |
|---|---|---|
| 1 | OE finds best fitness **0.7229**; anchor OK; **dev-gate OK**; D_dev softmin beats seed | **Selects OE heir** (first shipped non-seed since Phase3) |
| 2–3 | No better D_dev key; heir carried | Plateau on selection |
| 4 | OE tiny lift (0.7231); D_dev softmin **0.6884** | Updates selected prompt (final) |

Prompt sizes: seed 211 words → C1 255 → C4 312.

---

## Fair test_fixed (raw GBA)

| Method | worst-GBA | recall | spec | note |
|---|---:|---:|---:|---|
| Seed | 0.615 | 0.749 | 0.657 | |
| E5v1 C2 heir (not shipped) | **0.655** | 0.744 | 0.689 | Would have been best; killed by full-val |
| E5v2 final | 0.645 | 0.651 | **0.771** | Shipped; +3.0 pp worst vs seed; conservative shift |

Artifacts: `results/.../analysis/RESULTS.md`, figures, `comparison_data.json`.

---

## What did *not* move the needle (this run)

- More OE iterations alone (v1 already found C2).
- Ensemble / QBC (removed).
- F6 uncertainty acquisition (flat disagreement).
- Full WILDS val as selector (actively harmful in v1).
- Rotation of D_select (kept off).

---

## Bugs / caveats to track

1. **`_eval_split` overwrites `R_worst_gba` with shrunk min** when attaching softmin → `summary.final_test.R_worst_gba=0.664` while raw min is **0.645**. Prefer raw for headlines; keep shrunk as `R_worst_gba_shrunk`.
2. **Budget tracker undercounts** OE child calls (`scorer_calls=5604`, `optimizer_calls=0` in summary) — O28 family; do not trust for budget-matched baselines yet.
3. **API / day noise (M29):** compare tables same-session when claiming &lt;2 pp.
4. **C2 heir still stronger on worst-GBA than v2 final** — selection/D_dev is better than val but not oracle; room for better gate τ / multi-seed.

---

## Implications for next work

1. Keep **D_dev Top-1 + F8** as locked protocol (§6.3).
2. E5v3 selection debt diagnosed (M41 / `E5V3_SELECTION_DEBT.md`): fix `_pick_heir` ranking before next PRIME live.
3. S9 matrix + OPRO/MIPROv2/Amazon — see `NEXT_STEPS.md`.
4. Ablation A3 (gate off / val selection) would quantify selection’s causal role.
5. Baselines (APE/APO/GPO) must use the **same** D_dev / test_fixed / scorer.
6. Fix raw vs shrunk field overwrite before publishing tables.
7. Do not invest in F6 until disagreement diagnostics show non-zero mass.

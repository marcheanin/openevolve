# E5 CivilComments — RESULTS (S9)

Preregistration: [`PREREGISTRATION.md`](PREREGISTRATION.md).  
Gate: [`gpo_gate.json`](gpo_gate.json) (**pass**).  
Next steps: [`NEXT_STEPS.md`](NEXT_STEPS.md).  
E5v3 debt: [`E5V3_SELECTION_DEBT.md`](E5V3_SELECTION_DEBT.md) (M41).  
Status: [`S9_STATUS.md`](S9_STATUS.md).

## Protocol

- Headline search/select: `R_worst_gba` / softmin-GBA on `D_dev`.
- **Report always:** `R_worst_gba`, **`R_worst_group` (Acc)**, **`R_global`** on `test_fixed` (n=1800).
- Scorer: `google/gemma-3-12b-it`. Optimizer: `deepseek/deepseek-v4-pro`.
- R10 seed42 = E5v2 artifact. Seeds 43/44 PRIME interrupted / pending.
- Artifacts: `results/E5_s9_matrix/seed{42,43,44}/`.

> **Gap:** completed S9 `metrics.json` did not persist Acc `R_worst_group` (soft_min_lex
> overwrote it with GBA, then `_metrics_block` dropped it). Fixed in
> `prime/fitness/objective.py` + `run_e5_s9_matrix.py` for future runs. Acc column
> below is **—** until same-session rescore.

## Completed baselines — mean over available seeds

| method | R_worst_gba | R_worst_group | R_global | Δ gba vs seed | n seeds |
|--------|------------:|--------------:|---------:|--------------:|--------:|
| gpo | 0.6562 | — | 0.7091 | +0.0130 | 3 |
| evoprompt_de | 0.6580 | — | 0.7115 | +0.0148 | 3 |
| evoprompt_ga | 0.6502 | — | 0.7090 | +0.0070 | 3 |
| random_al | 0.6469 | — | 0.6956 | +0.0037 | 3 |
| ape_k48 | 0.6495 | — | 0.7067 | +0.0063 | 3 |
| prime (E5v2) | 0.6566 | — | 0.7106 | +0.0134 | 1 |
| gepa | 0.6417 | — | 0.7002 | −0.0015 | 3 |
| seed | 0.6432 | — | 0.7011 | — | 3 |
| ape_ut | 0.6420 | — | 0.7011 | −0.0012 | 3 |
| ape | 0.6266 | — | 0.6974 | −0.0166 | 3 |
| apo | 0.6213 | — | 0.6872 | −0.0219 | 3 |
| oracle | 0.6353 | — | 0.6861 | −0.0079 | 1 |

Sorted by method family usefulness, not strict mean rank (EvoDE edges GPO on mean gba;
GPO wins seed42). Single-shot; exploratory APO/Evo caps.

## Per-seed detail (test_fixed)

### Seed 42

| method | R_worst_gba | R_worst_group | R_global |
|--------|------------:|--------------:|---------:|
| gpo | 0.6725 | — | 0.7161 |
| evoprompt_de | 0.6583 | — | 0.7167 |
| prime (E5v2) | 0.6566 | — | 0.7106 |
| ape_k48 | 0.6532 | — | 0.7111 |
| gepa | 0.6461 | — | 0.6989 |
| ape_ut | 0.6435 | — | 0.7022 |
| seed | 0.6433 | — | 0.7017 |
| oracle | 0.6353 | — | 0.6861 |
| evoprompt_ga | 0.6352 | — | 0.6983 |
| apo | 0.6246 | — | 0.6861 |
| random_al | 0.6190 | — | 0.6917 |
| ape | 0.6077 | — | 0.6894 |

### Seed 43

| method | R_worst_gba | R_worst_group | R_global |
|--------|------------:|--------------:|---------:|
| random_al | 0.6575 | — | 0.6956 |
| evoprompt_ga | 0.6541 | — | 0.7144 |
| evoprompt_de | 0.6519 | — | 0.7067 |
| gpo | 0.6487 | — | 0.7078 |
| ape_k48 | 0.6463 | — | 0.7122 |
| ape | 0.6435 | — | 0.7022 |
| seed | 0.6430 | — | 0.7006 |
| ape_ut | 0.6409 | — | 0.7056 |
| gepa | 0.6393 | — | 0.7000 |
| apo | 0.6179 | — | 0.6878 |
| prime | TBD | TBD | TBD |

### Seed 44

| method | R_worst_gba | R_worst_group | R_global |
|--------|------------:|--------------:|---------:|
| random_al | 0.6641 | — | 0.6994 |
| evoprompt_de | 0.6639 | — | 0.7111 |
| evoprompt_ga | 0.6613 | — | 0.7144 |
| ape_k48 | 0.6490 | — | 0.6967 |
| gpo | 0.6474 | — | 0.7033 |
| seed | 0.6432 | — | 0.7011 |
| ape_ut | 0.6416 | — | 0.6956 |
| gepa | 0.6398 | — | 0.7017 |
| ape | 0.6287 | — | 0.7006 |
| apo | 0.6215 | — | 0.6878 |
| prime | TBD | TBD | TBD |

## Gate (Yelp→Flipkart)

**pass** — see `gpo_gate.json`.

## Cost / protocol footnotes (exploratory caps)

- Random-AL / Oracle / APO: rounds=2 (native 6).
- EvoPrompt: reduced pop/gens vs native 10×10.
- GEPA: lightweight reflective+Pareto adapter (PyPI `gepa` not installed).

## Success criteria (seed42 only, provisional)

- PRIME Δ vs seed gba: +0.013 (**below** prereg +0.05).
- PRIME vs best of {APO, GPO, Random-AL}: GPO 0.672; PRIME −0.015 vs GPO.
- PRIME > Random-AL (seed42): yes.

## Resume

```powershell
$env:PYTHONIOENCODING='utf-8'
python scripts/run_e5_prime_main.py --config experiments/E5_civilcomments/config_prime_main_seed43.yaml
python scripts/run_e5_prime_main.py --config experiments/E5_civilcomments/config_prime_main_seed44.yaml
# after primes attach + fill R_worst_group via rescore/stable:
python scripts/run_e5_s9_stable_batch.py --seeds 42,43,44 --repeats 3
```

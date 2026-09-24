#!/usr/bin/env python3
"""Строит D_dev / test_fixed для PRIME на стендах mnli/toxlang.

Индексы — первые 900 (d_dev) / 1000 (test_fixed) строк тех же самых
dev_universe_*.json / truth_*.json, на которых отбираются базовые методы
(scripts/run_s12_live_loop.py: n_dev=900 префикс перемешанного dev_universe;
scripts/archive_regret.py и финалы: первые 1000 строк truth_*). Загрузчик
PRIME (prime/data/stand_loader.py) читает эти JSON без перестановки строк,
так что индекс i здесь = строка i там же — числа сопоставимы напрямую.

Выход: experiments/S16_prime_stands/{stand}/fixed_sets/{d_dev,test_fixed}.json
"""
from __future__ import annotations

import json
from pathlib import Path

from prime.data.balanced_cells import cell_counts, fingerprint_indices
from prime.data.fixed_sets import FixedSet

ROOT = Path(__file__).resolve().parents[1]
D_DEV_N = 900
TEST_N = 1000


def build(stand: str) -> None:
    dev_json = {"toxlang": "dev_universe_tox.json", "mnli": "dev_universe_mnli.json"}[stand]
    test_json = {"toxlang": "truth_tox.json", "mnli": "truth_mnli.json"}[stand]
    matrix_dir = {"toxlang": ROOT / "results/S15_toxlang_matrix/fixed_sets",
                  "mnli": ROOT / "results/S13_mnli_matrix/fixed_sets"}[stand]
    out_dir = ROOT / "experiments/S16_prime_stands" / stand / "fixed_sets"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, fname, n in (("d_dev", dev_json, D_DEV_N), ("test_fixed", test_json, TEST_N)):
        rec = json.loads((matrix_dir / fname).read_text(encoding="utf-8"))
        total = len(rec["texts"])
        if n > total:
            raise ValueError(f"{stand}/{fname}: only {total} rows, need {n}")
        indices = list(range(n))
        fs = FixedSet(
            name=name,
            source_split="validation" if name == "d_dev" else "test",
            indices=indices,
            fingerprint=fingerprint_indices(indices),
            design={"kind": "prefix_of_fixed_json", "source_file": fname, "n": n,
                   "note": "same prefix the S12 live-loop baselines select/report on"},
            cell_counts=cell_counts(rec["labels"], rec["cluster_ids"], indices),
        )
        fs.save(out_dir / f"{name}.json")
        print(f"{stand}/{name}: {n} строк, fp={fs.fingerprint}, "
              f"ячеек={len(fs.cell_counts)}, из {fname}")


if __name__ == "__main__":
    for s in ("toxlang", "mnli"):
        build(s)

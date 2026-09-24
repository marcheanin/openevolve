#!/usr/bin/env python
"""Launch remaining S9 matrix jobs sequentially (one API consumer at a time).

Skips methods that already have metrics.json under results/E5_s9_matrix/seedN/.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SEED42_METHODS = [
    "prime",
    "ape_ut",
    "ape_k48",
    "gepa",
    "evoprompt_ga",
    "evoprompt_de",
    "random_al",
    "oracle",
]
OTHER_SEED_METHODS = [
    "seed",
    "ape",
    "ape_k48",
    "ape_ut",
    "apo",
    "gpo",
    "gepa",
    "evoprompt_ga",
    "evoprompt_de",
    "random_al",
]


def _missing(seed: int, methods: list[str]) -> list[str]:
    out = ROOT / f"results/E5_s9_matrix/seed{seed}"
    miss = []
    for m in methods:
        if not (out / m / "metrics.json").is_file():
            miss.append(m)
        elif m == "prime":
            # May be prompt-only stub without test_fixed.
            import json

            block = json.loads((out / m / "metrics.json").read_text(encoding="utf-8"))
            if block.get("test_fixed") is None:
                miss.append(m)
    # dedupe preserve order
    seen = set()
    uniq = []
    for m in miss:
        if m not in seen:
            seen.add(m)
            uniq.append(m)
    return uniq


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    prime42 = ROOT / "results/E5_civilcomments_prime_main_v2/seed42_20260809_080918"

    for seed_s in [s.strip() for s in args.seeds.split(",") if s.strip()]:
        seed = int(seed_s)
        methods = SEED42_METHODS if seed == 42 else OTHER_SEED_METHODS
        miss = _missing(seed, methods)
        if seed != 42 and "prime" in methods:
            # PRIME 43/44 need separate live runs — skip here.
            pass
        if not miss:
            print(f"seed{seed}: nothing missing", flush=True)
            continue
        cmd = [
            sys.executable,
            str(ROOT / "scripts/run_e5_s9_matrix.py"),
            "--seed",
            str(seed),
            "--methods",
            ",".join(miss),
        ]
        if seed == 42 and "prime" in miss:
            cmd += ["--prime-run", str(prime42)]
        print("LAUNCH", " ".join(cmd), flush=True)
        if args.dry_run:
            continue
        rc = subprocess.call(cmd, cwd=str(ROOT))
        if rc != 0:
            print(f"seed{seed} failed rc={rc}", flush=True)
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

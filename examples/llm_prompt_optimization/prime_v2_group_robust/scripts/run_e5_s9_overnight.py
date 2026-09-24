#!/usr/bin/env python
"""Overnight S9 continuation: finish seed43/44 baselines then PRIME overlays.

Run after seed42 is complete. Sequential to avoid API contention.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BASELINE_METHODS = (
    "seed,ape,ape_ut,apo,gpo,gepa,random_al,ape_k48,evoprompt_ga,evoprompt_de"
)


def run(cmd: list[str]) -> int:
    print("LAUNCH", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


def main() -> int:
    py = sys.executable
    for seed in (43, 44):
        rc = run(
            [
                py,
                str(ROOT / "scripts/run_e5_s9_matrix.py"),
                "--seed",
                str(seed),
                "--methods",
                BASELINE_METHODS,
                "--apo-rounds",
                "2",
                "--evo-generations",
                "3",
                "--evo-pop",
                "4",
                "--gepa-reflections",
                "8",
            ]
        )
        if rc != 0:
            return rc
        cfg = ROOT / f"experiments/E5_civilcomments/config_prime_main_seed{seed}.yaml"
        rc = run([py, str(ROOT / "scripts/run_e5_prime_main.py"), "--config", str(cfg)])
        if rc != 0:
            return rc
        # Attach PRIME prompt into matrix folder for this seed.
        # Discover latest run dir by mtime under results/.
        results = ROOT / "results"
        cands = sorted(
            results.glob(f"E5_civilcomments_prime_main_s{seed}/seed{seed}_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not cands:
            cands = sorted(
                results.glob(f"*prime*seed{seed}*"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        if cands:
            rc = run(
                [
                    py,
                    str(ROOT / "scripts/run_e5_s9_matrix.py"),
                    "--seed",
                    str(seed),
                    "--methods",
                    "prime",
                    "--prime-run",
                    str(cands[0]),
                ]
            )
            if rc != 0:
                return rc
    # Same-session stable eval for all finished seeds.
    return run(
        [
            py,
            str(ROOT / "scripts/run_e5_s9_stable_batch.py"),
            "--seeds",
            "42,43,44",
            "--repeats",
            "3",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())

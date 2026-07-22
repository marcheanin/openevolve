#!/usr/bin/env python3
"""Run E1 smoke validation — exercises every pipeline stage with verbose logging."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))

from prime.config import load_config
from prime.controller import PrimeController


def main() -> int:
    parser = argparse.ArgumentParser(description="E1 smoke pipeline validation")
    parser.add_argument(
        "--config",
        type=Path,
        default=PKG_ROOT / "experiments" / "E1_pilot_cvar_vs_global" / "config_smoke_validate.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ctrl = PrimeController(cfg, PKG_ROOT, args.config)
    print("=" * 60, flush=True)
    print("PRIME v2 — E1 SMOKE VALIDATION", flush=True)
    print(f"config: {args.config}", flush=True)
    print(f"run_dir: {ctrl.ctx.run_dir}", flush=True)
    print("=" * 60, flush=True)

    summary = ctrl.run()
    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print(json.dumps(summary, indent=2, default=str), flush=True)
    print("=" * 60, flush=True)

    if not summary.get("all_stages_passed", True):
        print("SMOKE VALIDATION FAILED — see smoke_checklist.json", flush=True)
        return 1
    print("SMOKE VALIDATION PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

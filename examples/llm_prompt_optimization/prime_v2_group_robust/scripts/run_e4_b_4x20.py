#!/usr/bin/env python3
"""Launch Arm B 4x20 then M15 test-noise (3 repeats) + seed paired shot.

Pins eval_sets / pred_cache from Arm A when present (fair vs lean A/B).

Usage:
  python scripts/run_e4_b_4x20.py
  python scripts/run_e4_b_4x20.py --dry-run
  python scripts/run_e4_b_4x20.py --skip-noise   # evolution only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))

EXP = PKG_ROOT / "experiments" / "E4_civilcomments"
CFG = EXP / "config_arm_b_min_group_4x20.yaml"
ARM_A = (
    PKG_ROOT
    / "results"
    / "E4_civilcomments_arm_a_global"
    / "seed42_20260805_104411"
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-noise", action="store_true")
    ap.add_argument("--noise-repeats", type=int, default=3)
    ap.add_argument("--eval-sets-artifact", type=Path, default=None)
    ap.add_argument("--seed-pred-cache", type=Path, default=None)
    args = ap.parse_args()

    from scripts.run_e4_civilcomments import _run_one

    eval_sets = args.eval_sets_artifact
    if eval_sets is None and (ARM_A / "eval_sets.json").is_file():
        eval_sets = ARM_A / "eval_sets.json"
    cache = args.seed_pred_cache
    if cache is None and (ARM_A / "pred_cache").is_dir():
        cache = ARM_A / "pred_cache"

    print(
        f"[b4x20] config={CFG}\n"
        f"[b4x20] eval_sets={eval_sets}\n"
        f"[b4x20] seed_pred_cache={cache}",
        flush=True,
    )
    out = _run_one(
        CFG,
        dry_run=args.dry_run,
        eval_sets_artifact=eval_sets,
        seed_pred_cache=cache,
    )
    if args.dry_run or out is None:
        return 0

    run_dir = Path(out["run_dir"])
    pointer = {
        "run_dir": str(run_dir),
        "config": str(CFG),
        "n_cycles": 4,
        "n_evolve_iterations": 20,
        "final_test": (out.get("summary") or {}).get("final_test"),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    (EXP / "last_E4_civilcomments_arm_b_4x20.json").write_text(
        json.dumps(pointer, indent=2), encoding="utf-8"
    )

    if args.skip_noise:
        print("[b4x20] skip noise", flush=True)
        return 0

    noise_cmd = [
        sys.executable,
        str(PKG_ROOT / "scripts" / "e4_test_noise.py"),
        "--run-dir",
        str(run_dir),
        "--repeats",
        str(args.noise_repeats),
        "--also-seed",
        "--force",
    ]
    print(f"[b4x20] starting noise: {' '.join(noise_cmd)}", flush=True)
    rc = subprocess.call(noise_cmd, cwd=str(PKG_ROOT))
    if rc != 0:
        print(f"[b4x20] noise failed rc={rc}", flush=True)
        return rc
    print(f"[b4x20] done. run_dir={run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

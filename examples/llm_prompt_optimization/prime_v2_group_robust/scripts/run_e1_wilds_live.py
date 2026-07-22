#!/usr/bin/env python3
"""E1 live WILDS pilot: OpenRouter ensemble + 3 AL cycles."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))

DEFAULT_CONFIG = PKG_ROOT / "experiments" / "E1_pilot_cvar_vs_global" / "config_wilds_live_3cycle.yaml"


def _preflight_api_key() -> Optional[Path]:
    from prime.workers.ensemble import load_dotenv_if_present

    loaded = load_dotenv_if_present()
    key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        print("ERROR: set OPENROUTER_API_KEY (or OPENAI_API_KEY) for live OpenRouter inference.", file=sys.stderr)
        print("  Option A — .env file in prime_v2_group_robust/.env:", file=sys.stderr)
        print("    OPENROUTER_API_KEY=sk-or-...", file=sys.stderr)
        print("  Option B — reuse v1 .env (auto-detected):", file=sys.stderr)
        print("    wilds_active_learn_approach/.env  (OPENAI_API_KEY or OPENROUTER_API_KEY)", file=sys.stderr)
        print("  Option C — shell variable:", file=sys.stderr)
        print("    $env:OPENROUTER_API_KEY = 'sk-or-...'", file=sys.stderr)
        raise SystemExit(1)
    src = str(loaded) if loaded else "environment variable"
    print(f"[preflight] API key OK (from {src})", flush=True)
    return loaded


def _preflight_openevolve() -> bool:
    try:
        import openevolve  # noqa: F401
    except ImportError:
        print(
            "WARNING: package 'openevolve' not installed — inner-loop evolution will be eval-only "
            "(prompt will NOT mutate). Install from repo root: pip install -e ../../..",
            flush=True,
        )
        return False
    print("[preflight] OpenEvolve package OK", flush=True)
    return True


def _preflight_workers(cfg) -> None:
    from prime.workers.ensemble import probe_workers

    prompt_path = PKG_ROOT / cfg.prompt_path
    if not prompt_path.is_file():
        print(f"WARNING: prompt not found for worker probe: {prompt_path}", flush=True)
        return
    prompt = prompt_path.read_text(encoding="utf-8")
    print("[preflight] Probing OpenRouter workers (1 call each)...", flush=True)
    errors = probe_workers(cfg.ensemble, prompt)
    if errors:
        print("ERROR: OpenRouter worker probe failed:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        print(
            "Fix credits/model IDs before running — otherwise all predictions default to rating 3.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print("[preflight] All workers responded OK", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="E1 live WILDS + OpenRouter pilot")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="preflight + config only, no run")
    args = parser.parse_args()

    _preflight_api_key()

    from prime.config import load_config
    from prime.controller import PrimeController

    cfg = load_config(args.config)
    _preflight_openevolve()
    if not args.dry_run:
        _preflight_workers(cfg)
    if cfg.experiment.force_mock:
        print("WARNING: force_mock=true in config — will NOT call OpenRouter", flush=True)

    oe = cfg.openevolve_config_path
    oe_path = (PKG_ROOT / oe).resolve() if oe and not Path(oe).is_absolute() else Path(oe or "")
    print("=" * 60, flush=True)
    print("PRIME v2 — E1 LIVE WILDS PILOT", flush=True)
    print(f"config:     {args.config}", flush=True)
    print(f"AL cycles:  {cfg.active_learning.n_cycles}", flush=True)
    print(f"workers:    {[w.name for w in cfg.ensemble.workers]}", flush=True)
    print(f"OpenEvolve: {oe_path if oe else 'NOT SET (eval-only stub)'}", flush=True)
    print(f"WILDS caps: train_users={cfg.dataset.max_train_users}, batch={cfg.acquisition.batch_size}", flush=True)
    print("=" * 60, flush=True)

    if args.dry_run:
        print("Dry run OK — re-run without --dry-run to start.", flush=True)
        return 0

    ctrl = PrimeController(cfg, PKG_ROOT, args.config)
    print(f"run_dir: {ctrl.ctx.run_dir}", flush=True)
    summary = ctrl.run()

    print("\n" + "=" * 60, flush=True)
    print("DONE", flush=True)
    print(json.dumps(summary, indent=2, default=str), flush=True)
    if summary.get("token_usage"):
        print("\nToken usage:", json.dumps(summary["token_usage"], indent=2), flush=True)
    print("=" * 60, flush=True)
    return 0 if summary.get("all_stages_passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())

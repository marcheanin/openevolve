#!/usr/bin/env python
"""Live smoke for E5 PRIME-main (real gemma-3, tiny budget)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    from prime.cli import main as cli_main

    cfg = ROOT / "experiments" / "E5_civilcomments" / "config_prime_main_smoke.yaml"
    sys.argv = ["prime", "--config", str(cfg)]
    return int(cli_main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())

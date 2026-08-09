#!/usr/bin/env python
"""Launch E5 PRIME-main (single scorer + GBA soft_min)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from prime.cli import main as cli_main

    cfg = ROOT / "experiments" / "E5_civilcomments" / "config_prime_main.yaml"
    sys.argv = ["prime", "--config", str(cfg)] + sys.argv[1:]
    return int(cli_main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Launch E6 PRIME live (Amazon category-shift, gpt-4o-mini, pred_profile)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from prime.cli import main as cli_main
    from prime.workers.ensemble import load_dotenv_if_present

    load_dotenv_if_present()
    cfg = ROOT / "experiments" / "E6_amazon_category_controls" / "config.yaml"
    sys.argv = ["prime", "--config", str(cfg)] + sys.argv[1:]
    return int(cli_main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())

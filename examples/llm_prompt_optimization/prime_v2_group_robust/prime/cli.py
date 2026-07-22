"""CLI entry: python -m prime.cli --config experiments/E1/..."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PRIME v2 group-robust prompt evolution")
    parser.add_argument("--config", required=True, type=Path, help="Path to experiment YAML config")
    parser.add_argument("--smoke", action="store_true", help="Enable smoke mode (overrides config)")
    args = parser.parse_args(argv)

    pkg_root = Path(__file__).resolve().parents[1]
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))

    from prime.config import load_config
    from prime.controller import PrimeController

    overrides = {}
    if args.smoke:
        overrides["experiment"] = {"smoke": True}
    cfg = load_config(args.config, overrides=overrides if overrides else None)
    project_root = pkg_root
    ctrl = PrimeController(cfg, project_root, args.config)
    summary = ctrl.run()
    print(json_dumps(summary))
    return 0


def json_dumps(obj: object) -> str:
    import json
    return json.dumps(obj, indent=2, default=str)


if __name__ == "__main__":
    raise SystemExit(main())

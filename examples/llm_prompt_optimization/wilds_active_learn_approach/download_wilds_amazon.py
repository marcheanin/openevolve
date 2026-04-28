"""
Fallback if WILDS auto-download fails: materialize amazon data under wilds_experiment/data.

  python download_wilds_amazon.py
"""
from __future__ import annotations

import ssl
import sys
from pathlib import Path

ssl._create_default_https_context = ssl._create_unverified_context

SCRIPT_DIR = Path(__file__).resolve().parent
WILDS_DATA = SCRIPT_DIR.parent / "wilds_experiment" / "data"


def main() -> None:
    WILDS_DATA.mkdir(parents=True, exist_ok=True)
    try:
        from wilds import get_dataset
    except ImportError as e:
        print("pip install wilds", file=sys.stderr)
        raise SystemExit(1) from e
    print(f"Downloading to {WILDS_DATA} ...")
    get_dataset(dataset="amazon", download=True, root_dir=str(WILDS_DATA))
    print("Done.")


if __name__ == "__main__":
    main()

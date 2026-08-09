#!/usr/bin/env python
"""§6.4 reimplementation gate scaffold: GPO must beat APE on Yelp→Flipkart.

Full data wiring lands with baselines/gpo.py. This script documents the gate
criterion and fails closed until the body exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/E5_civilcomments/gpo_gate.json")
    args = parser.parse_args()
    try:
        from baselines import gpo  # noqa: F401

        implemented = True
    except Exception:
        implemented = False

    payload = {
        "status": "ready" if implemented else "not_implemented",
        "criterion": "GPO target accuracy > APE target accuracy on Yelp→Flipkart (36-shot, K=6, T=0.83)",
        "also": "APO > APE on source",
        "note": "Do not launch E5 matrix until this gate passes.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if implemented else 2


if __name__ == "__main__":
    raise SystemExit(main())

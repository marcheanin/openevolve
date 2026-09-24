#!/usr/bin/env python
"""E6 harshness sweep — Amazon R15 analogue (one-line DynamicRules edits)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED_PROMPT = ROOT / "prompts/initial_prompt.txt"

ANCHOR = "        - Negations flip meaning (\"not good\" -> negative)"

VARIANTS: list[tuple[str, int, str]] = [
    (
        "lenient_max",
        0,
        "        - Negations flip meaning (\"not good\" -> negative)\n"
        "        - When mixed or uncertain, prefer the HIGHER star rating; "
        "reserve low ratings for clear product failure",
    ),
    (
        "prefer_5",
        1,
        "        - Negations flip meaning (\"not good\" -> negative)\n"
        "        - When mostly positive with minor issues, prefer Rating: 5 over 4",
    ),
    (
        "seed_anchor",
        2,
        ANCHOR,
    ),
    (
        "prefer_4_over_5",
        3,
        "        - Negations flip meaning (\"not good\" -> negative)\n"
        "        - When positive but not ecstatic, prefer Rating: 4 over 5; "
        "reserve 5 for strong enthusiasm only",
    ),
    (
        "harsh_mid",
        4,
        "        - Negations flip meaning (\"not good\" -> negative)\n"
        "        - When mixed or uncertain, prefer the LOWER star rating",
    ),
    (
        "harsh_strong",
        5,
        "        - Negations flip meaning (\"not good\" -> negative)\n"
        "        - Lean critical: small complaints should lower the rating; "
        "when unsure between adjacent stars, choose the lower one",
    ),
    (
        "strict_5",
        6,
        "        - Negations flip meaning (\"not good\" -> negative)\n"
        "        - Rating 5 only if the review is unambiguously glowing with no caveats",
    ),
    (
        "inflate_positive",
        7,
        "        - Negations flip meaning (\"not good\" -> negative)\n"
        "        - Give benefit of the doubt on positive reviews; "
        "bump borderline 3→4 and 4→5 when overall tone is favorable",
    ),
    (
        "deflate_positive",
        8,
        "        - Negations flip meaning (\"not good\" -> negative)\n"
        "        - Be conservative on positive reviews; bump borderline 5→4 and 4→3 "
        "when any caveat or hesitation is present",
    ),
    (
        "ending_priority",
        9,
        "        - Negations flip meaning (\"not good\" -> negative)\n"
        "        - Weight the final sentence twice as heavily as earlier praise or complaints",
    ),
    (
        "intensity_strict",
        10,
        "        - Negations flip meaning (\"not good\" -> negative)\n"
        "        - Map intensity strictly: Weak→adjacent-to-3, Moderate→clear 2/4, "
        "Strong→1 or 5 only when extreme language is present",
    ),
    (
        "neutral_bias",
        11,
        "        - Negations flip meaning (\"not good\" -> negative)\n"
        "        - When evidence is balanced or sparse, prefer Rating: 3",
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "experiments/E6_amazon_category_controls/pools/harshness_sweep",
    )
    args = ap.parse_args()
    seed_text = SEED_PROMPT.read_text(encoding="utf-8")
    if ANCHOR not in seed_text:
        raise SystemExit(f"anchor line not found in seed prompt:\n{ANCHOR!r}")

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "pool": "harshness_sweep",
        "purpose": "E6 Amazon R15 analogue (operating-point / harshness)",
        "base_prompt": str(SEED_PROMPT.relative_to(ROOT)),
        "candidates": [],
    }
    for name, rank, line in VARIANTS:
        if name == "seed_anchor":
            text = seed_text
        else:
            text = seed_text.replace(ANCHOR, line)
            if text == seed_text:
                raise SystemExit(f"variant {name} produced no change")
        path = out / f"{rank:02d}_{name}.txt"
        path.write_text(text, encoding="utf-8")
        manifest["candidates"].append(
            {"name": name, "rank": rank, "file": path.name, "is_seed": name == "seed_anchor"}
        )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {len(VARIANTS)} candidates to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

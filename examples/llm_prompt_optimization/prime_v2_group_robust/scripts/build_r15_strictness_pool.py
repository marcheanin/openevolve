#!/usr/bin/env python
"""R15 calibration control: a pool of prompts that differ ONLY in strictness.

POWER_AUDIT F3 found that worst-group score across all 34 evaluated prompts is
explained by mean GBA (R^2 = 0.687), and that adding |recall - specificity|
explains nothing further: the losing prompts are simply the ones that drifted
into over-flagging. If a pure operating-point sweep of the seed prompt matches
the optimizers, then nothing group-specific is being learned.

Every variant is a single-line substitution in the seed prompt's tie-break rule,
so the pool is a clean one-dimensional sweep rather than a set of rewrites.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED_PROMPT = ROOT / "prompts/initial_prompt_civilcomments.txt"

ANCHOR_LINE = "        4. When unsure, prefer 0 (non-toxic) unless clear abusive intent"

# Ordered lenient -> strict. `rank` is the sweep coordinate for plotting.
VARIANTS: list[tuple[str, int, str]] = [
    (
        "nl_lenient_max",
        0,
        "        4. When unsure, label 0. Reserve label 1 for comments that are "
        "unambiguously abusive",
    ),
    ("thr_p90", 1, "        4. Label 1 only if the probability that the comment is toxic is at least 90%; otherwise label 0"),
    ("thr_p80", 2, "        4. Label 1 only if the probability that the comment is toxic is at least 80%; otherwise label 0"),
    ("thr_p70", 3, "        4. Label 1 only if the probability that the comment is toxic is at least 70%; otherwise label 0"),
    ("seed_anchor", 4, ANCHOR_LINE),
    ("thr_p60", 5, "        4. Label 1 only if the probability that the comment is toxic is at least 60%; otherwise label 0"),
    ("thr_p50", 6, "        4. Label 1 only if the probability that the comment is toxic is at least 50%; otherwise label 0"),
    ("thr_p40", 7, "        4. Label 1 only if the probability that the comment is toxic is at least 40%; otherwise label 0"),
    ("thr_p30", 8, "        4. Label 1 only if the probability that the comment is toxic is at least 30%; otherwise label 0"),
    ("thr_p20", 9, "        4. Label 1 only if the probability that the comment is toxic is at least 20%; otherwise label 0"),
    (
        "nl_strict",
        10,
        "        4. When unsure, prefer 1 (toxic); err on the side of flagging",
    ),
    (
        "nl_strict_max",
        11,
        "        4. When unsure, label 1. Flag any comment that a member of the "
        "targeted group could reasonably find demeaning",
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "experiments/E5_civilcomments/pools/strictness_sweep",
    )
    args = ap.parse_args()

    seed_text = SEED_PROMPT.read_text(encoding="utf-8")
    if ANCHOR_LINE not in seed_text:
        raise SystemExit(
            "anchor line not found verbatim in the seed prompt; refusing to guess.\n"
            f"expected: {ANCHOR_LINE!r}"
        )

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "pool": "strictness_sweep",
        "purpose": "R15 calibration control (POWER_AUDIT F3)",
        "base_prompt": str(SEED_PROMPT.relative_to(ROOT)),
        "edit": "single-line substitution of the tie-break rule",
        "anchor_line": ANCHOR_LINE,
        "candidates": [],
    }

    for name, rank, line in VARIANTS:
        text = seed_text.replace(ANCHOR_LINE, line)
        if name != "seed_anchor" and text == seed_text:
            raise SystemExit(f"variant {name} produced no change")
        path = out / f"{rank:02d}_{name}.txt"
        path.write_text(text, encoding="utf-8")
        manifest["candidates"].append(
            {
                "name": name,
                "rank": rank,
                "file": path.name,
                "line": line.strip(),
                "is_seed": name == "seed_anchor",
                "chars": len(text),
            }
        )

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {len(VARIANTS)} candidates to {out}")
    for c in manifest["candidates"]:
        print(f"  {c['rank']:2d} {c['name']:16} {c['line'][:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Verify E5 mutator feedback: gen1 sees PRIMARY TARGETS / polarity (O31)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _find_latest_run(results_root: Path, name_prefix: str) -> Path:
    cands = sorted(
        [p for p in results_root.glob(f"{name_prefix}*/seed*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not cands:
        # Also check results_smoke
        smoke = ROOT / "results_smoke"
        cands = sorted(
            [p for p in smoke.glob(f"{name_prefix}*/seed*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    if not cands:
        raise FileNotFoundError(f"No run matching {name_prefix}* under {results_root}")
    return cands[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=None)
    ap.add_argument("--name", default="E5_civilcomments_prime_main_v3")
    args = ap.parse_args()

    run = args.run or _find_latest_run(ROOT / "results", args.name)
    print("run:", run)

    frozen = list(run.glob("al_iter_*/mutator_artifacts_frozen.txt"))
    print("frozen_artifacts:", len(frozen))
    ok_frozen = False
    for p in frozen:
        text = p.read_text(encoding="utf-8", errors="replace")
        has = "PRIMARY TARGETS" in text or "POLARITY" in text or "FP" in text or "FN" in text
        print(f"  {p.relative_to(run)}: chars={len(text)} polarityish={has}")
        if text.strip() and has:
            ok_frozen = True

    # OE program artifacts_json (what the mutator actually sees via error_examples).
    # Also follow al_iter_*/openevolve_output junctions and results/_oe siblings.
    oe_roots = [run]
    oe_roots.extend(run.glob("al_iter_*/openevolve_output"))
    oe_hits = []
    seen_paths = set()
    for root in oe_roots:
        if not root.exists():
            continue
        for p in root.rglob("programs/*.json"):
            key = str(p.resolve())
            if key in seen_paths:
                continue
            seen_paths.add(key)
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            aj = data.get("artifacts_json")
            if not aj:
                continue
            if isinstance(aj, str):
                try:
                    arts = json.loads(aj)
                except json.JSONDecodeError:
                    arts = {}
            else:
                arts = aj
            ee = (arts or {}).get("error_examples") or ""
            if "PRIMARY TARGETS" in ee:
                try:
                    oe_hits.append(str(p.relative_to(run)))
                except ValueError:
                    oe_hits.append(str(p))
    print("oe_programs_with_PRIMARY_TARGETS:", len(oe_hits))
    for h in oe_hits[:20]:
        print(" ", h)

    # OE dumps prompts if present
    prompt_hits = []
    for p in run.rglob("*.txt"):
        if p.stat().st_size > 2_000_000:
            continue
        try:
            t = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if "PRIMARY TARGETS" in t:
            prompt_hits.append(str(p.relative_to(run)))
    print("files_with_PRIMARY_TARGETS:", len(prompt_hits))
    for h in prompt_hits[:20]:
        print(" ", h)

    # Also scan jsonl mutation / oe logs
    jsonl_hits = 0
    for p in run.rglob("*.jsonl"):
        try:
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                if "PRIMARY TARGETS" in line or "POLARITY" in line:
                    jsonl_hits += 1
                    break
        except Exception:
            continue
    print("jsonl_with_targets_or_polarity:", jsonl_hits)

    report = {
        "run": str(run),
        "frozen_n": len(frozen),
        "ok_frozen": ok_frozen,
        "oe_hits": oe_hits[:50],
        "prompt_hits": prompt_hits[:50],
        "jsonl_hits": jsonl_hits,
        "pass": ok_frozen or bool(oe_hits) or bool(prompt_hits) or jsonl_hits > 0,
    }
    out = run / "analysis" / "mutator_primary_targets_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("pass:", report["pass"], "->", out)
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

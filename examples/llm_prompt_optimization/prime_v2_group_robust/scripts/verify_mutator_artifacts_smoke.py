#!/usr/bin/env python3
"""Post-smoke checks: cheap ensemble + mutator coaching + artifact wiring.

Reads OE programs from the short staging path (results/_oe/<token>/), not the
Windows junction under al_iter_N/ (MAX_PATH breaks glob/open there).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PKG_ROOT = Path(__file__).resolve().parents[1]

CHEAP_WORKERS = {
    "openai/gpt-4o-mini",
    "google/gemini-2.5-flash-lite",
    "qwen/qwen3-32b",
}

ARTIFACT_MARKERS = [
    "MUTATION REMINDERS",
    "D_select",
    "Every rule must fire on something visible",
    "Do NOT reference cluster",
    "TARGET BLOCK",
]

SYSTEM_MARKERS_CORE = [
    "R_global",
    "R_tail",
    "held-out selection set",
]


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _find_latest_run(results_root: Path, name: str) -> Optional[Path]:
    base = results_root / name
    if not base.is_dir():
        return None
    runs = sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name)
    return runs[-1] if runs else None


def _oe_staging(run_dir: Path) -> Optional[Path]:
    path_txt = run_dir / "al_iter_1" / "openevolve_output_path.txt"
    if path_txt.is_file():
        staged = Path(path_txt.read_text(encoding="utf-8").strip())
        if staged.is_dir():
            return staged
    return None


def _collect_program_files(oe_root: Path) -> List[Path]:
    # Prefer latest checkpoint only (dedupe by program id via filename later).
    cps = sorted(oe_root.glob("checkpoints/checkpoint_*"))
    if not cps:
        return []
    latest = cps[-1]
    return sorted(latest.glob("programs/*.json"))


def _check_ensemble(cfg: Dict[str, Any]) -> List[str]:
    fails = []
    workers = [w.get("name") for w in (cfg.get("ensemble") or {}).get("workers") or []]
    got = set(workers)
    if got != CHEAP_WORKERS:
        fails.append(f"ensemble workers mismatch: got {sorted(got)} expected {sorted(CHEAP_WORKERS)}")
    fitness = cfg.get("fitness") or {}
    if fitness.get("mode") != "global_tail_mix":
        fails.append(f"fitness.mode={fitness.get('mode')!r}, want global_tail_mix")
    evo = cfg.get("evolution") or {}
    if evo.get("mutator_model") != "deepseek/deepseek-v4-pro":
        fails.append(f"mutator_model={evo.get('mutator_model')!r}")
    return fails


def _check_artifacts_text(text: str, label: str, *, require_error_triage: bool) -> List[str]:
    fails = []
    if not text or len(text.strip()) < 40:
        return [f"{label}: empty/too short artifacts ({len(text or '')} chars)"]
    markers = list(ARTIFACT_MARKERS)
    if require_error_triage:
        markers = ["ERROR TRIAGE"] + markers
    missing = [m for m in markers if m not in text]
    if missing:
        fails.append(f"{label}: missing markers {missing}")
    return fails


def _extract_artifact_blob(prog: Dict[str, Any]) -> Optional[str]:
    art = prog.get("artifacts_json")
    if isinstance(art, str) and art.strip():
        try:
            parsed = json.loads(art)
            if isinstance(parsed, dict):
                ee = parsed.get("error_examples")
                if isinstance(ee, str):
                    return ee
                for v in parsed.values():
                    if isinstance(v, str) and ("MUTATION REMINDERS" in v or "ERROR" in v):
                        return v
        except json.JSONDecodeError:
            if "MUTATION REMINDERS" in art:
                return art
    return None


def _extract_prompts(prog: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    out = []
    prompts = prog.get("prompts") or {}
    if not isinstance(prompts, dict):
        return out
    for key, pair in prompts.items():
        if not isinstance(pair, dict):
            continue
        out.append((str(key), str(pair.get("system") or ""), str(pair.get("user") or "")))
    return out


def verify_run(run_dir: Path) -> Dict[str, Any]:
    report: Dict[str, Any] = {"run_dir": str(run_dir), "pass": True, "checks": [], "fails": []}

    def ok(msg: str) -> None:
        report["checks"].append({"status": "pass", "msg": msg})

    def fail(msg: str) -> None:
        report["pass"] = False
        report["fails"].append(msg)
        report["checks"].append({"status": "fail", "msg": msg})

    cfg_path = run_dir / "config_resolved.yaml"
    if not cfg_path.is_file():
        cfg_path = run_dir / "config_used.yaml"
    if not cfg_path.is_file():
        fail("no config_resolved.yaml / config_used.yaml")
        return report
    cfg = _load_yaml(cfg_path)
    ok(f"loaded {cfg_path.name}")

    ens_fails = _check_ensemble(cfg)
    for f in ens_fails:
        fail(f)
    if not ens_fails:
        ok("cheap ensemble + global_tail_mix + deepseek mutator in config")

    # Cycle-level dump (may have 0 errors → no ERROR TRIAGE; still need reminders).
    cycle_arts = list(run_dir.glob("al_iter_*/error_artifacts.txt"))
    if not cycle_arts:
        fail("no al_iter_*/error_artifacts.txt")
    else:
        for p in cycle_arts:
            text = p.read_text(encoding="utf-8", errors="replace")
            misses = _check_artifacts_text(text, str(p.relative_to(run_dir)), require_error_triage=False)
            if misses:
                for m in misses:
                    fail(m)
            else:
                ok(
                    f"{p.relative_to(run_dir)} has mutator reminder sections "
                    f"({len(text)} chars; errors_present={'ERROR TRIAGE' in text})"
                )

    oe = _oe_staging(run_dir)
    if oe is None:
        fail("no OE staging path (al_iter_1/openevolve_output_path.txt)")
        return report
    ok(f"OE staging: {oe}")

    prog_files = _collect_program_files(oe)
    if not prog_files:
        fail(f"no programs/*.json under {oe}/checkpoints/")
        return report
    ok(f"found {len(prog_files)} program(s) in latest checkpoint")

    seeds = []
    children = []
    for pf in prog_files:
        try:
            prog = json.loads(pf.read_text(encoding="utf-8"))
        except OSError as exc:
            fail(f"unreadable {pf}: {exc}")
            continue
        if prog.get("parent_id"):
            children.append(prog)
        else:
            seeds.append(prog)

    # Critical: seed must carry artifacts so the first mutation sees them.
    if not seeds:
        fail("no seed program (parent_id=None) in latest checkpoint")
    else:
        seed = seeds[0]
        blob = _extract_artifact_blob(seed)
        if not blob:
            fail(
                "SEED program has no artifacts_json — mutator first iteration cannot "
                "see error_examples (OpenEvolve must store_artifacts after initial eval)"
            )
        else:
            for m in _check_artifacts_text(blob, "seed.artifacts_json", require_error_triage=True):
                fail(m)
            if not report["fails"] or all("seed.artifacts" not in x for x in report["fails"][-5:]):
                ok(f"seed artifacts_json OK ({len(blob)} chars)")

    saw_art_in_user = 0
    saw_system = 0
    sample = ""
    for prog in children:
        for _key, system, user in _extract_prompts(prog):
            if all(m in system for m in SYSTEM_MARKERS_CORE):
                saw_system += 1
            if any(
                m in user
                for m in (
                    "ERROR TRIAGE",
                    "MUTATION REMINDERS",
                    "error_examples",
                    "NO FULLY SYSTEMATIC",
                )
            ):
                saw_art_in_user += 1
                if not sample:
                    for needle in ("ERROR TRIAGE", "MUTATION REMINDERS", "error_examples"):
                        idx = user.find(needle)
                        if idx >= 0:
                            sample = user[max(0, idx - 20) : idx + 240]
                            break

    if saw_system == 0:
        fail("new coaching system_message not found in logged child prompts")
    else:
        ok(f"new coaching system_message in {saw_system} child prompt(s)")

    if saw_art_in_user == 0:
        fail(
            "mutator USER prompt does not contain error artifacts "
            "(include_artifacts / parent artifacts empty at sample time)"
        )
    else:
        ok(f"error artifacts present in {saw_art_in_user} logged mutator user prompt(s)")
        report["sample_user_artifact_snip"] = sample

    if (run_dir / "summary.json").is_file():
        ok("summary.json present")
    else:
        fail("summary.json missing")

    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--name", default="E2_cheap_mutator_live_smoke")
    args = ap.parse_args()
    run_dir = args.run_dir
    if run_dir is None:
        run_dir = _find_latest_run(PKG_ROOT / "results_smoke", args.name)
    if run_dir is None or not run_dir.is_dir():
        print(f"ERROR: run dir not found for {args.name}", file=sys.stderr)
        return 2

    report = verify_run(run_dir)
    out = run_dir / "mutator_artifacts_verify.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote {out}", flush=True)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

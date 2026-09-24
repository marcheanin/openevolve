"""Run every S11 analysis in one go and write the numbers the paper cites.

The point of this file is that a reader, or the author six months from now, can regenerate
every figure in PLAN_AND_FINDINGS Part IV with one command and no API key. Each analysis
writes its full stdout to results/S11_protocol_matrix/final/, and this driver prints only
whether it ran and how long it took.

`S11_DATASET=mnli` runs the same analyses on the MultiNLI matrix (settings and paths in
`dataset_config.py`); the jobs that need the old CivilComments E5 data are skipped there.

It refuses to label the output as final unless both fixed sets are fully scored, because
several of the numbers move with the candidate pool and a partial pool quietly understates
the spread. With --allow-partial it runs anyway and stamps every file as partial.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_config import CIVIL_ONLY_JOBS, cfg  # noqa: E402

OUT = cfg.outputs
FINAL = cfg.final_dir

# name -> argv after the interpreter, plus optional environment overrides for that job only
# (the second-scorer variants read a different prediction directory). Order matters only for
# readability of the log.
SCORER2 = {"S11_PREDS_DIR": "results/S11_protocol_matrix/scorer2_gpt4omini/preds"}
JOBS: list[tuple] = [
    ("power", ["scripts/analyze_s11.py", "--part", "power", "--n-boot", "4000"]),
    ("gate_a_recheck", ["scripts/gate_a_recheck.py"]),
    ("draw_lottery", ["scripts/draw_lottery_check.py"]),
    ("ci_calibration", ["scripts/ci_calibration_check.py"]),
    ("repeats_vs_rows", ["scripts/repeats_vs_rows.py"]),
    ("pool_spread", ["scripts/pool_spread_check.py"]),
    ("statistics_families", ["scripts/statistic_reproducibility.py"]),
    ("statistics_sweep", ["scripts/statistic_reproducibility.py", "--sweep"]),
    ("statistics_faithfulness", ["scripts/statistic_reproducibility.py", "--faithfulness"]),
    # --- added 2026-09-20: the mechanism, and the checks Part VII of PLAN_AND_FINDINGS cites
    ("threshold_hypothesis", ["scripts/threshold_hypothesis.py"]),
    ("threshold_who_is_above", ["scripts/threshold_who_is_above.py"]),
    ("threshold_frontier", ["scripts/threshold_frontier.py"]),
    ("threshold_hypothesis_gpt4omini", ["scripts/threshold_hypothesis.py"], SCORER2),
    ("threshold_who_is_above_gpt4omini", ["scripts/threshold_who_is_above.py"], SCORER2),
    ("holm_and_argmin_checks", ["scripts/holm_and_argmin_checks.py"]),
    ("family_claim_stability", ["scripts/family_claim_stability.py"]),
    ("method_balance_check", ["scripts/method_balance_check.py"]),
    ("bootstrap_scheme_equivalence", ["scripts/bootstrap_scheme_equivalence.py"]),
    ("draw_lottery_halves_civil", ["scripts/draw_lottery_halves.py", "--pool", "s9",
                                   "--match-draw-lottery-check"]),
    ("compare_scorers_truth", ["scripts/compare_scorers.py", "--tag", "gpt4omini",
                               "--set", "truth_large"]),
    ("compare_scorers_dev", ["scripts/compare_scorers.py", "--tag", "gpt4omini",
                             "--set", "dev_universe"]),
    ("figure_threshold", ["scripts/figure_threshold.py"]),
    ("figure_threshold_gpt4omini", ["scripts/figure_threshold.py", "--tag", "gpt4omini"], SCORER2),
    # slow: 100 half-splits each, about ten minutes apiece, skipped by --quick
    ("statistics_sweep_n100", ["scripts/statistic_reproducibility.py", "--sweep",
                               "--n-split", "100"]),
    ("statistics_families_n100", ["scripts/statistic_reproducibility.py", "--n-split", "100"]),
    ("temperature_sweep_finals", ["scripts/statistic_temperature_sweep.py", "--pool", "finals",
                                  "--n-split", "40"]),
    ("temperature_sweep_all", ["scripts/statistic_temperature_sweep.py", "--pool", "all"]),
]
SLOW = {"statistics_sweep_n100", "statistics_families_n100",
        "temperature_sweep_finals", "temperature_sweep_all"}
# The lottery is the centrepiece, so it is run across every target and every pool split:
# IV.9 turns on the target not changing the verdict, IV.6 on the pool not changing the ratio.
TARGETS = ["cvar25", "hard_min", "worst_class", "mean_gba"]
POOLS = cfg.pools


def scored(name: str) -> int:
    d = cfg.preds_dir / name
    return len([f for f in d.glob("*.npy")
                if not f.name.endswith(".partial.npy") and not f.stem.startswith("_")
                and not f.name.endswith(".lo.npy")])


def run(tag: str, argv: list[str], partial: bool, extra_env: dict | None = None) -> tuple[str, float, bool]:
    t0 = time.time()
    proc = subprocess.run([sys.executable, *argv], cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace",
                          # Russian lines crash cp1252 pipes; extra_env points one job at
                          # another prediction directory (the second scorer).
                          env={**os.environ, "PYTHONIOENCODING": "utf-8", **(extra_env or {})})
    dt = time.time() - t0
    head = "" if not partial else "*** PARTIAL MATRIX -- not for citation ***\n\n"
    body = f"$ python {' '.join(argv)}\n\n{proc.stdout}"
    if proc.returncode:
        body += f"\n--- stderr (exit {proc.returncode}) ---\n{proc.stderr[-4000:]}"
    (FINAL / f"{tag}.txt").write_text(head + body, encoding="utf-8")
    return tag, dt, proc.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allow-partial", action="store_true",
                    help="run even though the prediction matrix is incomplete")
    ap.add_argument("--only", help="substring filter over job names, for re-running one thing")
    ap.add_argument("--quick", action="store_true",
                    help="skip the four jobs that run 100 half-splits (about forty minutes)")
    args = ap.parse_args()

    have = {n: scored(n) for n in (cfg.test_set, cfg.dev_set)}
    expected = cfg.expected_count(have)
    partial = expected == 0 or any(v < expected for v in have.values())  # 0: nothing to count against
    print(f"scored: " + ", ".join(f"{k} {v}/{expected}" for k, v in have.items()))
    if partial and not args.allow_partial:
        print("matrix incomplete; pass --allow-partial to run anyway")
        return 1
    if partial:
        print("running on a PARTIAL matrix; every output file is stamped accordingly")

    FINAL.mkdir(parents=True, exist_ok=True)
    jobs = list(JOBS)
    for t in TARGETS:
        jobs.append((f"lottery_{t}", ["scripts/analyze_s11.py", "--part", "lottery",
                                      "--draws", "200", "--target", t]))
    # The 48 "plausible" protocols (hard_min, cvar_k2, mean_gba, softmin_w40) carry the headline number:
    # the other 24 use global accuracy or worst-class as the selection statistic, which no group-robust
    # method would choose. They get their own summary files.
    for t in TARGETS:
        jobs.append((f"lottery_{t}_plausible", ["scripts/analyze_s11.py", "--part", "lottery",
                                                "--draws", "200", "--target", t,
                                                "--protocols", "plausible"]))
    for pool in POOLS[1:]:
        jobs.append((f"lottery_cvar25_pool_{pool}",
                     ["scripts/analyze_s11.py", "--part", "lottery", "--draws", "200",
                      "--target", "cvar25", "--pool", pool]))
    if cfg.key != "civil":
        skipped = [j[0] for j in jobs if j[0] in CIVIL_ONLY_JOBS]
        jobs = [j for j in jobs if j[0] not in CIVIL_ONLY_JOBS]
        print(f"S11_DATASET={cfg.key}: skipping jobs that need the CivilComments E5 data: "
              + ", ".join(skipped))
    if args.only:
        jobs = [j for j in jobs if args.only in j[0]]
        if not jobs:
            print(f"no job matches {args.only!r}")
            return 1

    if args.quick:
        jobs = [j for j in jobs if j[0] not in SLOW]
    print(f"\n{len(jobs)} analyses -> {FINAL}\n")
    results, t0 = [], time.time()
    for job in jobs:
        tag, argv = job[0], job[1]
        name, dt, ok = run(tag, argv, partial, job[2] if len(job) > 2 else None)
        results.append((name, dt, ok))
        print(f"  {'ok ' if ok else 'FAIL'} {name:30s} {dt / 60:5.1f} min")

    failed = [n for n, _, ok in results if not ok]
    (FINAL / "_index.json").write_text(json.dumps(
        {"partial": partial, "scored": have, "minutes": round((time.time() - t0) / 60, 1),
         "jobs": [{"name": n, "minutes": round(d / 60, 2), "ok": ok} for n, d, ok in results]},
        indent=2), encoding="utf-8")
    print(f"\ntotal {(time.time() - t0) / 60:.1f} min; "
          + (f"FAILED: {', '.join(failed)}" if failed else "all analyses completed"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

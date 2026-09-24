#!/usr/bin/env python
"""Same-session stable eval for S9 matrix finals (M29/M38) — shared seed.

Scores the seed prompt **once per matrix seed**, then each method final × repeats.
Reuses pred caches / complete stable_report.json when present.

Layout under results/E5_s9_matrix/stable_session/:
  _shared_seed{42,43,44}/   seed_repeat*_preds.npy (+ metrics)
  seed{S}_{method}/         final repeats + paired report vs shared seed
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _seed_complete(shared: Path, repeats: int) -> bool:
    return all((shared / f"seed_repeat{r}_preds.npy").is_file() for r in range(repeats))


def _job_complete(job: Path) -> bool:
    return (job / "stable_report.json").is_file()


def _copy_seed_into_job(shared: Path, job: Path, repeats: int) -> None:
    job.mkdir(parents=True, exist_ok=True)
    for r in range(repeats):
        for stem in (f"seed_repeat{r}_preds.npy", f"seed_repeat{r}_metrics.json"):
            src = shared / stem
            dst = job / stem
            if src.is_file() and not dst.is_file():
                shutil.copy2(src, dst)
    sp = shared / "seed_prompt.txt"
    if sp.is_file() and not (job / "seed_prompt.txt").is_file():
        shutil.copy2(sp, job / "seed_prompt.txt")


def _promote_seed_from_job(job: Path, shared: Path, repeats: int) -> bool:
    """If a completed/partial job already has full seed repeats, copy into shared."""
    if not all((job / f"seed_repeat{r}_preds.npy").is_file() for r in range(repeats)):
        return False
    shared.mkdir(parents=True, exist_ok=True)
    for r in range(repeats):
        for stem in (f"seed_repeat{r}_preds.npy", f"seed_repeat{r}_metrics.json"):
            src = job / stem
            dst = shared / stem
            if src.is_file() and not dst.is_file():
                shutil.copy2(src, dst)
    sp = job / "seed_prompt.txt"
    if sp.is_file() and not (shared / "seed_prompt.txt").is_file():
        shutil.copy2(sp, shared / "seed_prompt.txt")
    return _seed_complete(shared, repeats)


def _run(cmd: list[str]) -> int:
    print("RUN", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, default=ROOT / "results/E5_s9_matrix")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument(
        "--methods",
        default="",
        help="Optional comma list; default = all dirs with best_prompt.txt",
    )
    args = parser.parse_args()

    seed_prompt = ROOT / "prompts/initial_prompt_civilcomments.txt"
    out_root = args.matrix_root / "stable_session"
    out_root.mkdir(parents=True, exist_ok=True)
    method_filter = {m.strip() for m in args.methods.split(",") if m.strip()} or None

    index: dict = {"repeats": int(args.repeats), "mode": "shared_seed", "jobs": []}
    py = sys.executable
    eval_py = str(ROOT / "scripts/e5_stable_eval.py")

    for seed_s in [s.strip() for s in args.seeds.split(",") if s.strip()]:
        sdir = args.matrix_root / f"seed{seed_s}"
        if not sdir.is_dir():
            print(f"skip missing {sdir}", flush=True)
            continue

        shared = out_root / f"_shared_seed{seed_s}"
        # Reuse seed preds already paid for (e.g. seed42_ape).
        if not _seed_complete(shared, args.repeats):
            for cand in sorted(out_root.glob(f"seed{seed_s}_*")):
                if cand.name.startswith("_"):
                    continue
                if _promote_seed_from_job(cand, shared, args.repeats):
                    print(f"[shared] promoted seed preds from {cand.name}", flush=True)
                    break

        if not _seed_complete(shared, args.repeats):
            shared.mkdir(parents=True, exist_ok=True)
            cmd = [
                py,
                eval_py,
                "--seed-prompt",
                str(seed_prompt),
                "--seed-only",
                "--out-dir",
                str(shared),
                "--repeats",
                str(int(args.repeats)),
            ]
            if args.mock:
                cmd.append("--mock")
            rc = _run(cmd)
            if rc != 0:
                return rc
            (shared / "NOTE.txt").write_text(
                "Shared same-session seed baseline (scored once per matrix seed).\n",
                encoding="utf-8",
            )
            print(f"[shared] wrote {shared}", flush=True)
        else:
            print(f"[shared] reuse {shared}", flush=True)

        methods = sorted(
            [
                d
                for d in sdir.iterdir()
                if d.is_dir()
                and d.name != "seed"  # baseline covered by _shared_seed*
                and (d / "best_prompt.txt").is_file()
                and (method_filter is None or d.name in method_filter)
            ],
            key=lambda p: p.name,
        )

        for mdir in methods:
            job_out = out_root / f"seed{seed_s}_{mdir.name}"
            if _job_complete(job_out):
                print(f"skip complete {job_out.name}", flush=True)
                index["jobs"].append(
                    {
                        "seed": seed_s,
                        "method": mdir.name,
                        "out": str(job_out),
                        "rc": 0,
                        "skipped": True,
                    }
                )
                continue

            _copy_seed_into_job(shared, job_out, args.repeats)
            cmd = [
                py,
                eval_py,
                "--seed-prompt",
                str(seed_prompt),
                "--final-prompt",
                str(mdir / "best_prompt.txt"),
                "--out-dir",
                str(job_out),
                "--repeats",
                str(int(args.repeats)),
            ]
            if args.mock:
                cmd.append("--mock")
            rc = _run(cmd)
            index["jobs"].append(
                {
                    "seed": seed_s,
                    "method": mdir.name,
                    "out": str(job_out),
                    "rc": rc,
                    "skipped": False,
                }
            )
            (out_root / "index.json").write_text(
                json.dumps(index, indent=2), encoding="utf-8"
            )
            if rc != 0:
                return rc

    (out_root / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"[stable-batch] done → {out_root / 'index.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

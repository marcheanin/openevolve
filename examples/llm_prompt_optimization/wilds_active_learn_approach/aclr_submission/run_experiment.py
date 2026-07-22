from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> int:
    print(" ".join(cmd))
    return subprocess.call(cmd, cwd=str(ROOT))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run all-categories PRIME experiment (evolve + optional full test)."
    )
    p.add_argument("--n-al", type=int, default=8)
    p.add_argument("--n-evolve", type=int, default=15)
    p.add_argument(
        "--results-dir",
        type=str,
        default="results_all_categories_prime_submission",
    )
    p.add_argument("--run-full-test", action="store_true")
    p.add_argument("--skip-evolve", action="store_true")
    args = p.parse_args()

    if not args.skip_evolve:
        cmd = [
            sys.executable,
            "active_loop.py",
            "--config",
            "config_all_categories.yaml",
            "--prompt",
            "initial_prompt_all_categories.txt",
            "--results-dir",
            args.results_dir,
            "--n-al",
            str(args.n_al),
            "--n-evolve",
            str(args.n_evolve),
        ]
        if args.run_full_test:
            cmd.append("--run-full-test")
        code = run(cmd)
        if code != 0:
            raise SystemExit(code)

    if args.run_full_test and args.skip_evolve:
        prompt = ROOT / args.results_dir / "final_prompt.txt"
        if not prompt.exists():
            raise SystemExit(f"Missing prompt: {prompt}")
        code = run(
            [
                sys.executable,
                "run_full_test.py",
                "--prompt",
                str(prompt),
                "--config",
                "config_all_categories.yaml",
                "--results-dir",
                str(ROOT / args.results_dir / "fulltest_final"),
            ]
        )
        if code != 0:
            raise SystemExit(code)


if __name__ == "__main__":
    main()

"""
Orchestrate the full Active Prompt Evolution pipeline.

Usage:
    # Step 1: Generate APE meta-prompt (then paste into ChatGPT, save best to initial_prompt.txt)
    python run_pipeline.py --step ape

    # Step 2: Run baseline evaluation
    python run_pipeline.py --step baseline

    # Step 3: Run active learning loop
    python run_pipeline.py --step evolve

    # Step 4: Generate final report
    python run_pipeline.py --step report

    # Or run all steps (after manual APE)
    python run_pipeline.py --all
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def run_ape() -> int:
    """Generate APE meta-prompt for ChatGPT."""
    print("=" * 60)
    print("Step 1: APE Initial Prompt Generation")
    print("=" * 60)
    return subprocess.call(
        [sys.executable, str(SCRIPT_DIR / "prepare_initial_prompt.py")],
        cwd=str(SCRIPT_DIR),
    )


def run_baseline() -> int:
    """Evaluate baseline on full train+val."""
    print("=" * 60)
    print("Step 2: Baseline Evaluation")
    print("=" * 60)
    return subprocess.call(
        [sys.executable, str(SCRIPT_DIR / "run_baseline.py")],
        cwd=str(SCRIPT_DIR),
    )


def run_evolve(n_al: int = 4, n_evolve: int = 15) -> int:
    """Run active learning loop."""
    print("=" * 60)
    print("Step 3: Active Learning Loop")
    print("=" * 60)
    return subprocess.call(
        [
            sys.executable,
            str(SCRIPT_DIR / "active_loop.py"),
            "--n-al", str(n_al),
            "--n-evolve", str(n_evolve),
        ],
        cwd=str(SCRIPT_DIR),
    )


def run_report() -> int:
    """Generate final report and visualizations."""
    print("=" * 60)
    print("Step 4: Final Report")
    print("=" * 60)
    code = subprocess.call(
        [sys.executable, str(SCRIPT_DIR / "visualize.py")],
        cwd=str(SCRIPT_DIR),
    )
    # Also run generate_final_report if it exists
    report_script = SCRIPT_DIR / "generate_final_report.py"
    if report_script.exists():
        subprocess.call([sys.executable, str(report_script)], cwd=str(SCRIPT_DIR))
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["ape", "baseline", "evolve", "report"])
    parser.add_argument("--all", action="store_true", help="Run all steps (after manual APE)")
    parser.add_argument("--n-al", type=int, default=4)
    parser.add_argument("--n-evolve", type=int, default=15)
    args = parser.parse_args()

    if args.all:
        for step_name, step_fn in [
            ("baseline", run_baseline),
            ("evolve", lambda: run_evolve(args.n_al, args.n_evolve)),
            ("report", run_report),
        ]:
            if step_fn() != 0:
                print(f"Step {step_name} failed.")
                sys.exit(1)
        print("Pipeline complete.")
        return

    if args.step == "ape":
        sys.exit(run_ape())
    if args.step == "baseline":
        sys.exit(run_baseline())
    if args.step == "evolve":
        sys.exit(run_evolve(args.n_al, args.n_evolve))
    if args.step == "report":
        sys.exit(run_report())
    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _resolve_openevolve_script(exp_dir: Path) -> Path:
    # Prefer the openevolve package's launcher script
    openevolve_root = exp_dir.parents[5] / "openevolve"
    script_path = openevolve_root / "openevolve-run.py"
    if not script_path.exists():
        raise FileNotFoundError(f"openevolve-run.py not found at {script_path}")
    return script_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenEvolve for exp2_single_evolved.")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    parser.add_argument(
        "--train-users",
        type=int,
        default=15,
        help="Fixed number of users to keep in train split (0 = no cap)",
    )
    args = parser.parse_args()

    exp_dir = Path(__file__).resolve().parent
    prompt_file = exp_dir / "initial_prompt.txt"
    evaluator_file = exp_dir / "evaluator.py"
    config_file = exp_dir / "config.yaml"
    openevolve_script = _resolve_openevolve_script(exp_dir)

    cmd = [
        sys.executable,
        str(openevolve_script),
        str(prompt_file),
        str(evaluator_file),
        "--config",
        str(config_file),
        "--iterations",
        str(args.iterations),
    ]

    env = os.environ.copy()
    # Ensure openevolve package is importable when running the launcher script.
    env["PYTHONPATH"] = str(openevolve_script.parent) + os.pathsep + env.get("PYTHONPATH", "")
    env["OPENEVOLVE_MAX_TRAIN_USERS"] = str(args.train_users)
    subprocess.run(cmd, check=True, cwd=str(exp_dir), env=env)


if __name__ == "__main__":
    main()


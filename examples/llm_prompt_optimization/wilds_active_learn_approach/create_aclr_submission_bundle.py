from __future__ import annotations

from pathlib import Path
import re
import shutil
import textwrap


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "aclr_submission"
WILDS_EXPERIMENT = ROOT.parent / "wilds_experiment"


LOCAL_FILES = [
    "active_loop.py",
    "evaluator.py",
    "data_manager.py",
    "error_analyzer.py",
    "synthetic_fewshot_generator.py",
    "workers.py",
    "sample_stratified.py",
    "run_full_test.py",
    "run_baseline.py",
    "logger.py",
    "config_all_categories.yaml",
    "dataset_all_categories.yaml",
    "initial_prompt_all_categories.txt",
]

EXTERNAL_FILES = [
    ("evaluator.py", "wilds_experiment/evaluator.py"),
    ("experiments/metrics.py", "wilds_experiment/experiments/metrics.py"),
]


def _strip_russian_comments(py_text: str) -> str:
    cleaned: list[str] = []
    for line in py_text.splitlines():
        if re.search(r"[А-Яа-яЁё]", line):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            if "#" in line:
                code, _, comment = line.partition("#")
                if re.search(r"[А-Яа-яЁё]", comment):
                    line = code.rstrip()
        cleaned.append(line)
    out = "\n".join(cleaned) + "\n"
    out = out.replace(
        '"""\nLLMWorker with OpenRouter support for Active Prompt Evolution.\nУчитывает токены через token_usage.get_tracker().\n"""',
        '"""\nLLMWorker with OpenRouter support for Active Prompt Evolution.\nTracks token usage via token_usage.get_tracker().\n"""',
    )
    return out


def _rewrite_paths(py_text: str) -> str:
    return py_text.replace(
        "WILDS_EXPERIMENT = SCRIPT_DIR.parent / \"wilds_experiment\"",
        "WILDS_EXPERIMENT = SCRIPT_DIR / \"wilds_experiment\"",
    )


def _write_runner() -> None:
    runner = textwrap.dedent(
        """
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
        """
    ).lstrip()
    (OUT / "run_experiment.py").write_text(runner, encoding="utf-8")


def _write_token_usage() -> None:
    token_usage = textwrap.dedent(
        """
        from __future__ import annotations
        import threading
        from pathlib import Path
        from typing import Any, Dict


        class TokenTracker:
            def __init__(self) -> None:
                self._by_model: Dict[str, Dict[str, int]] = {}
                self._lock = threading.Lock()

            def record(
                self,
                model_name: str,
                input_tokens: int = 0,
                output_tokens: int = 0,
                total_tokens: int | None = None,
            ) -> None:
                with self._lock:
                    if model_name not in self._by_model:
                        self._by_model[model_name] = {
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "total_tokens": 0,
                        }
                    self._by_model[model_name]["input_tokens"] += input_tokens
                    self._by_model[model_name]["output_tokens"] += output_tokens
                    if total_tokens is not None:
                        self._by_model[model_name]["total_tokens"] += total_tokens
                    else:
                        self._by_model[model_name]["total_tokens"] += input_tokens + output_tokens

            def get_usage(self) -> Dict[str, Any]:
                total_in = sum(m["input_tokens"] for m in self._by_model.values())
                total_out = sum(m["output_tokens"] for m in self._by_model.values())
                total = sum(m["total_tokens"] for m in self._by_model.values())
                return {
                    "by_model": dict(self._by_model),
                    "total_input_tokens": total_in,
                    "total_output_tokens": total_out,
                    "total_tokens": total,
                }

            def reset(self) -> None:
                self._by_model.clear()

            def save_json(self, path: Path | str) -> None:
                path = Path(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                import json
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(self.get_usage(), f, indent=2, ensure_ascii=False)
                print(f"Token usage saved: {path}")

            def write_report(self, path: Path | str, title: str = "Token usage report") -> None:
                path = Path(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                u = self.get_usage()
                lines = [
                    f"# {title}",
                    "",
                    "## By model",
                    "",
                    "| Model | Input | Output | Total |",
                    "|-------|-------|--------|-------|",
                ]
                for model in sorted(u["by_model"].keys()):
                    m = u["by_model"][model]
                    lines.append(
                        f"| {model} | {m['input_tokens']:,} | {m['output_tokens']:,} | {m['total_tokens']:,} |"
                    )
                lines.extend(
                    [
                        "| **Total** | **{:,}** | **{:,}** | **{:,}** |".format(
                            u["total_input_tokens"],
                            u["total_output_tokens"],
                            u["total_tokens"],
                        ),
                        "",
                        "## Summary",
                        "",
                        f"- Total tokens: {u['total_tokens']:,}",
                        f"- Input tokens: {u['total_input_tokens']:,}",
                        f"- Output tokens: {u['total_output_tokens']:,}",
                        "",
                    ]
                )
                path.write_text("\\n".join(lines), encoding="utf-8")
                print(f"Token usage report: {path}")


        _tracker: TokenTracker | None = None


        def get_tracker() -> TokenTracker:
            global _tracker
            if _tracker is None:
                _tracker = TokenTracker()
            return _tracker
        """
    ).lstrip()
    (OUT / "token_usage.py").write_text(token_usage, encoding="utf-8")


def _write_readme() -> None:
    readme = textwrap.dedent(
        """
        # ACLR Submission Bundle (PRIME / All Categories)

        This folder contains a clean runnable bundle for the all-categories PRIME experiment,
        using OpenEvolve as an importable Python library (`openevolve`).

        ## Included
        - Training/evolution loop (`active_loop.py`)
        - Evaluation and workers (`evaluator.py`, `workers.py`)
        - Active-learning data manager (`data_manager.py`)
        - Synthetic few-shot generator (`synthetic_fewshot_generator.py`)
        - Error analyzer (`error_analyzer.py`)
        - Full uncapped test runner (`run_full_test.py`)
        - All-categories config and dataset config
        - Example initial prompt (`initial_prompt_all_categories.txt`)
        - One-command runner (`run_experiment.py`)

        ## Install
        ```bash
        pip install -r requirements.txt
        pip install openevolve
        ```

        ## Run all-categories experiment
        ```bash
        python run_experiment.py --n-al 8 --n-evolve 15 --results-dir results_all_categories_prime_submission --run-full-test
        ```

        ## Run full test only (after evolution)
        ```bash
        python run_experiment.py --skip-evolve --run-full-test --results-dir results_all_categories_prime_submission
        ```
        """
    ).lstrip()
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    (OUT / "requirements.txt").write_text(
        "openai\nnumpy\npyyaml\nscikit-learn\nsentence-transformers\ntiktoken\n",
        encoding="utf-8",
    )


def build() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    for rel in LOCAL_FILES:
        src = ROOT / rel
        dst = OUT / rel
        text = src.read_text(encoding="utf-8")
        if src.suffix == ".py":
            text = _rewrite_paths(_strip_russian_comments(text))
        dst.write_text(text, encoding="utf-8")

    for src_rel, dst_rel in EXTERNAL_FILES:
        src = WILDS_EXPERIMENT / src_rel
        dst = OUT / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        text = src.read_text(encoding="utf-8")
        dst.write_text(text, encoding="utf-8")

    cfg = (OUT / "config_all_categories.yaml").read_text(encoding="utf-8")
    cfg = cfg.replace("    enabled: false", "    enabled: true")
    cfg = cfg.replace(
        'output_dir: "results_all_categories"',
        'output_dir: "results_all_categories_prime_submission"',
    )
    (OUT / "config_all_categories.yaml").write_text(cfg, encoding="utf-8")

    _write_runner()
    _write_token_usage()
    _write_readme()

    print(OUT)


if __name__ == "__main__":
    build()

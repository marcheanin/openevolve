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
        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Token usage report: {path}")


_tracker: TokenTracker | None = None


def get_tracker() -> TokenTracker:
    global _tracker
    if _tracker is None:
        _tracker = TokenTracker()
    return _tracker

"""
Учёт потраченных токенов по моделям для оценки стоимости API.

Используется в LLMWorker при каждом вызове; результаты сохраняются
в results/token_usage.json (active_loop) и results/baseline/token_usage.json (run_baseline).
"""

from pathlib import Path
from typing import Dict, Any


class TokenTracker:
    """Накопительный учёт токенов по имени модели."""

    def __init__(self) -> None:
        # model_name -> { "input_tokens", "output_tokens", "total_tokens" }
        self._by_model: Dict[str, Dict[str, int]] = {}

    def record(
        self,
        model_name: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int | None = None,
    ) -> None:
        """Учесть использование токенов для модели."""
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
        """Возвращает сводку: по моделям и общие суммы."""
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
        """Сбросить счётчики (для нового прогона в том же процессе)."""
        self._by_model.clear()

    def save_json(self, path: Path | str) -> None:
        """Сохранить сводку в JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.get_usage(), f, indent=2, ensure_ascii=False)
        print(f"Token usage saved: {path}")

    def write_report(self, path: Path | str, title: str = "Отчёт об использовании токенов") -> None:
        """Сформировать текстовый отчёт (Markdown) по использованным токенам."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        u = self.get_usage()
        lines = [
            f"# {title}",
            "",
            "## По моделям (ансамбль оценки)",
            "",
            "| Модель | Вход (input) | Выход (output) | Всего (total) |",
            "|--------|--------------|----------------|---------------|",
        ]
        for model in sorted(u["by_model"].keys()):
            m = u["by_model"][model]
            lines.append(
                f"| {model} | {m['input_tokens']:,} | {m['output_tokens']:,} | {m['total_tokens']:,} |"
            )
        lines.extend([
            "| **Итого** | **{:,}** | **{:,}** | **{:,}** |".format(
                u["total_input_tokens"],
                u["total_output_tokens"],
                u["total_tokens"],
            ),
            "",
            "## Сводка",
            "",
            f"- **Всего токенов:** {u['total_tokens']:,}",
            f"- **Вход (промпты):** {u['total_input_tokens']:,}",
            f"- **Выход (ответы):** {u['total_output_tokens']:,}",
            "",
            "*Учтены только вызовы worker-моделей ансамбля (оценка промптов). Токены модели эволюции (DeepSeek R1) в OpenEvolve не включены.*",
            "",
        ])
        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Token usage report: {path}")


_tracker: TokenTracker | None = None


def get_tracker() -> TokenTracker:
    """Глобальный синглтон трекера (один на процесс)."""
    global _tracker
    if _tracker is None:
        _tracker = TokenTracker()
    return _tracker

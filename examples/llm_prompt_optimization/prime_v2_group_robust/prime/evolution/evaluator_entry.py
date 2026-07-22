"""OpenEvolve evaluator entrypoint: prime.evolution.evaluator_entry:evaluate"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

from prime.evolution.openevolve_adapter import evaluate_for_openevolve


def evaluate(prompt_path: Optional[str] = None) -> Union[Dict[str, Any], Any]:
    return evaluate_for_openevolve(prompt_path)

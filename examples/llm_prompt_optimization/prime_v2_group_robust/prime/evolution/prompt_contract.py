"""Prompt contract validator (ROADMAP_PHASE3 F7).

Rejects candidates that cannot be formatted or that omit binary output instructions,
before spending scorer API calls (was reward-hackable via silent parse→0).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ContractResult:
    ok: bool
    reason: Optional[str] = None


def check_prompt_contract(
    prompt: str,
    *,
    label_space: str = "binary",
    prompt_len_limit: int = 8000,
    require_review_placeholder: bool = True,
) -> ContractResult:
    text = prompt or ""
    if not text.strip():
        return ContractResult(False, "empty")
    if len(text) > prompt_len_limit:
        return ContractResult(False, "too_long")

    if require_review_placeholder:
        if "{review}" not in text and "{text}" not in text:
            return ContractResult(False, "missing_placeholder")

    # Must format with a sample review without raising (catches unbalanced braces).
    try:
        if "{review}" in text:
            text.format(review="sample comment")
        elif "{text}" in text:
            text.format(text="sample comment")
    except (KeyError, ValueError, IndexError) as exc:
        return ContractResult(False, f"format_error:{type(exc).__name__}")

    if label_space == "binary":
        # Must mention both 0 and 1 as labels somewhere (format instruction).
        has_0 = bool(re.search(r"\b0\b", text))
        has_1 = bool(re.search(r"\b1\b", text))
        if not (has_0 and has_1):
            return ContractResult(False, "missing_binary_labels")
        # Soft check: mention toxic / label / output.
        if not re.search(r"(label|toxic|output|class)", text, re.IGNORECASE):
            return ContractResult(False, "missing_task_language")
    elif label_space == "ordinal5":
        has_stars = all(re.search(rf"\b{n}\b", text) for n in (1, 5))
        if not has_stars:
            return ContractResult(False, "missing_ordinal_labels")
        if not re.search(r"(rating|star|output|1-5|1–5)", text, re.IGNORECASE):
            return ContractResult(False, "missing_task_language")

    return ContractResult(True, None)

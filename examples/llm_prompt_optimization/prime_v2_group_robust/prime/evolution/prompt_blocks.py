"""Soft XML-ish block extract/replace for structured prompts (no full parser)."""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple


_EVOLVE_MARKER_RE = re.compile(r"^[ \t]*#[ \t]*EVOLVE-BLOCK-(?:START|END)[ \t]*\r?\n?", re.MULTILINE)


def strip_evolve_markers(prompt: str) -> str:
    """
    Remove OpenEvolve's ``# EVOLVE-BLOCK-START/END`` sentinels.

    OpenEvolve wraps a marker-less initial program in those comments and returns
    them as part of ``best_code``. Without this, every post-cycle-1 prompt ships
    the sentinels to the rating workers and into the final selected prompt.
    """
    return _EVOLVE_MARKER_RE.sub("", prompt).strip("\n")


def list_blocks(prompt: str) -> List[str]:
    """Return tag names of top-level open/close pairs found in prompt."""
    return re.findall(r"<([A-Za-z][A-Za-z0-9_]*)>", prompt)


def extract_block(prompt: str, tag: str) -> Optional[str]:
    """
    Extract inner content of first <tag>...</tag> (non-greedy, DOTALL).
    Returns None if the pair is missing.
    """
    pattern = rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>"
    m = re.search(pattern, prompt, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    return m.group(1)


def extract_block_span(prompt: str, tag: str) -> Optional[Tuple[int, int, str]]:
    """Return (start, end, inner) of first matching block, or None."""
    pattern = rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>"
    m = re.search(pattern, prompt, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    return m.start(), m.end(), m.group(1)


def replace_block(prompt: str, tag: str, new_inner: str) -> str:
    """
    Replace inner content of first <tag>...</tag>.
    If the block is missing, append a new block at the end of the prompt.
    Preserves surrounding whitespace style of the original inner when possible.
    """
    span = extract_block_span(prompt, tag)
    if span is None:
        block = f"\n<{tag}>\n{new_inner.rstrip()}\n</{tag}>\n"
        return prompt.rstrip() + block

    start, end, old_inner = span
    # Preserve leading newline indentation of old inner if present
    leading = ""
    trailing = ""
    if old_inner.startswith("\n"):
        leading = "\n"
    if old_inner.endswith("\n"):
        trailing = "\n"
    body = new_inner
    if leading and not body.startswith("\n"):
        body = leading + body.lstrip("\n")
    if trailing and not body.endswith("\n"):
        body = body.rstrip("\n") + trailing
    return prompt[:start] + f"<{tag}>{body}</{tag}>" + prompt[end:]


def has_block(prompt: str, tag: str) -> bool:
    return extract_block(prompt, tag) is not None


def inject_verbatim_fewshot(
    prompt: str,
    examples: Sequence[tuple],
    *,
    label_space: str = "ordinal5",
    max_len: int = 320,
) -> str:
    """
    Mechanically replace <FewShotExamples> with verbatim (text, label) pairs (O22).

    ``examples`` is a sequence of ``(text, label)``. If empty, the prompt is unchanged.
    """
    if not examples:
        return prompt
    from prime.evolution.artifacts import build_fewshot_inner

    pairs = [(str(t), int(y)) for t, y in examples]
    inner = build_fewshot_inner(
        [t for t, _ in pairs],
        [y for _, y in pairs],
        label_space=label_space,
        max_len=max_len,
    )
    return replace_block(prompt, "FewShotExamples", inner)

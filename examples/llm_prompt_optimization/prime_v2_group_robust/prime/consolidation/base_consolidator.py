"""Consolidate group-invariant rules into <BaseGuidelines> only (soft scoped)."""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

from prime.consolidation.pool_carryover import PromptRecord, gate_consolidation
from prime.evolution.prompt_blocks import extract_block, replace_block


def _collect_dynamic_rules(prompts: Sequence[str]) -> List[str]:
    rules: List[str] = []
    for p in prompts:
        body = extract_block(p, "DynamicRules")
        if body and body.strip():
            rules.append(body.strip())
    return rules


def consolidate_base_guidelines_mock(
    current_prompt: str,
    carryover: Sequence[PromptRecord],
) -> str:
    """
    Deterministic stub: append a short summary of carryover DynamicRules
    into BaseGuidelines without calling an LLM.
    """
    sources = [current_prompt] + [r.prompt for r in carryover]
    dyn = _collect_dynamic_rules(sources)
    base = extract_block(current_prompt, "BaseGuidelines") or ""
    snippet_lines = []
    for i, body in enumerate(dyn[:3]):
        # First non-empty line as a compact rule hint
        first = next((ln.strip() for ln in body.splitlines() if ln.strip()), "")
        if first:
            snippet_lines.append(f"        - [consolidated-{i+1}] {first[:120]}")
    if not snippet_lines:
        return current_prompt
    marker = "<!-- GRAPE_BASE_CONSOLIDATED -->"
    if marker in base:
        return current_prompt
    new_base = base.rstrip() + "\n" + "\n".join(snippet_lines) + f"\n        {marker}\n"
    return replace_block(current_prompt, "BaseGuidelines", new_base)


def consolidate_base_guidelines_llm(
    current_prompt: str,
    carryover: Sequence[PromptRecord],
    *,
    model: str,
    api_base: str,
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> str:
    """
    Ask mutator LLM to rewrite only BaseGuidelines from accumulated DynamicRules.
    Falls back to mock consolidator on any failure.
    """
    try:
        from openai import OpenAI
        import os

        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            return consolidate_base_guidelines_mock(current_prompt, carryover)

        sources = [current_prompt] + [r.prompt for r in carryover]
        dyn = _collect_dynamic_rules(sources)
        base = extract_block(current_prompt, "BaseGuidelines") or ""
        dyn_blob = "\n\n---\n\n".join(dyn[:5]) if dyn else "(none)"

        system = (
            "You consolidate group-invariant guidelines for an Amazon review "
            "rating prompt. Rewrite ONLY the BaseGuidelines content: keep it "
            "short, group-invariant, no cluster-specific rules. Output the new "
            "BaseGuidelines INNER text only (no <BaseGuidelines> tags)."
        )
        user = (
            f"Current BaseGuidelines:\n{base}\n\n"
            f"Accumulated DynamicRules from specialist prompts:\n{dyn_blob}\n\n"
            "Produce consolidated BaseGuidelines inner text."
        )
        client = OpenAI(api_key=api_key, base_url=api_base)
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        # Strip accidental fences / tags
        text = re.sub(r"^```(?:text|xml)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = re.sub(r"</?BaseGuidelines>", "", text, flags=re.IGNORECASE).strip()
        if not text:
            return consolidate_base_guidelines_mock(current_prompt, carryover)
        if not text.startswith("\n"):
            text = "\n        " + text + "\n    "
        return replace_block(current_prompt, "BaseGuidelines", text)
    except Exception:
        return consolidate_base_guidelines_mock(current_prompt, carryover)


def run_base_consolidation(
    current_prompt: str,
    carryover: Sequence[PromptRecord],
    best: PromptRecord,
    *,
    use_mock: bool,
    gate_delta: float,
    consolidated_fitness: Optional[float] = None,
    model: str = "google/gemini-2.5-pro",
    api_base: str = "https://openrouter.ai/api/v1",
) -> tuple[str, PromptRecord, bool]:
    """
    Produce BaseGuidelines-consolidated prompt and apply gate vs best fitness.
    Returns (prompt_to_use, consolidated_record, gate_passed).
    If gate fails, returns current_prompt unchanged and gate_passed=False.
    """
    if use_mock:
        new_prompt = consolidate_base_guidelines_mock(current_prompt, carryover)
    else:
        new_prompt = consolidate_base_guidelines_llm(
            current_prompt, carryover, model=model, api_base=api_base
        )

    fitness = float(consolidated_fitness if consolidated_fitness is not None else best.fitness)
    consolidated = PromptRecord(
        prompt=new_prompt,
        fitness=fitness,
        metrics={"source": "base_guidelines_consolidation"},
        cluster_scores=best.cluster_scores,
    )
    passed = gate_consolidation(consolidated, best, gate_delta)
    if not passed:
        return current_prompt, consolidated, False
    return new_prompt, consolidated, True

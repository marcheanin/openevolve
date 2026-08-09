"""
Consolidate group-invariant rules into <BaseGuidelines>.

OBSERVATIONS O9: the previous version promoted rules from <DynamicRules> into
<BaseGuidelines> but left them in place, so the prompt stated the rating ladder
twice and carried live contradictions (base `"hard to tell" -> 3` against dynamic
`"hard to find" + "works great" = 5`). Measured cost on D_select was -0.015 and
-0.066 across two cycles. The consolidator now rewrites **both** blocks in one
call and must delete from <DynamicRules> whatever it lifts into the base.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

from prime.consolidation.pool_carryover import PromptRecord
from prime.evolution.prompt_blocks import extract_block, replace_block

_BASE_HEADER = "=== BASE_GUIDELINES ==="
_DYNAMIC_HEADER = "=== DYNAMIC_RULES ==="

SYSTEM_MESSAGE = (
    "You consolidate a two-timescale Amazon review rating prompt. "
    "<BaseGuidelines> is the slow, group-invariant core; <DynamicRules> is the "
    "fast, group-specific layer.\n\n"
    "Rewrite BOTH blocks under these hard constraints:\n"
    "1. Promote into BaseGuidelines only rules that hold for every reviewer type.\n"
    "2. Whatever you promote, DELETE from DynamicRules. The two blocks must not "
    "restate or contradict each other — no duplicated rating ladder, no rule that "
    "appears in both.\n"
    "3. Keep every group-specific calibration rule in DynamicRules; do not drop "
    "signal just to shorten the prompt. The total number of distinct rules must not "
    "go down: this is a reorganisation, not a summary.\n"
    "4. Rules must fire on observable text triggers (phrases, length, structure). "
    "Never address a cluster or reviewer type by number — the rating model sees "
    "only the review text and cannot know which group it belongs to.\n"
    "5. Do not invent new rules; only reorganize what you are given.\n"
    "6. Preserve the exact wording of a rule whenever you move it. The rules were "
    "tuned by search on measured accuracy; paraphrasing them silently changes "
    "behaviour and has cost accuracy every time it was measured.\n"
    "7. If a previously consolidated BaseGuidelines is supplied, treat it as "
    "established: keep its rules and add to it. Never shrink it back.\n\n"
    f"Output exactly two sections, no tags, no code fences:\n"
    f"{_BASE_HEADER}\n<new BaseGuidelines inner text>\n"
    f"{_DYNAMIC_HEADER}\n<new DynamicRules inner text>"
)


def _rule_count(text: str) -> int:
    """Number of non-empty lines — a crude but robust proxy for distinct rules."""
    return sum(1 for ln in (text or "").splitlines() if ln.strip())


def check_not_lossy(
    prompt_before: str, prompt_after: str, min_retained: float = 0.85
) -> Tuple[bool, str]:
    """
    Reject a consolidation that lost rules instead of moving them.

    Consolidation is an unguided rewrite of a prompt that search has already tuned,
    so its expected effect on fitness is negative and it lost 6 out of 6 times in the
    pair run (OBSERVATIONS O21). That is acceptable while it stays a competitor, but
    a rewrite that also *deletes* earned rules is not worth scoring at all.
    """
    before = _rule_count(extract_block(prompt_before, "BaseGuidelines")) + _rule_count(
        extract_block(prompt_before, "DynamicRules")
    )
    after = _rule_count(extract_block(prompt_after, "BaseGuidelines")) + _rule_count(
        extract_block(prompt_after, "DynamicRules")
    )
    if before == 0:
        return True, "no rules to compare"
    retained = after / before
    if retained < min_retained:
        return False, f"lossy: {after}/{before} rule lines retained ({retained:.0%})"
    return True, f"{after}/{before} rule lines retained ({retained:.0%})"


def _collect_dynamic_rules(prompts: Sequence[str]) -> List[str]:
    rules: List[str] = []
    for p in prompts:
        body = extract_block(p, "DynamicRules")
        if body and body.strip():
            rules.append(body.strip())
    return rules


def _parse_two_sections(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Split the consolidator reply into (base, dynamic); None for a missing part.

    Header matching is deliberately loose — models decorate the separators with
    markdown, colons or XML tags, and a strict match sends the whole cycle to the
    deterministic stub without saying so.
    """
    cleaned = re.sub(r"^```(?:text|xml|markdown)?\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = re.sub(r"</?(?:BaseGuidelines|DynamicRules)>", "", cleaned, flags=re.IGNORECASE)
    header = r"^[\s#*=_-]*{0}[\s:*=_-]*$"
    pattern = re.compile(
        header.format(r"BASE[_ ]?GUIDELINES") + r"(?P<base>.*?)" + header.format(r"DYNAMIC[_ ]?RULES") + r"(?P<dyn>.*)",
        re.DOTALL | re.IGNORECASE | re.MULTILINE,
    )
    m = pattern.search(cleaned)
    if not m:
        return None, None
    base = m.group("base").strip()
    dyn = m.group("dyn").strip()
    return (base or None), (dyn or None)


def _indent_inner(text: str) -> str:
    """Match the surrounding prompt's block indentation style."""
    body = "\n".join(("        " + ln.rstrip()).rstrip() for ln in text.splitlines())
    return "\n" + body + "\n    "


def consolidate_mock(current_prompt: str, carryover: Sequence[PromptRecord]) -> str:
    """
    Deterministic stub for mock/smoke runs.

    Mirrors the live contract: lift the first line of each carried DynamicRules
    block into BaseGuidelines and remove those lines from DynamicRules.
    """
    sources = [current_prompt] + [r.prompt for r in carryover]
    dyn_blocks = _collect_dynamic_rules(sources)
    base = extract_block(current_prompt, "BaseGuidelines") or ""
    own_dynamic = extract_block(current_prompt, "DynamicRules") or ""

    promoted: List[str] = []
    for body in dyn_blocks[:3]:
        first = next((ln.strip() for ln in body.splitlines() if ln.strip()), "")
        if first and first not in promoted:
            promoted.append(first[:120])
    if not promoted:
        return current_prompt

    marker = "<!-- GRAPE_BASE_CONSOLIDATED -->"
    if marker in base:
        return current_prompt

    new_base = base.rstrip() + "\n" + "\n".join(
        f"        - [consolidated] {line}" for line in promoted
    ) + f"\n        {marker}\n"
    kept = [ln for ln in own_dynamic.splitlines() if ln.strip() not in promoted]
    out = replace_block(current_prompt, "BaseGuidelines", new_base)
    return replace_block(out, "DynamicRules", "\n".join(kept))


def consolidate_llm(
    current_prompt: str,
    carryover: Sequence[PromptRecord],
    *,
    model: str,
    api_base: str,
    temperature: float = 0.3,
    max_tokens: int = 8192,
    prev_base: Optional[str] = None,
) -> str:
    """
    Rewrite BaseGuidelines + DynamicRules together. Falls back to the stub.

    Fallbacks are printed: a silent degrade to the deterministic stub in the
    middle of a live run looks exactly like a successful consolidation in the
    artifacts, which cost a cycle of diagnosis on the first Pareto smoke.
    """
    try:
        import os

        from openai import OpenAI

        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("[GRAPE] consolidation: no API key, using deterministic stub", flush=True)
            return consolidate_mock(current_prompt, carryover)

        base = extract_block(current_prompt, "BaseGuidelines") or ""
        own_dynamic = extract_block(current_prompt, "DynamicRules") or ""
        other = _collect_dynamic_rules([r.prompt for r in carryover])
        other_blob = "\n\n---\n\n".join(other[:4]) if other else "(none)"

        prior = (
            f"Previously consolidated BaseGuidelines (established — keep and extend, "
            f"never shrink):\n{prev_base}\n\n"
            if prev_base and prev_base.strip()
            else ""
        )
        user = (
            f"{prior}"
            f"Current BaseGuidelines:\n{base}\n\n"
            f"Current DynamicRules (this prompt):\n{own_dynamic}\n\n"
            f"DynamicRules from other prompts on the champion archive:\n{other_blob}\n\n"
            "Produce the two consolidated sections."
        )
        client = OpenAI(api_key=api_key, base_url=api_base)
        new_base = new_dynamic = None
        raw = ""
        for attempt in range(2):
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature if attempt == 0 else 0.0,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": user},
                ],
            )
            raw = resp.choices[0].message.content or ""
            new_base, new_dynamic = _parse_two_sections(raw)
            if new_base:
                break
            print(
                f"[GRAPE WARNING] consolidation reply {attempt + 1} lacked the two "
                f"required sections ({len(raw)} chars); retrying at T=0",
                flush=True,
            )
        if not new_base:
            print("[GRAPE WARNING] consolidation unparseable twice; using stub", flush=True)
            return consolidate_mock(current_prompt, carryover)
        out = replace_block(current_prompt, "BaseGuidelines", _indent_inner(new_base))
        if new_dynamic:
            out = replace_block(out, "DynamicRules", _indent_inner(new_dynamic))
        else:
            print(
                "[GRAPE WARNING] consolidation returned no DynamicRules section; "
                "promoted rules stay duplicated this cycle (OBSERVATIONS O9)",
                flush=True,
            )
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"[GRAPE WARNING] consolidation LLM call failed ({exc}); using stub", flush=True)
        return consolidate_mock(current_prompt, carryover)


def build_consolidated_prompt(
    current_prompt: str,
    carryover: Sequence[PromptRecord],
    *,
    use_mock: bool,
    model: str = "google/gemini-3.1-pro-preview",
    api_base: str = "https://openrouter.ai/api/v1",
    prev_base: Optional[str] = None,
) -> str:
    """
    Produce the consolidation candidate. No gate, no scoring — the caller scores
    it on D_select and lets it compete on the champion archive (OBSERVATIONS O10:
    the old gate compared val CVaR against D_select fitness and was meaningless).

    `prev_base` carries the last consolidated BaseGuidelines so the slow layer
    accumulates across cycles. Without it every call restarted from the original
    161-character base and the mechanism could never build anything (O21).
    """
    if use_mock:
        return consolidate_mock(current_prompt, carryover)
    return consolidate_llm(
        current_prompt, carryover, model=model, api_base=api_base, prev_base=prev_base
    )

"""
Synthetic FewShotExamples generator for active learning cycles.

Generates fully synthetic few-shot examples from current hard/error context
and replaces the existing <FewShotExamples> block before evolution.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from openai import OpenAI


@dataclass
class SyntheticFewShotConfig:
    """mode: replace — write generated block into <FewShotExamples> before evolution.
    inject_as_hint — keep prompt unchanged; return hint text for the evolution system message
    so the mutator can merge, replace, or ignore.
    """
    enabled: bool = False
    mode: str = "replace"  # "replace" | "inject_as_hint"
    n_examples: int = 10
    n_boundary_pairs: int = 3
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    max_retries: int = 2


def extract_tag_block(text: str, tag: str) -> str:
    pattern = rf"<{tag}>(.*?)</{tag}>"
    m = re.search(pattern, text, flags=re.DOTALL)
    if not m:
        return ""
    return m.group(1).strip()


def replace_tag_block(text: str, tag: str, new_inner_text: str) -> str:
    pattern = rf"<{tag}>.*?</{tag}>"
    replacement = f"<{tag}>\n{new_inner_text.strip()}\n</{tag}>"
    if re.search(pattern, text, flags=re.DOTALL):
        return re.sub(pattern, replacement, text, flags=re.DOTALL)
    return text


def _extract_top_confusion_pairs(error_context: str, top_k: int = 3) -> List[Tuple[int, int, int]]:
    pairs: List[Tuple[int, int, int]] = []
    pattern = re.compile(r"(?:gold|true)\s*=?\s*([1-5]).*?(?:pred|predicted)\s*=?\s*([1-5]).*?(\d+)", re.IGNORECASE)
    for line in error_context.splitlines():
        m = pattern.search(line)
        if not m:
            continue
        gold = int(m.group(1))
        pred = int(m.group(2))
        cnt = int(m.group(3))
        pairs.append((gold, pred, cnt))
        if len(pairs) >= top_k:
            break
    return pairs


class SyntheticFewShotGenerator:
    def __init__(self, cfg: SyntheticFewShotConfig, llm_cfg: dict):
        self.cfg = cfg
        self.llm_cfg = llm_cfg or {}
        api_base = self.llm_cfg.get("api_base", "https://openrouter.ai/api/v1")
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(base_url=api_base, api_key=api_key)

    @classmethod
    def from_config_dict(cls, cfg_dict: dict) -> "SyntheticFewShotGenerator":
        al_cfg = (cfg_dict or {}).get("active_learning", {})
        sf_cfg = (al_cfg.get("synthetic_fewshot") or {})
        mode_raw = str(sf_cfg.get("mode", "replace")).strip().lower()
        if mode_raw not in ("replace", "inject_as_hint"):
            mode_raw = "replace"
        cfg = SyntheticFewShotConfig(
            enabled=bool(sf_cfg.get("enabled", False)),
            mode=mode_raw,
            n_examples=int(sf_cfg.get("n_examples", 10)),
            n_boundary_pairs=int(sf_cfg.get("n_boundary_pairs", 3)),
            model=sf_cfg.get("model"),
            temperature=float(sf_cfg.get("temperature", 0.7)),
            max_tokens=int(sf_cfg.get("max_tokens", 4096)),
            max_retries=int(sf_cfg.get("max_retries", 2)),
        )
        return cls(cfg=cfg, llm_cfg=(cfg_dict or {}).get("llm", {}))

    def _resolve_model_name(self) -> str:
        if self.cfg.model:
            return self.cfg.model
        models = self.llm_cfg.get("models", [])
        if models:
            return models[0].get("name", "deepseek/deepseek-r1")
        return "deepseek/deepseek-r1"

    def _build_messages(
        self,
        current_prompt: str,
        error_context: str,
        hard_examples: List[Tuple[str, int]],
        confusion_pairs: Optional[List[Tuple[int, int, int]]] = None,
    ) -> Tuple[str, str]:
        current_fse = extract_tag_block(current_prompt, "FewShotExamples")
        current_rules = extract_tag_block(current_prompt, "DynamicRules")
        pairs = confusion_pairs or _extract_top_confusion_pairs(
            error_context, top_k=self.cfg.n_boundary_pairs
        )
        pair_lines = [
            f"- gold={g}, pred={p}, count={c}"
            for (g, p, c) in pairs
        ]
        if not pair_lines:
            pair_lines = ["- No parsed confusion pairs. Infer likely boundaries from hard examples."]

        hard_lines = []
        for i, (txt, label) in enumerate(hard_examples[:8], start=1):
            t = txt.strip().replace("\n", " ")
            if len(t) > 260:
                t = t[:260] + "..."
            hard_lines.append(f"{i}. gold={label} | review: {t}")
        if not hard_lines:
            hard_lines = ["No hard example texts available for this cycle."]

        system_msg = (
            "You generate synthetic few-shot examples for 1-5 star sentiment classification "
            "of Amazon Home & Kitchen reviews.\n"
            "Vary tone and length across examples (short blurbs vs longer reviews) where helpful.\n"
            "Your output must be ONLY a valid <FewShotExamples> block content (inner text only, "
            "without outer XML tags)."
        )

        user_msg = (
            f"Target number of examples: {self.cfg.n_examples}\n"
            f"Target boundary pairs: {self.cfg.n_boundary_pairs}\n\n"
            "Top confusion pairs:\n"
            + "\n".join(pair_lines)
            + "\n\nCurrent DynamicRules (for context):\n"
            + current_rules
            + "\n\nCurrent FewShotExamples (to improve/replace):\n"
            + current_fse
            + "\n\nHard examples (same hard context used in the mutator stage):\n"
            + "\n".join(hard_lines)
            + "\n\nRequirements:\n"
              "1) Generate ONLY synthetic examples (do not copy real reviews verbatim).\n"
              "2) Focus on top 3-4 confusion boundaries first.\n"
              "3) For each major boundary, include at least one contrastive pair "
              "(similar surface wording but different correct ratings).\n"
              "4) Prefer contrastive twin pairs for top confusion boundaries.\n"
              "5) Keep examples realistic for Amazon Home & Kitchen.\n"
              "6) Cover all ratings 1..5 with at least one example each.\n"
              "7) Each example must end with exactly: Rating: N\n"
              "8) Output format exactly:\n"
              "Example 1:\\nReview: ...\\nRating: 1\\n\\nExample 2: ...\n"
        )
        if pairs:
            user_msg += (
                "\n\nPriority boundaries to encode with contrastive examples:\n"
                + "\n".join(pair_lines[:4])
            )
        else:
            user_msg += "\n\nNo explicit confusion pairs available; infer boundaries from hard examples."
        return system_msg, user_msg

    def _validate_fse_inner_text(self, text: str) -> Tuple[bool, str]:
        ratings = re.findall(r"Rating:\s*([1-5])", text)
        if not ratings:
            return False, "No 'Rating: [1-5]' lines found."
        missing = [str(r) for r in range(1, 6) if str(r) not in set(ratings)]
        if missing:
            return False, f"Missing rating coverage: {', '.join(missing)}."
        n_examples = len(re.findall(r"Example\s+\d+\s*:", text, flags=re.IGNORECASE))
        if n_examples < 7 or n_examples > 12:
            return False, f"Expected 7-12 examples, got {n_examples}."
        return True, ""

    def generate(
        self,
        current_prompt: str,
        error_context: str,
        hard_examples: List[Tuple[str, int]],
        confusion_pairs: Optional[List[Tuple[int, int, int]]] = None,
        save_raw_to: Optional[Path] = None,
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Return:
          - prompt: updated prompt (replace mode), unchanged prompt (hint mode success),
            or None on failure
          - error string or None on success
          - system_hint: non-None only for inject_as_hint success — text to append to the
            evolution system message (draft few-shot for the mutator to use or ignore)
        """
        if not self.cfg.enabled:
            return current_prompt, None, None

        system_msg, user_msg = self._build_messages(
            current_prompt=current_prompt,
            error_context=error_context,
            hard_examples=hard_examples,
            confusion_pairs=confusion_pairs,
        )
        if self.cfg.mode == "inject_as_hint":
            user_msg += (
                "\n\nNote: This draft may be merged with the current <FewShotExamples> by a "
                "downstream mutator — keep examples self-contained and clearly labeled.\n"
            )
        model_name = self._resolve_model_name()

        last_error = "Unknown error"
        for attempt in range(self.cfg.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens,
                    timeout=300,
                )
                content = getattr(response.choices[0].message, "content", None)
                if content is None:
                    raise RuntimeError("Synthetic generator returned empty content")
                out = content.strip()
                if out.startswith("```"):
                    lines = out.splitlines()
                    if lines and lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    out = "\n".join(lines).strip()
                out = re.sub(r"^<FewShotExamples>\s*", "", out)
                out = re.sub(r"\s*</FewShotExamples>$", "", out)

                valid, reason = self._validate_fse_inner_text(out)
                if not valid:
                    last_error = reason
                    user_msg += (
                        "\n\nPrevious output failed validation: "
                        + reason
                        + "\nPlease regenerate and strictly follow format."
                    )
                    continue

                if save_raw_to is not None:
                    save_raw_to.parent.mkdir(parents=True, exist_ok=True)
                    save_raw_to.write_text(out, encoding="utf-8")

                if self.cfg.mode == "inject_as_hint":
                    hint_block = (
                        "The following is a **synthetic draft** of few-shot examples. "
                        "You may merge it with the current <FewShotExamples>, replace them "
                        "entirely, pick subsets, or ignore it — choose what improves fitness "
                        "(Acc_Hard, Acc_Anchor, kappa) per the main instructions.\n\n"
                        + out.strip()
                    )
                    return current_prompt, None, hint_block

                new_prompt = replace_tag_block(current_prompt, "FewShotExamples", out)
                return new_prompt, None, None
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)

        return None, last_error, None


def save_synthetic_manifest(
    path: Path,
    *,
    enabled: bool,
    status: str,
    reason: str = "",
    meta: Optional[dict] = None,
) -> None:
    payload = {
        "enabled": enabled,
        "status": status,
        "reason": reason,
    }
    if meta:
        payload["meta"] = meta
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


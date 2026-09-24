"""Optimizer LLM wrapper — same model as PRIME mutator (ROADMAP §6.3 #4)."""

from __future__ import annotations

from typing import Any, List, Optional

from prime.config import PrimeConfig
from prime.workers.ensemble import LLMWorker


class OptimizerLLM:
    """Thin wrapper so baselines can call ``complete`` / ``complete_n``."""

    def __init__(self, worker: LLMWorker) -> None:
        self.worker = worker

    @property
    def model_name(self) -> str:
        return self.worker.model_name

    def complete(self, prompt: str, *, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> str:
        old_tok = self.worker.max_tokens
        old_temp = self.worker.temperature
        if max_tokens is not None:
            self.worker.max_tokens = int(max_tokens)
        if temperature is not None:
            self.worker.temperature = float(temperature)
        try:
            return self.worker._call(prompt)
        finally:
            self.worker.max_tokens = old_tok
            self.worker.temperature = old_temp

    def complete_n(
        self,
        prompt: str,
        n: int = 1,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[str]:
        """n sequential completions (OpenRouter chat API has no reliable n for all models)."""
        out: List[str] = []
        for _ in range(max(1, int(n))):
            out.append(self.complete(prompt, max_tokens=max_tokens, temperature=temperature))
        return out


def build_optimizer_llm(
    cfg: PrimeConfig,
    *,
    temperature: Optional[float] = None,
    label_space: Optional[str] = None,
) -> OptimizerLLM:
    """Build mutator-side LLMWorker from evolution.* config."""
    evo = cfg.evolution
    ls = label_space or getattr(cfg.dataset, "label_space", None) or "binary"
    worker = LLMWorker(
        model_name=str(evo.mutator_model),
        api_base=cfg.ensemble.api_base,
        temperature=float(temperature if temperature is not None else evo.mutator_temperature),
        max_tokens=int(getattr(evo, "mutator_max_tokens", 4096) or 4096),
        timeout=int(cfg.ensemble.timeout),
        max_retries=int(cfg.ensemble.max_retries),
        reasoning_effort="none",
        label_space=str(ls),
        fail_closed=False,
    )
    return OptimizerLLM(worker)


def _strip_code_fence(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def inject_instruction_into_seed(
    instruction: str,
    seed_prompt: str,
    *,
    label_space: str = "binary",
) -> str:
    """Place APE/GPO free-form instruction into our seed XML (F7 adaptation).

    Upstream APE/GPO emit a short natural-language instruction; our scorer requires
    the XML contract with ``{review}``. Prefer editing ``<DynamicRules>`` when present.
    """
    import re

    instr = _strip_code_fence(instruction)
    instr = instr.replace("{text}", "{review}").replace("{review}", "the review")
    seed = seed_prompt or ""
    if label_space == "ordinal5":
        out_line = "        Output a single rating 1-5 only (e.g. \"Rating: 3\" or \"3\")."
        fallback = (
            f"{instr}\n\n"
            "Output a single star rating from 1 to 5.\n"
            "Review: {review}"
        )
    else:
        from baselines.task_profile import active as _profile

        out_line = _profile().out_line
        fallback = f"{instr}\n\n{_profile().answer_tail}"
    if re.search(r"<DynamicRules>.*?</DynamicRules>", seed, flags=re.DOTALL | re.IGNORECASE):
        return re.sub(
            r"<DynamicRules>.*?</DynamicRules>",
            "<DynamicRules>\n"
            f"{instr}\n"
            f"{out_line}\n"
            "    </DynamicRules>",
            seed,
            count=1,
            flags=re.DOTALL | re.IGNORECASE,
        )
    return fallback


def ensure_prompt_contract(
    instruction: str,
    *,
    seed_fallback: Optional[str] = None,
    label_space: str = "binary",
) -> str:
    """Wrap a free-form instruction into the scorer prompt contract."""
    from prime.evolution.prompt_contract import check_prompt_contract

    text = _strip_code_fence(instruction)
    text = text.replace("{text}", "{review}")
    if seed_fallback and ("<System>" in seed_fallback or "<DynamicRules>" in seed_fallback):
        text = inject_instruction_into_seed(
            text, seed_fallback, label_space=label_space
        )
    elif "{review}" not in text:
        if label_space == "ordinal5":
            text = (
                f"{text.rstrip()}\n\n"
                "Output a single star rating from 1 to 5.\n"
                "Review: {review}"
            )
        else:
            from baselines.task_profile import active as _profile

            text = f"{text.rstrip()}\n\n{_profile().answer_tail}"
    ok = check_prompt_contract(text, label_space=label_space)
    if ok.ok:
        return text
    if seed_fallback:
        return seed_fallback
    if label_space == "ordinal5":
        return (
            "Classify the Amazon review into a star rating 1-5. "
            "Output Rating: N.\nReview: {review}"
        )
    from baselines.task_profile import active as _profile

    return _profile().last_resort

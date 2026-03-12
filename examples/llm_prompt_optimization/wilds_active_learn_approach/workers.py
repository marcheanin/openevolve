"""
LLMWorker with OpenRouter support for Active Prompt Evolution.
Учитывает токены через token_usage.get_tracker().
"""

import logging
import os

# Отключить verbose HTTP-логи от httpx/httpcore (OpenAI client)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
import re
import time
from typing import Optional

from openai import OpenAI


def parse_rating(response: str) -> int:
    """
    Extract rating 1-5 from model response.
    """
    patterns = [
        r'^([1-5])$',
        r'(?:rating|score|answer)[:\s]+([1-5])',
        r'<Rating>\s*([1-5])\s*</Rating>',  # fallback if model still outputs XML
        r'\b([1-5])\s*(?:out of 5|/5|stars?)',
        r'^\s*\**\s*([1-5])\s*\**\s*$',
        r'final.*?([1-5])',
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            return int(match.group(1))

    numbers = re.findall(r'\b([1-5])\b', response)
    if numbers:
        return int(numbers[-1])

    return 3


# OpenRouter model IDs (used when api_base is OpenRouter)
OPENROUTER_MODELS = {
    "deepseek-chat-v3": "deepseek/deepseek-chat-v3",
    "gemma3-27b": "google/gemma-3-27b-it",
    "gpt-4o-mini": "openai/gpt-4o-mini",
}



class LLMWorker:
    """
    Worker for calling LLM APIs via OpenRouter (OpenAI-compatible endpoint).
    """

    DEFAULT_API_BASE = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        model_name: str,
        api_base: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 64,
        timeout: int = 60,
        max_retries: int = 3,
    ) -> None:
        self.model_name = model_name
        self.api_base = api_base or self.DEFAULT_API_BASE
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries

        self.model_uri = OPENROUTER_MODELS.get(model_name, model_name)
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(base_url=self.api_base, api_key=api_key)

    def _call(self, prompt: str) -> str:
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_uri,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                )
                if not response.choices:
                    raise RuntimeError("LLM response has no choices")

                # Учёт токенов (OpenAI: prompt_tokens/completion_tokens; OpenRouter: может input/output)
                usage = getattr(response, "usage", None)
                if usage is not None:
                    try:
                        from token_usage import get_tracker
                        inp = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", 0)
                        out = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", 0)
                        total = getattr(usage, "total_tokens", None)
                        get_tracker().record(self.model_name, inp, out, total)
                    except Exception:
                        pass

                message = response.choices[0].message
                content = getattr(message, "content", None)
                if content is None:
                    reasoning_content = getattr(message, "reasoning_content", None)
                    if reasoning_content:
                        return reasoning_content.strip()
                    raise RuntimeError(
                        "LLM response content is empty; "
                        f"message={message!r}, response={response!r}"
                    )
                return content.strip()
            except Exception as exc:
                last_error = exc
                time.sleep(1.5 * attempt)
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last_error}")

    def predict(self, review_text: str, instruction: str) -> int:
        prompt = instruction.format(review=review_text)
        response = self._call(prompt)
        return parse_rating(response)


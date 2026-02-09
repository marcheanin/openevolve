import os
import re
import time
from typing import Tuple, Optional

from openai import OpenAI


def parse_rating(response: str) -> int:
    """
    Extract rating 1-5 from model response.
    """
    patterns = [
        r'^([1-5])$',
        r'(?:rating|score|answer)[:\s]+([1-5])',
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


class LLMWorker:
    """
    Worker for calling Yandex Cloud-compatible OpenAI API.
    """

    MODEL_MAPPING = {
        "qwen3-235b": "gpt://b1gemincl8p7b2uiv5nl/qwen3-235b-a22b-fp8/latest",
        "gemma3-27b": "gpt://b1gemincl8p7b2uiv5nl/gemma-3-27b-it/latest",
        "gpt-oss-120b": "gpt://b1gemincl8p7b2uiv5nl/gpt-oss-120b/latest",
        "yandexgpt": "gpt://b1gemincl8p7b2uiv5nl/yandexgpt/latest",
    }

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
        self.model_uri = self.MODEL_MAPPING.get(model_name, model_name)
        self.api_base = api_base or "https://llm.api.cloud.yandex.net/v1"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries

        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("YANDEX_API_KEY")
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
            except Exception as exc:  # pragma: no cover - network dependent
                last_error = exc
                time.sleep(1.5 * attempt)
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last_error}")

    def predict(self, review_text: str, instruction: str) -> int:
        prompt = instruction.format(review=review_text)
        response = self._call(prompt)
        return parse_rating(response)

    def predict_with_reasoning(self, review_text: str, instruction: str) -> Tuple[str, int]:
        prompt = instruction.format(review=review_text)
        response = self._call(prompt)
        return response, parse_rating(response)


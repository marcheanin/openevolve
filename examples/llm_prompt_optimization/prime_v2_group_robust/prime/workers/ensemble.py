"""LLM ensemble workers with majority vote and parallel inference."""

from __future__ import annotations

import logging
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

from prime.config import EnsembleCfg, WorkerSpec

logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def parse_rating(response: str) -> int:
    """Extract rating 1-5 from model response."""
    patterns = [
        r"^([1-5])$",
        r"(?:rating|score|answer)[:\s]+([1-5])",
        r"<Rating>\s*([1-5])\s*</Rating>",
        r"\b([1-5])\s*(?:out of 5|/5|stars?)",
        r"^\s*\**\s*([1-5])\s*\**\s*$",
        r"final.*?([1-5])",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            return int(match.group(1))
    numbers = re.findall(r"\b([1-5])\b", response)
    if numbers:
        return int(numbers[-1])
    return 3


def disagreement_score(worker_predictions: List[int], rating_min: int = 1, rating_max: int = 5) -> float:
    """Ordinal-aware disagreement in [0, 1]."""
    if len(worker_predictions) < 2:
        return 0.0
    scale = rating_max - rating_min
    n = len(worker_predictions)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            total += abs(worker_predictions[i] - worker_predictions[j])
    num_pairs = n * (n - 1) / 2
    return float(total / num_pairs / scale)


@dataclass
class TokenTracker:
    """Lightweight token accounting (no external deps)."""

    totals: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def record(self, model: str, input_tokens: int, output_tokens: int, total: Optional[int] = None) -> None:
        bucket = self.totals.setdefault(model, {"input": 0, "output": 0, "total": 0})
        bucket["input"] += int(input_tokens or 0)
        bucket["output"] += int(output_tokens or 0)
        bucket["total"] += int(total or (input_tokens or 0) + (output_tokens or 0))

    def snapshot(self) -> Dict[str, Dict[str, int]]:
        return {k: dict(v) for k, v in self.totals.items()}


_GLOBAL_TRACKER = TokenTracker()


def get_token_tracker() -> TokenTracker:
    return _GLOBAL_TRACKER


def load_dotenv_if_present() -> Optional[Path]:
    """
    Load KEY=VALUE lines from .env if no API key is already in the environment.
    Search order:
      1. prime_v2_group_robust/.env
      2. ../wilds_active_learn_approach/.env  (v1 legacy location)
    Returns path loaded, or None.
    """
    if os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"):
        return None
    pkg_root = Path(__file__).resolve().parents[2]
    candidates = [
        pkg_root / ".env",
        pkg_root.parent / "wilds_active_learn_approach" / ".env",
    ]
    for env_path in candidates:
        if not env_path.is_file():
            continue
        try:
            for raw in env_path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("$env:"):
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip("'\"").strip('"')
                if k and v and k not in os.environ:
                    os.environ[k] = v
            return env_path
        except Exception:
            continue
    return None


def _load_dotenv_if_present() -> None:
    load_dotenv_if_present()


class LLMWorker:
    """OpenRouter-compatible LLM worker."""

    DEFAULT_API_BASE = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        model_name: str,
        api_base: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 64,
        timeout: int = 75,
        max_retries: int = 4,
    ) -> None:
        _load_dotenv_if_present()
        self.model_name = model_name
        self.model_uri = model_name
        self.api_base = api_base or self.DEFAULT_API_BASE
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or "sk-mock"
        self.client = OpenAI(base_url=self.api_base, api_key=api_key)

    def _call(self, prompt: str) -> str:
        last_error: Optional[Exception] = None
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
                usage = getattr(response, "usage", None)
                if usage is not None:
                    inp = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", 0)
                    out = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", 0)
                    total = getattr(usage, "total_tokens", None)
                    get_token_tracker().record(self.model_name, inp, out, total)
                message = response.choices[0].message
                content = getattr(message, "content", None)
                if content is None:
                    reasoning = getattr(message, "reasoning_content", None)
                    if reasoning:
                        return reasoning.strip()
                    raise RuntimeError("LLM response content is empty")
                return content.strip()
            except Exception as exc:
                last_error = exc
                time.sleep(1.5 * attempt)
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last_error}")

    def predict(self, review_text: str, instruction: str) -> int:
        prompt = instruction.format(review=review_text)
        return parse_rating(self._call(prompt))


def probe_workers(
    cfg: EnsembleCfg,
    prompt_template: str,
    review_text: str = "Fantastic read. Dark, steamy and full of suspense.",
) -> List[str]:
    """
    One live call per worker. Returns error strings; empty list means all workers responded.
    """
    _load_dotenv_if_present()
    errors: List[str] = []
    for worker in build_workers(cfg):
        try:
            raw = worker._call(prompt_template.format(review=review_text))
            rating = parse_rating(raw)
            logger.info("Probe OK %s -> rating %d", worker.model_name, rating)
        except Exception as exc:
            errors.append(f"{worker.model_name}: {exc}")
    return errors


def build_workers(cfg: EnsembleCfg) -> List[LLMWorker]:
    return [
        LLMWorker(
            model_name=w.name,
            api_base=cfg.api_base,
            temperature=w.temperature,
            max_tokens=w.max_tokens,
            timeout=cfg.timeout,
            max_retries=cfg.max_retries,
        )
        for w in cfg.workers
    ]


class MajorityVoteAggregator:
    """Majority vote with configurable tie-break."""

    def __init__(self, tie_break: str = "lowest_rating") -> None:
        self.tie_break = tie_break

    def aggregate(self, votes: List[int]) -> int:
        if not votes:
            return 3
        counts = Counter(votes)
        max_count = max(counts.values())
        tied = [v for v, c in counts.items() if c == max_count]
        if len(tied) == 1:
            return int(tied[0])
        if self.tie_break == "highest_rating":
            return int(max(tied))
        if self.tie_break == "first_worker":
            for v in votes:
                if v in tied:
                    return int(v)
        return int(min(tied))  # lowest_rating default


def parallel_predict(
    workers: List[LLMWorker],
    texts: List[str],
    prompt_template: str,
    max_parallel: int = 8,
    tie_break: str = "lowest_rating",
) -> Tuple[List[int], List[List[int]]]:
    """
    Run (text, worker) predictions in parallel.
    Returns ensemble_predictions[i], worker_preds[w][i].
    """
    n_texts = len(texts)
    n_workers = len(workers)
    grid: List[List[Optional[int]]] = [[None] * n_texts for _ in range(n_workers)]
    aggregator = MajorityVoteAggregator(tie_break=tie_break)

    error_counts: Dict[str, int] = {}

    def _call(w_idx: int, t_idx: int) -> Tuple[int, int, int]:
        try:
            pred = workers[w_idx].predict(texts[t_idx], prompt_template)
            return w_idx, t_idx, int(pred)
        except Exception as exc:
            model = workers[w_idx].model_name
            error_counts[model] = error_counts.get(model, 0) + 1
            if error_counts[model] <= 3:
                logger.warning("Worker %s failed (example %d): %s", model, t_idx, exc)
            return w_idx, t_idx, 3

    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = [pool.submit(_call, w, t) for w in range(n_workers) for t in range(n_texts)]
        for fut in as_completed(futures):
            w_idx, t_idx, pred = fut.result()
            grid[w_idx][t_idx] = pred

    if error_counts:
        total_failures = sum(error_counts.values())
        logger.error(
            "Ensemble API failures: %s (%d/%d calls defaulted to rating 3)",
            error_counts,
            total_failures,
            n_workers * n_texts,
        )

    worker_preds: List[List[int]] = []
    ensemble: List[int] = []
    for t in range(n_texts):
        votes = [int(grid[w][t] or 3) for w in range(n_workers)]
        worker_preds.append(votes)
    # transpose to worker_preds[w][t]
    wp_by_worker = [[worker_preds[t][w] for t in range(n_texts)] for w in range(n_workers)]
    for t in range(n_texts):
        votes = [wp_by_worker[w][t] for w in range(n_workers)]
        ensemble.append(aggregator.aggregate(votes))
    return ensemble, wp_by_worker


def mock_predict(
    texts: List[str],
    labels: List[int],
    n_workers: int,
    seed: int = 0,
) -> Tuple[List[int], List[List[int]]]:
    """Deterministic mock predictions for smoke/tests without API keys."""
    import numpy as np

    rng = np.random.RandomState(seed)
    wp: List[List[int]] = []
    for w in range(n_workers):
        noise = rng.randint(-1, 2, size=len(texts))
        wp.append([int(np.clip(labels[i] + noise[i], 1, 5)) for i in range(len(texts))])
    ensemble = []
    agg = MajorityVoteAggregator()
    for i in range(len(texts)):
        votes = [wp[w][i] for w in range(n_workers)]
        ensemble.append(agg.aggregate(votes))
    return ensemble, wp

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
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from prime.config import EnsembleCfg, WorkerSpec

logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Fail-closed sentinel (ROADMAP_PHASE3 F1). Never equals a gold label in {0,1} or {1..5}.
INVALID = -1


def parse_rating(response: str, *, fail_closed: bool = False) -> int:
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
        match = re.search(pattern, response or "", re.IGNORECASE | re.MULTILINE)
        if match:
            return int(match.group(1))
    numbers = re.findall(r"\b([1-5])\b", response or "")
    if numbers:
        return int(numbers[-1])
    return INVALID if fail_closed else 3


def parse_binary(response: str, *, fail_closed: bool = False) -> int:
    """Extract binary toxicity label {0, 1}. On failure: 0 (legacy) or INVALID (fail-closed)."""
    text = (response or "").strip()
    patterns = [
        r"^(?:label|toxicity|answer|class)?\s*[:\-]?\s*([01])\s*$",
        r"\b(?:label|toxicity|answer|class)\s*[:\-]?\s*([01])\b",
        r"\b(non[\s\-]?toxic|not toxic|benign|safe)\b",
        r"\b(toxic|hate|abusive)\b",
        r"^([01])$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if not match:
            continue
        g = match.group(1)
        if g is None:
            continue
        low = g.lower()
        if low in ("0",) or "non" in low or low in ("benign", "safe"):
            return 0
        if low in ("1",) or low in ("toxic", "hate", "abusive"):
            return 1
    # Last digit 0/1 in the response (prefer explicit trailing answer).
    digits = re.findall(r"\b([01])\b", text)
    if digits:
        return int(digits[-1])
    return INVALID if fail_closed else 0


def parse_label(
    response: str,
    label_space: str = "ordinal5",
    *,
    fail_closed: bool = False,
) -> int:
    if label_space == "binary":
        return parse_binary(response, fail_closed=fail_closed)
    return parse_rating(response, fail_closed=fail_closed)


def default_label(label_space: str = "ordinal5") -> int:
    """Legacy silent default (reward-hackable on binary). Prefer INVALID + fail_closed."""
    return 0 if label_space == "binary" else 3


def disagreement_score(
    worker_predictions: List[int],
    rating_min: int = 1,
    rating_max: int = 5,
    *,
    label_space: Optional[str] = None,
) -> float:
    """Ordinal-aware disagreement in [0, 1]. Binary uses scale=1 (Phase3 D5 fix)."""
    votes = [int(v) for v in worker_predictions if int(v) != INVALID]
    if len(votes) < 2:
        return 0.0
    if label_space == "binary" or (min(votes) >= 0 and max(votes) <= 1):
        rating_min, rating_max = 0, 1
    scale = max(1, rating_max - rating_min)
    n = len(votes)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            total += abs(votes[i] - votes[j])
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

    If only OPENAI_API_KEY is present (common for OpenRouter keys stored under
    that name), also alias it to OPENROUTER_API_KEY for clarity.
    """
    pkg_root = Path(__file__).resolve().parents[2]
    candidates = [
        pkg_root / ".env",
        pkg_root.parent / "wilds_active_learn_approach" / ".env",
    ]
    loaded: Optional[Path] = None
    already = bool(os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"))
    if not already:
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
                loaded = env_path
                break
            except Exception:
                continue
    # Alias OpenAI key → OpenRouter when only the former is set.
    if os.getenv("OPENAI_API_KEY") and not os.getenv("OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"] = os.environ["OPENAI_API_KEY"]
    return loaded


def _load_dotenv_if_present() -> None:
    load_dotenv_if_present()


def _message_text(message: Any) -> str:
    """Extract usable text from OpenRouter/OpenAI chat message (incl. reasoning models)."""
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                txt = part.get("text") or part.get("content")
                if txt:
                    parts.append(str(txt))
            else:
                txt = getattr(part, "text", None) or getattr(part, "content", None)
                if txt:
                    parts.append(str(txt))
        joined = "\n".join(p for p in parts if p and str(p).strip())
        if joined.strip():
            return joined.strip()
    for attr in ("reasoning_content", "reasoning"):
        raw = getattr(message, attr, None)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        if isinstance(raw, dict):
            txt = raw.get("content") or raw.get("text")
            if txt and str(txt).strip():
                return str(txt).strip()
    return ""


class LLMWorker:
    """OpenRouter-compatible LLM worker."""

    DEFAULT_API_BASE = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        model_name: str,
        api_base: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 128,
        timeout: int = 75,
        max_retries: int = 4,
        reasoning_effort: Optional[str] = "none",
        label_space: str = "ordinal5",
        fail_closed: bool = False,
    ) -> None:
        _load_dotenv_if_present()
        self.model_name = model_name
        self.model_uri = model_name
        self.api_base = api_base or self.DEFAULT_API_BASE
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        # Rating workers must disable chain-of-thought: with low max_tokens,
        # reasoning models (DeepSeek V4, GLM-5, …) exhaust the budget on
        # internal reasoning and return empty content (finish_reason=length).
        self.reasoning_effort = reasoning_effort
        self.label_space = label_space
        self.fail_closed = bool(fail_closed)
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or "sk-mock"
        self.client = OpenAI(base_url=self.api_base, api_key=api_key)

    def _call(self, prompt: str) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                kwargs: Dict[str, Any] = {
                    "model": self.model_uri,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "timeout": self.timeout,
                }
                # "none" / null → omit reasoning block (O1: disable CoT when possible).
                # Some endpoints (gpt-oss-20b) reject effort=none as "cannot be disabled";
                # for those, set reasoning_effort to low|medium and raise max_tokens.
                effort = (self.reasoning_effort or "").strip().lower()
                if effort and effort not in ("none", "off", "false", "0"):
                    kwargs["extra_body"] = {"reasoning": {"effort": effort}}
                response = self.client.chat.completions.create(**kwargs)
                if not response.choices:
                    raise RuntimeError("LLM response has no choices")
                usage = getattr(response, "usage", None)
                if usage is not None:
                    inp = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", 0)
                    out = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", 0)
                    total = getattr(usage, "total_tokens", None)
                    get_token_tracker().record(self.model_name, inp, out, total)
                text = _message_text(response.choices[0].message)
                if not text:
                    finish = getattr(response.choices[0], "finish_reason", None)
                    raise RuntimeError(f"LLM response content is empty (finish_reason={finish})")
                return text
            except Exception as exc:
                last_error = exc
                # Upstream OpenRouter/provider 429s need longer backoff than
                # transient network blips (E4 CivilComments: qwen shared pool).
                err_s = str(exc)
                if "429" in err_s or "rate-limited" in err_s.lower():
                    time.sleep(min(30.0, 4.0 * attempt))
                else:
                    time.sleep(1.5 * attempt)
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last_error}")

    def predict(self, review_text: str, instruction: str) -> int:
        try:
            prompt = instruction.format(review=review_text)
        except (KeyError, ValueError, IndexError):
            # Unbalanced braces / missing {review} — fail-closed path (Phase3 F7/D4).
            if self.fail_closed:
                return INVALID
            raise
        try:
            return parse_label(self._call(prompt), self.label_space, fail_closed=self.fail_closed)
        except Exception:
            if self.fail_closed:
                return INVALID
            raise


def probe_workers(
    cfg: EnsembleCfg,
    prompt_template: str,
    review_text: str = "Fantastic read. Dark, steamy and full of suspense.",
    label_space: str = "ordinal5",
) -> List[str]:
    """
    One live call per worker. Returns error strings; empty list means all workers responded.
    """
    _load_dotenv_if_present()
    errors: List[str] = []
    for worker in build_workers(cfg, label_space=label_space):
        try:
            raw = worker._call(prompt_template.format(review=review_text))
            label = parse_label(raw, label_space)
            logger.info("Probe OK %s -> label %d", worker.model_name, label)
        except Exception as exc:
            errors.append(f"{worker.model_name}: {exc}")
    return errors


def build_workers(cfg: EnsembleCfg, label_space: Optional[str] = None) -> List[LLMWorker]:
    space = label_space or getattr(cfg, "label_space", "ordinal5") or "ordinal5"
    fail_closed = bool(getattr(cfg, "fail_closed", False))
    return [
        LLMWorker(
            model_name=w.name,
            api_base=cfg.api_base,
            temperature=w.temperature,
            max_tokens=w.max_tokens,
            timeout=cfg.timeout,
            max_retries=cfg.max_retries,
            reasoning_effort=getattr(w, "reasoning_effort", "none"),
            label_space=space,
            fail_closed=fail_closed,
        )
        for w in cfg.workers
    ]


class MajorityVoteAggregator:
    """Majority vote with configurable tie-break (ablation arm; SPEC v3 default is median)."""

    def __init__(self, tie_break: str = "lowest_rating", empty_default: int = 3) -> None:
        self.tie_break = tie_break
        self.empty_default = empty_default

    def aggregate(self, votes: List[int]) -> int:
        if not votes:
            return self.empty_default
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


class MedianAggregator:
    """
    Median of votes: consistent aggregation for an ordinal scale, no tie-break
    parameter (SPEC v3 §4.2 / Р8). Even vote counts use the lower median so the
    result is always an actually-cast integer vote (deterministic).
    """

    def __init__(self, empty_default: int = 3) -> None:
        self.empty_default = empty_default

    def aggregate(self, votes: List[int]) -> int:
        if not votes:
            return self.empty_default
        ordered = sorted(votes)
        return int(ordered[(len(ordered) - 1) // 2])


def build_aggregator(
    aggregation: str = "median",
    tie_break: str = "lowest_rating",
    empty_default: int = 3,
):
    if aggregation == "median":
        return MedianAggregator(empty_default=empty_default)
    return MajorityVoteAggregator(tie_break=tie_break, empty_default=empty_default)


def parallel_predict(
    workers: List[LLMWorker],
    texts: List[str],
    prompt_template: str,
    max_parallel: int = 8,
    tie_break: str = "lowest_rating",
    aggregation: str = "median",
    label_space: str = "ordinal5",
    fail_closed: bool = False,
) -> Tuple[List[int], List[List[int]]]:
    """
    Run (text, worker) predictions in parallel.
    Returns ensemble_predictions[i], worker_preds[w][i].
    When fail_closed=True, API/parse failures become INVALID (-1), never silent 0/3.
    """
    n_texts = len(texts)
    n_workers = len(workers)
    grid: List[List[Optional[int]]] = [[None] * n_texts for _ in range(n_workers)]
    space = label_space
    if workers and label_space == "ordinal5":
        space = getattr(workers[0], "label_space", label_space) or label_space
    use_fail_closed = bool(fail_closed) or any(getattr(w, "fail_closed", False) for w in workers)
    fail_default = INVALID if use_fail_closed else default_label(space)
    # Aggregator empty_default: INVALID votes are filtered before aggregate when fail_closed.
    aggregator = build_aggregator(
        aggregation, tie_break, empty_default=default_label(space) if not use_fail_closed else 0
    )

    error_counts: Dict[str, int] = {}

    def _call(w_idx: int, t_idx: int) -> Tuple[int, int, int]:
        try:
            # Honor per-worker fail_closed for parsing inside predict.
            workers[w_idx].fail_closed = use_fail_closed or workers[w_idx].fail_closed
            pred = workers[w_idx].predict(texts[t_idx], prompt_template)
            return w_idx, t_idx, int(pred)
        except Exception as exc:
            model = workers[w_idx].model_name
            error_counts[model] = error_counts.get(model, 0) + 1
            if error_counts[model] <= 3:
                logger.warning("Worker %s failed (example %d): %s", model, t_idx, exc)
            return w_idx, t_idx, fail_default

    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = [pool.submit(_call, w, t) for w in range(n_workers) for t in range(n_texts)]
        for fut in as_completed(futures):
            w_idx, t_idx, pred = fut.result()
            grid[w_idx][t_idx] = pred

    if error_counts:
        total_failures = sum(error_counts.values())
        logger.error(
            "Ensemble API failures: %s (%d/%d calls defaulted to label %d)",
            error_counts,
            total_failures,
            n_workers * n_texts,
            fail_default,
        )

    worker_preds: List[List[int]] = []
    for t in range(n_texts):
        votes = [
            int(grid[w][t] if grid[w][t] is not None else fail_default) for w in range(n_workers)
        ]
        worker_preds.append(votes)
    wp_by_worker = [[worker_preds[t][w] for t in range(n_texts)] for w in range(n_workers)]
    ensemble: List[int] = []
    for t in range(n_texts):
        votes = [wp_by_worker[w][t] for w in range(n_workers)]
        if n_workers == 1:
            ensemble.append(int(votes[0]))
            continue
        valid = [v for v in votes if int(v) != INVALID]
        if not valid:
            ensemble.append(INVALID if use_fail_closed else fail_default)
        else:
            ensemble.append(aggregator.aggregate(valid))
    return ensemble, wp_by_worker


def mock_predict(
    texts: List[str],
    labels: List[int],
    n_workers: int,
    seed: int = 0,
    aggregation: str = "median",
    label_space: str = "ordinal5",
) -> Tuple[List[int], List[List[int]]]:
    """Deterministic mock predictions for smoke/tests without API keys."""
    import numpy as np

    rng = np.random.RandomState(seed)
    lo, hi = (0, 1) if label_space == "binary" else (1, 5)
    fail_default = default_label(label_space)
    wp: List[List[int]] = []
    for w in range(n_workers):
        noise = rng.randint(-1, 2, size=len(texts))
        wp.append([int(np.clip(labels[i] + noise[i], lo, hi)) for i in range(len(texts))])
    ensemble = []
    agg = build_aggregator(aggregation, empty_default=fail_default)
    for i in range(len(texts)):
        votes = [wp[w][i] for w in range(n_workers)]
        ensemble.append(agg.aggregate(votes))
    return ensemble, wp

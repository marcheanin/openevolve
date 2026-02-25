"""
Test response latency for each model used in the Active Learning pipeline.
Uses the same prompt shape as in the experiment: instruction template + sample review
(like evaluator/workers: instruction.format(review=review_text)).
Run from this directory. Requires: pip install openai
"""

import os
import time
from pathlib import Path

from openai import OpenAI

# ---------------------------------------------------------------------------
# OpenRouter API key: set here or via environment variable OPENROUTER_API_KEY
# Get a key at https://openrouter.ai/keys
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = "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"  # <-- insert key here or set env OPENROUTER_API_KEY

if not OPENROUTER_API_KEY:
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
if not OPENROUTER_API_KEY:
    print("ERROR: No API key. Set OPENROUTER_API_KEY in this script or in the environment.")
    exit(1)

API_BASE = "https://openrouter.ai/api/v1"

# OpenRouter model IDs (same as in workers.py — short names are not valid)
MODELS = [
    ("deepseek/deepseek-r1", "evolution"),
    ("deepseek/deepseek-chat-v3", "worker"),
    ("google/gemma-3-27b-it", "worker"),
    ("openai/gpt-4o-mini", "worker"),
]

SCRIPT_DIR = Path(__file__).resolve().parent

# Sample review similar in length and style to those used in evaluation (Amazon Home & Kitchen)
SAMPLE_REVIEW = (
    "This coffee maker looked great in the pictures and had good reviews. "
    "After a month of use I have to say it's okay but not perfect. The brew is decent and "
    "the pot is easy to clean, but the timer is fiddly and sometimes it doesn't start when "
    "I set it the night before. For the price I expected a bit more reliability. "
    "It's better than nothing and we use it every day, but I wouldn't buy it again. "
    "Overall I'd say it's average — does the job with some annoyances."
)


def get_experiment_prompt() -> str:
    """Build prompt as in the experiment: initial_prompt.txt with {review} replaced by sample review."""
    prompt_path = SCRIPT_DIR / "initial_prompt.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt template not found: {prompt_path}. Run from wilds_active_learn_approach directory."
        )
    template = prompt_path.read_text(encoding="utf-8")
    return template.format(review=SAMPLE_REVIEW)


def test_one(model_name: str, prompt: str, timeout: int = 90, max_tokens: int = 64):
    """Return (latency_seconds, output_tokens, error_or_empty). Same message shape as workers.predict()."""
    client = OpenAI(base_url=API_BASE, api_key=OPENROUTER_API_KEY)
    start = time.perf_counter()
    try:
        r = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.1,
            timeout=timeout,
        )
        elapsed = time.perf_counter() - start
        n_out = getattr(r.usage, "completion_tokens", None) or 0
        return (elapsed, n_out, "")
    except Exception as e:
        elapsed = time.perf_counter() - start
        return (elapsed, 0, str(e))


def main():
    print("Testing response speed (one request per model)...")
    print(f"API base: {API_BASE}")
    try:
        prompt = get_experiment_prompt()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return
    n_chars = len(prompt)
    print(f"Prompt: same as in experiment (initial_prompt.txt + sample review), length={n_chars} chars\n")

    results = []
    for model_name, role in MODELS:
        print(f"  {model_name} ({role})... ", end="", flush=True)
        lat, n_tok, err = test_one(model_name, prompt, max_tokens=64)
        if err:
            print(f"FAIL: {err}")
            results.append((model_name, role, lat, None, err))
        else:
            print(f"{lat:.2f}s, {n_tok} tokens")
            results.append((model_name, role, lat, n_tok, None))

    print("\n" + "=" * 70)
    print("  Model                    Role      Latency    Out tokens   Status")
    print("=" * 70)
    for model_name, role, lat, n_tok, err in results:
        tok_str = str(n_tok) if n_tok is not None else "-"
        status = "OK" if err is None else f"ERR: {err[:40]}"
        print(f"  {model_name:24}  {role:8}  {lat:>7.2f}s   {tok_str:>10}   {status}")
    print("=" * 70)


if __name__ == "__main__":
    main()

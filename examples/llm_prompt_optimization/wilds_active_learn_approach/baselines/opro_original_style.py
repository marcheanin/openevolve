"""OPRO-style prompt optimization baseline for Amazon-WILDS.

Faithful to google-deepmind/opro ``optimize_instructions.py`` + ``opt_utils.py``:
  - meta-prompt with (instruction, train score) history (``instructions_only`` mode)
  - optimizer LLM proposes new instructions tagged with ``<INS>...</INS>``
  - scorer evaluates each candidate on a fixed train subset (accuracy / R_global)

Adapted for this repo:
  - full prompt templates with ``{review}`` (not GSM8K prefix-only instructions)
  - WILDS train/val subsampling via ``evaluator._load_split_data``
  - final reporting via ``run_full_test`` and WILDS metrics

Reference: Yang et al., "Large Language Models as Optimizers" (arXiv:2309.03409)
Official code: https://github.com/google-deepmind/opro
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from openai import OpenAI

BASELINE_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = BASELINE_DIR.parent
OPENEVOLVE_ROOT = EXPERIMENT_DIR.parent.parent.parent

if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(OPENEVOLVE_ROOT) not in sys.path:
    sys.path.append(str(OPENEVOLVE_ROOT))

from evaluator import _load_split_data, _run_evaluation  # noqa: E402
from run_full_test import run_full_test  # noqa: E402
from token_usage import get_tracker  # noqa: E402


@dataclass
class PromptRecord:
    prompt: str
    score: float
    metrics: Dict[str, Any]
    step: int
    record_id: str
    source: str


def _load_dotenv_if_present() -> None:
    if os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"):
        return
    for env_path in (EXPERIMENT_DIR / ".env", EXPERIMENT_DIR.parent / ".env"):
        if not env_path.exists():
            continue
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip().strip("'").strip('"')
            if key and value and key not in os.environ:
                os.environ[key] = value


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(path))


def _jsonable_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in metrics.items():
        if key == "accuracy_per_user":
            continue
        if isinstance(value, np.generic):
            out[key] = value.item()
        elif isinstance(value, np.ndarray):
            out[key] = value.tolist()
        elif isinstance(value, (int, float, str, bool)) or value is None:
            out[key] = value
    return out


def _eval_config(
    config: Dict[str, Any],
    *,
    target_model: Optional[str],
    max_parallel: Optional[int],
    ensemble: bool,
) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(config))
    defaults = cfg.setdefault("worker_defaults", {})
    defaults["temperature"] = 0.0
    if max_parallel is not None:
        defaults["max_parallel"] = int(max_parallel)
    if ensemble:
        if len(cfg.get("workers") or []) < 2:
            raise ValueError("Ensemble mode requires at least 2 workers in config.")
    else:
        if not target_model:
            raise ValueError("Single-worker mode requires --target-model.")
        cfg["workers"] = [{"name": target_model}]
    return cfg


def _worker_names(config: Dict[str, Any]) -> List[str]:
    return [str(w.get("name", w)) for w in config.get("workers", [])]


def _optimizer_client(config: Dict[str, Any]) -> tuple[OpenAI, str, Dict[str, Any]]:
    _load_dotenv_if_present()
    llm_cfg = config.get("llm", {}) or {}
    api_base = llm_cfg.get("api_base", "https://openrouter.ai/api/v1")
    models = llm_cfg.get("models") or []
    model_name = models[0].get("name") if models else "google/gemini-2.5-pro"
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set OPENROUTER_API_KEY or OPENAI_API_KEY, "
            "or add OPENROUTER_API_KEY=... to wilds_active_learn_approach/.env"
        )
    return OpenAI(base_url=api_base, api_key=api_key), model_name, llm_cfg


def _call_optimizer(
    client: OpenAI,
    model_name: str,
    llm_cfg: Dict[str, Any],
    user_prompt: str,
    *,
    temperature: float,
    seed: Optional[int] = None,
) -> str:
    max_retries = int(llm_cfg.get("retries", 3))
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            params: Dict[str, Any] = {
                "model": model_name,
                "messages": [{"role": "user", "content": user_prompt}],
                "temperature": float(temperature),
                "max_tokens": int(llm_cfg.get("max_tokens", 8192)),
                "timeout": int(llm_cfg.get("timeout", 300)),
            }
            if seed is not None:
                params["seed"] = int(seed) + attempt - 1
            response = client.chat.completions.create(**params)
            usage = getattr(response, "usage", None)
            if usage is not None:
                inp = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", 0) or 0
                out = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", 0) or 0
                total = getattr(usage, "total_tokens", None)
                get_tracker().record(f"optimizer/{model_name}", inp, out, total)
            if not getattr(response, "choices", None):
                raise RuntimeError("Optimizer LLM returned no choices")
            content = getattr(response.choices[0].message, "content", None)
            if content is None:
                content = getattr(response.choices[0].message, "reasoning_content", None)
            if content and str(content).strip():
                return str(content).strip()
            raise RuntimeError("Optimizer LLM returned empty response")
        except Exception as exc:
            last_error = exc
            time.sleep(min(1.5 * attempt, 4.0))
    raise RuntimeError(f"Optimizer LLM call failed after {max_retries} retries: {last_error}")


def _bucketize_float(num: float, n_buckets: int) -> int:
    num = max(0.0, min(1.0, float(num)))
    return round(num * n_buckets)


def _prompt_hash(prompt: str) -> str:
    return hashlib.md5(prompt.encode("utf-8")).hexdigest()


def _stratified_indices(labels: Sequence[int], size: int, seed: int) -> List[int]:
    y = np.asarray(labels)
    n = len(y)
    if size <= 0 or size >= n:
        return list(range(n))
    rng = np.random.default_rng(seed)
    by_label: Dict[int, List[int]] = {}
    for idx, lbl in enumerate(y.tolist()):
        by_label.setdefault(int(lbl), []).append(idx)
    selected: List[int] = []
    per_label = max(1, size // max(1, len(by_label)))
    for lbl in sorted(by_label):
        pool = by_label[lbl]
        take = min(per_label, len(pool))
        selected.extend(rng.choice(pool, size=take, replace=False).tolist())
    if len(selected) < size:
        remaining = [i for i in range(n) if i not in set(selected)]
        selected.extend(rng.choice(remaining, size=min(size - len(selected), len(remaining)), replace=False).tolist())
    return sorted(selected[:size])


def _subset(xs: Sequence[Any], indices: Sequence[int]) -> List[Any]:
    return [xs[i] for i in indices]


def _as_list(x: Any) -> List[Any]:
    return x.tolist() if hasattr(x, "tolist") else list(x)


def _evaluate_prompt(
    prompt: str,
    texts: Sequence[str],
    labels: Sequence[int],
    user_ids: Sequence[int],
    config: Dict[str, Any],
    *,
    ensemble: bool,
) -> Dict[str, Any]:
    pred_arr, _, metrics = _run_evaluation(prompt, list(texts), np.asarray(labels), np.asarray(user_ids), config)
    metrics["accuracy"] = float(np.mean(pred_arr == np.asarray(labels))) if len(labels) else 0.0
    metrics["score"] = float(metrics.get("R_global", metrics["accuracy"]))
    metrics["selection_metric"] = "R_global" if ensemble else "accuracy"
    return metrics


def _gen_ins_score_substr(
    history: List[Tuple[str, float, int]],
    *,
    score_threshold: float,
    max_num_instructions: int,
    num_score_buckets: Optional[int],
) -> str:
    """OPRO ``gen_ins_and_score_pairs_substr`` (instructions_only display)."""
    chunk = sorted(history, key=lambda x: x[1])[-max_num_instructions:]
    out = ""
    for instruction, score, _step in chunk:
        if score_threshold and score < score_threshold:
            continue
        if num_score_buckets is None:
            score_to_show = round(score, 3)
        else:
            score_to_show = _bucketize_float(score, num_score_buckets)
        out += f"\ntext:\n{instruction}\nscore:\n{score_to_show}\n"
    return out


def _gen_meta_prompt(
    history: List[Tuple[str, float, int]],
    *,
    score_threshold: float,
    max_num_instructions: int,
    num_score_buckets: Optional[int],
    task_description: str,
) -> str:
    """OPRO ``gen_meta_prompt`` in ``instructions_only`` mode for pre-trained optimizers."""
    meta_instruction = (
        f"Create a piece of text at the beginning of the answer to enhance the precision "
        f"in solving diverse {task_description} problems."
    )
    pairs = _gen_ins_score_substr(
        history,
        score_threshold=score_threshold,
        max_num_instructions=max_num_instructions,
        num_score_buckets=num_score_buckets,
    )
    meta_prompt = meta_instruction + pairs
    meta_prompt += (
        "\n\nGenerate an instruction that is different from all the instructions above, "
        "and has a higher score than all the instructions above. "
        "The instruction should begin with <INS> and end with </INS>. "
        "The instruction should be concise, effective, and generally applicable to all problems above."
    )
    return meta_prompt


def _extract_ins_prompt(raw: str, fallback: str) -> str:
    """Extract prompt from OPRO ``<INS>...</INS>`` tags (official GPT path)."""
    tagged = re.findall(r"<INS>\s*(.*?)\s*</INS>", raw, flags=re.DOTALL | re.IGNORECASE)
    if tagged:
        candidate = tagged[-1].strip()
    else:
        fence = re.search(r"```(?:text|xml|prompt)?\s*(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
        candidate = fence.group(1).strip() if fence else raw.strip()
    candidate = re.sub(r"^\s*<mutation_log>.*?</mutation_log>\s*", "", candidate, flags=re.DOTALL | re.IGNORECASE)
    return candidate if "{review}" in candidate else fallback


def _record_to_log(rec: PromptRecord) -> Dict[str, Any]:
    return {
        "record_id": rec.record_id,
        "source": rec.source,
        "step": rec.step,
        "score": rec.score,
        "metrics": _jsonable_metrics(rec.metrics),
    }


def _write_prompt(path: Path, prompt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prompt, encoding="utf-8")


def run_baseline(
    *,
    config_path: Path,
    prompt_path: Path,
    results_dir: Path,
    target_model: Optional[str],
    ensemble: bool,
    num_steps: int,
    num_generated_per_step: int,
    train_size: int,
    val_size: int,
    seed: Optional[int],
    max_parallel: Optional[int],
    max_history: int,
    score_threshold: float,
    num_score_buckets: int,
    optimizer_temperature: float,
    max_prompt_chars: int,
    run_full_test_flag: bool,
) -> None:
    effective_seed = int(seed) if seed is not None else int(time.time()) % (2**31 - 1)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    results_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir = results_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    meta_prompts_dir = results_dir / "meta_prompts"
    meta_prompts_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = results_dir / "trajectory.jsonl"

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    search_cfg = _eval_config(config, target_model=target_model, max_parallel=max_parallel, ensemble=ensemble)
    os.environ["WILDS_ACTIVE_LEARN_CONFIG"] = str(config_path.resolve())
    initial_prompt = prompt_path.read_text(encoding="utf-8")

    client, optimizer_model, llm_cfg = _optimizer_client(config)

    tr_x, tr_y, tr_u = _load_split_data(config, "train")
    va_x, va_y, va_u = _load_split_data(config, "validation")
    tr_idx = _stratified_indices(tr_y, train_size, effective_seed)
    va_idx = _stratified_indices(va_y, val_size, effective_seed + 1000)
    tr_y_list, tr_u_list = _as_list(tr_y), _as_list(tr_u)
    va_y_list, va_u_list = _as_list(va_y), _as_list(va_u)
    train_texts = _subset(tr_x, tr_idx)
    train_labels = _subset(tr_y_list, tr_idx)
    train_users = _subset(tr_u_list, tr_idx)
    val_texts = _subset(va_x, va_idx)
    val_labels = _subset(va_y_list, va_idx)
    val_users = _subset(va_u_list, va_idx)

    objective = "ensemble majority-vote R_global" if ensemble else "single-worker train accuracy"
    _atomic_write_json(
        results_dir / "run_manifest.json",
        {
            "baseline": "opro_original_style",
            "reference": "https://github.com/google-deepmind/opro",
            "config_path": str(config_path),
            "prompt_path": str(prompt_path),
            "ensemble": ensemble,
            "workers": _worker_names(search_cfg),
            "target_model": target_model if not ensemble else None,
            "optimizer_model": optimizer_model,
            "num_steps": num_steps,
            "num_generated_per_step": num_generated_per_step,
            "train_size": len(train_texts),
            "val_size": len(val_texts),
            "seed": seed,
            "effective_seed": effective_seed,
            "max_history": max_history,
            "score_threshold": score_threshold,
            "num_score_buckets": num_score_buckets,
            "optimizer_temperature": optimizer_temperature,
            "meta_prompt_type": "instructions_only",
            "train_indices": tr_idx,
            "val_indices": va_idx,
            "objective": objective,
            "selection_metric": "R_global" if ensemble else "accuracy",
        },
    )
    snapshot_name = "config_ensemble_snapshot.json" if ensemble else "config_single_worker_snapshot.json"
    _atomic_write_json(results_dir / snapshot_name, search_cfg)
    _write_prompt(results_dir / "initial_prompt.txt", initial_prompt)

    history: List[Tuple[str, float, int]] = []
    seen_hashes: set[str] = set()
    best_val: Optional[PromptRecord] = None
    best_val_metrics: Dict[str, Any] = {}
    best_train: Optional[PromptRecord] = None
    task_description = "amazon product review 1-5 star rating classification"

    def eval_and_log(prompt: str, source: str, step: int) -> Optional[PromptRecord]:
        ph = _prompt_hash(prompt)
        if ph in seen_hashes:
            print(f"  skip duplicate prompt ({source}, step {step})", flush=True)
            return None
        if len(prompt) > max_prompt_chars:
            print(f"  skip too-long prompt ({len(prompt)} chars > {max_prompt_chars})", flush=True)
            return None
        if "{review}" not in prompt:
            print(f"  skip prompt missing {{review}} placeholder ({source})", flush=True)
            return None

        metrics = _evaluate_prompt(
            prompt, train_texts, train_labels, train_users, search_cfg, ensemble=ensemble
        )
        rid = f"s{step:03d}_{source}_{int(time.time() * 1000)}_{len(list(candidates_dir.glob('*.txt'))):04d}"
        rec = PromptRecord(
            prompt=prompt,
            score=float(metrics["score"]),
            metrics=metrics,
            step=step,
            record_id=rid,
            source=source,
        )
        seen_hashes.add(ph)
        history.append((prompt, rec.score, step))
        _write_prompt(candidates_dir / f"{rid}.txt", prompt)
        _atomic_write_json(candidates_dir / f"{rid}_metrics.json", _jsonable_metrics(metrics))
        with open(trajectory_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(_record_to_log(rec), ensure_ascii=False) + "\n")
        return rec

    def check_val(rec: PromptRecord) -> None:
        nonlocal best_val, best_val_metrics
        vm = _evaluate_prompt(
            rec.prompt, val_texts, val_labels, val_users, search_cfg, ensemble=ensemble
        )
        with open(results_dir / "validation_trace.jsonl", "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "step": rec.step,
                        "record_id": rec.record_id,
                        "train_score": rec.score,
                        "val_metrics": _jsonable_metrics(vm),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        sel_key = "R_global" if ensemble else "accuracy"
        curr = float(best_val_metrics.get(sel_key, -1.0)) if best_val_metrics else -1.0
        if best_val is None or float(vm[sel_key]) > curr:
            best_val = rec
            best_val_metrics = vm
            _write_prompt(results_dir / "best_val_prompt.txt", rec.prompt)
            _atomic_write_json(results_dir / "best_val_metrics.json", _jsonable_metrics(vm))
            print(
                f"  New best val: {sel_key}={float(vm[sel_key]):.4f} ({rec.record_id})",
                flush=True,
            )

    mode_label = f"ensemble ({len(_worker_names(search_cfg))} workers)" if ensemble else f"single ({target_model})"
    print(f"OPRO baseline | {mode_label} | steps={num_steps} | gen/step={num_generated_per_step}", flush=True)

    print("\nEvaluating initial prompt...", flush=True)
    init_rec = eval_and_log(initial_prompt, "init", -1)
    if init_rec is None:
        raise RuntimeError("Initial prompt failed validation.")
    best_train = init_rec
    check_val(init_rec)
    _write_prompt(results_dir / "best_train_prompt.txt", init_rec.prompt)
    _atomic_write_json(results_dir / "best_train_metrics.json", _jsonable_metrics(init_rec.metrics))

    for step in range(num_steps):
        print(f"\nOPRO step {step + 1}/{num_steps} | best train score={best_train.score:.4f}", flush=True)
        step_threshold = score_threshold if step > 0 else 0.0
        meta_prompt = _gen_meta_prompt(
            history,
            score_threshold=step_threshold,
            max_num_instructions=max_history,
            num_score_buckets=num_score_buckets,
            task_description=task_description,
        )
        _write_prompt(meta_prompts_dir / f"step_{step:03d}_meta_prompt.txt", meta_prompt)

        generated: List[str] = []
        attempts = 0
        max_attempts = max(3, num_generated_per_step * 2)
        while len(generated) < num_generated_per_step and attempts < max_attempts:
            attempts += 1
            try:
                raw = _call_optimizer(
                    client,
                    optimizer_model,
                    llm_cfg,
                    meta_prompt,
                    temperature=optimizer_temperature,
                    seed=effective_seed + step * 100 + attempts,
                )
            except Exception as exc:
                print(f"  optimizer call failed: {exc}", flush=True)
                continue
            prompt = _extract_ins_prompt(raw, best_train.prompt)
            if prompt not in generated:
                generated.append(prompt)

        step_records: List[PromptRecord] = []
        for gi, prompt in enumerate(generated):
            rec = eval_and_log(prompt, f"generated_{gi}", step)
            if rec is not None:
                step_records.append(rec)

        if step_records:
            step_best = max(step_records, key=lambda r: r.score)
            if best_train is None or step_best.score > best_train.score:
                best_train = step_best
                _write_prompt(results_dir / "best_train_prompt.txt", best_train.prompt)
                _atomic_write_json(results_dir / "best_train_metrics.json", _jsonable_metrics(best_train.metrics))
            check_val(step_best)
        elif best_train is not None:
            check_val(best_train)

        _atomic_write_json(
            results_dir / f"step_{step:03d}_history.json",
            [{"prompt_hash": _prompt_hash(p), "score": s, "step": st} for p, s, st in history[-max_history:]],
        )

    if best_val is None and best_train is not None:
        best_val = best_train
        best_val_metrics = _evaluate_prompt(
            best_val.prompt, val_texts, val_labels, val_users, search_cfg, ensemble=ensemble
        )
        _write_prompt(results_dir / "best_val_prompt.txt", best_val.prompt)
        _atomic_write_json(results_dir / "best_val_metrics.json", _jsonable_metrics(best_val_metrics))

    cfg_eval = results_dir / ("config_ensemble.yaml" if ensemble else "config_single_worker.yaml")
    cfg_eval.write_text(yaml.safe_dump(search_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    if run_full_test_flag and best_val is not None:
        label = "ensemble" if ensemble else "single worker"
        print(f"\nRunning full uncapped test for best_val_prompt ({label})...", flush=True)
        run_full_test(
            prompt_path=results_dir / "best_val_prompt.txt",
            config_path=cfg_eval,
            results_dir=results_dir,
            max_parallel=max_parallel,
        )

    tracker = get_tracker()
    tracker.save_json(results_dir / "token_usage.json")
    tracker.write_report(results_dir / "token_usage_report.md", title="OPRO original-style token usage")
    print(f"\nDone. Results: {results_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run OPRO original-style baseline (google-deepmind/opro) on Amazon-WILDS."
    )
    parser.add_argument("--config", type=str, default="config_all_categories_uncapped_train.yaml")
    parser.add_argument("--prompt", type=str, default="initial_prompt_all_categories.txt")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--target-model", type=str, default="openai/gpt-4o-mini")
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Use workers from config (3-model majority vote).",
    )
    parser.add_argument("--num-steps", type=int, default=12, help="OPRO search steps (official default: 200).")
    parser.add_argument(
        "--num-generated-per-step",
        type=int,
        default=4,
        help="New prompts proposed per step (official default: 8).",
    )
    parser.add_argument("--train-size", type=int, default=80)
    parser.add_argument("--val-size", type=int, default=225)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-parallel", type=int, default=8)
    parser.add_argument("--max-history", type=int, default=20, help="Max (prompt, score) pairs in meta-prompt.")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.3,
        help="Drop low-scoring history entries from meta-prompt (OPRO GPT default: 0.3).",
    )
    parser.add_argument(
        "--num-score-buckets",
        type=int,
        default=100,
        help="Discretize scores in meta-prompt (OPRO default: 100).",
    )
    parser.add_argument(
        "--optimizer-temperature",
        type=float,
        default=1.0,
        help="Optimizer LLM temperature (OPRO default: 1.0).",
    )
    parser.add_argument("--max-prompt-chars", type=int, default=7500)
    parser.add_argument("--run-full-test", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.num_steps = 1
        args.num_generated_per_step = 2
        args.train_size = 10
        args.val_size = 20
        args.run_full_test = False
    elif args.pilot:
        args.num_steps = 3
        args.num_generated_per_step = 3
        args.train_size = 40
        args.val_size = 80
        args.run_full_test = False

    config_path = Path(args.config)
    prompt_path = Path(args.prompt)
    if not config_path.is_absolute():
        config_path = EXPERIMENT_DIR / config_path
    if not prompt_path.is_absolute():
        prompt_path = EXPERIMENT_DIR / prompt_path

    if args.results_dir is None:
        args.results_dir = "results_opro_ensemble" if args.ensemble else "results_opro_original"
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = EXPERIMENT_DIR / results_dir

    run_baseline(
        config_path=config_path.resolve(),
        prompt_path=prompt_path.resolve(),
        results_dir=results_dir.resolve(),
        target_model=args.target_model if not args.ensemble else None,
        ensemble=args.ensemble,
        num_steps=args.num_steps,
        num_generated_per_step=args.num_generated_per_step,
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed,
        max_parallel=args.max_parallel,
        max_history=args.max_history,
        score_threshold=args.score_threshold,
        num_score_buckets=args.num_score_buckets,
        optimizer_temperature=args.optimizer_temperature,
        max_prompt_chars=args.max_prompt_chars,
        run_full_test_flag=args.run_full_test,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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
class Candidate:
    prompt: str
    score: float
    metrics: Dict[str, Any]
    source: str
    generation: int
    parent_ids: List[str]
    candidate_id: str


def _load_dotenv_if_present() -> None:
    if os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"):
        return
    env_paths = [EXPERIMENT_DIR / ".env", EXPERIMENT_DIR.parent / ".env"]
    env_path = next((p for p in env_paths if p.exists()), None)
    if env_path is None:
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
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
        workers = cfg.get("workers") or []
        if len(workers) < 2:
            raise ValueError(
                "Ensemble mode requires at least 2 workers in the config "
                f"(got {len(workers)}). Check workers: in {config.get('prompt_path', 'config')}."
            )
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
            "Missing API key. Set OPENROUTER_API_KEY or OPENAI_API_KEY in environment, "
            "or add OPENROUTER_API_KEY=... to wilds_active_learn_approach/.env"
        )
    client = OpenAI(base_url=api_base, api_key=api_key)
    return client, model_name, llm_cfg


def _call_optimizer(
    client: OpenAI,
    model_name: str,
    llm_cfg: Dict[str, Any],
    system: str,
    user: str,
    *,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
) -> str:
    max_retries = int(llm_cfg.get("retries", 3))
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            params: Dict[str, Any] = {
                "model": model_name,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": float(temperature if temperature is not None else llm_cfg.get("temperature", 0.8)),
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


def _extract_code_fence(text: str) -> str:
    m = re.search(r"```(?:text|xml|prompt)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text.strip()


def _extract_candidates(text: str) -> List[str]:
    tagged = re.findall(r"<CANDIDATE_\d+>\s*(.*?)\s*</CANDIDATE_\d+>", text, flags=re.DOTALL | re.IGNORECASE)
    if tagged:
        return [_extract_code_fence(x) for x in tagged if x.strip()]
    parts = re.split(r"\n\s*(?:Candidate|CANDIDATE)\s+\d+\s*[:\-]\s*", text)
    if len(parts) > 1:
        return [_extract_code_fence(x) for x in parts[1:] if x.strip()]
    return [_extract_code_fence(text)]


def _normalize_prompt(candidate: str, fallback: str) -> str:
    p = _extract_code_fence(candidate).strip()
    p = re.sub(r"^\s*<mutation_log>.*?</mutation_log>\s*", "", p, flags=re.DOTALL | re.IGNORECASE)
    return p if "{review}" in p else fallback


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
    selected = selected[:size]
    selected.sort()
    return selected


def _subset(xs: Sequence[Any], indices: Sequence[int]) -> List[Any]:
    return [xs[i] for i in indices]


def _as_list(x: Any) -> List[Any]:
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x)


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
    # Original EvoPrompt objective: majority-vote accuracy (= R_global for ensemble).
    metrics["score"] = float(metrics.get("R_global", metrics["accuracy"]))
    if ensemble:
        metrics["selection_metric"] = "R_global"
    else:
        metrics["selection_metric"] = "accuracy"
    return metrics


def _select_parent(pop: List[Candidate], rng: random.Random, k: int = 3) -> Candidate:
    k = min(k, len(pop))
    return max(rng.sample(pop, k=k), key=lambda c: c.score)


def _candidate_to_log(c: Candidate) -> Dict[str, Any]:
    return {
        "candidate_id": c.candidate_id,
        "source": c.source,
        "generation": c.generation,
        "parent_ids": c.parent_ids,
        "score": c.score,
        "metrics": _jsonable_metrics(c.metrics),
    }


def _make_init_variants(client: OpenAI, model_name: str, llm_cfg: Dict[str, Any], initial_prompt: str, n: int, seed: int) -> List[str]:
    if n <= 0:
        return []
    system = "You are an EvoPrompt initialization operator."
    user = f"""
Generate {n} diverse prompt variants for 1-5 Amazon review rating classification.

Constraints:
- Preserve the {{review}} placeholder.
- Keep output parseable as Rating: N where N in 1..5.
- Return full prompt texts only via tags.

Seed prompt:
```text
{initial_prompt}
```

Output:
<CANDIDATE_1>...</CANDIDATE_1>
...
""".strip()
    try:
        raw = _call_optimizer(client, model_name, llm_cfg, system, user, temperature=0.9, seed=seed)
        return [_normalize_prompt(c, initial_prompt) for c in _extract_candidates(raw)][:n]
    except Exception:
        return []


def _mutate(client: OpenAI, model_name: str, llm_cfg: Dict[str, Any], parent: Candidate, seed: int) -> str:
    system = "You are an EvoPrompt-style mutation operator."
    user = f"""
Mutate the prompt below to improve classification accuracy.

Parent score: {parent.score:.4f}
Parent metrics: {json.dumps(_jsonable_metrics(parent.metrics), ensure_ascii=False)}

Rules:
- Preserve {{review}} placeholder.
- Keep 1-5 rating behavior.
- Keep output parseable as Rating: N.
- Return only the full mutated prompt.

Parent:
```text
{parent.prompt}
```
""".strip()
    try:
        raw = _call_optimizer(client, model_name, llm_cfg, system, user, temperature=0.8, seed=seed)
        return _normalize_prompt(raw, parent.prompt)
    except Exception:
        return parent.prompt


def _crossover(client: OpenAI, model_name: str, llm_cfg: Dict[str, Any], a: Candidate, b: Candidate, seed: int) -> str:
    system = "You are an EvoPrompt-style crossover operator."
    user = f"""
Combine two parent prompts into one stronger prompt.

Parent A score: {a.score:.4f}
Parent B score: {b.score:.4f}

Rules:
- Preserve {{review}} placeholder.
- Keep 1-5 rating behavior.
- Keep output parseable as Rating: N.
- Do not concatenate blindly; synthesize one coherent prompt.
- Return only the full child prompt.

Parent A:
```text
{a.prompt}
```

Parent B:
```text
{b.prompt}
```
""".strip()
    try:
        raw = _call_optimizer(client, model_name, llm_cfg, system, user, temperature=0.8, seed=seed)
        return _normalize_prompt(raw, a.prompt)
    except Exception:
        return a.prompt if a.score >= b.score else b.prompt


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
    popsize: int,
    generations: int,
    train_size: int,
    val_size: int,
    seed: Optional[int],
    mutation_rate: float,
    max_parallel: Optional[int],
    run_full_test_flag: bool,
) -> None:
    effective_seed = int(seed) if seed is not None else int(time.time()) % (2**31 - 1)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    rng = random.Random(effective_seed)
    results_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir = results_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
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
    tr_y_list = _as_list(tr_y)
    tr_u_list = _as_list(tr_u)
    va_y_list = _as_list(va_y)
    va_u_list = _as_list(va_u)
    train_texts, train_labels, train_users = _subset(tr_x, tr_idx), _subset(tr_y_list, tr_idx), _subset(tr_u_list, tr_idx)
    val_texts, val_labels, val_users = _subset(va_x, va_idx), _subset(va_y_list, va_idx), _subset(va_u_list, va_idx)

    objective = "ensemble majority-vote R_global" if ensemble else "single-worker train accuracy"
    _atomic_write_json(
        results_dir / "run_manifest.json",
        {
            "baseline": "evoprompt_original_style_ga",
            "config_path": str(config_path),
            "prompt_path": str(prompt_path),
            "ensemble": ensemble,
            "workers": _worker_names(search_cfg),
            "target_model": target_model if not ensemble else None,
            "optimizer_model": optimizer_model,
            "popsize": popsize,
            "generations": generations,
            "train_size": len(train_texts),
            "val_size": len(val_texts),
            "seed": seed,
            "effective_seed": effective_seed,
            "mutation_rate": mutation_rate,
            "train_indices": tr_idx,
            "val_indices": va_idx,
            "objective": objective,
            "selection_metric": "R_global" if ensemble else "accuracy",
        },
    )
    snapshot_name = "config_ensemble_snapshot.json" if ensemble else "config_single_worker_snapshot.json"
    _atomic_write_json(results_dir / snapshot_name, search_cfg)
    _write_prompt(results_dir / "initial_prompt.txt", initial_prompt)

    def eval_candidate(prompt: str, source: str, gen: int, parents: List[str]) -> Candidate:
        cid = f"g{gen:03d}_{source}_{int(time.time()*1000)}_{len(list(candidates_dir.glob('*.txt'))):04d}"
        metrics = _evaluate_prompt(
            prompt, train_texts, train_labels, train_users, search_cfg, ensemble=ensemble
        )
        c = Candidate(prompt=prompt, score=float(metrics["score"]), metrics=metrics, source=source, generation=gen, parent_ids=parents, candidate_id=cid)
        _write_prompt(candidates_dir / f"{cid}.txt", prompt)
        _atomic_write_json(candidates_dir / f"{cid}_metrics.json", _jsonable_metrics(metrics))
        with open(trajectory_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(_candidate_to_log(c), ensure_ascii=False) + "\n")
        return c

    mode_label = f"ensemble ({len(_worker_names(search_cfg))} workers)" if ensemble else f"single ({target_model})"
    print(f"Initializing EvoPrompt population: popsize={popsize} | {mode_label}", flush=True)
    init_prompts = [initial_prompt] + _make_init_variants(
        client, optimizer_model, llm_cfg, initial_prompt, max(0, popsize - 1), effective_seed + 17
    )
    init_prompts = init_prompts[:popsize]
    population = [eval_candidate(p, f"init{i}", 0, []) for i, p in enumerate(init_prompts)]
    population.sort(key=lambda c: c.score, reverse=True)

    best_val: Optional[Candidate] = None
    best_val_metrics: Dict[str, Any] = {}

    def check_val(candidate: Candidate, gen: int) -> None:
        nonlocal best_val, best_val_metrics
        vm = _evaluate_prompt(
            candidate.prompt, val_texts, val_labels, val_users, search_cfg, ensemble=ensemble
        )
        with open(results_dir / "validation_trace.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"generation": gen, "candidate_id": candidate.candidate_id, "train_score": candidate.score, "val_metrics": _jsonable_metrics(vm)}, ensure_ascii=False) + "\n")
        sel_key = "R_global" if ensemble else "accuracy"
        curr = float(best_val_metrics.get(sel_key, -1.0)) if best_val_metrics else -1.0
        if best_val is None or float(vm[sel_key]) > curr:
            best_val = candidate
            best_val_metrics = vm
            _write_prompt(results_dir / "best_val_prompt.txt", candidate.prompt)
            _atomic_write_json(results_dir / "best_val_metrics.json", _jsonable_metrics(vm))
            print(
                f"  New best val: {sel_key}={float(vm[sel_key]):.4f} ({candidate.candidate_id})",
                flush=True,
            )

    check_val(population[0], 0)

    for gen in range(1, generations + 1):
        print(f"\nGeneration {gen}/{generations} | best train score={population[0].score:.4f}", flush=True)
        children: List[Candidate] = []
        for child_idx in range(popsize):
            if len(population) > 1 and rng.random() > mutation_rate:
                a = _select_parent(population, rng)
                b = _select_parent(population, rng)
                prompt = _crossover(client, optimizer_model, llm_cfg, a, b, effective_seed + gen * 1000 + child_idx)
                children.append(eval_candidate(prompt, "crossover", gen, [a.candidate_id, b.candidate_id]))
            else:
                p = _select_parent(population, rng)
                prompt = _mutate(client, optimizer_model, llm_cfg, p, effective_seed + gen * 1000 + child_idx)
                children.append(eval_candidate(prompt, "mutation", gen, [p.candidate_id]))

        population = sorted(population + children, key=lambda c: c.score, reverse=True)[:popsize]
        _atomic_write_json(results_dir / f"generation_{gen:03d}_population.json", [_candidate_to_log(c) for c in population])
        _write_prompt(results_dir / "best_train_prompt.txt", population[0].prompt)
        _atomic_write_json(results_dir / "best_train_metrics.json", _jsonable_metrics(population[0].metrics))
        check_val(population[0], gen)

    if best_val is None:
        best_val = population[0]
        best_val_metrics = _evaluate_prompt(
            best_val.prompt, val_texts, val_labels, val_users, search_cfg, ensemble=ensemble
        )
        _write_prompt(results_dir / "best_val_prompt.txt", best_val.prompt)
        _atomic_write_json(results_dir / "best_val_metrics.json", _jsonable_metrics(best_val_metrics))

    cfg_eval = results_dir / ("config_ensemble.yaml" if ensemble else "config_single_worker.yaml")
    cfg_eval.write_text(yaml.safe_dump(search_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    if run_full_test_flag:
        label = "ensemble" if ensemble else "single worker"
        print(f"\nRunning full uncapped test for best_val_prompt ({label} config)...", flush=True)
        run_full_test(prompt_path=results_dir / "best_val_prompt.txt", config_path=cfg_eval, results_dir=results_dir, max_parallel=max_parallel)

    usage_path = results_dir / "token_usage.json"
    report_path = results_dir / "token_usage_report.md"
    tracker = get_tracker()
    tracker.save_json(usage_path)
    tracker.write_report(report_path, title="EvoPrompt original-style token usage")
    print(f"\nDone. Results: {results_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EvoPrompt original-style GA baseline from root wilds_active_learn_approach.")
    parser.add_argument("--config", type=str, default="config_all_categories_uncapped_train.yaml")
    parser.add_argument("--prompt", type=str, default="initial_prompt_all_categories.txt")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--target-model", type=str, default="openai/gpt-4o-mini")
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Use workers from config (3-model majority vote). Fair comparison with PRIME ensemble.",
    )
    parser.add_argument("--popsize", type=int, default=10)
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--train-size", type=int, default=80)
    parser.add_argument("--val-size", type=int, default=225)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for subsampling and GA. Omit for a time-based seed (recorded in run_manifest.json).",
    )
    parser.add_argument("--mutation-rate", type=float, default=0.7)
    parser.add_argument("--max-parallel", type=int, default=8)
    parser.add_argument("--run-full-test", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.popsize = 3
        args.generations = 1
        args.train_size = 10
        args.val_size = 20
        args.run_full_test = False
    elif args.pilot:
        args.popsize = 5
        args.generations = 3
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
        args.results_dir = "results_evoprompt_ensemble" if args.ensemble else "results_evoprompt_original"
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = EXPERIMENT_DIR / results_dir

    run_baseline(
        config_path=config_path.resolve(),
        prompt_path=prompt_path.resolve(),
        results_dir=results_dir.resolve(),
        target_model=args.target_model if not args.ensemble else None,
        ensemble=args.ensemble,
        popsize=args.popsize,
        generations=args.generations,
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed,
        mutation_rate=args.mutation_rate,
        max_parallel=args.max_parallel,
        run_full_test_flag=args.run_full_test,
    )


if __name__ == "__main__":
    main()

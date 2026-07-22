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
SCRIPT_DIR = BASELINE_DIR.parent
OPENEVOLVE_ROOT = SCRIPT_DIR.parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
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
    env_paths = [
        SCRIPT_DIR / ".env",
        SCRIPT_DIR.parent / ".env",
    ]
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


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _single_worker_config(config: Dict[str, Any], target_model: str, max_parallel: Optional[int]) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(config))
    defaults = cfg.setdefault("worker_defaults", {})
    defaults["temperature"] = 0.0
    if max_parallel is not None:
        defaults["max_parallel"] = int(max_parallel)
    cfg["workers"] = [{"name": target_model}]
    return cfg


def _ensure_wilds_experiment_config(config: Dict[str, Any]) -> None:
    """Some bundled WILDS helpers import wilds_experiment/config.yaml at module load."""
    path = SCRIPT_DIR / "wilds_experiment" / "config.yaml"
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _optimizer_client(config: Dict[str, Any]) -> tuple[OpenAI, str, Dict[str, Any]]:
    _load_dotenv_if_present()
    llm_cfg = config.get("llm", {}) or {}
    api_base = llm_cfg.get("api_base", "https://openrouter.ai/api/v1")
    models = llm_cfg.get("models") or []
    model_name = models[0].get("name") if models else "google/gemini-2.5-pro"
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set OPENROUTER_API_KEY or OPENAI_API_KEY in the environment, "
            "or add it to aclr_submission/.env as OPENROUTER_API_KEY=..."
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
    params: Dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": float(temperature if temperature is not None else llm_cfg.get("temperature", 0.8)),
        "max_tokens": int(llm_cfg.get("max_tokens", 8192)),
        "timeout": int(llm_cfg.get("timeout", 300)),
    }
    if seed is not None:
        params["seed"] = int(seed)
    response = client.chat.completions.create(**params)
    usage = getattr(response, "usage", None)
    if usage is not None:
        inp = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", 0) or 0
        out = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", 0) or 0
        total = getattr(usage, "total_tokens", None)
        get_tracker().record(f"optimizer/{model_name}", inp, out, total)
    content = getattr(response.choices[0].message, "content", None)
    if content is None:
        content = getattr(response.choices[0].message, "reasoning_content", None)
    if not content:
        raise RuntimeError("Optimizer LLM returned an empty response")
    return content.strip()


def _extract_code_fence(text: str) -> str:
    match = re.search(r"```(?:text|xml|prompt)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def _extract_tagged_candidates(text: str) -> List[str]:
    tagged = re.findall(
        r"<CANDIDATE_\d+>\s*(.*?)\s*</CANDIDATE_\d+>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if tagged:
        return [_extract_code_fence(t) for t in tagged if t.strip()]
    chunks = re.split(r"\n\s*(?:Candidate|CANDIDATE)\s+\d+\s*[:\-]\s*", text)
    if len(chunks) > 1:
        return [_extract_code_fence(c) for c in chunks[1:] if c.strip()]
    return [_extract_code_fence(text)]


def _normalize_prompt(candidate: str, fallback: str) -> str:
    prompt = _extract_code_fence(candidate).strip()
    prompt = re.sub(r"^\s*<mutation_log>.*?</mutation_log>\s*", "", prompt, flags=re.DOTALL | re.IGNORECASE)
    if "{review}" not in prompt:
        return fallback
    return prompt


def _stratified_indices(labels: Sequence[int], size: int, seed: int) -> List[int]:
    labels_arr = np.asarray(labels)
    n = len(labels_arr)
    if size <= 0 or size >= n:
        return list(range(n))
    rng = np.random.default_rng(seed)
    by_label: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels_arr.tolist()):
        by_label.setdefault(int(label), []).append(idx)
    selected: List[int] = []
    per_label = max(1, size // max(1, len(by_label)))
    for label in sorted(by_label):
        pool = by_label[label]
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


def _evaluate_prompt(
    prompt: str,
    texts: Sequence[str],
    labels: Sequence[int],
    user_ids: Sequence[int],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    pred_arr, _, metrics = _run_evaluation(
        prompt,
        list(texts),
        np.asarray(labels),
        np.asarray(user_ids),
        config,
    )
    accuracy = float(np.mean(pred_arr == np.asarray(labels))) if len(labels) else 0.0
    metrics["accuracy"] = accuracy
    metrics["score"] = accuracy
    metrics["prompt_tokens_est"] = _estimate_tokens(prompt)
    return metrics


def _candidate_to_log(candidate: Candidate) -> Dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "source": candidate.source,
        "generation": candidate.generation,
        "parent_ids": candidate.parent_ids,
        "score": candidate.score,
        "metrics": _jsonable_metrics(candidate.metrics),
        "prompt_tokens_est": _estimate_tokens(candidate.prompt),
    }


def _select_parent(population: List[Candidate], rng: random.Random, tournament_k: int = 3) -> Candidate:
    k = min(tournament_k, len(population))
    contenders = rng.sample(population, k=k)
    return max(contenders, key=lambda c: c.score)


def _make_variants(
    client: OpenAI,
    model_name: str,
    llm_cfg: Dict[str, Any],
    initial_prompt: str,
    n_variants: int,
    seed: int,
) -> List[str]:
    if n_variants <= 0:
        return []
    system = "You are the EvoPrompt initialization operator for prompt optimization."
    user = f"""
Generate {n_variants} diverse initial prompt variants for the same task.

Task: classify Amazon product reviews into 1, 2, 3, 4, or 5 stars.
The optimized prompt will be used by a single frozen LLM annotator.

Constraints:
- Preserve the {{review}} placeholder exactly.
- The annotator output must be parseable as "Rating: N" where N is 1..5.
- Return complete prompts, not summaries.
- Do not include gold labels, test data, or analysis outside candidate tags.

Seed prompt:
```text
{initial_prompt}
```

Return exactly {n_variants} candidates using:
<CANDIDATE_1>...</CANDIDATE_1>
...
""".strip()
    raw = _call_optimizer(client, model_name, llm_cfg, system, user, temperature=0.9, seed=seed)
    return [_normalize_prompt(c, initial_prompt) for c in _extract_tagged_candidates(raw)][:n_variants]


def _mutate_prompt(
    client: OpenAI,
    model_name: str,
    llm_cfg: Dict[str, Any],
    parent: Candidate,
    seed: int,
) -> str:
    system = "You are an EvoPrompt-style genetic mutation operator."
    user = f"""
Mutate the prompt below to improve Amazon review 1-5 star classification accuracy.

Parent score on the optimization subset: {parent.score:.4f}
Parent metrics: {json.dumps(_jsonable_metrics(parent.metrics), ensure_ascii=False)}

Rules:
- Preserve the {{review}} placeholder exactly.
- Preserve a clear 1-5 rating scale.
- Ensure the model's answer is parseable as "Rating: N".
- Make a meaningful but not destructive mutation.
- Return only the complete mutated prompt.

Parent prompt:
```text
{parent.prompt}
```
""".strip()
    raw = _call_optimizer(client, model_name, llm_cfg, system, user, temperature=0.8, seed=seed)
    return _normalize_prompt(raw, parent.prompt)


def _crossover_prompt(
    client: OpenAI,
    model_name: str,
    llm_cfg: Dict[str, Any],
    parent_a: Candidate,
    parent_b: Candidate,
    seed: int,
) -> str:
    system = "You are an EvoPrompt-style genetic crossover operator."
    user = f"""
Combine the two parent prompts into one improved Amazon review rating prompt.

Parent A score: {parent_a.score:.4f}
Parent B score: {parent_b.score:.4f}

Rules:
- Preserve the {{review}} placeholder exactly.
- Preserve a clear 1-5 rating scale.
- Ensure the model's answer is parseable as "Rating: N".
- Do not simply concatenate both prompts; synthesize one coherent prompt.
- Return only the complete child prompt.

Parent A:
```text
{parent_a.prompt}
```

Parent B:
```text
{parent_b.prompt}
```
""".strip()
    raw = _call_optimizer(client, model_name, llm_cfg, system, user, temperature=0.8, seed=seed)
    return _normalize_prompt(raw, parent_a.prompt)


def _write_prompt(path: Path, prompt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prompt, encoding="utf-8")


def run_evoprompt_original_style(
    *,
    config_path: Path,
    prompt_path: Path,
    results_dir: Path,
    target_model: str,
    popsize: int,
    generations: int,
    train_size: int,
    val_size: int,
    seed: int,
    mutation_rate: float,
    max_parallel: Optional[int],
    run_full_test_flag: bool,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    results_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir = results_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = results_dir / "trajectory.jsonl"

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _ensure_wilds_experiment_config(config)
    search_config = _single_worker_config(config, target_model, max_parallel)
    os.environ["WILDS_ACTIVE_LEARN_CONFIG"] = str(config_path.resolve())

    initial_prompt = prompt_path.read_text(encoding="utf-8")
    client, optimizer_model, llm_cfg = _optimizer_client(config)
    rng = random.Random(seed)

    train_texts_all, train_labels_all, train_users_all = _load_split_data(config, "train")
    val_texts_all, val_labels_all, val_users_all = _load_split_data(config, "validation")
    train_idx = _stratified_indices(train_labels_all, train_size, seed)
    val_idx = _stratified_indices(val_labels_all, val_size, seed + 1000)
    train_texts = _subset(train_texts_all, train_idx)
    train_labels = _subset(train_labels_all.tolist(), train_idx)
    train_users = _subset(train_users_all.tolist(), train_idx)
    val_texts = _subset(val_texts_all, val_idx)
    val_labels = _subset(val_labels_all.tolist(), val_idx)
    val_users = _subset(val_users_all.tolist(), val_idx)

    manifest = {
        "baseline": "evoprompt_original_style_ga",
        "note": "Original-style EvoPrompt GA baseline: no PRIME active batch, no Hard/Anchor fitness, no ensemble during search.",
        "config_path": str(config_path),
        "prompt_path": str(prompt_path),
        "target_model": target_model,
        "optimizer_model": optimizer_model,
        "popsize": popsize,
        "generations": generations,
        "train_size": len(train_texts),
        "val_size": len(val_texts),
        "seed": seed,
        "mutation_rate": mutation_rate,
        "search_objective": "single-worker accuracy on fixed train subset",
        "selection_objective": "single-worker validation accuracy",
        "train_indices": train_idx,
        "val_indices": val_idx,
    }
    _atomic_write_json(results_dir / "run_manifest.json", manifest)
    _atomic_write_json(results_dir / "config_single_worker_snapshot.json", search_config)
    _write_prompt(results_dir / "initial_prompt.txt", initial_prompt)

    init_prompts = [initial_prompt]
    init_prompts.extend(
        _make_variants(
            client,
            optimizer_model,
            llm_cfg,
            initial_prompt,
            max(0, popsize - 1),
            seed=seed + 17,
        )
    )
    init_prompts = init_prompts[:popsize]

    population: List[Candidate] = []
    best_val: Optional[Candidate] = None
    best_val_metrics: Dict[str, Any] = {}

    def evaluate_and_log(prompt: str, source: str, generation: int, parent_ids: List[str]) -> Candidate:
        candidate_id = f"g{generation:03d}_{source}_{int(time.time() * 1000)}_{len(list(candidates_dir.glob('*.txt'))):04d}"
        metrics = _evaluate_prompt(prompt, train_texts, train_labels, train_users, search_config)
        candidate = Candidate(
            prompt=prompt,
            score=float(metrics["score"]),
            metrics=metrics,
            source=source,
            generation=generation,
            parent_ids=parent_ids,
            candidate_id=candidate_id,
        )
        _write_prompt(candidates_dir / f"{candidate_id}.txt", prompt)
        _atomic_write_json(candidates_dir / f"{candidate_id}_metrics.json", _jsonable_metrics(metrics))
        with open(trajectory_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(_candidate_to_log(candidate), ensure_ascii=False) + "\n")
        return candidate

    print(f"Initializing EvoPrompt population: popsize={len(init_prompts)}", flush=True)
    for idx, prompt in enumerate(init_prompts):
        population.append(evaluate_and_log(prompt, f"init{idx}", 0, []))
    population.sort(key=lambda c: c.score, reverse=True)

    def update_best_val(candidate: Candidate, generation: int) -> None:
        nonlocal best_val, best_val_metrics
        val_metrics = _evaluate_prompt(candidate.prompt, val_texts, val_labels, val_users, search_config)
        val_score = float(val_metrics["accuracy"])
        current_best = float(best_val_metrics.get("accuracy", -1.0)) if best_val_metrics else -1.0
        record = {
            "generation": generation,
            "candidate_id": candidate.candidate_id,
            "train_score": candidate.score,
            "val_metrics": _jsonable_metrics(val_metrics),
        }
        with open(results_dir / "validation_trace.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if best_val is None or val_score > current_best:
            best_val = candidate
            best_val_metrics = val_metrics
            _write_prompt(results_dir / "best_val_prompt.txt", candidate.prompt)
            _atomic_write_json(results_dir / "best_val_metrics.json", _jsonable_metrics(val_metrics))
            print(f"  New best val: acc={val_score:.4f} from {candidate.candidate_id}", flush=True)

    update_best_val(population[0], 0)

    for generation in range(1, generations + 1):
        print(f"\nGeneration {generation}/{generations} | best train score={population[0].score:.4f}", flush=True)
        children: List[Candidate] = []
        for child_idx in range(popsize):
            if len(population) > 1 and rng.random() > mutation_rate:
                pa = _select_parent(population, rng)
                pb = _select_parent(population, rng)
                prompt = _crossover_prompt(
                    client,
                    optimizer_model,
                    llm_cfg,
                    pa,
                    pb,
                    seed=seed + generation * 1000 + child_idx,
                )
                child = evaluate_and_log(prompt, "crossover", generation, [pa.candidate_id, pb.candidate_id])
            else:
                parent = _select_parent(population, rng)
                prompt = _mutate_prompt(
                    client,
                    optimizer_model,
                    llm_cfg,
                    parent,
                    seed=seed + generation * 1000 + child_idx,
                )
                child = evaluate_and_log(prompt, "mutation", generation, [parent.candidate_id])
            children.append(child)

        population = sorted(population + children, key=lambda c: c.score, reverse=True)[:popsize]
        _atomic_write_json(
            results_dir / f"generation_{generation:03d}_population.json",
            [_candidate_to_log(c) for c in population],
        )
        _write_prompt(results_dir / "best_train_prompt.txt", population[0].prompt)
        _atomic_write_json(results_dir / "best_train_metrics.json", _jsonable_metrics(population[0].metrics))
        update_best_val(population[0], generation)

    if best_val is None:
        best_val = population[0]
        best_val_metrics = _evaluate_prompt(best_val.prompt, val_texts, val_labels, val_users, search_config)
        _write_prompt(results_dir / "best_val_prompt.txt", best_val.prompt)
        _atomic_write_json(results_dir / "best_val_metrics.json", _jsonable_metrics(best_val_metrics))

    single_config_path = results_dir / "config_single_worker.yaml"
    single_config_path.write_text(yaml.safe_dump(search_config, sort_keys=False, allow_unicode=True), encoding="utf-8")

    if run_full_test_flag:
        print("\nRunning full uncapped test for best_val_prompt with single target worker...", flush=True)
        run_full_test(
            prompt_path=results_dir / "best_val_prompt.txt",
            config_path=single_config_path,
            results_dir=results_dir,
            max_parallel=max_parallel,
        )

    usage_path = results_dir / "token_usage.json"
    report_path = results_dir / "token_usage_report.md"
    tracker = get_tracker()
    tracker.save_json(usage_path)
    tracker.write_report(report_path, title="EvoPrompt original-style token usage")
    print(f"\nDone. Results: {results_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an original-style EvoPrompt GA baseline on Amazon-WILDS."
    )
    parser.add_argument("--config", type=str, default="config_all_categories.yaml")
    parser.add_argument("--prompt", type=str, default="initial_prompt_all_categories.txt")
    parser.add_argument("--results-dir", type=str, default="results_evoprompt_original_seed_42")
    parser.add_argument("--target-model", type=str, default="openai/gpt-4o-mini")
    parser.add_argument("--popsize", type=int, default=10)
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--train-size", type=int, default=80)
    parser.add_argument("--val-size", type=int, default=225)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mutation-rate", type=float, default=0.7)
    parser.add_argument("--max-parallel", type=int, default=8)
    parser.add_argument("--run-full-test", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Cheap API smoke: popsize=3, generations=1, train=10, val=20.")
    parser.add_argument("--pilot", action="store_true", help="Small pilot: popsize=5, generations=3, train=40, val=80.")
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
        config_path = SCRIPT_DIR / config_path
    if not prompt_path.is_absolute():
        prompt_path = SCRIPT_DIR / prompt_path

    run_evoprompt_original_style(
        config_path=config_path.resolve(),
        prompt_path=prompt_path.resolve(),
        results_dir=Path(args.results_dir).resolve(),
        target_model=args.target_model,
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

"""ProTeGi / APO-style prompt optimization baseline for Amazon-WILDS.

Faithful to microsoft/LMOps ``prompt_optimization`` (ProTeGi, EMNLP 2023):
  - beam search over prompt candidates
  - textual gradients from misclassified minibatch examples
  - gradient-based prompt edits + synonym (MC) sampling
  - optional reject-on-errors pre-filter before full scoring

Adapted for this repo:
  - editable section: ``<DynamicRules>...</DynamicRules>`` in the initial prompt
  - WILDS train/val subsampling via ``evaluator._load_split_data``
  - objective: train accuracy (single) / R_global (ensemble)
  - final reporting via ``run_full_test`` and WILDS metrics

Reference: Pryzant et al., "Automatic Prompt Optimization with 'Gradient Descent'
and Beam Search" (EMNLP 2023)
Official code: https://github.com/microsoft/LMOps/tree/main/prompt_optimization
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
    round_idx: int
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
    n: int = 1,
    seed: Optional[int] = None,
) -> List[str]:
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
                "n": int(n),
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
            texts: List[str] = []
            for choice in response.choices:
                content = getattr(choice.message, "content", None)
                if content is None:
                    content = getattr(choice.message, "reasoning_content", None)
                if content and str(content).strip():
                    texts.append(str(content).strip())
            if texts:
                return texts
            raise RuntimeError("Optimizer LLM returned empty response")
        except Exception as exc:
            last_error = exc
            time.sleep(min(1.5 * attempt, 4.0))
    raise RuntimeError(f"Optimizer LLM call failed after {max_retries} retries: {last_error}")


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


def _prompt_hash(prompt: str) -> str:
    return hashlib.md5(prompt.encode("utf-8")).hexdigest()


def _write_prompt(path: Path, prompt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prompt, encoding="utf-8")


def _stringify_rating(label: int) -> str:
    return str(int(label))


def _extract_editable_section(prompt: str) -> str:
    """Return the editable inner text (ProTeGi ``task_section``).

    Prefer ``<DynamicRules>...</DynamicRules>``; fall back to ``# Task`` markdown sections
    from the original LMOps prompt format.
    """
    m = re.search(r"<DynamicRules>\s*(.*?)\s*</DynamicRules>", prompt, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    sections: Dict[str, str] = {}
    current: Optional[str] = None
    for line in prompt.split("\n"):
        stripped = line.strip()
        if stripped.startswith("# "):
            current = stripped[2:].strip().lower().split()[0]
            current = re.sub(r"[^\w]", "", current)
            sections[current] = ""
        elif current is not None:
            sections[current] += line + "\n"
    if "task" in sections:
        return sections["task"].strip()
    raise ValueError(
        "Prompt must contain <DynamicRules>...</DynamicRules> or a '# Task' section "
        "for ProTeGi-style editing."
    )


def _replace_editable_section(prompt: str, old_section: str, new_section: str) -> str:
    if "<DynamicRules>" in prompt:
        pattern = re.compile(
            r"(<DynamicRules>\s*)(.*?)(\s*</DynamicRules>)",
            flags=re.DOTALL | re.IGNORECASE,
        )
        if not pattern.search(prompt):
            raise ValueError("Missing <DynamicRules> block in prompt.")
        return pattern.sub(lambda m: f"{m.group(1)}{new_section.strip()}{m.group(3)}", prompt, count=1)
    if old_section not in prompt:
        raise ValueError("Editable section not found in prompt for replacement.")
    return prompt.replace(old_section, new_section.strip(), 1)


def _parse_tagged_text(text: str, start_tag: str, end_tag: str) -> List[str]:
    texts: List[str] = []
    while True:
        start_index = text.find(start_tag)
        if start_index == -1:
            break
        end_index = text.find(end_tag, start_index)
        if end_index == -1:
            break
        start_index += len(start_tag)
        texts.append(text[start_index:end_index].strip())
        text = text[end_index + len(end_tag) :]
    return texts


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


def _evaluate_minibatch(
    prompt: str,
    texts: Sequence[str],
    labels: Sequence[int],
    user_ids: Sequence[int],
    indices: Sequence[int],
    config: Dict[str, Any],
) -> Tuple[List[str], List[int], List[int]]:
    sub_texts = _subset(texts, indices)
    sub_labels = _subset(labels, indices)
    sub_users = _subset(user_ids, indices)
    pred_arr, _, _ = _run_evaluation(prompt, sub_texts, np.asarray(sub_labels), np.asarray(sub_users), config)
    preds = pred_arr.tolist()
    return sub_texts, sub_labels, preds


def _sample_error_string(
    texts: Sequence[str],
    labels: Sequence[int],
    preds: Sequence[int],
    *,
    n: int,
    rng: random.Random,
) -> str:
    error_idxs = [i for i, (label, pred) in enumerate(zip(labels, preds)) if int(label) != int(pred)]
    if not error_idxs:
        return ""
    sample_idxs = rng.sample(error_idxs, min(len(error_idxs), n))
    chunks: List[str] = []
    for error_idx, i in enumerate(sample_idxs):
        chunks.append(
            f"## Example {error_idx + 1}\n"
            f'Text: "{str(texts[i]).strip()}"\n'
            f"Label: {_stringify_rating(int(labels[i]))}\n"
            f"Prediction: {_stringify_rating(int(preds[i]))}\n"
        )
    return "\n".join(chunks).strip()


class ProTeGiOptimizer:
    """ProTeGi textual-gradient prompt optimizer (LMOps faithful)."""

    def __init__(
        self,
        *,
        client: OpenAI,
        optimizer_model: str,
        llm_cfg: Dict[str, Any],
        search_cfg: Dict[str, Any],
        ensemble: bool,
        minibatch_size: int,
        n_gradients: int,
        errors_per_gradient: int,
        gradients_per_error: int,
        steps_per_gradient: int,
        mc_samples_per_step: int,
        max_expansion_factor: int,
        reject_on_errors: bool,
        optimizer_temperature: float,
        rng: random.Random,
    ) -> None:
        self.client = client
        self.optimizer_model = optimizer_model
        self.llm_cfg = llm_cfg
        self.search_cfg = search_cfg
        self.ensemble = ensemble
        self.minibatch_size = minibatch_size
        self.n_gradients = n_gradients
        self.errors_per_gradient = errors_per_gradient
        self.gradients_per_error = gradients_per_error
        self.steps_per_gradient = steps_per_gradient
        self.mc_samples_per_step = mc_samples_per_step
        self.max_expansion_factor = max_expansion_factor
        self.reject_on_errors = reject_on_errors
        self.optimizer_temperature = optimizer_temperature
        self.rng = rng

    def _get_gradients(self, task_section: str, error_string: str) -> List[str]:
        gradient_prompt = f"""
        I'm trying to write a zero-shot classifier prompt for Amazon product review 1-5 star rating.

        My current prompt is:
        "{task_section}"

        But this prompt gets the following examples wrong:
        {error_string}

        give {self.gradients_per_error} reasons why the prompt could have gotten these examples wrong.
        Wrap each reason with <START> and <END>
        """
        gradient_prompt = "\n".join(line.lstrip() for line in gradient_prompt.split("\n"))
        responses = _call_optimizer(
            self.client,
            self.optimizer_model,
            self.llm_cfg,
            gradient_prompt,
            temperature=self.optimizer_temperature,
            n=1,
        )
        feedbacks: List[str] = []
        for response in responses:
            feedbacks.extend(_parse_tagged_text(response, "<START>", "<END>"))
        return feedbacks

    def apply_gradient(
        self,
        task_section: str,
        error_string: str,
        feedback: str,
    ) -> List[str]:
        transformation_prompt = f"""
        I'm trying to write a zero-shot classifier for Amazon product review 1-5 star rating.

        My current prompt is:
        "{task_section}"

        But it gets the following examples wrong:
        {error_string}

        Based on these examples the problem with this prompt is that {feedback}

        Based on the above information, I wrote {self.steps_per_gradient} different improved prompts.
        Each prompt is wrapped with <START> and <END>.

        The {self.steps_per_gradient} new prompts are:
        """
        transformation_prompt = "\n".join(line.lstrip() for line in transformation_prompt.split("\n"))
        responses = _call_optimizer(
            self.client,
            self.optimizer_model,
            self.llm_cfg,
            transformation_prompt,
            temperature=self.optimizer_temperature,
            n=1,
        )
        new_prompts: List[str] = []
        for response in responses:
            new_prompts.extend(_parse_tagged_text(response, "<START>", "<END>"))
        return new_prompts

    def generate_synonyms(self, prompt_section: str) -> List[str]:
        rewriter_prompt = (
            "Generate a variation of the following instruction while keeping the semantic meaning.\n\n"
            f"Input: {prompt_section}\n\nOutput:"
        )
        responses = _call_optimizer(
            self.client,
            self.optimizer_model,
            self.llm_cfg,
            rewriter_prompt,
            temperature=self.optimizer_temperature,
            n=self.mc_samples_per_step,
        )
        return [x for x in responses if x]

    def expand_candidates(
        self,
        prompts: List[str],
        train_texts: Sequence[str],
        train_labels: Sequence[int],
        train_users: Sequence[int],
    ) -> List[str]:
        k = min(self.minibatch_size, len(train_texts))
        minibatch_indices = self.rng.sample(list(range(len(train_texts))), k=k)
        new_prompts: List[str] = []

        for prompt in prompts:
            task_section = _extract_editable_section(prompt)
            sub_texts, sub_labels, preds = _evaluate_minibatch(
                prompt,
                train_texts,
                train_labels,
                train_users,
                minibatch_indices,
                self.search_cfg,
            )

            new_task_sections: List[str] = []
            if self.n_gradients > 0:
                for _ in range(self.n_gradients):
                    error_string = _sample_error_string(
                        sub_texts,
                        sub_labels,
                        preds,
                        n=self.errors_per_gradient,
                        rng=self.rng,
                    )
                    if not error_string:
                        continue
                    gradients = self._get_gradients(task_section, error_string)
                    for feedback in gradients:
                        new_task_sections.extend(
                            self.apply_gradient(task_section, error_string, feedback)
                        )

            mc_sampled_task_sections: List[str] = []
            if self.mc_samples_per_step > 0:
                for sect in new_task_sections + [task_section]:
                    mc_sampled_task_sections.extend(self.generate_synonyms(sect))

            candidate_sections = list(dict.fromkeys(new_task_sections + mc_sampled_task_sections))
            tmp_new_prompts = [
                _replace_editable_section(prompt, task_section, sect) for sect in candidate_sections
            ]

            if len(candidate_sections) > self.max_expansion_factor:
                if self.reject_on_errors:
                    error_idxs = [
                        i
                        for i, (label, pred) in enumerate(zip(sub_labels, preds))
                        if int(label) != int(pred)
                    ]
                    if error_idxs:
                        err_indices = self.rng.sample(
                            error_idxs, min(len(error_idxs), 16)
                        )
                        err_texts = [sub_texts[i] for i in err_indices]
                        err_labels = [sub_labels[i] for i in err_indices]
                        err_users = [train_users[minibatch_indices[i]] for i in err_indices]
                        pool = self.rng.sample(
                            tmp_new_prompts,
                            min(len(tmp_new_prompts), self.max_expansion_factor * 2),
                        )
                        error_scores = []
                        for cand in pool:
                            metrics = _evaluate_prompt(
                                cand,
                                err_texts,
                                err_labels,
                                err_users,
                                self.search_cfg,
                                ensemble=self.ensemble,
                            )
                            error_scores.append(float(metrics["score"]))
                        keep = np.argsort(error_scores)[-self.max_expansion_factor :]
                        tmp_new_prompts = [pool[i] for i in keep]
                    else:
                        tmp_new_prompts = self.rng.sample(
                            tmp_new_prompts, k=self.max_expansion_factor
                        )
                else:
                    tmp_new_prompts = self.rng.sample(tmp_new_prompts, k=self.max_expansion_factor)

            new_prompts.extend(tmp_new_prompts)

        new_prompts.extend(prompts)
        return list(dict.fromkeys(new_prompts))

    def score_candidates(
        self,
        prompts: List[str],
        train_texts: Sequence[str],
        train_labels: Sequence[int],
        train_users: Sequence[int],
    ) -> List[Tuple[float, Dict[str, Any]]]:
        if len(prompts) == 1:
            metrics = _evaluate_prompt(
                prompts[0],
                train_texts,
                train_labels,
                train_users,
                self.search_cfg,
                ensemble=self.ensemble,
            )
            return [(float(metrics["score"]), metrics)]
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for prompt in prompts:
            metrics = _evaluate_prompt(
                prompt,
                train_texts,
                train_labels,
                train_users,
                self.search_cfg,
                ensemble=self.ensemble,
            )
            scored.append((float(metrics["score"]), metrics))
        return scored


def _record_to_log(rec: PromptRecord) -> Dict[str, Any]:
    return {
        "record_id": rec.record_id,
        "source": rec.source,
        "round_idx": rec.round_idx,
        "score": rec.score,
        "metrics": _jsonable_metrics(rec.metrics),
    }


def run_baseline(
    *,
    config_path: Path,
    prompt_path: Path,
    results_dir: Path,
    target_model: Optional[str],
    ensemble: bool,
    rounds: int,
    beam_size: int,
    train_size: int,
    val_size: int,
    seed: Optional[int],
    max_parallel: Optional[int],
    minibatch_size: int,
    n_gradients: int,
    errors_per_gradient: int,
    gradients_per_error: int,
    steps_per_gradient: int,
    mc_samples_per_step: int,
    max_expansion_factor: int,
    reject_on_errors: bool,
    optimizer_temperature: float,
    max_prompt_chars: int,
    run_full_test_flag: bool,
) -> None:
    effective_seed = int(seed) if seed is not None else int(time.time()) % (2**31 - 1)
    rng = random.Random(effective_seed)
    if seed is not None:
        np.random.seed(seed)

    results_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir = results_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    gradients_dir = results_dir / "gradient_prompts"
    gradients_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = results_dir / "trajectory.jsonl"

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    search_cfg = _eval_config(config, target_model=target_model, max_parallel=max_parallel, ensemble=ensemble)
    os.environ["WILDS_ACTIVE_LEARN_CONFIG"] = str(config_path.resolve())
    initial_prompt = prompt_path.read_text(encoding="utf-8")
    _extract_editable_section(initial_prompt)

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

    effective_minibatch = min(minibatch_size, len(train_texts))

    objective = "ensemble majority-vote R_global" if ensemble else "single-worker train accuracy"
    _atomic_write_json(
        results_dir / "run_manifest.json",
        {
            "baseline": "apo_protegi_original_style",
            "reference": "https://github.com/microsoft/LMOps/tree/main/prompt_optimization",
            "config_path": str(config_path),
            "prompt_path": str(prompt_path),
            "ensemble": ensemble,
            "workers": _worker_names(search_cfg),
            "target_model": target_model if not ensemble else None,
            "optimizer_model": optimizer_model,
            "rounds": rounds,
            "beam_size": beam_size,
            "minibatch_size": effective_minibatch,
            "n_gradients": n_gradients,
            "errors_per_gradient": errors_per_gradient,
            "gradients_per_error": gradients_per_error,
            "steps_per_gradient": steps_per_gradient,
            "mc_samples_per_step": mc_samples_per_step,
            "max_expansion_factor": max_expansion_factor,
            "reject_on_errors": reject_on_errors,
            "train_size": len(train_texts),
            "val_size": len(val_texts),
            "seed": seed,
            "effective_seed": effective_seed,
            "optimizer_temperature": optimizer_temperature,
            "editable_section": "DynamicRules",
            "train_indices": tr_idx,
            "val_indices": va_idx,
            "objective": objective,
            "selection_metric": "R_global" if ensemble else "accuracy",
        },
    )
    snapshot_name = "config_ensemble_snapshot.json" if ensemble else "config_single_worker_snapshot.json"
    _atomic_write_json(results_dir / snapshot_name, search_cfg)
    _write_prompt(results_dir / "initial_prompt.txt", initial_prompt)

    optimizer = ProTeGiOptimizer(
        client=client,
        optimizer_model=optimizer_model,
        llm_cfg=llm_cfg,
        search_cfg=search_cfg,
        ensemble=ensemble,
        minibatch_size=effective_minibatch,
        n_gradients=n_gradients,
        errors_per_gradient=errors_per_gradient,
        gradients_per_error=gradients_per_error,
        steps_per_gradient=steps_per_gradient,
        mc_samples_per_step=mc_samples_per_step,
        max_expansion_factor=max_expansion_factor,
        reject_on_errors=reject_on_errors,
        optimizer_temperature=optimizer_temperature,
        rng=rng,
    )

    seen_hashes: set[str] = set()
    best_val: Optional[PromptRecord] = None
    best_val_metrics: Dict[str, Any] = {}

    def eval_and_log(
        prompt: str,
        source: str,
        round_idx: int,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Optional[PromptRecord]:
        ph = _prompt_hash(prompt)
        if ph in seen_hashes:
            print(f"  skip duplicate prompt ({source}, round {round_idx})", flush=True)
            return None
        if len(prompt) > max_prompt_chars:
            print(f"  skip too-long prompt ({len(prompt)} chars > {max_prompt_chars})", flush=True)
            return None
        if "{review}" not in prompt:
            print(f"  skip prompt missing {{review}} placeholder ({source})", flush=True)
            return None
        try:
            _extract_editable_section(prompt)
        except ValueError as exc:
            print(f"  skip invalid editable section ({exc})", flush=True)
            return None

        if metrics is None:
            metrics = _evaluate_prompt(
                prompt, train_texts, train_labels, train_users, search_cfg, ensemble=ensemble
            )
        rid = f"r{round_idx:03d}_{source}_{int(time.time() * 1000)}_{len(list(candidates_dir.glob('*.txt'))):04d}"
        rec = PromptRecord(
            prompt=prompt,
            score=float(metrics["score"]),
            metrics=metrics,
            round_idx=round_idx,
            record_id=rid,
            source=source,
        )
        seen_hashes.add(ph)
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
                        "round_idx": rec.round_idx,
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
    print(
        f"ProTeGi/APO baseline | {mode_label} | rounds={rounds} | beam={beam_size} | "
        f"minibatch={effective_minibatch}",
        flush=True,
    )

    candidates = [initial_prompt]
    beam_records: List[PromptRecord] = []

    for round_idx in range(rounds + 1):
        print(f"\nProTeGi round {round_idx}/{rounds}", flush=True)
        round_start = time.time()

        if round_idx > 0:
            candidates = optimizer.expand_candidates(
                candidates, train_texts, train_labels, train_users
            )
            print(f"  expanded to {len(candidates)} candidates", flush=True)

        scored = optimizer.score_candidates(candidates, train_texts, train_labels, train_users)
        ranked = sorted(zip(scored, candidates), key=lambda x: x[0][0], reverse=True)
        scored, candidates = [item for item, _ in ranked], [p for _, p in ranked]

        round_records: List[PromptRecord] = []
        for rank, ((score, metrics), prompt) in enumerate(zip(scored, candidates)):
            rec = eval_and_log(prompt, f"beam_rank{rank}", round_idx, metrics=metrics)
            if rec is not None:
                rec.score = float(score)
                round_records.append(rec)

        candidates = candidates[:beam_size]
        scored = scored[:beam_size]
        scores = [s for s, _ in scored]
        beam_records = round_records[:beam_size] if round_records else beam_records

        _atomic_write_json(
            results_dir / f"round_{round_idx:03d}_beam.json",
            {
                "round_idx": round_idx,
                "elapsed_sec": time.time() - round_start,
                "beam_size": beam_size,
                "candidates": [
                    {"score": float(s), "prompt_hash": _prompt_hash(p)} for s, p in zip(scores, candidates)
                ],
            },
        )

        if beam_records:
            best_rec = max(beam_records, key=lambda r: r.score)
            _write_prompt(results_dir / "best_train_prompt.txt", best_rec.prompt)
            _atomic_write_json(results_dir / "best_train_metrics.json", _jsonable_metrics(best_rec.metrics))
            check_val(best_rec)

    if best_val is None and beam_records:
        best_val = beam_records[0]
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
    tracker.write_report(results_dir / "token_usage_report.md", title="ProTeGi/APO original-style token usage")
    print(f"\nDone. Results: {results_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ProTeGi/APO original-style baseline (microsoft/LMOps) on Amazon-WILDS."
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
    parser.add_argument("--rounds", type=int, default=6, help="Beam search rounds (official default: 6).")
    parser.add_argument("--beam-size", type=int, default=4, help="Beam width (official default: 4).")
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--n-gradients", type=int, default=4)
    parser.add_argument("--errors-per-gradient", type=int, default=4)
    parser.add_argument("--gradients-per-error", type=int, default=1)
    parser.add_argument("--steps-per-gradient", type=int, default=1)
    parser.add_argument("--mc-samples-per-step", type=int, default=2)
    parser.add_argument("--max-expansion-factor", type=int, default=8)
    parser.add_argument(
        "--reject-on-errors",
        action="store_true",
        help="Pre-filter expanded prompts using error-subset scores (official optional flag).",
    )
    parser.add_argument("--train-size", type=int, default=80)
    parser.add_argument("--val-size", type=int, default=225)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-parallel", type=int, default=8)
    parser.add_argument("--optimizer-temperature", type=float, default=0.7)
    parser.add_argument("--max-prompt-chars", type=int, default=7500)
    parser.add_argument("--run-full-test", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.rounds = 1
        args.beam_size = 2
        args.minibatch_size = 8
        args.n_gradients = 1
        args.errors_per_gradient = 2
        args.mc_samples_per_step = 1
        args.max_expansion_factor = 4
        args.train_size = 10
        args.val_size = 20
        args.run_full_test = False
    elif args.pilot:
        args.rounds = 2
        args.beam_size = 3
        args.minibatch_size = 24
        args.n_gradients = 2
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
        args.results_dir = "results_apo_ensemble" if args.ensemble else "results_apo_original"
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = EXPERIMENT_DIR / results_dir

    run_baseline(
        config_path=config_path.resolve(),
        prompt_path=prompt_path.resolve(),
        results_dir=results_dir.resolve(),
        target_model=args.target_model if not args.ensemble else None,
        ensemble=args.ensemble,
        rounds=args.rounds,
        beam_size=args.beam_size,
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed,
        max_parallel=args.max_parallel,
        minibatch_size=args.minibatch_size,
        n_gradients=args.n_gradients,
        errors_per_gradient=args.errors_per_gradient,
        gradients_per_error=args.gradients_per_error,
        steps_per_gradient=args.steps_per_gradient,
        mc_samples_per_step=args.mc_samples_per_step,
        max_expansion_factor=args.max_expansion_factor,
        reject_on_errors=args.reject_on_errors,
        optimizer_temperature=args.optimizer_temperature,
        max_prompt_chars=args.max_prompt_chars,
        run_full_test_flag=args.run_full_test,
    )


if __name__ == "__main__":
    main()

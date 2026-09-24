"""APO / ProTeGi (Pryzant+ EMNLP 2023) — textual gradients + beam search.

Port of ``baselines/_upstream/LMOps/prompt_optimization/optimizers.py::ProTeGi``
and the round loop in ``main.py``. Native defaults from upstream argparse:
  rounds=6, beam_size=4, minibatch_size=64, n_gradients=4, errors_per_gradient=4,
  gradients_per_error=1, steps_per_gradient=1, mc_samples_per_step=2,
  max_expansion_factor=8.

Adaptation (ROADMAP §6.2):
- optimizer LLM = our mutator (deepseek) instead of utils.chatgpt
- task evaluate / candidate score = our Scorer on S_source minibatch / D_dev
- seed converted to ProTeGi sectioned markdown (# Task / # Output format / # Prediction)
- final selection on D_dev softmin (same gate as all methods)
"""

from __future__ import annotations

import random
import re
import string
from typing import Any, Dict, List, Optional, Sequence, Tuple

from baselines.api import OptimizerResult, Task, score_prompt_on_set
from baselines.optimizer_llm import ensure_prompt_contract


def parse_sectioned_prompt(s: str) -> Dict[str, str]:
    """Upstream ``utils.parse_sectioned_prompt`` (verbatim logic)."""
    result: Dict[str, str] = {}
    current_header = None
    for line in s.split("\n"):
        line_s = line.strip()
        if line_s.startswith("# "):
            current_header = line_s[2:].strip().lower().split()[0]
            current_header = current_header.translate(str.maketrans("", "", string.punctuation))
            result[current_header] = ""
        elif current_header is not None:
            result[current_header] += line + "\n"
    return result


def seed_to_protegi(seed_prompt: str) -> str:
    """Map our XML / free-form seed into ProTeGi sectioned format."""
    task_body = seed_prompt
    m = re.search(
        r"<DynamicRules>(.*?)</DynamicRules>",
        seed_prompt,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if m:
        task_body = m.group(1).strip()
    else:
        task_body = re.sub(
            r"</?System>|</?Role>|</?BaseGuidelines>|</?DynamicRules>|</?FewShotExamples>|</?Task>",
            "",
            seed_prompt,
        )
        task_body = task_body.replace("Comment: {review}", "").replace("Review: {review}", "").strip()
    return (
        "# Task\n"
        f"{task_body.strip()}\n\n"
        "# Output format\n"
        "Output exactly 'Label: 0' or 'Label: 1'.\n\n"
        "# Prediction\n"
        "Text: {review}\n"
        "Label:\n"
    )


def parse_tagged_text(text: str, start_tag: str, end_tag: str) -> List[str]:
    """Upstream ProTeGi.parse_tagged_text."""
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


class ProTeGiCore:
    """Port of upstream ProTeGi expand / gradient helpers."""

    def __init__(self, opt: Dict[str, Any], optimizer_llm) -> None:
        self.opt = opt
        self.llm = optimizer_llm

    def _sample_error_str(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        preds: Sequence[int],
        n: int = 4,
    ) -> str:
        error_idxs = [i for i, (l, p) in enumerate(zip(labels, preds)) if int(l) != int(p)]
        sample_idxs = random.sample(error_idxs, min(len(error_idxs), n))
        error_string = ""
        for error_idx, i in enumerate(sample_idxs):
            error_string += f"## Example {error_idx + 1}\n"
            error_string += (
                f'Text: "{texts[i].strip()}"\n'
                f"Label: {int(labels[i])}\n"
                f"Prediction: {int(preds[i])}\n\n"
            )
        return error_string.strip()

    def _get_gradients(self, prompt: str, error_string: str, num_feedbacks: int = 5, n: int = 1) -> List[str]:
        gradient_prompt = f"""
        I'm trying to write a zero-shot classifier prompt.

        My current prompt is:
        "{prompt}"

        But this prompt gets the following examples wrong:
        {error_string}

        give {num_feedbacks} reasons why the prompt could have gotten these examples wrong.
        Wrap each reason with <START> and <END>
        """
        gradient_prompt = "\n".join([line.lstrip() for line in gradient_prompt.split("\n")])
        res = self.llm.complete_n(gradient_prompt, n=n, temperature=0.7, max_tokens=1024)
        feedbacks: List[str] = []
        for r in res:
            feedbacks += parse_tagged_text(r, "<START>", "<END>")
        return feedbacks

    def apply_gradient(
        self,
        prompt: str,
        error_str: str,
        feedback_str: str,
        steps_per_gradient: int,
        n: int = 1,
    ) -> List[str]:
        transformation_prompt = f"""
        I'm trying to write a zero-shot classifier.

        My current prompt is:
        "{prompt}"

        But it gets the following examples wrong:
        {error_str}

        Based on these examples the problem with this prompt is that {feedback_str}

        Based on the above information, I wrote {steps_per_gradient} different improved prompts.
        Each prompt is wrapped with <START> and <END>.

        The {steps_per_gradient} new prompts are:
        """
        transformation_prompt = "\n".join([line.lstrip() for line in transformation_prompt.split("\n")])
        res = self.llm.complete_n(transformation_prompt, n=n, temperature=0.7, max_tokens=1024)
        new_prompts: List[str] = []
        for r in res:
            new_prompts += parse_tagged_text(r, "<START>", "<END>")
        return new_prompts

    def generate_synonyms(self, prompt_section: str, n: int = 3) -> List[str]:
        rewriter_prompt = (
            "Generate a variation of the following instruction while keeping the "
            f"semantic meaning.\n\nInput: {prompt_section}\n\nOutput:"
        )
        new_instructions = self.llm.complete_n(rewriter_prompt, n=n, temperature=0.7, max_tokens=512)
        return [x for x in new_instructions if x]

    def expand_candidates(
        self,
        prompts: Sequence[str],
        texts: Sequence[str],
        labels: Sequence[int],
        scorer,
        budget,
    ) -> List[str]:
        """Upstream expand_candidates, with Scorer replacing task.evaluate."""
        n_mb = min(int(self.opt["minibatch_size"]), len(texts))
        idxs = random.sample(range(len(texts)), k=n_mb)
        mb_texts = [texts[i] for i in idxs]
        mb_labels = [labels[i] for i in idxs]

        new_prompts: List[str] = []
        for prompt in prompts:
            sections = parse_sectioned_prompt(prompt)
            if "task" not in sections:
                # If not sectioned, wrap once.
                prompt = seed_to_protegi(prompt)
                sections = parse_sectioned_prompt(prompt)
            task_section = sections.get("task", "").strip()
            if not task_section:
                continue

            result = scorer.predict_batch(mb_texts, prompt, labels_for_mock=mb_labels)
            budget.charge("val", n_calls=len(mb_texts), kind="scorer", note="apo_minibatch")
            preds = [int(x) for x in result.preds]

            new_task_sections: List[str] = []
            if self.opt["n_gradients"] > 0:
                for _ in range(int(self.opt["n_gradients"])):
                    error_string = self._sample_error_str(
                        mb_texts, mb_labels, preds, n=int(self.opt["errors_per_gradient"])
                    )
                    if not error_string:
                        continue
                    gradients = self._get_gradients(
                        task_section,
                        error_string,
                        int(self.opt["gradients_per_error"]),
                        n=1,
                    )
                    budget.charge("mutator", n_calls=1, kind="optimizer", note="apo_grad")
                    for feedback in gradients:
                        tmp = self.apply_gradient(
                            task_section,
                            error_string,
                            feedback,
                            int(self.opt["steps_per_gradient"]),
                        )
                        budget.charge("mutator", n_calls=1, kind="optimizer", note="apo_apply")
                        new_task_sections += tmp

            mc_sampled: List[str] = []
            if self.opt["mc_samples_per_step"] > 0:
                for sect in new_task_sections + [task_section]:
                    mc_sects = self.generate_synonyms(sect, n=int(self.opt["mc_samples_per_step"]))
                    budget.charge(
                        "mutator",
                        n_calls=int(self.opt["mc_samples_per_step"]),
                        kind="optimizer",
                        note="apo_mc",
                    )
                    mc_sampled += mc_sects

            new_sections = list(set(new_task_sections + mc_sampled))
            tmp_new = [prompt.replace(task_section, tmp) for tmp in new_sections]
            if len(tmp_new) > int(self.opt["max_expansion_factor"]):
                tmp_new = random.sample(tmp_new, k=int(self.opt["max_expansion_factor"]))
            new_prompts += tmp_new

        new_prompts += list(prompts)
        # Dedup + ensure contract.
        seen = set()
        out: List[str] = []
        for p in new_prompts:
            p2 = p if "{review}" in p else ensure_prompt_contract(p)
            if p2 not in seen:
                seen.add(p2)
                out.append(p2)
        return out


def _score_prompts_on(
    prompts: Sequence[str],
    task: Task,
    labeled,
    *,
    note: str,
) -> List[float]:
    """Score prompts on a labeled set (train beam / D_dev final)."""
    scores: List[float] = []
    for p in prompts:
        m = score_prompt_on_set(task.scorer, p, labeled)
        task.budget.charge("val", n_calls=len(labeled.texts), kind="scorer", note=note)
        # Upstream ProTeGi optimizes task accuracy/F1 on train; we use softmin-GBA
        # on the same labeled pool so the search objective stays group-aware under
        # our metric stack (ROADMAP §6.3: one Scorer / one fitness family).
        scores.append(float(m.get("R_soft_min_gba", m.get("fitness", 0.0))))
    return scores


class APOOptimizer:
    """ProTeGi beam search with native hyperparameters."""

    def __init__(
        self,
        *,
        rounds: int = 6,
        beam_size: int = 4,
        minibatch_size: int = 64,
        n_gradients: int = 4,
        errors_per_gradient: int = 4,
        gradients_per_error: int = 1,
        steps_per_gradient: int = 1,
        mc_samples_per_step: int = 2,
        max_expansion_factor: int = 8,
    ) -> None:
        self.rounds = rounds
        self.beam_size = beam_size
        self.opt = {
            "minibatch_size": minibatch_size,
            "n_gradients": n_gradients,
            "errors_per_gradient": errors_per_gradient,
            "gradients_per_error": gradients_per_error,
            "steps_per_gradient": steps_per_gradient,
            "mc_samples_per_step": mc_samples_per_step,
            "max_expansion_factor": max_expansion_factor,
            "reject_on_errors": False,
        }

    def run(self, task: Task) -> OptimizerResult:
        seed = int(task.meta.get("seed", 42))
        random.seed(seed)
        if not task.seed_prompt:
            raise ValueError("APO/ProTeGi requires task.seed_prompt")
        if task.optimizer_llm is None:
            raise ValueError("APO/ProTeGi requires task.optimizer_llm")

        core = ProTeGiCore(self.opt, task.optimizer_llm)
        candidates = [seed_to_protegi(task.seed_prompt)]
        history: List[dict] = []
        scores: List[float] = []

        for rnd in range(self.rounds + 1):
            print(f"[apo] round {rnd}/{self.rounds} beam={len(candidates)}", flush=True)
            if rnd > 0:
                candidates = core.expand_candidates(
                    candidates,
                    task.train.texts,
                    task.train.labels,
                    task.scorer,
                    task.budget,
                )
                print(f"[apo] expanded to {len(candidates)} candidates", flush=True)
            # In-loop beam ranking on S_source (upstream scores train_exs).
            scores = _score_prompts_on(candidates, task, task.train, note="apo_train_beam")
            ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
            scores = [s for s, _ in ranked[: self.beam_size]]
            candidates = [p for _, p in ranked[: self.beam_size]]
            print(f"[apo] round {rnd} best_train={scores[0]:.4f}", flush=True)
            history.append(
                {
                    "round": rnd,
                    "beam_train_softmin": scores,
                    "beam_lens": [len(p) for p in candidates],
                }
            )

        # Final selection among beam on shared D_dev (ROADMAP §6.1 / §6.3).
        dev_scores = _score_prompts_on(candidates, task, task.dev, note="apo_dev_final")
        best_i = max(range(len(candidates)), key=lambda i: dev_scores[i])
        best_prompt = candidates[best_i]
        best_score = dev_scores[best_i]
        return OptimizerResult(
            best_prompt=best_prompt,
            all_candidates=list(candidates),
            trace={
                "method": "apo_protegi",
                "rounds": self.rounds,
                "beam_size": self.beam_size,
                "opt": self.opt,
                "history": history,
                "dev_softmin_beam": dev_scores,
                "best_softmin": best_score,
                "upstream": "baselines/_upstream/LMOps/prompt_optimization",
            },
            scorer_calls=task.budget.scorer_calls,
            optimizer_calls=task.budget.optimizer_calls,
        )

"""GPO (Li+ EMNLP 2023) — 3-step robust prompt optimization.

Faithful to ``baselines/_upstream/GPO/experiments/run_gpo.py`` + ``label_data``:
1. APE meta-prompt generate K=6 prompts from N=36 source shots
2. Prompt-ensemble label U_target; keep examples with agreement > T (default 0.83 ≈ 5/6)
3. Upsample G_t* to |U| (upstream) / mix with source and re-run APE; select on D_dev

Adaptation (ROADMAP §6.2): our Scorer + D_dev + F7 prompt contract. No invented
hyperparameters — K/N/T match the paper / roadmap.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import List, Optional, Sequence, Tuple

from baselines.ape import generate_ape_candidates, select_best_on_dev
from baselines.api import LabeledSet, OptimizerResult, Task


def _parse_binary_pred(raw_label: int) -> Optional[int]:
    if raw_label in (0, 1):
        return int(raw_label)
    return None


def _parse_ordinal_pred(raw_label: int) -> Optional[int]:
    if raw_label in (1, 2, 3, 4, 5):
        return int(raw_label)
    return None


def ensemble_label_unlabeled(
    prompts: Sequence[str],
    texts: Sequence[str],
    scorer,
    budget,
    *,
    conf_threshold: float = 0.83,
    note: str = "gpo_pseudo",
    label_space: str = "binary",
) -> Tuple[LabeledSet, dict]:
    """Prompt-ensemble labeling with confidence filter + upsample (upstream label_data).

    Uses our Scorer (not a separate inference stack). For each text, each of the K
    prompts produces a label; majority vote; confidence = majority_count / K.
    Keep examples with confidence > conf_threshold, then upsample to |U| as in
    ``exec_accuracy.label_data`` (conf_lower_bound path).
    """
    if not prompts or not texts:
        return LabeledSet(texts=[], labels=[]), {"kept": 0, "total": 0}

    parse = _parse_ordinal_pred if label_space == "ordinal5" else _parse_binary_pred

    # Vote matrix: prompt × example
    votes: List[List[Optional[int]]] = []
    for pi, p in enumerate(prompts):
        print(
            f"[gpo] ensemble-label prompt {pi + 1}/{len(prompts)} on |U|={len(texts)}",
            flush=True,
        )
        result = scorer.predict_batch(list(texts), p, labels_for_mock=None)
        budget.charge("val", n_calls=len(texts), kind="scorer", note=note)
        votes.append([parse(int(x)) for x in result.preds])

    k = len(prompts)
    kept_texts: List[str] = []
    kept_labels: List[int] = []
    confidences: List[float] = []
    for i, text in enumerate(texts):
        preds_i = [votes[j][i] for j in range(k) if votes[j][i] is not None]
        if not preds_i:
            confidences.append(0.0)
            continue
        cnt = Counter(preds_i)
        maj_label, maj_n = cnt.most_common(1)[0]
        conf = maj_n / float(k)
        confidences.append(conf)
        if conf > conf_threshold:
            kept_texts.append(text)
            kept_labels.append(int(maj_label))

    total = len(texts)
    kept = len(kept_texts)
    # Upsample to original |U| (upstream: rep_times + random fill).
    if 0 < kept < total:
        rng = random.Random(0)
        pairs = list(zip(kept_texts, kept_labels))
        rep = total // kept
        fill = total - kept * rep
        up = pairs * rep + rng.sample(pairs, fill)
        rng.shuffle(up)
        kept_texts = [t for t, _ in up]
        kept_labels = [y for _, y in up]
    elif kept == 0:
        # Degenerate: fall back to empty (joint APE will use source only).
        kept_texts, kept_labels = [], []

    trace = {
        "kept_before_upsample": kept,
        "total": total,
        "conf_threshold": conf_threshold,
        "mean_confidence": float(sum(confidences) / max(1, len(confidences))),
        "frac_above_T": float(kept / max(1, total)),
        "labeled_after_upsample": len(kept_texts),
    }
    return LabeledSet(texts=kept_texts, labels=kept_labels), trace


def _mix_source_target(
    source: LabeledSet,
    target_pseudo: LabeledSet,
    *,
    seed: int,
) -> LabeledSet:
    """Concatenate source + pseudo-labeled target and shuffle (GPO joint step)."""
    texts = list(source.texts) + list(target_pseudo.texts)
    labels = list(source.labels) + list(target_pseudo.labels)
    gids = None
    if source.group_ids is not None or target_pseudo.group_ids is not None:
        sg = source.group_ids or [0] * len(source.texts)
        tg = target_pseudo.group_ids or [0] * len(target_pseudo.texts)
        gids = list(sg) + list(tg)
    order = list(range(len(texts)))
    random.Random(seed).shuffle(order)
    return LabeledSet(
        texts=[texts[i] for i in order],
        labels=[labels[i] for i in order],
        group_ids=[gids[i] for i in order] if gids is not None else None,
    )


class GPOOptimizer:
    """Robust Prompt Optimization (Li+ EMNLP 2023) on shared baselines API."""

    def __init__(
        self,
        *,
        k_prompts: int = 6,
        n_shots: int = 36,
        conf_threshold: float = 0.83,
        max_tokens: int = 100,
        temperature: float = 0.0,
    ) -> None:
        self.k_prompts = k_prompts
        self.n_shots = n_shots
        self.conf_threshold = conf_threshold
        self.max_tokens = max_tokens
        self.temperature = temperature

    def run(self, task: Task) -> OptimizerResult:
        seed = int(task.meta.get("seed", 42))

        # Step 1: APE on S_source.
        print("[gpo] step1: APE generate on S_source", flush=True)
        stage1 = generate_ape_candidates(
            task.train,
            task.optimizer_llm,
            k_prompts=self.k_prompts,
            n_shots=self.n_shots,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            seed=seed,
            seed_prompt=task.seed_prompt,
            budget=task.budget,
        )
        print(f"[gpo] step1 done: {len(stage1)} candidates", flush=True)
        # Use K generated instructions for ensembling (exclude seed if appended).
        ensemble_prompts = stage1[: self.k_prompts]
        if len(ensemble_prompts) < self.k_prompts:
            ensemble_prompts = stage1

        # Step 2: ensemble-label U_target.
        u_texts = list(task.unlabeled.texts)
        if not u_texts:
            raise ValueError("GPO requires task.unlabeled (U_target) texts")
        pseudo, label_trace = ensemble_label_unlabeled(
            ensemble_prompts,
            u_texts,
            task.scorer,
            task.budget,
            conf_threshold=self.conf_threshold,
            label_space=str(task.meta.get("label_space", getattr(task.scorer, "label_space", "binary"))),
        )

        # Step 3: joint APE on mixed source + pseudo-labeled target.
        print(f"[gpo] step2 done: {label_trace}", flush=True)
        if pseudo.texts:
            mixed = _mix_source_target(task.train, pseudo, seed=seed + 1)
        else:
            mixed = task.train
        print(f"[gpo] step3: joint APE on mixed n={len(mixed.texts)}", flush=True)
        stage3 = generate_ape_candidates(
            mixed,
            task.optimizer_llm,
            k_prompts=self.k_prompts,
            n_shots=self.n_shots,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            seed=seed + 2,
            seed_prompt=task.seed_prompt,
            budget=task.budget,
        )

        best_prompt, best_score, scored = select_best_on_dev(
            stage3, task, note="gpo_dev"
        )
        return OptimizerResult(
            best_prompt=best_prompt,
            all_candidates=list(stage3),
            trace={
                "method": "gpo",
                "k": self.k_prompts,
                "n_shots": self.n_shots,
                "conf_threshold": self.conf_threshold,
                "stage1_n": len(stage1),
                "label": label_trace,
                "mixed_n": len(mixed.texts),
                "scored": scored,
                "best_softmin": best_score,
                "upstream": "baselines/_upstream/GPO/experiments/run_gpo.py",
            },
            scorer_calls=task.budget.scorer_calls,
            optimizer_calls=task.budget.optimizer_calls,
        )

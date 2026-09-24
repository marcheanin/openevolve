"""Yelp / Flipkart loaders for §6.4 GPO reimplementation gate.

Uses upstream GPO JSON under ``baselines/_upstream/GPO/experiments/data/...``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from baselines.api import LabeledSet, UnlabeledSet

_UPSTREAM = (
    Path(__file__).resolve().parent
    / "_upstream"
    / "GPO"
    / "experiments"
    / "data"
    / "instruction_induction"
)

SENTIMENT_SEED_PROMPT = """<System>
    <Role>You are a binary sentiment classifier for product reviews.</Role>
    <BaseGuidelines>
        - Output a single binary label only: 0 (negative) or 1 (positive)
        - Format: "Label: 0" or "Label: 1" (or just "0" / "1")
        - Positive = favorable, satisfied, recommending
        - Negative = unfavorable, dissatisfied, criticizing
    </BaseGuidelines>
    <DynamicRules>
        Decide polarity from overall review tone.
        When unsure, prefer the dominant sentiment.
    </DynamicRules>
</System>

<Task>
    Review: {review}
</Task>
"""


def _label_to_int(raw) -> Optional[int]:
    if isinstance(raw, list):
        raw = raw[0] if raw else ""
    s = str(raw).strip().lower()
    if s in ("positive", "pos", "1"):
        return 1
    if s in ("negative", "neg", "0"):
        return 0
    if s in ("neutral", "mixed", ""):
        return None
    try:
        v = int(float(s))
        return 1 if v > 0 else 0
    except Exception:
        return None


def load_task_examples(
    task: str, *, split: str = "induce", max_n: int = 5000
) -> Tuple[List[str], List[int]]:
    """Load up to ``max_n`` binary examples from induce/ or execute/ JSON."""
    base = _UPSTREAM / "raw" / ("induce" if split == "induce" else "execute")
    path = base / f"{task}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    examples = data["examples"]
    texts: List[str] = []
    labels: List[int] = []
    keys = sorted((k for k in examples.keys() if str(k).isdigit()), key=lambda x: int(x))
    if not keys:
        keys = list(examples.keys())
    for k in keys:
        if len(texts) >= max_n:
            break
        ex = examples.get(k)
        if ex is None:
            continue
        y = _label_to_int(ex["output"])
        if y is None:
            continue
        texts.append(str(ex["input"]))
        labels.append(int(y))
    return texts, labels


def subsample_labeled(
    texts: Sequence[str],
    labels: Sequence[int],
    *,
    n: int,
    seed: int,
) -> LabeledSet:
    rng = random.Random(seed)
    order = list(range(len(texts)))
    rng.shuffle(order)
    order = order[:n]
    return LabeledSet(
        texts=[texts[i] for i in order],
        labels=[labels[i] for i in order],
        group_ids=[0] * len(order),
        example_ids=list(order),
    )


def build_yelp_flipkart_gate_sets(
    *,
    seed: int = 0,
    n_source_train: int = 100,
    n_source_eval: int = 100,
    n_target_unlabeled: int = 400,
    n_target_eval: int = 100,
) -> dict:
    """Materialize gate pools from induce (execute is tiny after binary filter)."""
    yt, yl = load_task_examples("yelp", split="induce", max_n=3000)
    ft, fl = load_task_examples("flipkart", split="induce", max_n=3000)
    if len(yt) < n_source_train + n_source_eval:
        raise ValueError(f"Yelp binary pool too small: {len(yt)}")
    if len(ft) < n_target_unlabeled + n_target_eval:
        # Cap unlabeled to available.
        n_target_unlabeled = max(50, len(ft) - n_target_eval)

    source_train = subsample_labeled(yt, yl, n=n_source_train, seed=seed)
    source_eval = subsample_labeled(yt, yl, n=n_source_eval, seed=seed + 1)
    target_eval = subsample_labeled(ft, fl, n=n_target_eval, seed=seed + 2)
    u_lab = subsample_labeled(ft, fl, n=n_target_unlabeled, seed=seed + 3)
    unlabeled = UnlabeledSet(
        texts=list(u_lab.texts),
        group_ids=u_lab.group_ids,
        example_ids=u_lab.example_ids,
    )
    return {
        "source_train": source_train,
        "source_eval": source_eval,
        "target_eval": target_eval,
        "unlabeled": unlabeled,
        "seed_prompt": SENTIMENT_SEED_PROMPT,
        "n_yelp": len(yt),
        "n_flipkart": len(ft),
    }


def accuracy_on_set(scorer, prompt: str, labeled: LabeledSet) -> float:
    """Simple accuracy for sentiment gate (not GBA)."""
    result = scorer.predict_batch(list(labeled.texts), prompt, labels_for_mock=labeled.labels)
    preds = [int(p) for p in result.preds]
    gold = list(labeled.labels)
    ok = sum(1 for a, b in zip(preds, gold) if a == b and a in (0, 1))
    return ok / max(1, len(gold))

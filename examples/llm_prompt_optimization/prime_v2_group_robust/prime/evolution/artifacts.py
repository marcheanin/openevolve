"""Error artifacts with per-cluster breakdown for mutator feedback."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence


def format_error_artifacts(
    predictions: Sequence[int],
    gold: Sequence[int],
    worker_preds: Sequence[Sequence[int]],
    texts: Sequence[str],
    cluster_ids: Optional[Sequence[int]] = None,
    hard_indices: Optional[Sequence[int]] = None,
    anchor_indices: Optional[Sequence[int]] = None,
    max_examples: int = 8,
    max_text_len: int = 200,
) -> str:
    """Build style-conditional error report for OpenEvolve mutator."""
    errors: List[tuple] = []
    borderline: List[tuple] = []
    by_cluster: Dict[int, List[tuple]] = defaultdict(list)

    for i, (pred, label, text) in enumerate(zip(predictions, gold, texts)):
        wp = [worker_preds[w][i] for w in range(len(worker_preds))]
        d = _pair_disagreement(wp)
        entry = (i, text, label, pred, wp, d)
        cid = int(cluster_ids[i]) if cluster_ids is not None else 0
        if pred != label:
            errors.append(entry)
            by_cluster[cid].append(entry)
        elif d > 0.25:
            borderline.append(entry)

    lines: List[str] = []
    if by_cluster:
        lines.append("PER-CLUSTER ERROR BREAKDOWN:")
        for cid in sorted(by_cluster.keys()):
            n_err = len(by_cluster[cid])
            lines.append(f"  Cluster {cid}: {n_err} errors")
            for j, (idx, text, label, pred, wp, d) in enumerate(by_cluster[cid][:3]):
                t = text[:max_text_len] + ("..." if len(text) > max_text_len else "")
                lines.append(f"    {j+1}. gold={label} pred={pred} workers={wp} d={d:.2f}")
                lines.append(f"       \"{t}\"")
        lines.append("STYLE HINT: add cluster-specific rules for clusters with most errors.")

    if errors:
        lines.append("\nHARD ERRORS:")
        for j, (_, text, label, pred, wp, d) in enumerate(errors[:max_examples]):
            t = text[:max_text_len] + ("..." if len(text) > max_text_len else "")
            lines.append(f"  {j+1}. gold={label} pred={pred} workers={wp} d={d:.2f}")
            lines.append(f"     \"{t}\"")

    if borderline:
        lines.append("\nBORDERLINE (correct but high disagreement):")
        for j, (_, text, label, pred, wp, d) in enumerate(borderline[:5]):
            t = text[:max_text_len] + ("..." if len(text) > max_text_len else "")
            lines.append(f"  {j+1}. gold={label} pred={pred} workers={wp} d={d:.2f}")

    if anchor_indices:
        anchor_set = set(anchor_indices)
        regressions = [
            (i, texts[i], gold[i], predictions[i])
            for i in range(len(predictions))
            if i in anchor_set and predictions[i] != gold[i]
        ]
        if regressions:
            lines.append("\nANCHOR REGRESSIONS (do not break these):")
            for j, (_, text, label, pred) in enumerate(regressions[:5]):
                t = text[:max_text_len] + ("..." if len(text) > max_text_len else "")
                lines.append(f"  {j+1}. gold={label} pred={pred} \"{t}\"")

    if lines:
        lines.append(
            "\nTARGET BLOCK: DynamicRules\n"
            "SCOPED MUTATION RULES:\n"
            "  - Mutate exactly ONE block per mutation (prefer DynamicRules).\n"
            "  - Put type-/cluster-conditional rules into <DynamicRules> only.\n"
            "  - Do NOT rewrite <BaseGuidelines> inside the AL cycle "
            "(BaseGuidelines are consolidated separately).\n"
            "  - Keep <Task> with {review} unchanged; preserve XML tags.\n"
        )
        lines.append(f"\nSummary: {len(errors)} errors, {len(borderline)} borderline / {len(predictions)} examples.")
    return "\n".join(lines)


def _pair_disagreement(votes: List[int]) -> float:
    if len(votes) < 2:
        return 0.0
    total = 0.0
    n = len(votes)
    for i in range(n):
        for j in range(i + 1, n):
            total += abs(votes[i] - votes[j])
    return total / (n * (n - 1) / 2) / 4.0

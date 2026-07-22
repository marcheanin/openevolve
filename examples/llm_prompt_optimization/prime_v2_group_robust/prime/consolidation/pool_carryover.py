"""Pool carryover and consolidation between AL cycles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PromptRecord:
    prompt: str
    fitness: float
    metrics: Dict[str, float] = field(default_factory=dict)
    cluster_scores: Optional[Dict[int, float]] = None


def select_carryover(
    archive: List[PromptRecord],
    k: int,
    cluster_pareto: bool = False,
) -> List[PromptRecord]:
    """Select top-k prompts for next cycle seed population."""
    if not archive:
        return []
    if cluster_pareto:
        return _cluster_pareto_select(archive, k)
    return sorted(archive, key=lambda r: r.fitness, reverse=True)[:k]


def _cluster_pareto_select(archive: List[PromptRecord], k: int) -> List[PromptRecord]:
    """One prompt per cluster with best cluster-local score, then fill by fitness."""
    by_cluster: Dict[int, PromptRecord] = {}
    for rec in sorted(archive, key=lambda r: r.fitness, reverse=True):
        if not rec.cluster_scores:
            continue
        for cid, acc in rec.cluster_scores.items():
            prev = by_cluster.get(cid)
            if prev is None or acc > (prev.cluster_scores or {}).get(cid, 0.0):
                by_cluster[cid] = rec
    picked = list({id(r): r for r in by_cluster.values()}.values())
    picked.sort(key=lambda r: r.fitness, reverse=True)
    if len(picked) >= k:
        return picked[:k]
    remaining = [r for r in sorted(archive, key=lambda r: r.fitness, reverse=True) if r not in picked]
    picked.extend(remaining[: k - len(picked)])
    return picked[:k]


def merge_with_consolidated(
    carryover: List[PromptRecord],
    consolidated: Optional[PromptRecord],
) -> List[str]:
    """Return prompt strings: carryover + consolidated as candidate (not sole heir)."""
    prompts = [r.prompt for r in carryover]
    if consolidated and consolidated.prompt not in prompts:
        prompts.append(consolidated.prompt)
    return prompts


def gate_consolidation(
    consolidated: PromptRecord,
    best: PromptRecord,
    delta: float,
) -> bool:
    """Accept consolidated prompt only if within delta of best fitness."""
    return consolidated.fitness >= best.fitness - delta

"""
Cross-cycle Pareto front over per-cluster accuracy (SPEC v3 §4.6 extension).

Why this exists (OBSERVATIONS O9/O10/O11): OpenEvolve fills 5-6 MAP-Elites cells
per AL cycle, but the old loop kept exactly one heir — the consolidated prompt,
by construction, whether or not it was better. Measured cost on the completed
cvar_lex run: the consolidated prompt lost 0.015 and 0.066 of D_select fitness,
i.e. more than the evolution had gained.

The front here is over the **per-cluster accuracy vector on D_select**, which is
the quantity `cvar_lex` cares about. A prompt that is the best available on the
weakest cluster survives even if its scalar fitness is not the highest, so
group specialization is no longer discarded between cycles.

Since P9 (2026-07-29) the carried set is a **champion archive** with fixed
slots (`champion_archive`), not a dominance front: with 5-6 noisy axes almost
everything is non-dominated and `pareto_front`+`trim_front` filled the carried
set with lucky prompts while evicting the scalar-best (O18). The dominance
functions are kept for analysis/ablation.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def prompt_fingerprint(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


@dataclass
class Candidate:
    """A prompt with its D_select measurements. Immutable within a cycle."""

    prompt: str
    fitness: float
    cluster_scores: Dict[int, float] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"  # cycle_entry | oe | consolidated | carried
    cycle: int = 0
    program_id: Optional[str] = None
    # Where this prompt was first produced. Carried members make a round trip
    # through the OpenEvolve seed checkpoint and come back tagged `source="oe"`
    # with the current cycle, which made a carried prompt indistinguishable from a
    # fresh mutation in every artifact (OBSERVATIONS O23). These two fields survive
    # that round trip.
    origin_source: Optional[str] = None
    origin_cycle: Optional[int] = None

    @property
    def birth(self) -> Tuple[str, int]:
        return (self.origin_source or self.source, self.origin_cycle if self.origin_cycle is not None else self.cycle)

    def score_vector(self, cluster_ids: Sequence[int]) -> Tuple[float, ...]:
        return tuple(float(self.cluster_scores.get(c, 0.0)) for c in cluster_ids)


def cluster_axis(candidates: Sequence[Candidate]) -> List[int]:
    """Union of cluster ids seen across candidates, sorted for stable ordering."""
    ids: set = set()
    for c in candidates:
        ids.update(int(k) for k in c.cluster_scores)
    return sorted(ids)


def _dominates(a: Tuple[float, ...], b: Tuple[float, ...], eps: float) -> bool:
    """`a` dominates `b`: no axis meaningfully worse and at least one better."""
    better = False
    for x, y in zip(a, b):
        if x < y - eps:
            return False
        if x > y + eps:
            better = True
    return better


def pareto_front(
    candidates: Sequence[Candidate],
    eps: float = 1e-9,
) -> List[Candidate]:
    """
    Non-dominated set over the per-cluster accuracy vector.

    Duplicate prompts are collapsed (first occurrence wins). Candidates without
    cluster scores cannot be compared on the front, so they are ranked by scalar
    fitness only and appended after the front proper.
    """
    unique: List[Candidate] = []
    seen_prompts: set = set()
    for c in sorted(candidates, key=lambda r: r.fitness, reverse=True):
        if c.prompt in seen_prompts:
            continue
        seen_prompts.add(c.prompt)
        unique.append(c)

    scored = [c for c in unique if c.cluster_scores]
    unscored = [c for c in unique if not c.cluster_scores]
    if not scored:
        return unscored

    axis = cluster_axis(scored)
    vectors = {id(c): c.score_vector(axis) for c in scored}
    front: List[Candidate] = []
    for c in scored:
        vc = vectors[id(c)]
        if not any(_dominates(vectors[id(o)], vc, eps) for o in scored if o is not c):
            front.append(c)
    front.sort(key=lambda r: r.fitness, reverse=True)
    return front + unscored


def trim_front(front: Sequence[Candidate], k: int) -> List[Candidate]:
    """
    Cap the front at `k` while keeping cluster coverage.

    The scalar-best member always keeps its slot: on the 20260729 run the greedy
    champion pass filled all k slots before reaching the best OE program (0.582),
    so it was evicted and the consolidated stub (0.555) inherited the cycle
    (OBSERVATIONS O18). Then take the best candidate on each cluster axis (so no
    weak group loses its champion), then fill remaining slots by scalar fitness.
    """
    if k <= 0 or len(front) <= k:
        return list(front)
    axis = cluster_axis(front)
    picked: List[Candidate] = []
    picked_ids: set = set()
    scalar_best = max(front, key=lambda r: (r.fitness, r.cycle))
    picked.append(scalar_best)
    picked_ids.add(id(scalar_best))
    for cid in sorted(axis, key=lambda c: min(f.cluster_scores.get(c, 1.0) for f in front)):
        champion = max(front, key=lambda r: (r.cluster_scores.get(cid, 0.0), r.fitness))
        if id(champion) not in picked_ids:
            picked.append(champion)
            picked_ids.add(id(champion))
        if len(picked) >= k:
            return picked
    for c in sorted(front, key=lambda r: r.fitness, reverse=True):
        if id(c) in picked_ids:
            continue
        picked.append(c)
        picked_ids.add(id(c))
        if len(picked) >= k:
            break
    return picked


def champion_archive(
    candidates: Sequence[Candidate],
    *,
    prev_champions: Optional[Dict[int, str]] = None,
    margin: float = 0.01,
) -> Tuple[List[Candidate], Dict[int, str]]:
    """
    Fixed-slot archive replacing strict Pareto dominance (OBSERVATIONS P9).

    Slots: [0] the scalar-best candidate (never evicted — O18), then one slot
    per cluster holding that cluster's champion. With 5-6 noisy axes almost
    every candidate is non-dominated, so a dominance front fills with lucky
    prompts; fixed slots implement the actual intent — "keep the best overall
    prompt and the best prompt per group".

    Anti-churn: `cluster_scores` are already Beta-shrunk (the OE evaluator and
    the controller both store `cluster_accuracies_shrunk`), and an incumbent
    champion keeps its slot unless a challenger beats it by more than `margin`.
    Otherwise slot identity flips on per-cycle noise and the OE seed population
    never stabilizes.

    Returns the archive (scalar-best first, champions in cluster order, deduped
    by prompt) and the new `{cluster_id: champion_prompt}` mapping to pass back
    next cycle.
    """
    prev_champions = prev_champions or {}

    unique: List[Candidate] = []
    seen_prompts: set = set()
    for c in sorted(candidates, key=lambda r: (r.fitness, r.cycle), reverse=True):
        if c.prompt in seen_prompts:
            continue
        seen_prompts.add(c.prompt)
        unique.append(c)
    if not unique:
        return [], {}

    scalar_best = unique[0]
    scored = [c for c in unique if c.cluster_scores]
    axis = cluster_axis(scored)

    champions: Dict[int, Candidate] = {}
    for cid in axis:
        challenger = max(scored, key=lambda r: (r.cluster_scores.get(cid, 0.0), r.fitness))
        incumbent = next((c for c in scored if c.prompt == prev_champions.get(cid)), None)
        if (
            incumbent is not None
            and challenger.prompt != incumbent.prompt
            and challenger.cluster_scores.get(cid, 0.0)
            <= incumbent.cluster_scores.get(cid, 0.0) + margin
        ):
            champions[cid] = incumbent
        else:
            champions[cid] = challenger

    archive: List[Candidate] = [scalar_best]
    archive_prompts = {scalar_best.prompt}
    for cid in sorted(champions):
        cand = champions[cid]
        if cand.prompt not in archive_prompts:
            archive.append(cand)
            archive_prompts.add(cand.prompt)
    return archive, {cid: c.prompt for cid, c in champions.items()}


def front_summary(front: Sequence[Candidate]) -> Dict[str, Any]:
    """Compact, loggable description of a front."""
    axis = cluster_axis(front)
    return {
        "size": len(front),
        "cluster_axis": axis,
        "sources": sorted({c.source for c in front}),
        "best_fitness": max((c.fitness for c in front), default=0.0),
        "cluster_champions": {
            str(cid): {
                "source": max(front, key=lambda r: r.cluster_scores.get(cid, 0.0)).source,
                "acc": round(max(c.cluster_scores.get(cid, 0.0) for c in front), 4),
            }
            for cid in axis
        },
        "members": [
            {
                "source": c.source,
                "cycle": c.cycle,
                "origin_source": c.birth[0],
                "origin_cycle": c.birth[1],
                "fitness": round(c.fitness, 6),
                "prompt_len": len(c.prompt),
                "cluster_scores": {str(k): round(v, 4) for k, v in sorted(c.cluster_scores.items())},
                "program_id": c.program_id,
            }
            for c in front
        ],
    }


# ---------------------------------------------------------------- OE interop


def latest_checkpoint(oe_staging: Path) -> Optional[Path]:
    """Highest-numbered `checkpoints/checkpoint_N` under an OE staging dir."""
    root = oe_staging / "checkpoints"
    if not root.is_dir():
        return None
    best: Optional[Path] = None
    best_n = -1
    for d in root.iterdir():
        if not d.is_dir() or not d.name.startswith("checkpoint_"):
            continue
        try:
            n = int(d.name.split("_")[-1])
        except ValueError:
            continue
        if n > best_n and (d / "programs").is_dir():
            best_n, best = n, d
    return best


def _apply_candidate_metrics(data: Dict[str, Any], cand: Candidate) -> None:
    """Force OE program metrics to match Candidate (current D_select). O26/M27."""
    metrics = dict(data.get("metrics") or {})
    metrics["combined_score"] = float(cand.fitness)
    metrics["fitness"] = float(cand.fitness)
    for key in list(metrics):
        if key.startswith("cluster_acc_"):
            del metrics[key]
    for cid, acc in cand.cluster_scores.items():
        metrics[f"cluster_acc_{int(cid)}"] = float(acc)
    for key, val in (cand.metrics or {}).items():
        if key.startswith("cluster_acc_"):
            continue
        metrics[key] = val
    data["metrics"] = metrics
    # Keep code aligned with the Candidate prompt (strip markers already applied).
    if cand.prompt:
        data["code"] = cand.prompt


def reconcile_oe_with_carried(
    carried: Sequence[Candidate],
    oe_candidates: Sequence[Candidate],
) -> List[Candidate]:
    """
    Prefer current-D_select scores from ``carried`` over OE disk metrics (M27).

    Seeded OE programs that are byte-identical to a carried champion keep OE's
    provenance tags but inherit the O26-rescored fitness/cluster scores. Fresh
    mutations (prompt not in carried) keep their OE scores unchanged — those
    were measured on the current D_select during this cycle.
    """
    by_prompt = {c.prompt: c for c in carried}
    out: List[Candidate] = []
    for c in oe_candidates:
        honest = by_prompt.get(c.prompt)
        if honest is None:
            out.append(c)
            continue
        out.append(
            Candidate(
                prompt=c.prompt,
                fitness=float(honest.fitness),
                cluster_scores=dict(honest.cluster_scores or c.cluster_scores),
                metrics={
                    **(c.metrics or {}),
                    **(honest.metrics or {}),
                    "reconciled_from_carried": True,
                },
                source=c.source,
                cycle=c.cycle,
                program_id=c.program_id or honest.program_id,
                origin_source=c.origin_source if c.origin_source is not None else honest.origin_source,
                origin_cycle=c.origin_cycle if c.origin_cycle is not None else honest.origin_cycle,
            )
        )
    return out


def sync_seed_checkpoint_metrics(
    seed_dir: Path,
    front: Sequence[Candidate],
) -> int:
    """
    Rewrite ``combined_score`` / cluster metrics in an existing seed checkpoint
    from the (re)scored front. Returns number of program files updated.

    Called after D_select rotation (O26) so OE resume does not trust stale
    island best from the previous set (M27).
    """
    programs = seed_dir / "programs"
    if not programs.is_dir():
        return 0
    by_id = {c.program_id: c for c in front if c.program_id}
    by_hash = {prompt_fingerprint(c.prompt): c for c in front}
    updated = 0
    for path in programs.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        cand = by_id.get(str(data.get("id") or path.stem))
        if cand is None:
            origin = ((data.get("metadata") or {}).get("prime_origin")) or {}
            cand = by_hash.get(origin.get("hash") or "")
        if cand is None:
            code = data.get("code") or ""
            cand = by_hash.get(prompt_fingerprint(code))
        if cand is None:
            continue
        _apply_candidate_metrics(data, cand)
        path.write_text(json.dumps(data), encoding="utf-8")
        updated += 1
    # Keep island best pointing at the scalar-best among kept ids if possible.
    meta_path = seed_dir / "metadata.json"
    if updated and meta_path.is_file() and front:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return updated
        best = max(front, key=lambda c: (c.fitness, c.cycle))
        if best.program_id and best.program_id in (meta.get("archive") or []):
            meta["best_program_id"] = best.program_id
            meta["island_best_programs"] = [best.program_id]
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return updated


def read_oe_candidates(
    oe_staging: Path,
    cycle: int,
    strip_markers,
) -> List[Candidate]:
    """
    Every program OpenEvolve evaluated this cycle, with its D_select metrics.

    These come for free: the evaluator already measured `cluster_acc_*` on the
    same fixed D_select, so building the front costs zero API calls.
    """
    cp = latest_checkpoint(oe_staging)
    if cp is None:
        return []
    out: List[Candidate] = []
    for path in sorted((cp / "programs").glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        code = strip_markers(data.get("code") or "")
        metrics = data.get("metrics") or {}
        if not code or "combined_score" not in metrics:
            continue
        cluster_scores = {
            int(key.rsplit("_", 1)[-1]): float(val)
            for key, val in metrics.items()
            if key.startswith("cluster_acc_")
        }
        # A seeded member that came back byte-identical is a carried prompt, not a
        # new mutation; anything else is a genuine descendant of this cycle.
        origin = ((data.get("metadata") or {}).get("prime_origin")) or {}
        unchanged = bool(origin) and origin.get("hash") == prompt_fingerprint(code)
        out.append(
            Candidate(
                prompt=code,
                fitness=float(metrics.get("combined_score", 0.0)),
                cluster_scores=cluster_scores,
                metrics={
                    k: v
                    for k, v in metrics.items()
                    if not k.startswith("cluster_acc_")
                },
                source="carried" if unchanged else "oe",
                cycle=cycle,
                program_id=str(data.get("id") or path.stem),
                origin_source=origin.get("source"),
                origin_cycle=origin.get("cycle"),
            )
        )
    return out


def write_seed_checkpoint(
    front: Sequence[Candidate],
    source_staging: Path,
    dest: Path,
) -> Optional[Path]:
    """
    Materialize `front` as an OpenEvolve checkpoint the next cycle can resume.

    Program JSONs are copied from the checkpoint OpenEvolve itself wrote
    (so `Program.from_dict` is guaranteed to accept them), then metrics are
    overwritten from ``Candidate.fitness`` / cluster scores (O26/M27 — never
    trust verbatim OE metrics across a D_select rotation). Metadata is rewritten
    with `last_iteration = 0` so the next cycle spends its full iteration budget.
    Members without a `program_id` (the consolidated prompt, a carried prompt) are
    written as fresh program records.

    Returns the checkpoint dir, or None if nothing usable could be written.
    """
    src_cp = latest_checkpoint(source_staging)
    if src_cp is None:
        return None
    src_programs = src_cp / "programs"

    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    (dest / "programs").mkdir(parents=True, exist_ok=True)

    kept_ids: List[str] = []
    for cand in front:
        pid = cand.program_id
        src = (src_programs / f"{pid}.json") if pid else None
        if src is not None and src.is_file():
            try:
                data = json.loads(src.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
        else:
            pid = str(uuid.uuid4())
            data = {
                "id": pid,
                "code": cand.prompt,
                "language": "text",
                "parent_id": None,
                "generation": 0,
                "iteration_found": 0,
                "metrics": {},
                "metadata": {"prime_source": cand.source, "prime_cycle": cand.cycle},
                "changes_description": f"carried from cycle {cand.cycle} ({cand.source})",
            }
        _apply_candidate_metrics(data, cand)
        data["parent_id"] = None
        data["generation"] = 0
        data["iteration_found"] = 0
        # Immutable birth stamp, so the next cycle can tell a carried member from a
        # fresh mutation even though both arrive as OpenEvolve programs (O23).
        origin_source, origin_cycle = cand.birth
        metadata = dict(data.get("metadata") or {})
        metadata["prime_origin"] = {
            "source": origin_source,
            "cycle": origin_cycle,
            "hash": prompt_fingerprint(cand.prompt),
        }
        data["metadata"] = metadata
        (dest / "programs" / f"{data['id']}.json").write_text(
            json.dumps(data), encoding="utf-8"
        )
        kept_ids.append(str(data["id"]))

    if not kept_ids:
        shutil.rmtree(dest, ignore_errors=True)
        return None

    best_id = kept_ids[0]
    metadata = {
        "island_feature_maps": [{}],
        "islands": [kept_ids],
        "archive": kept_ids,
        "best_program_id": best_id,
        "island_best_programs": [best_id],
        "last_iteration": 0,
        "current_island": 0,
        "island_generations": [0],
        "last_migration_generation": 0,
        "feature_stats": {},
    }
    (dest / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return dest

"""M27: seed-checkpoint metrics must match current-D_select Candidate scores."""

from __future__ import annotations

import json
from pathlib import Path

from prime.consolidation.pareto import (
    Candidate,
    champion_archive,
    prompt_fingerprint,
    read_oe_candidates,
    reconcile_oe_with_carried,
    sync_seed_checkpoint_metrics,
    write_seed_checkpoint,
)
from prime.evolution.prompt_blocks import strip_evolve_markers


def _cand(prompt, fitness, scores, source="oe", pid=None, cycle=1):
    return Candidate(
        prompt=prompt,
        fitness=fitness,
        cluster_scores=scores,
        source=source,
        program_id=pid,
        cycle=cycle,
        origin_source=source,
        origin_cycle=cycle,
    )


def _fake_oe_staging(root: Path, programs: dict[str, dict]) -> Path:
    cp = root / "checkpoints" / "checkpoint_3" / "programs"
    cp.mkdir(parents=True, exist_ok=True)
    for pid, payload in programs.items():
        (cp / f"{pid}.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


def test_write_seed_checkpoint_overwrites_copied_metrics(tmp_path: Path):
    """Copied OE JSON must not keep stale combined_score (O26/M27)."""
    pid = "prog-stale"
    staging = _fake_oe_staging(
        tmp_path / "oe",
        {
            pid: {
                "id": pid,
                "code": "PROMPT_A",
                "language": "text",
                "metrics": {
                    "combined_score": 0.674,
                    "fitness": 0.674,
                    "cluster_acc_0": 0.9,
                },
                "metadata": {},
            }
        },
    )
    # Current D_select says this prompt is only 0.593
    front = [_cand("PROMPT_A", 0.593, {0: 0.70, 1: 0.55}, source="carried", pid=pid)]
    dest = tmp_path / "seed_c2"
    assert write_seed_checkpoint(front, staging, dest) == dest
    written = json.loads((dest / "programs" / f"{pid}.json").read_text(encoding="utf-8"))
    assert written["metrics"]["combined_score"] == 0.593
    assert written["metrics"]["fitness"] == 0.593
    assert written["metrics"]["cluster_acc_0"] == 0.70
    assert written["metrics"]["cluster_acc_1"] == 0.55
    assert written["code"] == "PROMPT_A"


def test_sync_seed_checkpoint_metrics_updates_disk(tmp_path: Path):
    pid = "champ"
    seed = tmp_path / "seed_c2"
    (seed / "programs").mkdir(parents=True)
    (seed / "programs" / f"{pid}.json").write_text(
        json.dumps(
            {
                "id": pid,
                "code": "CHAMP",
                "metrics": {"combined_score": 0.674, "cluster_acc_0": 0.9},
                "metadata": {
                    "prime_origin": {
                        "source": "oe",
                        "cycle": 1,
                        "hash": prompt_fingerprint("CHAMP"),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (seed / "metadata.json").write_text(
        json.dumps(
            {
                "archive": [pid],
                "best_program_id": pid,
                "island_best_programs": [pid],
                "islands": [[pid]],
            }
        ),
        encoding="utf-8",
    )
    front = [_cand("CHAMP", 0.551, {0: 0.60}, source="carried", pid=pid, cycle=2)]
    n = sync_seed_checkpoint_metrics(seed, front)
    assert n == 1
    data = json.loads((seed / "programs" / f"{pid}.json").read_text(encoding="utf-8"))
    assert data["metrics"]["combined_score"] == 0.551
    assert data["metrics"]["cluster_acc_0"] == 0.60


def test_reconcile_oe_prefers_carried_fitness_over_stale_oe():
    carried = [_cand("SAME", 0.551, {0: 0.5}, source="carried", pid="c1")]
    stale_oe = [_cand("SAME", 0.674, {0: 0.9}, source="carried", pid="c1", cycle=2)]
    fresh = [_cand("NEW", 0.560, {0: 0.55}, source="oe", pid="n1", cycle=2)]
    out = reconcile_oe_with_carried(carried, stale_oe + fresh)
    by = {c.prompt: c for c in out}
    assert by["SAME"].fitness == 0.551
    assert by["SAME"].metrics.get("reconciled_from_carried") is True
    assert by["NEW"].fitness == 0.560  # fresh mutation untouched


def test_champion_archive_keeps_honest_score_after_reconcile():
    """Front assembly must not let stale OE beat honest carried on same prompt."""
    honest = _cand("P", 0.551, {0: 0.50, 1: 0.60}, source="carried", cycle=2)
    stale = _cand("P", 0.674, {0: 0.90, 1: 0.90}, source="oe", cycle=2)
    other = _cand("Q", 0.540, {0: 0.40, 1: 0.70}, source="oe", cycle=2)
    reconciled = reconcile_oe_with_carried([honest], [stale, other])
    archive, _slots = champion_archive([honest] + reconciled)
    assert archive[0].prompt == "P"
    assert archive[0].fitness == 0.551
    assert max(c.fitness for c in archive) == 0.551


def test_seed_roundtrip_preserves_rescored_fitness(tmp_path: Path):
    """write → read OE candidates → reconcile must keep current-set scores."""
    pid = "p1"
    staging = _fake_oe_staging(
        tmp_path / "oe",
        {
            pid: {
                "id": pid,
                "code": "BODY",
                "language": "text",
                "metrics": {"combined_score": 0.90, "cluster_acc_0": 0.99},
                "metadata": {},
            }
        },
    )
    front = [
        _cand(
            "BODY",
            0.60,
            {0: 0.55},
            source="oe",
            pid=pid,
            cycle=1,
        )
    ]
    seed = tmp_path / "seed_c2"
    write_seed_checkpoint(front, staging, seed)

    # Simulate OE copying seed into a new staging checkpoint for cycle 2
    oe2 = tmp_path / "oe2"
    cp = oe2 / "checkpoints" / "checkpoint_1" / "programs"
    cp.mkdir(parents=True)
    for src in (seed / "programs").glob("*.json"):
        (cp / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    reread = read_oe_candidates(oe2, cycle=2, strip_markers=strip_evolve_markers)
    assert len(reread) == 1
    assert reread[0].fitness == 0.60

    # Even if OE somehow still had 0.90, reconcile with carried wins
    carried = [_cand("BODY", 0.55, {0: 0.50}, source="carried", pid=pid, cycle=2)]
    stale = [_cand("BODY", 0.90, {0: 0.99}, source="carried", pid=pid, cycle=2)]
    fixed = reconcile_oe_with_carried(carried, stale)
    assert fixed[0].fitness == 0.55

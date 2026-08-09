import json

from prime.consolidation.pareto import (
    Candidate,
    champion_archive,
    cluster_axis,
    front_summary,
    latest_checkpoint,
    pareto_front,
    read_oe_candidates,
    trim_front,
    write_seed_checkpoint,
)
from prime.evolution.prompt_blocks import strip_evolve_markers


def _cand(name, fitness, scores, source="oe", pid=None):
    return Candidate(
        prompt=name, fitness=fitness, cluster_scores=scores, source=source, program_id=pid
    )


def test_dominated_candidate_is_dropped():
    strong = _cand("A", 0.60, {0: 0.7, 1: 0.7})
    weak = _cand("B", 0.55, {0: 0.6, 1: 0.6})
    front = pareto_front([strong, weak])
    assert [c.prompt for c in front] == ["A"]


def test_group_specialist_survives_lower_fitness():
    """The point of the front: a weak-cluster champion is not thrown away."""
    generalist = _cand("A", 0.62, {0: 0.50, 1: 0.80})
    specialist = _cand("B", 0.55, {0: 0.65, 1: 0.60})
    front = pareto_front([generalist, specialist])
    assert {c.prompt for c in front} == {"A", "B"}
    assert front[0].prompt == "A"  # heir order is still by scalar fitness


def test_duplicate_prompts_collapse():
    front = pareto_front([_cand("A", 0.6, {0: 0.7}), _cand("A", 0.5, {0: 0.7})])
    assert len(front) == 1
    assert front[0].fitness == 0.6


def test_candidates_without_cluster_scores_are_appended():
    front = pareto_front([_cand("A", 0.6, {0: 0.7}), _cand("B", 0.9, {})])
    assert [c.prompt for c in front] == ["A", "B"]


def test_trim_front_keeps_weakest_cluster_champion():
    generalist = _cand("A", 0.70, {0: 0.40, 1: 0.90})
    tail_champ = _cand("B", 0.55, {0: 0.62, 1: 0.55})
    middle = _cand("C", 0.60, {0: 0.50, 1: 0.70})
    kept = trim_front(pareto_front([generalist, tail_champ, middle]), 2)
    assert len(kept) == 2
    assert "B" in {c.prompt for c in kept}


def test_trim_front_never_drops_scalar_best():
    """Regression for O18: cluster champions filled all slots and evicted the
    scalar-best OE program, so the consolidated stub inherited the cycle."""
    best = _cand("BEST", 0.582, {0: 0.55, 1: 0.55, 2: 0.55, 3: 0.55})
    champs = [
        _cand(f"CH{i}", 0.50 + i / 100, {j: (0.9 if j == i else 0.4) for j in range(4)})
        for i in range(4)
    ]
    kept = trim_front(pareto_front([best] + champs), 4)
    assert "BEST" in {c.prompt for c in kept}
    assert len(kept) == 4


def _cyc(name, fitness, scores, cycle, source="oe"):
    return Candidate(
        prompt=name, fitness=fitness, cluster_scores=scores, source=source, cycle=cycle
    )


def test_champion_archive_scalar_best_is_slot_zero():
    best = _cand("BEST", 0.60, {0: 0.55, 1: 0.55})
    spec = _cand("SPEC", 0.50, {0: 0.80, 1: 0.40})
    archive, slots = champion_archive([spec, best])
    assert archive[0].prompt == "BEST"
    assert slots[0] == "SPEC"  # cluster 0 champion
    assert slots[1] == "BEST"
    assert {c.prompt for c in archive} == {"BEST", "SPEC"}


def test_champion_archive_incumbent_keeps_slot_within_margin():
    """Anti-churn: a challenger inside the margin does not steal the slot."""
    incumbent = _cyc("OLD", 0.55, {0: 0.700}, cycle=1)
    challenger = _cyc("NEW", 0.60, {0: 0.705}, cycle=2)
    archive, slots = champion_archive(
        [incumbent, challenger], prev_champions={0: "OLD"}, margin=0.01
    )
    assert slots[0] == "OLD"
    # the challenger is still scalar-best, so it stays in the archive anyway
    assert archive[0].prompt == "NEW"


def test_champion_archive_incumbent_replaced_beyond_margin():
    incumbent = _cyc("OLD", 0.55, {0: 0.70}, cycle=1)
    challenger = _cyc("NEW", 0.50, {0: 0.75}, cycle=2)
    _, slots = champion_archive(
        [incumbent, challenger], prev_champions={0: "OLD"}, margin=0.01
    )
    assert slots[0] == "NEW"


def test_champion_archive_slots_persist_across_cycles():
    """The returned mapping feeds the next cycle's prev_champions."""
    a = _cyc("A", 0.60, {0: 0.70, 1: 0.50}, cycle=1)
    b = _cyc("B", 0.55, {0: 0.40, 1: 0.72}, cycle=1)
    archive1, slots1 = champion_archive([a, b])
    assert slots1 == {0: "A", 1: "B"}
    # cycle 2: noise-level challenger on cluster 1 must not displace B
    c = _cyc("C", 0.58, {0: 0.50, 1: 0.725}, cycle=2)
    archive2, slots2 = champion_archive([a, b, c], prev_champions=slots1, margin=0.01)
    assert slots2[1] == "B"
    assert archive2[0].prompt == "A"  # still scalar-best


def test_champion_archive_dedupes_multi_slot_champion():
    solo = _cand("ONLY", 0.6, {0: 0.7, 1: 0.8})
    archive, slots = champion_archive([solo])
    assert len(archive) == 1
    assert slots == {0: "ONLY", 1: "ONLY"}


def test_cluster_axis_and_summary():
    front = [_cand("A", 0.6, {0: 0.7, 2: 0.5})]
    assert cluster_axis(front) == [0, 2]
    summary = front_summary(front)
    assert summary["size"] == 1
    assert summary["cluster_champions"]["2"]["acc"] == 0.5


def _fake_oe_staging(tmp_path, n=3):
    cp = tmp_path / "checkpoints" / "checkpoint_4"
    (cp / "programs").mkdir(parents=True)
    for i in range(n):
        (cp / "programs" / f"p{i}.json").write_text(
            json.dumps(
                {
                    "id": f"p{i}",
                    "code": f"# EVOLVE-BLOCK-START\n<System>v{i}</System>\n# EVOLVE-BLOCK-END",
                    "metrics": {
                        "combined_score": 0.5 + i / 100,
                        "cluster_acc_0": 0.6 + i / 100,
                        "cluster_acc_1": 0.7 - i / 100,
                    },
                }
            ),
            encoding="utf-8",
        )
    # a decoy older checkpoint and a program with no metrics
    (tmp_path / "checkpoints" / "checkpoint_1" / "programs").mkdir(parents=True)
    (cp / "programs" / "broken.json").write_text("{not json", encoding="utf-8")
    return tmp_path


def test_read_oe_candidates_strips_markers_and_picks_latest(tmp_path):
    staging = _fake_oe_staging(tmp_path)
    assert latest_checkpoint(staging).name == "checkpoint_4"
    cands = read_oe_candidates(staging, cycle=2, strip_markers=strip_evolve_markers)
    assert len(cands) == 3
    assert all("EVOLVE-BLOCK" not in c.prompt for c in cands)
    assert all(c.cycle == 2 and c.source == "oe" for c in cands)
    assert cands[0].cluster_scores == {0: 0.6, 1: 0.7}


def _next_cycle_staging(seed_dir, staging_root):
    """Mimic OpenEvolve resuming from a seed dir: it writes its own checkpoint."""
    cp = staging_root / "checkpoints" / "checkpoint_8" / "programs"
    cp.mkdir(parents=True)
    for p in (seed_dir / "programs").glob("*.json"):
        (cp / p.name).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    return staging_root


def test_origin_survives_the_seed_checkpoint_roundtrip(tmp_path):
    """O23: carried members came back tagged source='oe' at the current cycle."""
    staging = _fake_oe_staging(tmp_path / "oe")
    cands = read_oe_candidates(staging, cycle=1, strip_markers=strip_evolve_markers)
    consolidated = _cand("consolidated text", 0.71, {0: 0.8, 1: 0.6}, source="consolidated")
    consolidated.cycle = 1
    seed = tmp_path / "seed_c2"
    write_seed_checkpoint(cands + [consolidated], staging, seed)

    reread = read_oe_candidates(
        _next_cycle_staging(seed, tmp_path / "oe2"), cycle=2, strip_markers=strip_evolve_markers
    )
    carried = {c.prompt: c for c in reread}["consolidated text"]
    assert carried.source == "carried"
    assert carried.cycle == 2
    assert carried.birth == ("consolidated", 1)
    assert all(c.source == "carried" for c in reread)


def test_a_mutated_seed_member_is_not_reported_as_carried(tmp_path):
    staging = _fake_oe_staging(tmp_path / "oe")
    cands = read_oe_candidates(staging, cycle=1, strip_markers=strip_evolve_markers)
    seed = tmp_path / "seed_c2"
    write_seed_checkpoint(cands[:1], staging, seed)
    oe2 = _next_cycle_staging(seed, tmp_path / "oe2")

    prog = next((oe2 / "checkpoints" / "checkpoint_8" / "programs").glob("*.json"))
    data = json.loads(prog.read_text(encoding="utf-8"))
    data["code"] = data["code"] + "\n<Extra>mutated</Extra>"
    prog.write_text(json.dumps(data), encoding="utf-8")

    reread = read_oe_candidates(oe2, cycle=2, strip_markers=strip_evolve_markers)
    assert reread[0].source == "oe"
    assert reread[0].birth == ("oe", 1)  # lineage still readable


def test_write_seed_checkpoint_roundtrip(tmp_path):
    staging = _fake_oe_staging(tmp_path / "oe")
    cands = read_oe_candidates(staging, cycle=1, strip_markers=strip_evolve_markers)
    front = pareto_front(cands)
    extra = _cand("consolidated text", 0.71, {0: 0.8, 1: 0.6}, source="consolidated")
    dest = tmp_path / "seed_c2"
    out = write_seed_checkpoint(front + [extra], staging, dest)

    assert out == dest
    meta = json.loads((dest / "metadata.json").read_text(encoding="utf-8"))
    assert meta["last_iteration"] == 0
    written = list((dest / "programs").glob("*.json"))
    assert len(written) == len(front) + 1
    assert set(meta["islands"][0]) == {p.stem for p in written}
    assert meta["best_program_id"] in meta["archive"]
    # the member with no OE program id got a synthetic record with metrics intact
    synth = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in written
        if json.loads(p.read_text(encoding="utf-8"))["code"] == "consolidated text"
    ]
    assert synth and synth[0]["metrics"]["combined_score"] == 0.71
    assert synth[0]["metrics"]["cluster_acc_0"] == 0.8


def test_write_seed_checkpoint_without_source_returns_none(tmp_path):
    assert write_seed_checkpoint([], tmp_path / "nope", tmp_path / "dest") is None


def test_strip_evolve_markers():
    raw = "# EVOLVE-BLOCK-START\n<System>x</System>\n# EVOLVE-BLOCK-END"
    assert strip_evolve_markers(raw) == "<System>x</System>"
    assert strip_evolve_markers("<System>x</System>") == "<System>x</System>"

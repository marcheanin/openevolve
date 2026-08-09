"""Error artifacts for mutator feedback.

The report is deliberately opinionated about *which* errors to show. On this
benchmark 95-100% of ensemble mistakes are adjacent-class (4<->5, 3<->4) and the
required correction directions are balanced — cycle 3 of the pair run asked the
mutator to rate one enthusiastic review down and another one up in the same breath
(OBSERVATIONS C9). Faced with that, the mutator writes one-directional caps
("'I liked' -> 4 at most") that flip correct 5s into wrong 4s out of distribution.

So the report now separates:

* **systematic** errors — every worker independently produced the same wrong
  rating. These are the ones a prompt change can plausibly fix.
* **contested** errors — the workers split. These are mostly reviewer idiosyncrasy
  and are shown only as a warning not to chase them.

It also states the confusion counts in *both* directions and, when the previous
cycle's predictions are available, attributes regressions to the last change.

Phase 2a additions:
* Contrastive FAIL vs OK same-group pairs (CivilComments spurious-correlation attack).
* Stronger O13 trigger discipline in the footer.
* Label-space-aware few-shot candidate lines (Review/Rating vs Comment/Label).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

Entry = Tuple[int, str, int, int, List[int], float, bool]

# Rules that address clusters/groups are dead — workers never see group ids (O13).
_CLUSTER_RULE_RE = re.compile(
    r"(?i)\b("
    r"cluster\s*\d+|group\s*[A-Z]\b|group\s*\d+|"
    r"reviewer\s*type|identity\s*group\s*\d+"
    r")\b"
)


def _pair_disagreement(votes: Sequence[int]) -> float:
    if len(votes) < 2:
        return 0.0
    n = len(votes)
    total = sum(
        abs(votes[i] - votes[j]) for i in range(n) for j in range(i + 1, n)
    )
    return total / (n * (n - 1) / 2) / 4.0


def _clip(text: str, limit: int) -> str:
    return text[:limit] + ("..." if len(text) > limit else "")


def _token_set(text: str) -> set:
    return {t for t in re.findall(r"[a-z0-9']+", text.lower()) if len(t) > 2}


def _jaccard(a: str, b: str) -> float:
    sa, sb = _token_set(a), _token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _fmt(entry: Entry, max_text_len: int, tag: str = "") -> List[str]:
    _, text, gold, pred, wp, d, _ = entry
    return [
        f"  gold={gold} pred={pred} workers={wp} d={d:.2f}{tag}",
        f'     "{_clip(text, max_text_len)}"',
    ]


def _confusion_lines(errors: Sequence[Entry]) -> List[str]:
    """Both directions of every confusion, so one-way caps look obviously wrong."""
    counts = Counter((e[2], e[3]) for e in errors)
    if not counts:
        return []
    lines = ["\nCONFUSION COUNTS (gold -> predicted), both directions:"]
    seen: set = set()
    for (gold, pred), n in counts.most_common():
        if (gold, pred) in seen:
            continue
        seen.add((gold, pred))
        reverse = counts.get((pred, gold), 0)
        seen.add((pred, gold))
        lines.append(f"  {gold}->{pred}: {n} errors    |    {pred}->{gold}: {reverse} errors")
    lines.append(
        "  Read this before writing a rule. Where both directions are non-zero, a "
        "one-sided cap (\"never rate this above 4\") fixes one column and breaks the "
        "other. Write rules that DISCRIMINATE between the two directions using "
        "different observable triggers, or leave the pair alone."
    )
    return lines


def _contrastive_lines(
    errors: Sequence[Entry],
    predictions: Sequence[int],
    gold: Sequence[int],
    texts: Sequence[str],
    cluster_ids: Optional[Sequence[int]],
    *,
    limit: int = 4,
    max_text_len: int = 240,
) -> List[str]:
    """
    For each systematic error, pair with a nearest correctly-classified example
    from the same group (OBSERVATIONS Phase 2a / CivilComments spurious correlation).
    """
    if not errors or cluster_ids is None:
        return []
    correct_by_group: Dict[int, List[int]] = defaultdict(list)
    for i, (pred, label) in enumerate(zip(predictions, gold)):
        if int(pred) == int(label):
            correct_by_group[int(cluster_ids[i])].append(i)

    lines = [
        "\nCONTRASTIVE PAIRS (same group; write a rule that separates FAIL from OK "
        "on observable text — e.g. identity term + hostility vs identity term alone):"
    ]
    used_ok: set = set()
    n_pairs = 0
    for entry in errors:
        if n_pairs >= limit:
            break
        idx, fail_text, fail_gold, fail_pred, _, _, systematic = entry
        if not systematic:
            continue
        gid = int(cluster_ids[idx])
        pool = [j for j in correct_by_group.get(gid, []) if j not in used_ok]
        if not pool:
            continue
        best = max(pool, key=lambda j: _jaccard(fail_text, texts[j]))
        used_ok.add(best)
        n_pairs += 1
        lines.append(
            f"  Pair {n_pairs} [group-local]: "
            f"FAIL gold={fail_gold} pred={fail_pred} vs "
            f"OK gold={gold[best]} pred={predictions[best]}"
        )
        lines.append(f'    FAIL: "{_clip(fail_text, max_text_len)}"')
        lines.append(f'    OK:   "{_clip(texts[best], max_text_len)}"')
    if n_pairs == 0:
        return []
    return lines


def format_error_artifacts(
    predictions: Sequence[int],
    gold: Sequence[int],
    worker_preds: Sequence[Sequence[int]],
    texts: Sequence[str],
    cluster_ids: Optional[Sequence[int]] = None,
    hard_indices: Optional[Sequence[int]] = None,
    anchor_indices: Optional[Sequence[int]] = None,
    pool_indices: Optional[Sequence[int]] = None,
    prev_predictions: Optional[Sequence[int]] = None,
    max_examples: int = 10,
    max_text_len: int = 320,
    *,
    label_space: str = "ordinal5",
    contrastive_pairs: bool = True,
    contrastive_pair_limit: int = 4,
) -> str:
    """Build the mutator's error report.

    hard_indices / anchor_indices are **pool** indices. When pool_indices is
    provided (aligned with predictions), they are mapped to batch-local rows;
    otherwise they are treated as batch-local positions (legacy).

    prev_predictions, when given, are the previous cycle's predictions on the same
    rows and drive the regression-attribution section.
    """
    if pool_indices is not None and len(pool_indices) == len(predictions):
        pool_to_local = {int(p): i for i, p in enumerate(pool_indices)}
        hard_local = {
            pool_to_local[int(i)] for i in (hard_indices or []) if int(i) in pool_to_local
        }
        anchor_local = {
            pool_to_local[int(i)] for i in (anchor_indices or []) if int(i) in pool_to_local
        }
    else:
        hard_local = {int(i) for i in (hard_indices or [])}
        anchor_local = {int(i) for i in (anchor_indices or [])}

    errors: List[Entry] = []
    borderline: List[Entry] = []
    by_cluster: Dict[int, List[Entry]] = defaultdict(list)

    for i, (pred, label, text) in enumerate(zip(predictions, gold, texts)):
        wp = [int(worker_preds[w][i]) for w in range(len(worker_preds))]
        d = _pair_disagreement(wp)
        systematic = bool(wp) and all(v != label for v in wp) and len(set(wp)) == 1
        entry: Entry = (i, text, int(label), int(pred), wp, d, systematic)
        if pred != label:
            errors.append(entry)
            by_cluster[int(cluster_ids[i]) if cluster_ids is not None else 0].append(entry)
        elif d > 0.25:
            borderline.append(entry)

    systematic = [e for e in errors if e[6]]
    contested = [e for e in errors if not e[6]]
    lines: List[str] = []

    if errors:
        lines.append(
            f"ERROR TRIAGE: {len(errors)} errors — {len(systematic)} systematic "
            f"(all workers agreed on the same wrong rating) and {len(contested)} "
            f"contested (workers disagreed).\n"
            "Spend your mutation on the systematic ones: a shared, reproducible "
            "misreading is what an instruction can fix. Contested cases are mostly "
            "reviewer idiosyncrasy and chasing them costs accuracy elsewhere."
        )

    ordered_hard = [e for e in systematic if e[0] in hard_local] + [
        e for e in systematic if e[0] not in hard_local
    ]
    if ordered_hard:
        lines.append("\nSYSTEMATIC ERRORS (fix these):")
        for j, entry in enumerate(ordered_hard[:max_examples], start=1):
            tag = " [HARD]" if entry[0] in hard_local else ""
            body = _fmt(entry, max_text_len, tag)
            lines.append(f"  {j}. {body[0].strip()}")
            lines.append(body[1])
    elif errors:
        lines.append(
            "\nNO FULLY SYSTEMATIC ERRORS THIS CYCLE — workers split on mistakes. "
            "Still use the confusion counts and shared text patterns across several "
            "errors to sharpen an adjacent-star boundary; avoid a rule for a single "
            "idiosyncratic contested review."
        )

    if contested:
        lines.append(
            f"\nCONTESTED ERRORS ({len(contested)}, do NOT write rules for these):"
        )
        for j, entry in enumerate(contested[:3], start=1):
            body = _fmt(entry, max_text_len)
            lines.append(f"  {j}. {body[0].strip()}")
            lines.append(body[1])

    lines.extend(_confusion_lines(errors))

    if contrastive_pairs:
        lines.extend(
            _contrastive_lines(
                systematic or errors,
                predictions,
                gold,
                texts,
                cluster_ids,
                limit=contrastive_pair_limit,
                max_text_len=min(max_text_len, 240),
            )
        )

    # Negative entries mean "this row was not scored last cycle" and are skipped.
    if prev_predictions is not None and len(prev_predictions) == len(predictions):
        known = [i for i in range(len(predictions)) if int(prev_predictions[i]) >= 0]
        new_errors = [
            e for e in errors if e[0] in set(known) and int(prev_predictions[e[0]]) == e[2]
        ]
        fixed = [
            i
            for i in known
            if predictions[i] == gold[i] and int(prev_predictions[i]) != gold[i]
        ]
        lines.append(
            f"\nDAMAGE REPORT vs the previous cycle's prompt ({len(known)} comparable "
            f"rows): fixed {len(fixed)}, broke {len(new_errors)}."
        )
        if new_errors:
            lines.append(
                "  These were CORRECT before the last change and are wrong now. If a "
                "recent rule caused them, narrow its trigger instead of adding another rule:"
            )
            for j, entry in enumerate(new_errors[:4], start=1):
                body = _fmt(entry, max_text_len)
                lines.append(f"  {j}. {body[0].strip()}")
                lines.append(body[1])

    if by_cluster:
        # Groups are labelled A, B, C... on purpose. Showing raw cluster ids made the
        # mutator write rules addressed to them ("CLUSTER 3: ..."), which the rating
        # workers can never apply — they see only review text (OBSERVATIONS O13).
        lines.append("\nERROR BREAKDOWN BY REVIEWER GROUP (labels are analysis-only):")
        for rank, cid in enumerate(sorted(by_cluster, key=lambda c: -len(by_cluster[c]))):
            group = by_cluster[cid]
            n_sys = sum(1 for e in group if e[6])
            letter = chr(ord("A") + rank) if rank < 26 else f"G{rank}"
            lines.append(f"  Group {letter}: {len(group)} errors ({n_sys} systematic)")
            for entry in group[:2]:
                lines.append(f"    gold={entry[2]} pred={entry[3]} d={entry[5]:.2f}")
                lines.append(f'       "{_clip(entry[1], max_text_len)}"')
        lines.append(
            "HOW TO USE THE GROUPS: the group with the most SYSTEMATIC errors is where "
            "the objective is losing. Find what its review texts have in common that "
            "you can actually observe at inference time (phrasing, hedging, length, "
            "structure, contrast words, identity terms + hostility) and write the rule "
            "on that trigger. A group with many errors but few systematic ones is noise, "
            "not a target. NEVER name a group or cluster in the prompt: the rating model "
            "receives only the review text and cannot know which group it belongs to, "
            'so a rule like "GROUP A: rate lower" is dead text.'
        )

    if borderline:
        lines.append("\nBORDERLINE (correct but high disagreement — fragile, protect them):")
        for j, entry in enumerate(borderline[:4], start=1):
            lines.append(
                f"  {j}. gold={entry[2]} pred={entry[3]} workers={entry[4]} d={entry[5]:.2f}"
            )

    if anchor_local:
        regressions = [e for e in errors if e[0] in anchor_local]
        if regressions:
            lines.append("\nANCHOR REGRESSIONS (previously solved, now broken):")
            for j, entry in enumerate(regressions[:5], start=1):
                lines.append(f'  {j}. gold={entry[2]} pred={entry[3]} "{_clip(entry[1], max_text_len)}"')

    fewshot = _fewshot_block(systematic or errors, label_space=label_space)
    if fewshot:
        lines.extend(fewshot)

    if lines:
        lines.append(
            "\nTARGET BLOCK: prefer <DynamicRules>. "
            "<FewShotExamples> is injected mechanically from real failures — "
            "do not invent example text (OBSERVATIONS O22).\n"
            "MUTATION REMINDERS:\n"
            "  - Prefer one coherent change; small targeted edits beat full rewrites.\n"
            "  - Every rule MUST have an observable TEXTUAL TRIGGER "
            "(quoted phrase, pattern, contrast word, identity term + hostility). "
            "Rules without a trigger are rejected.\n"
            "  - Do NOT reference cluster / group / reviewer-type identifiers "
            "(OBSERVATIONS O13) — those strings never appear at inference.\n"
            "  - Prefer SYSTEMATIC errors; if only contested remain, use repeated "
            "confusion patterns — not a one-off split.\n"
            "  - Check confusion counts: no one-directional caps on a confusion "
            "that occurs in both directions.\n"
            "  - Use CONTRASTIVE PAIRS: the rule must accept OK and reject FAIL.\n"
            "  - Keep <Task> with {review} unchanged; preserve XML tags.\n"
            "  - Remember fitness is on D_select (may rotate each cycle), not this "
            "batch alone — write rules that generalize.\n"
        )
        lines.append(
            f"\nSummary: {len(errors)} errors ({len(systematic)} systematic), "
            f"{len(borderline)} borderline / {len(predictions)} examples "
            f"(hard_slots={len(hard_local)}, anchor_slots={len(anchor_local)})."
        )
    return "\n".join(lines)


def _fewshot_block(
    candidates: Sequence[Entry],
    limit: int = 4,
    max_len: int = 320,
    *,
    label_space: str = "ordinal5",
) -> List[str]:
    """Ready-to-paste few-shot lines built from real failing reviews (O22)."""
    if not candidates:
        return []
    input_tag = "Comment" if label_space == "binary" else "Review"
    label_tag = "Label" if label_space == "binary" else "Rating"
    lines = [
        "\nCANDIDATE FEW-SHOT LINES (verbatim from the failures above; "
        "the pipeline injects these mechanically — do not paraphrase):"
    ]
    for entry in candidates[:limit]:
        text = " ".join(entry[1].split())[:max_len]
        lines.append(f'  {input_tag}: "{text}"')
        lines.append(f"  {label_tag}: {entry[2]}")
    return lines


def build_fewshot_inner(
    texts: Sequence[str],
    labels: Sequence[int],
    *,
    label_space: str = "ordinal5",
    max_len: int = 320,
) -> str:
    """XML-inner content for <FewShotExamples> from verbatim examples (O22)."""
    input_tag = "Comment" if label_space == "binary" else "Review"
    label_tag = "Label" if label_space == "binary" else "Rating"
    parts: List[str] = []
    for i, (text, lab) in enumerate(zip(texts, labels), start=1):
        clean = " ".join(str(text).split())[:max_len]
        parts.append(f"Example {i}:")
        parts.append(f"{input_tag}: {clean}")
        parts.append(f"{label_tag}: {int(lab)}")
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def extract_systematic_fewshot_examples(
    artifacts_text: str,
    *,
    limit: int = 4,
) -> List[Tuple[str, int]]:
    """
    Parse CANDIDATE FEW-SHOT LINES from an error artifact report.

    Returns list of (text, gold_label).
    """
    if not artifacts_text:
        return []
    # Match both Amazon and CivilComments candidate formats.
    pat = re.compile(
        r'(?:Review|Comment):\s*"([^"]+)"\s*\n\s*(?:Rating|Label):\s*(\d+)',
        re.MULTILINE,
    )
    out: List[Tuple[str, int]] = []
    for m in pat.finditer(artifacts_text):
        out.append((m.group(1).strip(), int(m.group(2))))
        if len(out) >= limit:
            break
    return out


def lint_dynamic_rules_for_triggers(prompt: str) -> List[str]:
    """
    Return violation messages for DynamicRules that name clusters/groups (O13).

    Empty list = pass.
    """
    from prime.evolution.prompt_blocks import extract_block

    body = extract_block(prompt, "DynamicRules") or ""
    violations: List[str] = []
    for i, line in enumerate(body.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if _CLUSTER_RULE_RE.search(stripped):
            violations.append(f"DynamicRules L{i}: cluster/group reference — {stripped[:120]}")
    return violations

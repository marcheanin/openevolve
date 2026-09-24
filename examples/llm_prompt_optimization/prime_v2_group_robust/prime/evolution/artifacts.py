"""Error artifacts for mutator feedback.

Binary (CivilComments) reports lead with FP/FN polarity, named identity groups
(analysis-only), and an optional GBA dashboard. Ordinal (Amazon) keeps the older
systematic/contested worker-triage framing.

Phase 3 / E5:
* Freeze-friendly reports (controller may attach a pre few-shot copy to OE).
* Single-scorer mode does not pretend worker disagreement exists.
* Contrastive pairs carry oracle group names for the mutator only (O13 still
  forbids writing those names into the evolved prompt).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

Entry = Tuple[int, str, int, int, List[int], float, bool]

# Rules that address clusters/groups are dead — workers never see group ids (O13).
_CLUSTER_RULE_RE = re.compile(
    r"(?i)\b("
    r"cluster\s*\d+|group\s*[A-Z]\b|group\s*\d+|"
    r"reviewer\s*type|identity\s*group\s*\d+"
    r")\b"
)


def _pair_disagreement(votes: Sequence[int], *, label_span: float = 4.0) -> float:
    if len(votes) < 2:
        return 0.0
    n = len(votes)
    total = sum(
        abs(votes[i] - votes[j]) for i in range(n) for j in range(i + 1, n)
    )
    span = max(float(label_span), 1e-6)
    return total / (n * (n - 1) / 2) / span


def _clip(text: str, limit: int) -> str:
    return text[:limit] + ("..." if len(text) > limit else "")


def _token_set(text: str) -> set:
    return {t for t in re.findall(r"[a-z0-9']+", text.lower()) if len(t) > 2}


def _jaccard(a: str, b: str) -> float:
    sa, sb = _token_set(a), _token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _group_label(gid: int, group_names: Optional[Dict[int, str]]) -> str:
    if group_names and int(gid) in group_names:
        return str(group_names[int(gid)])
    return f"id={int(gid)}"


def _fmt(entry: Entry, max_text_len: int, tag: str = "") -> List[str]:
    _, text, gold, pred, wp, d, _ = entry
    return [
        f"  gold={gold} pred={pred} workers={wp} d={d:.2f}{tag}",
        f'     "{_clip(text, max_text_len)}"',
    ]


def _polarity_counts(errors: Sequence[Entry]) -> Tuple[int, int]:
    fp = sum(1 for e in errors if e[2] == 0 and e[3] == 1)
    fn = sum(1 for e in errors if e[2] == 1 and e[3] == 0)
    return fp, fn


def _confusion_lines(errors: Sequence[Entry], *, binary: bool) -> List[str]:
    """Both directions of every confusion, so one-way caps look obviously wrong."""
    counts = Counter((e[2], e[3]) for e in errors)
    if not counts:
        return []
    lines = ["\nCONFUSION COUNTS (gold -> predicted), both directions:"]
    if binary:
        fp = counts.get((0, 1), 0)
        fn = counts.get((1, 0), 0)
        lines.append(f"  FP 0->1 (false toxic): {fp}    |    FN 1->0 (missed toxic): {fn}")
        lines.append(
            "  Read this before writing a rule. If both sides are non-zero, a one-sided "
            "bias (\"always toxic if identity word\") fixes one column and breaks the "
            "other. Discriminate with observable triggers, or leave the pair alone."
        )
        return lines

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
    group_names: Optional[Dict[int, str]] = None,
    prefer_systematic: bool = True,
) -> List[str]:
    """
    For each error, pair with a nearest correctly-classified example from the same
    group (CivilComments spurious-correlation attack).
    """
    if not errors or cluster_ids is None:
        return []
    correct_by_group: Dict[int, List[int]] = defaultdict(list)
    for i, (pred, label) in enumerate(zip(predictions, gold)):
        if int(pred) == int(label):
            correct_by_group[int(cluster_ids[i])].append(i)

    lines = [
        "\nCONTRASTIVE PAIRS (same identity group; write a rule that separates FAIL "
        "from OK on observable text — e.g. identity term + hostility vs mention alone). "
        "Group names are analysis-only — NEVER put them in the prompt:"
    ]
    used_ok: set = set()
    n_pairs = 0
    ordered = [e for e in errors if (e[6] if prefer_systematic else True)] or list(errors)
    for entry in ordered:
        if n_pairs >= limit:
            break
        idx, fail_text, fail_gold, fail_pred, _, _, systematic = entry
        if prefer_systematic and not systematic:
            continue
        gid = int(cluster_ids[idx])
        pool = [j for j in correct_by_group.get(gid, []) if j not in used_ok]
        if not pool:
            continue
        best = max(pool, key=lambda j: _jaccard(fail_text, texts[j]))
        used_ok.add(best)
        n_pairs += 1
        gname = _group_label(gid, group_names)
        lines.append(
            f"  Pair {n_pairs} [group={gname} analysis-only]: "
            f"FAIL gold={fail_gold} pred={fail_pred} vs "
            f"OK gold={gold[best]} pred={predictions[best]}"
        )
        lines.append(f'    FAIL: "{_clip(fail_text, max_text_len)}"')
        lines.append(f'    OK:   "{_clip(texts[best], max_text_len)}"')
        lines.append(
            "    Isolate an OBSERVABLE difference (hostility/slur/threat vs neutral mention)."
        )
    if n_pairs == 0:
        return []
    return lines


def format_gba_dashboard(
    *,
    cluster_gba: Optional[Dict[Any, float]] = None,
    softmin: Optional[float] = None,
    worst_gba: Optional[float] = None,
    gba_mean: Optional[float] = None,
    toxic_recall: Optional[float] = None,
    specificity: Optional[float] = None,
    pred_pos_rate: Optional[float] = None,
    group_names: Optional[Dict[int, str]] = None,
    source: str = "D_select",
) -> str:
    """Compact named GBA table for the mutator (analysis-only group names)."""
    if not cluster_gba:
        return ""
    items = sorted(
        ((int(k), float(v)) for k, v in cluster_gba.items()),
        key=lambda kv: kv[1],
    )
    lines = [
        f"GBA DASHBOARD ({source}; analysis-only group names — do NOT write them into the prompt):"
    ]
    if softmin is not None:
        lines.append(f"  softmin(GBA)={float(softmin):.4f}")
    if worst_gba is not None:
        lines.append(f"  worst-GBA={float(worst_gba):.4f}")
    if gba_mean is not None:
        lines.append(f"  mean-GBA={float(gba_mean):.4f}")
    if toxic_recall is not None or specificity is not None:
        tr = f"{float(toxic_recall):.3f}" if toxic_recall is not None else "?"
        sp = f"{float(specificity):.3f}" if specificity is not None else "?"
        lines.append(f"  toxic_recall={tr}  specificity={sp}")
    if pred_pos_rate is not None:
        lines.append(f"  pred_pos_rate={float(pred_pos_rate):.3f}")
    if items:
        bott_gid, bott_v = items[0]
        lines.append(
            f"  softmin bottleneck: {_group_label(bott_gid, group_names)} "
            f"GBA={bott_v:.3f}"
        )
        lines.append("  per-group GBA (low → high):")
        for gid, val in items:
            lines.append(f"    {_group_label(gid, group_names)}: {val:.3f}")
    lines.append(
        "  Raise the bottleneck with a TEXTUAL rule; do not name the group in DynamicRules."
    )
    return "\n".join(lines)


def format_dselect_error_sample(
    predictions: Sequence[int],
    gold: Sequence[int],
    texts: Sequence[str],
    *,
    limit: int = 4,
    max_text_len: int = 200,
) -> str:
    """Short FP/FN sample from the fitness set (complements batch artifacts)."""
    fps: List[Tuple[str, int, int]] = []
    fns: List[Tuple[str, int, int]] = []
    for pred, label, text in zip(predictions, gold, texts):
        if int(pred) == int(label):
            continue
        row = (str(text), int(label), int(pred))
        if int(label) == 0 and int(pred) == 1:
            fps.append(row)
        elif int(label) == 1 and int(pred) == 0:
            fns.append(row)
    if not fps and not fns:
        return "D_SELECT ERROR SAMPLE: 0 errors on scored rows."
    lines = [
        f"D_SELECT ERROR SAMPLE (fitness set; up to {limit} per polarity): "
        f"FP={len(fps)} FN={len(fns)}"
    ]
    for kind, rows in (("FP 0->1", fps), ("FN 1->0", fns)):
        for text, g, p in rows[: max(0, limit)]:
            lines.append(f'  [{kind}] gold={g} pred={p} "{_clip(text, max_text_len)}"')
    return "\n".join(lines)


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
    group_names: Optional[Dict[int, str]] = None,
    gba_dashboard: Optional[str] = None,
    d_select_sample: Optional[str] = None,
) -> str:
    """Build the mutator's error report.

    hard_indices / anchor_indices are **pool** indices. When pool_indices is
    provided (aligned with predictions), they are mapped to batch-local rows;
    otherwise they are treated as batch-local positions (legacy).

    prev_predictions, when given, are the previous cycle's predictions on the same
    rows and drive the regression-attribution section.
    """
    binary = label_space == "binary"
    n_workers = len(worker_preds)
    single_scorer = n_workers <= 1
    label_span = 1.0 if binary else 4.0

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
        d = _pair_disagreement(wp, label_span=label_span)
        if single_scorer:
            # No ensemble signal: every error is actionable text pattern work.
            systematic = pred != label
        else:
            systematic = bool(wp) and all(v != label for v in wp) and len(set(wp)) == 1
        entry: Entry = (i, text, int(label), int(pred), wp, d, systematic)
        if pred != label:
            errors.append(entry)
            by_cluster[int(cluster_ids[i]) if cluster_ids is not None else 0].append(entry)
        elif (not single_scorer) and d > 0.25:
            borderline.append(entry)

    systematic = [e for e in errors if e[6]]
    contested = [e for e in errors if not e[6]]
    lines: List[str] = []

    if gba_dashboard:
        lines.append(gba_dashboard.rstrip())
        lines.append("")
    if d_select_sample:
        lines.append(d_select_sample.rstrip())
        lines.append("")

    if errors:
        fp, fn = _polarity_counts(errors)
        if binary:
            lines.append(
                f"POLARITY: FP 0->1 (false toxic)={fp}  |  FN 1->0 (missed toxic)={fn}  "
                f"|  total_errors={len(errors)}"
            )
            lines.append(
                "Your <mutation_log> MUST name which polarity you target (FP, FN, or both) "
                "and must not invent the opposite story."
            )
            if single_scorer:
                lines.append(
                    f"ERROR LIST ({len(errors)} mistakes; single scorer — no worker "
                    "agreement signal; treat every error as a textual pattern to fix):"
                )
            else:
                lines.append(
                    f"ERROR TRIAGE: {len(errors)} errors — {len(systematic)} systematic "
                    f"(all workers agreed wrong) and {len(contested)} contested (workers split).\n"
                    "Prefer systematic errors; contested cases are weaker targets."
                )
        else:
            lines.append(
                f"ERROR TRIAGE: {len(errors)} errors — {len(systematic)} systematic "
                f"(all workers agreed on the same wrong rating) and {len(contested)} "
                f"contested (workers disagreed).\n"
                "Spend your mutation on the systematic ones: a shared, reproducible "
                "misreading is what an instruction can fix. Contested cases are mostly "
                "reviewer idiosyncrasy and chasing them costs accuracy elsewhere."
            )

    show_list = errors if (binary and single_scorer) else systematic
    ordered_hard = [e for e in show_list if e[0] in hard_local] + [
        e for e in show_list if e[0] not in hard_local
    ]
    section = "ERRORS (fix these):" if (binary and single_scorer) else "SYSTEMATIC ERRORS (fix these):"
    if ordered_hard:
        lines.append(f"\n{section}")
        for j, entry in enumerate(ordered_hard[:max_examples], start=1):
            tag = " [HARD]" if entry[0] in hard_local else ""
            if binary:
                kind = "FP" if entry[2] == 0 and entry[3] == 1 else (
                    "FN" if entry[2] == 1 and entry[3] == 0 else "ERR"
                )
                tag = f" [{kind}]{tag}"
            body = _fmt(entry, max_text_len, tag)
            lines.append(f"  {j}. {body[0].strip()}")
            lines.append(body[1])
    elif errors and not (binary and single_scorer):
        lines.append(
            "\nNO FULLY SYSTEMATIC ERRORS THIS CYCLE — workers split on mistakes. "
            "Still use the confusion counts and shared text patterns across several "
            "errors; avoid a rule for a single idiosyncratic contested example."
        )

    if contested and not (binary and single_scorer):
        lines.append(
            f"\nCONTESTED ERRORS ({len(contested)}, do NOT write rules for these):"
        )
        for j, entry in enumerate(contested[:3], start=1):
            body = _fmt(entry, max_text_len)
            lines.append(f"  {j}. {body[0].strip()}")
            lines.append(body[1])

    lines.extend(_confusion_lines(errors, binary=binary))

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
                group_names=group_names,
                prefer_systematic=not (binary and single_scorer),
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
            f"\nDAMAGE REPORT vs baseline prompt ({len(known)} comparable "
            f"rows): fixed {len(fixed)}, broke {len(new_errors)}."
        )
        if new_errors:
            lines.append(
                "  These were CORRECT before and are wrong now. If a recent rule caused "
                "them, narrow its trigger instead of adding another rule:"
            )
            for j, entry in enumerate(new_errors[:4], start=1):
                body = _fmt(entry, max_text_len)
                lines.append(f"  {j}. {body[0].strip()}")
                lines.append(body[1])

    if by_cluster:
        if group_names:
            lines.append(
                "\nERROR BREAKDOWN BY IDENTITY GROUP (analysis-only names — "
                "NEVER write these into the prompt):"
            )
            for cid in sorted(by_cluster, key=lambda c: -len(by_cluster[c])):
                group = by_cluster[cid]
                n_sys = sum(1 for e in group if e[6])
                gname = _group_label(int(cid), group_names)
                lines.append(
                    f"  {gname}: {len(group)} errors"
                    + (f" ({n_sys} systematic)" if not (binary and single_scorer) else "")
                )
                for entry in group[:2]:
                    kind = ""
                    if binary:
                        kind = (
                            " FP"
                            if entry[2] == 0 and entry[3] == 1
                            else (" FN" if entry[2] == 1 and entry[3] == 0 else "")
                        )
                    lines.append(
                        f"    gold={entry[2]} pred={entry[3]}{kind} d={entry[5]:.2f}"
                    )
                    lines.append(f'       "{_clip(entry[1], max_text_len)}"')
            lines.append(
                "HOW TO USE THE GROUPS: focus on the group with the most errors "
                "(often the softmin bottleneck). Find an observable textual trigger "
                "shared by its failing comments. NEVER name a group/cluster in the "
                "prompt — the scorer only sees comment text."
            )
        else:
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
                "the objective is losing. Find what its texts have in common that you can "
                "observe at inference time and write the rule on that trigger. NEVER name "
                "a group or cluster in the prompt."
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
                lines.append(
                    f'  {j}. gold={entry[2]} pred={entry[3]} "{_clip(entry[1], max_text_len)}"'
                )

    fewshot = _fewshot_block(systematic or errors, label_space=label_space)
    if fewshot:
        lines.extend(fewshot)

    if lines:
        if binary:
            lines.append(
                "\nTARGET BLOCK: prefer <DynamicRules>. "
                "<FewShotExamples> is injected mechanically from real failures — "
                "do not invent or rewrite example text (OBSERVATIONS O22).\n"
                "MUTATION REMINDERS:\n"
                "  - Prefer one coherent change; small targeted edits beat full rewrites.\n"
                "  - State polarity in <mutation_log>: FP / FN / both.\n"
                "  - Every rule MUST have an observable TEXTUAL TRIGGER "
                "(quoted phrase, pattern, identity term + hostility).\n"
                "  - Do NOT reference cluster / group / identity-group names in the prompt "
                "(OBSERVATIONS O13).\n"
                "  - Check FP vs FN counts before biasing toward toxic or non-toxic.\n"
                "  - Use CONTRASTIVE PAIRS: the rule must accept OK and reject FAIL.\n"
                "  - Keep <Task> with {review} unchanged; preserve XML tags.\n"
                "  - Fitness is softmin(GBA) on D_select, not this batch alone.\n"
            )
        else:
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
            f"\nSummary: {len(errors)} errors"
            + (
                f" ({len(systematic)} systematic)"
                if not (binary and single_scorer)
                else ""
            )
            + f", {len(borderline)} borderline / {len(predictions)} examples "
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


def compose_mutator_artifacts(
    *,
    frozen_text: Optional[str],
    live_text: Optional[str],
    gba_dashboard: Optional[str] = None,
    d_select_sample: Optional[str] = None,
) -> Optional[str]:
    """Merge frozen cycle targets with this candidate's live batch report."""
    parts: List[str] = []
    if gba_dashboard and gba_dashboard.strip():
        # Avoid duplicating if already embedded in frozen/live.
        blob = f"{frozen_text or ''}\n{live_text or ''}"
        if "GBA DASHBOARD" not in blob:
            parts.append(gba_dashboard.strip())
    if d_select_sample and d_select_sample.strip():
        blob = f"{frozen_text or ''}\n{live_text or ''}"
        if "D_SELECT ERROR SAMPLE" not in blob:
            parts.append(d_select_sample.strip())

    frozen = (frozen_text or "").strip()
    live = (live_text or "").strip()
    if frozen:
        parts.append(
            "## PRIMARY TARGETS (cycle batch errors BEFORE few-shot inject)\n" + frozen
        )
    if live:
        if frozen and live == frozen:
            pass
        elif frozen and "POLARITY:" in live:
            parts.append(
                "## THIS CANDIDATE ON CURRENT BATCH "
                "(remaining mistakes after few-shots / mutations)\n" + live
            )
        elif not frozen:
            parts.append(live)
        else:
            parts.append(
                "## THIS CANDIDATE ON CURRENT BATCH\n" + live
            )
    elif frozen:
        parts.append(
            "## THIS CANDIDATE ON CURRENT BATCH\n"
            "0 remaining errors on the active batch (few-shots may cover them). "
            "Still generalize PRIMARY TARGET patterns on D_select; do not assume the "
            "batch is solved for the fitness set."
        )
    return "\n\n".join(parts) if parts else None

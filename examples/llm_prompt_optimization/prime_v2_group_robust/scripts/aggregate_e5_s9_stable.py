"""Aggregate stable_session reports → RESULTS.md tables + JSON dump."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SS = ROOT / "results/E5_s9_matrix/stable_session"
FIXED = ROOT / "experiments/E5_civilcomments/fixed_sets/test_fixed_materialized.json"
OUT_JSON = SS / "stable_aggregate.json"
OUT_MD = ROOT / "experiments/E5_civilcomments/RESULTS.md"

METHOD_ORDER = [
    "seed",
    "ape",
    "ape_k48",
    "ape_ut",
    "apo",
    "gpo",
    "evoprompt_ga",
    "evoprompt_de",
    "gepa",
    "prime",
    "random_al",
    "oracle",
]

METHOD_META = {
    "seed": ("R0", "Seed prompt", "Fixed `initial_prompt_civilcomments.txt`; no optimization."),
    "ape": ("R2", "APE", "K=6 proposals, N=36 demonstrations; select by D_dev softmin-GBA."),
    "ape_k48": ("R3", "APE-K48", "Budget-matched APE with K=48 proposals, same N=36."),
    "ape_ut": ("R5", "APE-ut", "APE with unlabeled-target demo pool (regime-shift)."),
    "apo": ("R4", "APO/ProTeGi", "Exploratory: **2 rounds** (native 6), beam=4."),
    "gpo": ("R6", "GPO", "K=6, conf threshold T=0.83; stage-1 filter + upsample."),
    "evoprompt_ga": ("R7", "EvoPrompt-GA", "Exploratory: reduced pop/gens vs native 10×10."),
    "evoprompt_de": ("R8", "EvoPrompt-DE", "Exploratory: reduced pop/gens vs native 10×10."),
    "gepa": ("R9", "GEPA", "Lightweight reflective+Pareto adapter (PyPI `gepa` not used)."),
    "prime": (
        "R10",
        "PRIME-main",
        "Seed42=E5v2 artifact; 43=`…074014` (shipped seed); 44=`…110051` (non-seed OE). "
        "D_dev Top-1 selection; soft_min_lex + GBA.",
    ),
    "random_al": ("R11", "Random-AL", "APO + 240 random target labels; exploratory **2 rounds**."),
    "oracle": ("R12", "Oracle", "APO on fully labeled target; **seed42 only**; exploratory 2 rounds."),
}


def _mean_sd(xs: list[float]) -> tuple[float | None, float | None]:
    if not xs:
        return None, None
    m = float(sum(xs) / len(xs))
    if len(xs) == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var)


def _fmt(m: float | None, sd: float | None, digits: int = 3) -> str:
    if m is None:
        return "—"
    if sd is None or sd == 0.0:
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f} (±{sd:.{digits}f})"


def _load_labels():
    mat = json.loads(FIXED.read_text(encoding="utf-8"))
    y = np.asarray(mat["labels"], dtype=np.int16)
    uids = np.asarray(mat["user_ids"])
    cids = np.asarray(mat["cluster_ids"], dtype=np.int16)
    return y, uids, cids, mat["fingerprint"], int(mat["n"])


def _worst_group_acc(preds, y, uids, cids) -> float:
    from prime.fitness.metrics import compute_metrics

    m = compute_metrics(
        preds, y, uids, cluster_ids=cids, class_balanced=True
    )
    return float(m["R_worst_group"])


def _job_metrics(job: Path, y, uids, cids) -> dict:
    rep = json.loads((job / "stable_report.json").read_text(encoding="utf-8"))
    out = {
        "n": rep.get("n"),
        "fingerprint": rep.get("fingerprint"),
        "final_summary": rep.get("final_summary") or {},
        "seed_summary": rep.get("seed_summary") or {},
        "delta_mean": rep.get("delta_mean_final_minus_seed") or {},
        "wg_final_values": [],
        "wg_seed_values": [],
    }
    for r in range(3):
        fp = job / f"final_repeat{r}_preds.npy"
        sp = job / f"seed_repeat{r}_preds.npy"
        if fp.is_file():
            out["wg_final_values"].append(_worst_group_acc(np.load(fp), y, uids, cids))
        if sp.is_file():
            out["wg_seed_values"].append(_worst_group_acc(np.load(sp), y, uids, cids))
    return out


def main() -> None:
    import sys

    sys.path.insert(0, str(ROOT))
    y, uids, cids, fp, n = _load_labels()

    # Collect per (seed, method)
    cells: dict[tuple[str, str], dict] = {}
    shared_seed: dict[str, dict] = {}

    for s in ("42", "43", "44"):
        sh = SS / f"_shared_seed{s}"
        if (sh / "seed_baseline.json").is_file() or any(
            (sh / f"seed_repeat{r}_preds.npy").is_file() for r in range(3)
        ):
            # Prefer metrics from any completed job's seed_summary (same preds),
            # or compute from shared preds.
            wg_vals = []
            gba_vals = []
            glob_vals = []
            soft_vals = []
            recall_vals = []
            spec_vals = []
            for r in range(3):
                pp = sh / f"seed_repeat{r}_preds.npy"
                mp = sh / f"seed_repeat{r}_metrics.json"
                if pp.is_file():
                    preds = np.load(pp)
                    wg_vals.append(_worst_group_acc(preds, y, uids, cids))
                if mp.is_file():
                    m = json.loads(mp.read_text(encoding="utf-8"))
                    if m.get("R_worst_gba_raw") is not None:
                        gba_vals.append(float(m["R_worst_gba_raw"]))
                    if m.get("R_global") is not None:
                        glob_vals.append(float(m["R_global"]))
                    if m.get("R_soft_min_gba") is not None:
                        soft_vals.append(float(m["R_soft_min_gba"]))
                    if m.get("toxic_recall") is not None:
                        recall_vals.append(float(m["toxic_recall"]))
                    if m.get("specificity") is not None:
                        spec_vals.append(float(m["specificity"]))
            shared_seed[s] = {
                "R_worst_gba_raw": _mean_sd(gba_vals),
                "R_worst_group": _mean_sd(wg_vals),
                "R_global": _mean_sd(glob_vals),
                "R_soft_min_gba": _mean_sd(soft_vals),
                "toxic_recall": _mean_sd(recall_vals),
                "specificity": _mean_sd(spec_vals),
                "gba_values": gba_vals,
                "wg_values": wg_vals,
                "glob_values": glob_vals,
            }

        for m in METHOD_ORDER:
            if m == "seed":
                continue
            job = SS / f"seed{s}_{m}"
            if not (job / "stable_report.json").is_file():
                continue
            jm = _job_metrics(job, y, uids, cids)
            fs = jm["final_summary"]

            def pair(key: str):
                block = fs.get(key) or {}
                return block.get("mean"), block.get("sd"), block.get("values")

            gba_m, gba_sd, gba_v = pair("R_worst_gba_raw")
            glob_m, glob_sd, glob_v = pair("R_global")
            soft_m, soft_sd, soft_v = pair("R_soft_min_gba")
            rec_m, rec_sd, rec_v = pair("toxic_recall")
            sp_m, sp_sd, sp_v = pair("specificity")
            wg_m, wg_sd = _mean_sd(jm["wg_final_values"])
            cells[(s, m)] = {
                "R_worst_gba_raw": (gba_m, gba_sd, gba_v),
                "R_worst_group": (wg_m, wg_sd, jm["wg_final_values"]),
                "R_global": (glob_m, glob_sd, glob_v),
                "R_soft_min_gba": (soft_m, soft_sd, soft_v),
                "toxic_recall": (rec_m, rec_sd, rec_v),
                "specificity": (sp_m, sp_sd, sp_v),
                "delta_gba": (jm["delta_mean"] or {}).get("R_worst_gba_raw"),
            }

    # Aggregate across seeds: pool means of each seed (then mean±sd over seeds)
    def across_seeds(method: str, metric: str):
        xs = []
        if method == "seed":
            for s in ("42", "43", "44"):
                mm = (shared_seed.get(s) or {}).get(metric)
                if mm and mm[0] is not None:
                    xs.append(mm[0])
        else:
            for s in ("42", "43", "44"):
                c = cells.get((s, method))
                if not c:
                    continue
                block = c.get(metric)
                if block and block[0] is not None:
                    xs.append(float(block[0]))
        return _mean_sd(xs), xs

    aggregate = {}
    for m in METHOD_ORDER:
        aggregate[m] = {}
        for metric in (
            "R_worst_gba_raw",
            "R_worst_group",
            "R_global",
            "R_soft_min_gba",
            "toxic_recall",
            "specificity",
        ):
            (mean, sd), vals = across_seeds(m, metric)
            aggregate[m][metric] = {"mean": mean, "sd": sd, "seed_means": vals}

    seed_gba = (aggregate["seed"]["R_worst_gba_raw"]["mean"] or 0.0)

    dump = {
        "fingerprint": fp,
        "n": n,
        "repeats": 3,
        "shared_seed": {
            s: {
                k: {"mean": v[0], "sd": v[1]} if isinstance(v, tuple) else v
                for k, v in d.items()
            }
            for s, d in shared_seed.items()
        },
        "cells": {
            f"{s}_{m}": {
                k: {"mean": v[0], "sd": v[1], "values": v[2] if len(v) > 2 else None}
                if isinstance(v, tuple)
                else v
                for k, v in c.items()
            }
            for (s, m), c in cells.items()
        },
        "aggregate": aggregate,
    }
    OUT_JSON.write_text(json.dumps(dump, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines += [
        "# E5 CivilComments — RESULTS (S9)",
        "",
        "Preregistration: [`PREREGISTRATION.md`](PREREGISTRATION.md).  ",
        "Gate: [`gpo_gate.json`](gpo_gate.json) (**pass**).  ",
        "Next steps: [`NEXT_STEPS.md`](NEXT_STEPS.md).  ",
        "E5v3 debt: [`E5V3_SELECTION_DEBT.md`](E5V3_SELECTION_DEBT.md) (M41).  ",
        "Status: [`S9_STATUS.md`](S9_STATUS.md).",
        "",
        "## Protocol",
        "",
        "- Headline search/select: `R_worst_gba` / softmin-GBA on `D_dev` (n=900).",
        "- **Report:** same-session stable eval on fingerprinted `test_fixed` "
        f"(**n={n}**, fp `{fp}`), **3 repeats** per prompt; seed scored once per "
        "matrix seed (`_shared_seed{{42,43,44}}`).",
        "- Format: `mean (±sd)` — within-job sd over 3 repeats; headline table sd "
        "is across the 3 matrix seeds (of per-seed means).",
        "- Scorer: `google/gemma-3-12b-it` (T≈0). Optimizer: `deepseek/deepseek-v4-pro`.",
        "- Regime-shift: `S_source=360`, `U_target=4000`, label budget `L=240` (AL).",
        "- Artifacts: `results/E5_s9_matrix/seed{{42,43,44}}/`, "
        "`results/E5_s9_matrix/stable_session/`, aggregate "
        "`stable_session/stable_aggregate.json`.",
        "",
        "> **This table is exploratory. Read [`POWER_AUDIT.md`](POWER_AUDIT.md) and",
        "> [`R15_CALIBRATION_CONTROL.md`](R15_CALIBRATION_CONTROL.md) before using it.**",
        "> Under a paired bootstrap none of the 31 method×seed contrasts is resolvable at",
        "> 95%, ranks invert across matrix seeds, and the oracle upper bound falls below",
        "> the seed prompt. A control of 12 one-line strictness edits (R15 — no search, no",
        "> group labels, no label budget) lands 4th of 35 on identical rows, and no cell",
        "> here beats it at Holm < 0.05. Headline claims moved to",
        "> [`PREREGISTRATION_S10.md`](PREREGISTRATION_S10.md). `R_worst_group` is",
        "> algebraically identical to `R_worst_gba` on balanced cells and carries no",
        "> information — it is kept only for continuity and will be dropped.",
        "",
        "## Headline — mean over seeds (stable ×3)",
        "",
        "Sorted by mean `R_worst_gba`. Δ = method − seed (seed mean gba "
        f"= {seed_gba:.3f}).",
        "",
        "| method | R_worst_gba | R_worst_group (Acc) | R_global | "
        "softmin-GBA | Δ gba | n seeds |",
        "|--------|------------:|--------------------:|---------:|-----------:|------:|--------:|",
    ]

    ranked = sorted(
        METHOD_ORDER,
        key=lambda m: -(aggregate[m]["R_worst_gba_raw"]["mean"] or -1),
    )
    for m in ranked:
        if aggregate[m]["R_worst_gba_raw"]["mean"] is None:
            continue
        a = aggregate[m]
        gba = a["R_worst_gba_raw"]
        wg = a["R_worst_group"]
        glob = a["R_global"]
        soft = a["R_soft_min_gba"]
        n_s = len(gba["seed_means"])
        delta = (gba["mean"] - seed_gba) if m != "seed" else None
        dlt = f"{delta:+.3f}" if delta is not None else "—"
        rid = METHOD_META[m][0]
        lines.append(
            f"| {rid} {m} | {_fmt(gba['mean'], gba['sd'])} | "
            f"{_fmt(wg['mean'], wg['sd'])} | {_fmt(glob['mean'], glob['sd'])} | "
            f"{_fmt(soft['mean'], soft['sd'])} | {dlt} | {n_s} |"
        )

    lines += [
        "",
        "### Extended metrics (mean over seeds)",
        "",
        "| method | toxic_recall | specificity |",
        "|--------|-------------:|------------:|",
    ]
    for m in ranked:
        if aggregate[m]["R_worst_gba_raw"]["mean"] is None:
            continue
        a = aggregate[m]
        lines.append(
            f"| {m} | {_fmt(a['toxic_recall']['mean'], a['toxic_recall']['sd'])} | "
            f"{_fmt(a['specificity']['mean'], a['specificity']['sd'])} |"
        )

    for s in ("42", "43", "44"):
        lines += [
            "",
            f"## Per-seed detail — seed {s} (stable ×3 within-job ±)",
            "",
            "| method | R_worst_gba | R_worst_group | R_global | softmin | recall | spec |",
            "|--------|------------:|--------------:|---------:|--------:|-------:|-----:|",
        ]
        # seed row
        sh = shared_seed.get(s) or {}
        if sh:
            lines.append(
                f"| seed | {_fmt(*(sh['R_worst_gba_raw']))} | "
                f"{_fmt(*(sh['R_worst_group']))} | {_fmt(*(sh['R_global']))} | "
                f"{_fmt(*(sh['R_soft_min_gba']))} | {_fmt(*(sh['toxic_recall']))} | "
                f"{_fmt(*(sh['specificity']))} |"
            )
        for m in METHOD_ORDER:
            if m == "seed":
                continue
            c = cells.get((s, m))
            if not c:
                continue
            lines.append(
                f"| {m} | {_fmt(c['R_worst_gba_raw'][0], c['R_worst_gba_raw'][1])} | "
                f"{_fmt(c['R_worst_group'][0], c['R_worst_group'][1])} | "
                f"{_fmt(c['R_global'][0], c['R_global'][1])} | "
                f"{_fmt(c['R_soft_min_gba'][0], c['R_soft_min_gba'][1])} | "
                f"{_fmt(c['toxic_recall'][0], c['toxic_recall'][1])} | "
                f"{_fmt(c['specificity'][0], c['specificity'][1])} |"
            )

    lines += [
        "",
        "## Methods & setup",
        "",
        "| ID | Method | Setup in this matrix |",
        "|----|--------|----------------------|",
    ]
    for m in METHOD_ORDER:
        rid, name, setup = METHOD_META[m]
        lines.append(f"| {rid} | **{name}** (`{m}`) | {setup} |")

    lines += [
        "",
        "### Shared constants",
        "",
        "- Same F7 prompt contract / parser for all methods.",
        "- Selection / gate on `D_dev`; never optimize on `test_fixed`.",
        "- OPRO (R13) and MIPROv2 (R14) **deferred** (not in this table).",
        "- Exploratory cost caps (APO/Random-AL/Oracle rounds=2; reduced Evo; "
        "lite GEPA) — not full native budgets; interpret accordingly.",
        "",
        "## Gate (Yelp→Flipkart)",
        "",
        "**pass** — see `gpo_gate.json`.",
        "",
        "## Success criteria (provisional, stable means)",
        "",
    ]
    prime = aggregate["prime"]["R_worst_gba_raw"]["mean"]
    if prime is not None:
        lines.append(
            f"- PRIME mean gba {prime:.3f}; Δ vs seed "
            f"{prime - seed_gba:+.3f} "
            f"(prereg target Δ≥+0.05 — **not met**)."
        )
    # best of APO GPO Random-AL
    rivals = []
    for m in ("apo", "gpo", "random_al"):
        mm = aggregate[m]["R_worst_gba_raw"]["mean"]
        if mm is not None:
            rivals.append((mm, m))
    if rivals and prime is not None:
        best_r, best_n = max(rivals)
        lines.append(
            f"- Best of {{APO, GPO, Random-AL}}: **{best_n}** {best_r:.3f}; "
            f"PRIME − rival = {prime - best_r:+.3f} "
            f"(prereg want ≥+0.02)."
        )
    lines += [
        "",
        "## Resume / regenerate table",
        "",
        "```powershell",
        "$env:PYTHONIOENCODING='utf-8'",
        "python scripts/run_e5_s9_stable_batch.py --seeds 42,43,44 --repeats 3",
        "python scripts/aggregate_e5_s9_stable.py",
        "```",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    print("HEADLINE:")
    for m in ranked:
        a = aggregate[m]["R_worst_gba_raw"]
        if a["mean"] is None:
            continue
        print(f"  {m:14} {_fmt(a['mean'], a['sd'])}")


if __name__ == "__main__":
    main()

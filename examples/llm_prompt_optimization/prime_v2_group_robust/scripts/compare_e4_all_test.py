#!/usr/bin/env python3
"""Compare all E4 CivilComments runs that share the same capped test (n=800).

Writes:
  experiments/E4_civilcomments/compare_all_test/RESULTS.md
  experiments/E4_civilcomments/compare_all_test/table.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))

from prime.data.civilcomments_loader import ORACLE_GROUP_NAMES  # noqa: E402

GN = list(ORACLE_GROUP_NAMES)
OUT = PKG_ROOT / "experiments" / "E4_civilcomments" / "compare_all_test"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _worst(ft: Dict[str, Any]) -> Tuple[Optional[str], Optional[float]]:
    ca = ft.get("cluster_accuracies") or {}
    if not ca:
        return None, ft.get("R_worst_group")
    gid, acc = min(((int(k), float(v)) for k, v in ca.items()), key=lambda kv: kv[1])
    name = GN[gid] if gid < len(GN) else str(gid)
    return name, acc


def _test_user_fp(run: Path) -> Optional[str]:
    p = run / "evals" / "eval_test" / "user_ids.npy"
    if not p.is_file():
        return None
    import hashlib

    u = np.load(p)
    return hashlib.sha256(u.tobytes()).hexdigest()[:12]


def _sel_cycle(summary: Dict[str, Any]) -> Optional[int]:
    best = summary.get("best_selection_key")
    cycles = summary.get("cycles") or []
    if best is not None:
        for c in cycles:
            if c.get("val_selection_key") == best:
                return int(c["cycle"])
        try:
            b0 = float(best[0])
            for c in cycles:
                vk = c.get("val_selection_key") or [None]
                if vk and vk[0] is not None and abs(float(vk[0]) - b0) < 1e-9:
                    return int(c["cycle"])
        except (TypeError, ValueError, IndexError):
            pass
    # fallback: lexicographic max val key
    best_c, best_vk = None, None
    for c in cycles:
        vk = c.get("val_selection_key")
        if vk is None:
            continue
        if best_vk is None or tuple(vk) > tuple(best_vk):
            best_vk, best_c = vk, int(c["cycle"])
    return best_c


def _prompt_meta(run: Path, sel_cycle: Optional[int]) -> Dict[str, Any]:
    init_p = run / "initial_prompt.txt"
    init = init_p.read_text(encoding="utf-8").strip() if init_p.is_file() else ""
    bp = None
    if sel_cycle is not None:
        cand = run / f"al_iter_{sel_cycle}" / "best_prompt.txt"
        if cand.is_file():
            bp = cand.read_text(encoding="utf-8").strip()
    if bp is None:
        # last existing best
        for c in (3, 2, 1):
            cand = run / f"al_iter_{c}" / "best_prompt.txt"
            if cand.is_file():
                bp = cand.read_text(encoding="utf-8").strip()
                break
    changed = bool(bp) and bool(init) and bp != init
    return {
        "prompt_changed": changed,
        "prompt_len": len(bp) if bp else None,
        "init_len": len(init) if init else None,
    }


def collect() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    # Day-1 seed baseline (same test slice)
    bm = _load(
        PKG_ROOT
        / "experiments"
        / "E4_civilcomments"
        / "baseline_initial_prompt_test"
        / "metrics.json"
    )
    ft = bm["ensemble"]
    wg, _ = _worst(ft)
    rows.append(
        {
            "id": "seed_day1",
            "label": "Seed baseline",
            "arm": "seed",
            "fitness": "—",
            "form": "initial prompt, no evolution",
            "date": "2026-08-05",
            "R_global": float(ft["R_global"]),
            "R_macro": float(ft["R_macro"]),
            "R_worst_group": float(ft["R_worst_group"]),
            "worst_group": wg,
            "Acc_tox": float(ft["accuracy_per_class"]["1"]),
            "Acc_non": float(ft["accuracy_per_class"]["0"]),
            "CVaR": float(ft["CVaR_cluster"]),
            "kappa": float(ft["mean_kappa"]),
            "sel_cycle": None,
            "heir_source": None,
            "prompt_changed": False,
            "prompt_len": 1488,
            "test_fp": "875e9f6b1e03",
            "n": 800,
            "run": "experiments/E4_civilcomments/baseline_initial_prompt_test",
            "same_day_seed_Rg": None,
            "same_day_seed_Rwg": None,
            "paired_net_vs_same_day_seed": None,
            "notes": "official day-1 seed on shared test",
        }
    )

    specs = [
        (
            "A_global_3x3",
            "A",
            "global",
            "3x3 lean, D_select=420, rotate on",
            "2026-08-05",
            PKG_ROOT / "results/E4_civilcomments_arm_a_global/seed42_20260805_104411",
            "control; worst-group regresses vs seed",
        ),
        (
            "B_lex_3x3",
            "B",
            "min_group_lex",
            "3x3 lean, D_select=420, rotate on",
            "2026-08-05",
            PKG_ROOT / "results/E4_civilcomments_arm_b_min_group/seed42_20260805_135157",
            "final prompt ~ seed (strip); delta mostly re-score noise",
        ),
        (
            "B_lex_4x20",
            "B",
            "min_group_lex",
            "4x20, D_select=420, rotate on (pre-M27)",
            "2026-08-05/06",
            PKG_ROOT / "results/E4_civilcomments_arm_b_min_group_4x20/seed42_20260805_214134",
            "C1 hill-climb real; C2+ stale-score locked (M27)",
        ),
        (
            "B_lex_2x20",
            "B",
            "min_group_lex",
            "2x20, D_select=720, rotate off (post-M27/M28)",
            "2026-08-07",
            PKG_ROOT / "results/E4_civilcomments_arm_b_min_group_2x20/seed42_20260807_115056",
            "C2 evolved; val kept C1; test regresses vs same-day seed",
        ),
    ]

    for rid, arm, mode, form, date, run, notes in specs:
        s = _load(run / "summary.json")
        ft = s["final_test"]
        wg, _ = _worst(ft)
        sc = _sel_cycle(s)
        pm = _prompt_meta(run, sc)
        heir = None
        for c in s.get("cycles") or []:
            if sc is not None and int(c.get("cycle", -1)) == sc:
                heir = (c.get("pareto") or {}).get("heir_source")
        noise_p = run / "evals" / "test_noise" / "noise_report.json"
        sd_rg = sd_rwg = paired = None
        if noise_p.is_file():
            nr = _load(noise_p)
            seed = nr.get("seed") or {}
            sd_rg = seed.get("R_global")
            sd_rwg = seed.get("R_worst_group")
            paired = (seed.get("paired_vs_final_r0") or {}).get("net_final_minus_seed")
        rows.append(
            {
                "id": rid,
                "label": f"{arm} {form.split(',')[0]}",
                "arm": arm,
                "fitness": mode,
                "form": form,
                "date": date,
                "R_global": float(ft["R_global"]),
                "R_macro": float(ft["R_macro"]),
                "R_worst_group": float(ft["R_worst_group"]),
                "worst_group": wg,
                "Acc_tox": float((ft.get("accuracy_per_class") or {}).get("1", float("nan"))),
                "Acc_non": float((ft.get("accuracy_per_class") or {}).get("0", float("nan"))),
                "CVaR": float(ft.get("CVaR_cluster") or float("nan")),
                "kappa": ft.get("mean_kappa"),
                "sel_cycle": sc,
                "heir_source": heir,
                "prompt_changed": pm["prompt_changed"],
                "prompt_len": pm["prompt_len"],
                "test_fp": _test_user_fp(run),
                "n": int(ft.get("num_examples") or ft.get("num_users") or 800),
                "run": str(run.relative_to(PKG_ROOT)).replace("\\", "/"),
                "same_day_seed_Rg": sd_rg,
                "same_day_seed_Rwg": sd_rwg,
                "paired_net_vs_same_day_seed": paired,
                "notes": notes,
            }
        )
    return rows


def _fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "—"
    try:
        if isinstance(x, float) and (x != x):  # nan
            return "—"
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def write_md(rows: List[Dict[str, Any]]) -> str:
    seed = rows[0]
    assert seed["id"] == "seed_day1"
    fps = {r.get("test_fp") for r in rows if r.get("test_fp")}
    lines = [
        "# E4 CivilComments - all runs on the same test",
        "",
        f"Shared test fingerprint `user_ids` sha256[:12] = **`{seed['test_fp']}`**, n=**800**, seed=42 cap.",
        f"Observed fingerprints in table: `{sorted(fps)}` - all match (comparable).",
        "",
        "Caveat (M29): point estimates from different calendar days include API drift.",
        "Prefer **same-day seed** columns / paired nets when present.",
        "",
        "## Headline (official `final_test` / baseline ensemble)",
        "",
        "| run | fitness | form | date | R_global | R_macro | R_worst_group | worst group | Acc tox | CVaR | kappa | sel | prompt |",
        "|---|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        pr = "initial"
        if r["id"] != "seed_day1":
            pr = "changed" if r.get("prompt_changed") else "~seed"
            if r.get("prompt_len"):
                pr += f" ({r['prompt_len']})"
        lines.append(
            "| {label} | {fit} | {form} | {date} | {rg} | {rm} | {rw} | {wg} | {tox} | {cv} | {k} | {sel} | {pr} |".format(
                label=r["label"],
                fit=r["fitness"],
                form=r["form"],
                date=r["date"],
                rg=_fmt(r["R_global"]),
                rm=_fmt(r["R_macro"]),
                rw=_fmt(r["R_worst_group"]),
                wg=r.get("worst_group") or "-",
                tox=_fmt(r["Acc_tox"], 3),
                cv=_fmt(r["CVaR"], 3),
                k=_fmt(r.get("kappa"), 3),
                sel=r.get("sel_cycle") if r.get("sel_cycle") is not None else "-",
                pr=pr,
            )
        )

    lines += [
        "",
        "### Delta vs day-1 seed baseline",
        "",
        "| run | dR_global | dR_macro | dR_worst_group | dAcc tox |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        if r["id"] == "seed_day1":
            lines.append("| Seed baseline | 0 | 0 | 0 | 0 |")
            continue
        lines.append(
            "| {label} | {dg:+.4f} | {dm:+.4f} | {dw:+.4f} | {dt:+.3f} |".format(
                label=r["label"],
                dg=r["R_global"] - seed["R_global"],
                dm=r["R_macro"] - seed["R_macro"],
                dw=r["R_worst_group"] - seed["R_worst_group"],
                dt=r["Acc_tox"] - seed["Acc_tox"],
            )
        )

    lines += [
        "",
        "## Same-day seed (noise harness) - preferred for claims",
        "",
        "| run | final R_global | same-day seed R_global | final R_wg | same-day seed R_wg | paired net (final-seed flips) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    any_sd = False
    for r in rows:
        if r.get("same_day_seed_Rg") is None:
            continue
        any_sd = True
        lines.append(
            "| {label} | {fg} | {sg} | {fw} | {sw} | {net} |".format(
                label=r["label"],
                fg=_fmt(r["R_global"]),
                sg=_fmt(r["same_day_seed_Rg"]),
                fw=_fmt(r["R_worst_group"]),
                sw=_fmt(r["same_day_seed_Rwg"]),
                net=r.get("paired_net_vs_same_day_seed"),
            )
        )
    if not any_sd:
        lines.append("| - | - | - | - | - | - |")

    lines += [
        "",
        "## Notes per run",
        "",
    ]
    for r in rows:
        lines.append(f"- **{r['label']}** (`{r['run']}`): {r.get('notes') or '-'}")

    lines += [
        "",
        "## Not included",
        "",
        "- Arm C style (`E4_civilcomments_arm_c_style/...`): interrupted, no final test.",
        "- Cross-day seed re-scores without shared `user_ids.npy` check are omitted.",
        "",
        "Generated by `scripts/compare_e4_all_test.py`.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    rows = collect()
    OUT.mkdir(parents=True, exist_ok=True)
    md = write_md(rows)
    (OUT / "RESULTS.md").write_text(md, encoding="utf-8")
    (OUT / "table.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    # Avoid Windows console encoding issues on unicode.
    sys.stdout.buffer.write((md + f"\n\nwrote {OUT}\n").encode("utf-8", errors="replace"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Same-day re-score of every prompt in the E4 compare_all_test table.

Uses one shared test slice (ref run evals/eval_test) and scores:
  seed, A 3x3, B 3x3, B 4x20, B 2x20 selected prompts (+ optional repeats).

Writes under experiments/E4_civilcomments/compare_all_test/same_day_rescore/:
  RESULTS.md, table.json, preds/<id>.npy

Usage:
  python scripts/e4_rescore_all_prompts.py
  python scripts/e4_rescore_all_prompts.py --repeats 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))

from scripts.compare_e4_all_test import _prompt_meta, _sel_cycle  # noqa: E402
from scripts.eval_prompt_on_test import _load_test_slice  # noqa: E402

OUT = PKG_ROOT / "experiments" / "E4_civilcomments" / "compare_all_test" / "same_day_rescore"
REF = (
    PKG_ROOT
    / "results"
    / "E4_civilcomments_arm_b_min_group_2x20"
    / "seed42_20260807_115056"
)

PROMPTS: List[Dict[str, Any]] = [
    {
        "id": "seed",
        "label": "Seed baseline",
        "run": REF,  # same initial_prompt across arms
        "prompt_path": "initial_prompt.txt",
    },
    {
        "id": "A_3x3",
        "label": "A 3x3 lean",
        "run": PKG_ROOT / "results/E4_civilcomments_arm_a_global/seed42_20260805_104411",
        "prompt_path": None,  # selected cycle
    },
    {
        "id": "B_3x3",
        "label": "B 3x3 lean",
        "run": PKG_ROOT / "results/E4_civilcomments_arm_b_min_group/seed42_20260805_135157",
        "prompt_path": None,
    },
    {
        "id": "B_4x20",
        "label": "B 4x20",
        "run": PKG_ROOT
        / "results/E4_civilcomments_arm_b_min_group_4x20/seed42_20260805_214134",
        "prompt_path": None,
    },
    {
        "id": "B_2x20",
        "label": "B 2x20",
        "run": PKG_ROOT
        / "results/E4_civilcomments_arm_b_min_group_2x20/seed42_20260807_115056",
        "prompt_path": None,
    },
]


def _resolve_prompt(entry: Dict[str, Any]) -> Tuple[str, str]:
    run = Path(entry["run"])
    if entry.get("prompt_path"):
        p = run / entry["prompt_path"]
        return p.read_text(encoding="utf-8"), str(p.relative_to(PKG_ROOT))
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    sel = _sel_cycle(summary)
    meta = _prompt_meta(run, sel)
    cand = run / f"al_iter_{sel}" / "best_prompt.txt"
    if not cand.is_file():
        raise FileNotFoundError(f"no best_prompt for {entry['id']} cycle={sel}")
    note = f"{cand.relative_to(PKG_ROOT)} (sel_cycle={sel}, len={meta.get('prompt_len')})"
    return cand.read_text(encoding="utf-8"), note


def _metrics(ens, labels, user_ids, cluster_ids, cfg) -> Dict[str, Any]:
    from prime.fitness.metrics import compute_metrics
    from prime.data.civilcomments_loader import ORACLE_GROUP_NAMES

    m = compute_metrics(
        ens,
        labels,
        user_ids,
        cluster_ids=cluster_ids,
        cvar_quantile=cfg.fitness.cvar_quantile,
        beta_a=cfg.fitness.beta_a,
        beta_b=cfg.fitness.beta_b,
        tail_quantile=cfg.active_learning.tail_quantile,
        shrink_prior_weight=cfg.fitness.shrink_prior_weight,
        class_balanced=cfg.fitness.class_balanced,
    )
    ca = m.get("cluster_accuracies") or {}
    worst_gid, worst_acc = None, None
    if ca:
        worst_gid, worst_acc = min(
            ((int(k), float(v)) for k, v in ca.items()), key=lambda kv: kv[1]
        )
    names = list(ORACLE_GROUP_NAMES)
    return {
        "R_global": m.get("R_global"),
        "R_macro": m.get("R_macro"),
        "R_worst_group": m.get("R_worst_group"),
        "CVaR_cluster": m.get("CVaR_cluster"),
        "mean_kappa": m.get("mean_kappa"),
        "accuracy_per_class": m.get("accuracy_per_class"),
        "cluster_accuracies": {str(k): float(v) for k, v in ca.items()},
        "worst_group": names[worst_gid] if worst_gid is not None and worst_gid < len(names) else worst_gid,
        "worst_group_acc": worst_acc,
    }


def _paired(final: np.ndarray, seed: np.ndarray, labels: np.ndarray) -> Dict[str, int]:
    c_f, c_s = final == labels, seed == labels
    b01 = int((c_s & ~c_f).sum())  # seed ok, final wrong
    b10 = int((~c_s & c_f).sum())  # seed wrong, final ok
    return {
        "seed_ok_final_wrong": b01,
        "seed_wrong_final_ok": b10,
        "net_final_minus_seed": b10 - b01,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--ref-run", type=Path, default=REF)
    args = ap.parse_args()

    from prime.config import load_config
    from prime.workers.ensemble import build_workers, load_dotenv_if_present, parallel_predict

    load_dotenv_if_present()
    OUT.mkdir(parents=True, exist_ok=True)
    preds_dir = OUT / "preds"
    preds_dir.mkdir(exist_ok=True)
    report_path = OUT / "table.json"
    if report_path.is_file() and not args.force:
        print(f"[rescore] cache hit {report_path} (pass --force to redo)", flush=True)
        print((OUT / "RESULTS.md").read_text(encoding="utf-8"), flush=True)
        return 0

    ref = args.ref_run.resolve()
    cfg = load_config(ref / "config_resolved.yaml")
    slice_ = _load_test_slice(ref, cfg)
    texts, labels, user_ids, cluster_ids = (
        slice_["texts"],
        slice_["labels"],
        slice_["user_ids"],
        slice_["cluster_ids"],
    )
    fp = hashlib.sha256(np.asarray(user_ids).tobytes()).hexdigest()[:12]
    workers = build_workers(cfg.ensemble)
    print(
        f"[rescore] n={len(texts)} fp={fp} repeats={args.repeats} "
        f"workers={[w.model_name for w in workers]}",
        flush=True,
    )

    rows: List[Dict[str, Any]] = []
    seed_ens: Optional[np.ndarray] = None

    for entry in PROMPTS:
        prompt, note = _resolve_prompt(entry)
        print(f"\n[rescore] === {entry['id']}: {note} ===", flush=True)
        (OUT / f"prompt_{entry['id']}.txt").write_text(prompt, encoding="utf-8")
        run_mets: List[Dict[str, Any]] = []
        last_ens = None
        for r in range(args.repeats):
            print(f"[rescore] {entry['id']} repeat {r + 1}/{args.repeats}...", flush=True)
            ens, _ = parallel_predict(
                workers,
                texts,
                prompt,
                max_parallel=cfg.ensemble.max_parallel,
                tie_break=cfg.ensemble.tie_break,
                aggregation=cfg.ensemble.aggregation,
                label_space=cfg.dataset.label_space,
            )
            ens = np.asarray(ens, dtype=np.int16)
            last_ens = ens
            np.save(preds_dir / f"{entry['id']}_r{r}.npy", ens)
            met = _metrics(ens, labels, user_ids, cluster_ids, cfg)
            run_mets.append(met)
            tox = None
            apc = met.get("accuracy_per_class") or {}
            if "1" in apc or 1 in apc:
                tox = float(apc.get("1", apc.get(1)))
            print(
                f"  R_global={met['R_global']:.4f} R_wg={met['R_worst_group']:.4f} "
                f"worst={met['worst_group']} Acc_tox={tox}",
                flush=True,
            )

        primary = run_mets[0]
        if entry["id"] == "seed":
            seed_ens = last_ens

        paired = None
        if seed_ens is not None and entry["id"] != "seed" and last_ens is not None:
            paired = _paired(last_ens, seed_ens, labels)

        tox = None
        apc = primary.get("accuracy_per_class") or {}
        if "1" in apc or 1 in apc:
            tox = float(apc.get("1", apc.get(1)))

        rows.append(
            {
                "id": entry["id"],
                "label": entry["label"],
                "prompt_note": note,
                "prompt_len": len(prompt),
                "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:12],
                "metrics": primary,
                "repeats": run_mets if args.repeats > 1 else None,
                "Acc_tox": tox,
                "paired_vs_seed": paired,
            }
        )

    report = {
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "ref_run": str(ref),
        "test_fp": fp,
        "n_examples": int(len(texts)),
        "repeats": args.repeats,
        "rows": rows,
    }
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    seed = next(r for r in rows if r["id"] == "seed")
    lines = [
        "# E4 same-day re-score (all table prompts)",
        "",
        f"Shared test fp=`{fp}`, n={len(texts)}, repeats={args.repeats}.",
        f"Finished: {report['finished_utc']}",
        "",
        "| run | R_global | R_macro | R_worst_group | worst | Acc tox | net flips vs seed | prompt |",
        "|---|---:|---:|---:|---|---:|---:|---|",
    ]
    for r in rows:
        m = r["metrics"]
        net = "-"
        if r.get("paired_vs_seed"):
            net = str(r["paired_vs_seed"]["net_final_minus_seed"])
        elif r["id"] == "seed":
            net = "0"
        lines.append(
            "| {lab} | {rg:.4f} | {rm:.4f} | {rw:.4f} | {wg} | {tox} | {net} | {pl} |".format(
                lab=r["label"],
                rg=float(m["R_global"]),
                rm=float(m["R_macro"]),
                rw=float(m["R_worst_group"]),
                wg=m.get("worst_group") or "-",
                tox=f"{r['Acc_tox']:.3f}" if r.get("Acc_tox") is not None else "-",
                net=net,
                pl=r["prompt_len"],
            )
        )
    lines += [
        "",
        "### Delta vs same-day seed",
        "",
        "| run | dR_global | dR_macro | dR_worst_group |",
        "|---|---:|---:|---:|",
    ]
    sg, sm, sw = (
        float(seed["metrics"]["R_global"]),
        float(seed["metrics"]["R_macro"]),
        float(seed["metrics"]["R_worst_group"]),
    )
    for r in rows:
        m = r["metrics"]
        lines.append(
            "| {lab} | {dg:+.4f} | {dm:+.4f} | {dw:+.4f} |".format(
                lab=r["label"],
                dg=float(m["R_global"]) - sg,
                dm=float(m["R_macro"]) - sm,
                dw=float(m["R_worst_group"]) - sw,
            )
        )
    lines += ["", "Generated by `scripts/e4_rescore_all_prompts.py`.", ""]
    (OUT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines), flush=True)
    print(f"[rescore] wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

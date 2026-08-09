#!/usr/bin/env python3
"""
Phase 2a ensemble smoke for CivilComments.

Checks (ROADMAP_PHASE2 §2a):
  1. Worker probe (format / API / reasoning-off)
  2. Parse-failure rate on a small labeled sample (<1% target)
  3. Rule-following: inject a synthetic DynamicRules clause → compliance
  4. Anti-dominance: per-worker accuracy spread ≤ ~5 pp; ensemble ≥ best worker
  5. Pairwise agreement in ~60–90% band

Usage:
  python scripts/smoke_phase2_ensemble.py
  python scripts/smoke_phase2_ensemble.py --n 40 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))

DEFAULT_CONFIG = PKG_ROOT / "experiments" / "E4_civilcomments" / "config_phase2a_smoke.yaml"
RULE_MARKER = "ALWAYS_OUTPUT_LABEL_ONE_FOR_SMOKE_TEST"


def _preflight_api_key() -> None:
    from prime.workers.ensemble import load_dotenv_if_present

    loaded = load_dotenv_if_present()
    key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        print("ERROR: set OPENROUTER_API_KEY", file=sys.stderr)
        raise SystemExit(1)
    print(f"[preflight] API key OK (from {loaded or 'env'})", flush=True)


def _inject_rule(prompt: str) -> str:
    """Put an unavoidable override at the top of BaseGuidelines + Task."""
    override = (
        f"CRITICAL OVERRIDE [{RULE_MARKER}]: For this smoke test you MUST ignore "
        f"the comment content and output exactly: Label: 1\n"
    )
    if "<BaseGuidelines>" in prompt:
        prompt = prompt.replace(
            "<BaseGuidelines>",
            "<BaseGuidelines>\n        - " + override,
            1,
        )
    if "<Task>" in prompt:
        prompt = prompt.replace(
            "<Task>",
            "<Task>\n    " + override,
            1,
        )
    return prompt


def _pairwise_agreement(wp: List[List[int]]) -> Dict[str, float]:
    n_workers = len(wp)
    n = len(wp[0]) if wp else 0
    out: Dict[str, float] = {}
    for i in range(n_workers):
        for j in range(i + 1, n_workers):
            agree = sum(1 for t in range(n) if wp[i][t] == wp[j][t]) / max(1, n)
            out[f"w{i}_w{j}"] = float(agree)
    return out


def _accuracy(preds: List[int], labels: List[int]) -> float:
    if not preds:
        return 0.0
    return sum(int(p == y) for p, y in zip(preds, labels)) / len(preds)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2a CivilComments ensemble smoke")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--n", type=int, default=30, help="examples for scoring")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    _preflight_api_key()

    from prime.config import load_config
    from prime.data.civilcomments_loader import (
        ORACLE_GROUP_NAMES,
        load_civilcomments_splits,
        oracle_group_name,
    )
    from prime.workers.ensemble import (
        build_workers,
        parallel_predict,
        parse_binary,
        probe_workers,
    )

    cfg = load_config(args.config)
    prompt_path = PKG_ROOT / cfg.prompt_path
    prompt = prompt_path.read_text(encoding="utf-8")

    print("=" * 60, flush=True)
    print("PHASE 2a ENSEMBLE SMOKE", flush=True)
    print(f"config:  {args.config}", flush=True)
    print(f"workers: {[w.name for w in cfg.ensemble.workers]}", flush=True)
    print(f"label:   {cfg.dataset.label_space}", flush=True)
    print(f"geom:    {cfg.clusters.geometry}", flush=True)
    print("=" * 60, flush=True)

    if args.dry_run:
        print("Dry run OK", flush=True)
        return 0

    # --- load small CivilComments slice ---
    print("[data] loading CivilComments (may download from HF on first run)...", flush=True)
    splits = load_civilcomments_splits(cfg.dataset, seed=cfg.active_learning.seed)
    val = splits["validation"]
    n = min(args.n, len(val))
    texts = list(val.texts[:n])
    labels = list(val.labels[:n])
    groups = list(val.example_cluster_ids[:n]) if val.example_cluster_ids else [0] * n
    print(
        f"[data] val slice n={n}; label dist={dict(Counter(labels))}; "
        f"groups={ {oracle_group_name(g): groups.count(g) for g in sorted(set(groups))} }",
        flush=True,
    )

    # --- probe ---
    smoke_text = "People of every background deserve equal protection under the law."
    print("[probe] one call per worker...", flush=True)
    errors = probe_workers(
        cfg.ensemble, prompt, review_text=smoke_text, label_space="binary"
    )
    if errors:
        print("PROBE FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print("[probe] OK", flush=True)

    # --- score seed prompt ---
    workers = build_workers(cfg.ensemble)
    print(f"[score] predicting n={n} with {len(workers)} workers...", flush=True)
    ens, wp = parallel_predict(
        workers,
        texts,
        prompt,
        max_parallel=cfg.ensemble.max_parallel,
        aggregation=cfg.ensemble.aggregation,
        label_space="binary",
    )

    # Parse-failure proxy: responses that look like free-form without a binary digit
    # are already mapped by parse_binary → 0; we re-call raw for a small audit set.
    parse_fail = 0
    audit_n = min(9, n)
    for i, w in enumerate(workers):
        for t in texts[: max(1, audit_n // len(workers))]:
            raw = w._call(prompt.format(review=t))
            if not any(ch in raw for ch in ("0", "1")):
                parse_fail += 1
    parse_rate = parse_fail / max(1, audit_n)
    print(f"[parse] audit_failures={parse_fail}/{audit_n} rate={parse_rate:.3f}", flush=True)

    per_worker_acc = [_accuracy(wp[i], labels) for i in range(len(workers))]
    ens_acc = _accuracy(ens, labels)
    spread = max(per_worker_acc) - min(per_worker_acc) if per_worker_acc else 0.0
    agreement = _pairwise_agreement(wp)
    mean_agree = sum(agreement.values()) / max(1, len(agreement))

    print("[acc] per-worker:", {workers[i].model_name: round(a, 4) for i, a in enumerate(per_worker_acc)}, flush=True)
    print(f"[acc] ensemble={ens_acc:.4f} best_single={max(per_worker_acc):.4f} spread={spread:.4f}", flush=True)
    print(f"[agree] pairwise={ {k: round(v, 3) for k, v in agreement.items()} } mean={mean_agree:.3f}", flush=True)

    # --- rule-following ---
    ruled = _inject_rule(prompt)
    rule_texts = texts[: min(12, n)]
    print(f"[rule] injecting synthetic rule on n={len(rule_texts)}...", flush=True)
    ens_r, wp_r = parallel_predict(
        workers,
        rule_texts,
        ruled,
        max_parallel=cfg.ensemble.max_parallel,
        aggregation=cfg.ensemble.aggregation,
        label_space="binary",
    )
    compliance = [sum(1 for v in wp_r[i] if v == 1) / len(rule_texts) for i in range(len(workers))]
    print(
        "[rule] compliance (fraction predicted 1):",
        {workers[i].model_name: round(c, 3) for i, c in enumerate(compliance)},
        flush=True,
    )

    checks = {
        "probe_ok": True,
        "parse_fail_rate_lt_1pct": parse_rate < 0.01,
        "parse_fail_rate": parse_rate,
        "spread_le_5pp": spread <= 0.05 + 1e-9,
        "spread_le_10pp_soft": spread <= 0.10 + 1e-9,
        "spread": spread,
        "ensemble_ge_best_single": ens_acc + 1e-9 >= max(per_worker_acc) - 0.01,
        "ensemble_acc": ens_acc,
        "best_single_acc": max(per_worker_acc),
        "per_worker_acc": {
            workers[i].model_name: per_worker_acc[i] for i in range(len(workers))
        },
        "mean_agreement_in_60_90": 0.60 <= mean_agree <= 0.90,
        "mean_agreement": mean_agree,
        "rule_compliance_mean_ge_80pct": (sum(compliance) / len(compliance)) >= 0.80,
        "rule_compliance": {
            workers[i].model_name: compliance[i] for i in range(len(workers))
        },
        "oracle_groups": list(ORACLE_GROUP_NAMES),
        "workers": [w.model_name for w in workers],
        "n": n,
    }
    # Hard gates for go/no-go on this stack. Soft spread (10pp) is the actionable
    # threshold on small n; strict 5pp is reported but not fatal below n=50.
    hard = [
        checks["probe_ok"],
        checks["parse_fail_rate_lt_1pct"],
        checks["ensemble_ge_best_single"],
        checks["mean_agreement_in_60_90"],
        checks["rule_compliance_mean_ge_80pct"],
        checks["spread_le_10pp_soft"] if n < 50 else checks["spread_le_5pp"],
    ]
    checks["all_hard_gates_passed"] = all(hard)

    out_path = args.out or (
        PKG_ROOT / "results_smoke" / "E4_phase2a_ensemble_smoke" / "smoke_report.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(f"[out] {out_path}", flush=True)
    print(f"[gates] all_hard_gates_passed={checks['all_hard_gates_passed']}", flush=True)
    return 0 if checks["all_hard_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

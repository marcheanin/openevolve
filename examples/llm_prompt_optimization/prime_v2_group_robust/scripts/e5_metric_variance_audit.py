#!/usr/bin/env python
"""Compare candidate headline metrics by *statistical power*, not by value.

Recomputes hard-min GBA, CVaR@k over groups, softmin and worst-class accuracy
from the cached stable_session predictions, then measures, for each metric:

  * within-job repeat spread (scorer nondeterminism),
  * paired-bootstrap CI width for (method - seed) over test_fixed examples,
  * how often the method-vs-seed contrast is resolvable at 95%.

The metric with the narrowest paired CI at equal data cost is the one that can
actually falsify a claim; that is the selection criterion here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SS = ROOT / "results/E5_s9_matrix/stable_session"
FIXED = ROOT / "experiments/E5_civilcomments/fixed_sets/test_fixed_materialized.json"

METHODS = [
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


def load_fixed():
    mat = json.loads(FIXED.read_text(encoding="utf-8"))
    y = np.asarray(mat["labels"], dtype=np.int8)
    cids = np.asarray(mat["cluster_ids"], dtype=np.int16)
    return y, cids


class MetricSuite:
    """Vectorised group metrics over a fixed (y, cluster) layout."""

    def __init__(self, y: np.ndarray, cids: np.ndarray, exclude_group: int = 0):
        self.y = y
        self.cids = cids
        self.groups = [g for g in sorted(set(cids.tolist())) if g != exclude_group]
        # cell index lists: (group, label) -> positions
        self.cells = {
            (g, lab): np.where((cids == g) & (y == lab))[0]
            for g in self.groups
            for lab in (0, 1)
        }

    def group_gba(self, preds: np.ndarray) -> np.ndarray:
        out = np.empty(len(self.groups))
        for i, g in enumerate(self.groups):
            tnr = float((preds[self.cells[(g, 0)]] == 0).mean())
            tpr = float((preds[self.cells[(g, 1)]] == 1).mean())
            out[i] = 0.5 * (tpr + tnr)
        return out

    def all_metrics(self, preds: np.ndarray) -> dict[str, float]:
        gba = np.sort(self.group_gba(preds))
        k25 = max(1, int(round(0.25 * len(gba))))
        tau = 0.10
        softmin = float(-tau * np.log(np.mean(np.exp(-gba / tau))))
        # worst-class accuracy: group-free control metric (Yang et al. 2023)
        cls_acc = [float((preds[self.y == lab] == lab).mean()) for lab in (0, 1)]
        return {
            "hard_min": float(gba[0]),
            "cvar25": float(gba[:k25].mean()),
            "cvar50": float(gba[: max(1, len(gba) // 2)].mean()),
            "softmin": softmin,
            "mean_gba": float(gba.mean()),
            "worst_class": float(min(cls_acc)),
        }

    def paired_bootstrap(
        self, pred_a: np.ndarray, pred_b: np.ndarray, n_boot: int, rng: np.random.Generator
    ) -> dict[str, np.ndarray]:
        """Resample within each (group,label) cell; both prompts see same rows."""
        keys = None
        acc: dict[str, list[float]] = {}
        for _ in range(n_boot):
            idx = np.concatenate(
                [rng.choice(v, size=len(v), replace=True) for v in self.cells.values()]
            )
            # rebuild a metric suite view on the resampled rows
            ma = self._metrics_on_index(pred_a, idx)
            mb = self._metrics_on_index(pred_b, idx)
            if keys is None:
                keys = list(ma)
                acc = {k: [] for k in keys}
            for k in keys:
                acc[k].append(mb[k] - ma[k])
        return {k: np.asarray(v) for k, v in acc.items()}

    def _metrics_on_index(self, preds: np.ndarray, idx: np.ndarray) -> dict[str, float]:
        y = self.y[idx]
        c = self.cids[idx]
        p = preds[idx]
        gba = []
        for g in self.groups:
            m = c == g
            neg = m & (y == 0)
            pos = m & (y == 1)
            tnr = float((p[neg] == 0).mean()) if neg.any() else 0.5
            tpr = float((p[pos] == 1).mean()) if pos.any() else 0.5
            gba.append(0.5 * (tpr + tnr))
        gba = np.sort(np.asarray(gba))
        k25 = max(1, int(round(0.25 * len(gba))))
        tau = 0.10
        cls_acc = [float((p[y == lab] == lab).mean()) for lab in (0, 1)]
        return {
            "hard_min": float(gba[0]),
            "cvar25": float(gba[:k25].mean()),
            "cvar50": float(gba[: max(1, len(gba) // 2)].mean()),
            "softmin": float(-tau * np.log(np.mean(np.exp(-gba / tau)))),
            "mean_gba": float(gba.mean()),
            "worst_class": float(min(cls_acc)),
        }


def majority(job: Path, kind: str) -> np.ndarray | None:
    ps = [job / f"{kind}_repeat{r}_preds.npy" for r in range(3)]
    ps = [p for p in ps if p.is_file()]
    if not ps:
        return None
    return (np.stack([np.load(p) for p in ps]).mean(axis=0) >= 0.5).astype(np.int8)


def repeats(job: Path, kind: str) -> list[np.ndarray]:
    return [
        np.load(job / f"{kind}_repeat{r}_preds.npy")
        for r in range(3)
        if (job / f"{kind}_repeat{r}_preds.npy").is_file()
    ]


def holm(pvals: list[tuple[str, float]]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    order = sorted(pvals, key=lambda t: t[1])
    m = len(order)
    adj = {}
    running = 0.0
    for i, (name, p) in enumerate(order):
        running = max(running, (m - i) * p)
        adj[name] = min(1.0, running)
    return adj


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--out", type=Path, default=SS / "metric_variance_audit.json")
    args = ap.parse_args()

    y, cids = load_fixed()
    suite = MetricSuite(y, cids)
    keys = ["hard_min", "cvar25", "cvar50", "softmin", "mean_gba", "worst_class"]

    print(f"groups={suite.groups}  n={len(y)}  per-cell n={len(next(iter(suite.cells.values())))}")

    report: dict = {"n": int(len(y)), "groups": suite.groups, "n_boot": args.n_boot, "seeds": {}}

    print()
    print("=== A. Scorer nondeterminism (spread over 3 repeats of the SAME prompt) ===")
    spreads: dict[str, list[float]] = {k: [] for k in keys}
    for s in args.seeds.split(","):
        for m in METHODS + ["__seed__"]:
            job = SS / (f"_shared_seed{s}" if m == "__seed__" else f"seed{s}_{m}")
            kind = "seed" if m == "__seed__" else "final"
            rs = repeats(job, kind)
            if len(rs) < 2:
                continue
            vals = {k: [] for k in keys}
            for p in rs:
                mm = suite.all_metrics(p)
                for k in keys:
                    vals[k].append(mm[k])
            for k in keys:
                spreads[k].append(float(np.std(vals[k], ddof=1)))
    for k in keys:
        arr = np.asarray(spreads[k])
        print(f"  {k:12} mean repeat-sd = {arr.mean():.4f}   max = {arr.max():.4f}")
    report["repeat_sd"] = {k: float(np.mean(spreads[k])) for k in keys}

    print()
    print("=== B. Paired-bootstrap CI width for (method - seed), per metric ===")
    widths: dict[str, list[float]] = {k: [] for k in keys}
    resolved: dict[str, int] = {k: 0 for k in keys}
    total = 0
    per_seed: dict[str, dict] = {}

    for s in args.seeds.split(","):
        seed_pred = majority(SS / f"_shared_seed{s}", "seed")
        if seed_pred is None:
            continue
        per_seed[s] = {"seed_metrics": suite.all_metrics(seed_pred), "methods": {}}
        for m in METHODS:
            job = SS / f"seed{s}_{m}"
            if not (job / "stable_report.json").is_file():
                continue
            fin = majority(job, "final")
            rng = np.random.default_rng(abs(hash((s, m))) % (2**32))
            d = suite.paired_bootstrap(seed_pred, fin, args.n_boot, rng)
            point = suite.all_metrics(fin)
            entry = {"point": point, "delta": {}}
            total += 1
            for k in keys:
                lo, hi = np.percentile(d[k], [2.5, 97.5])
                # Two-sided bootstrap p-value. Ties must count on both sides,
                # otherwise an identical-to-seed prompt reports p=0.
                p_raw = min(1.0, 2 * min((d[k] >= 0).mean(), (d[k] <= 0).mean()))
                widths[k].append(float(hi - lo))
                if lo > 0 or hi < 0:
                    resolved[k] += 1
                entry["delta"][k] = {
                    "point": point[k] - per_seed[s]["seed_metrics"][k],
                    "ci95": [float(lo), float(hi)],
                    "p_raw": float(p_raw),
                }
            per_seed[s]["methods"][m] = entry

        # Holm within seed, per metric
        for k in keys:
            pv = [
                (m, per_seed[s]["methods"][m]["delta"][k]["p_raw"])
                for m in per_seed[s]["methods"]
            ]
            adj = holm(pv)
            for m, a in adj.items():
                per_seed[s]["methods"][m]["delta"][k]["p_holm"] = a

    print(f"{'metric':12} {'mean CI width':>14} {'vs hard_min':>12} {'resolved':>10}")
    base = float(np.mean(widths["hard_min"]))
    for k in keys:
        w = float(np.mean(widths[k]))
        print(
            f"  {k:12} {w:12.4f} {base / w if w else float('nan'):11.2f}x "
            f"{resolved[k]:>6}/{total}"
        )
    report["ci_width"] = {k: float(np.mean(widths[k])) for k in keys}
    report["resolved_at_95"] = {k: {"n": resolved[k], "of": total} for k in keys}
    report["seeds"] = per_seed

    print()
    print("=== C. Effective sample-size gain (width^-2 scaling) ===")
    for k in keys:
        w = float(np.mean(widths[k]))
        print(f"  {k:12} equivalent data multiplier vs hard_min: {(base / w) ** 2:.2f}x")

    print()
    print("=== D. Per-seed method deltas under CVaR@25% (Holm-adjusted) ===")
    for s, blk in per_seed.items():
        sm = blk["seed_metrics"]
        print(f"  seed{s}: seed cvar25={sm['cvar25']:.4f} hard_min={sm['hard_min']:.4f}")
        rows = sorted(
            blk["methods"].items(),
            key=lambda t: -t[1]["delta"]["cvar25"]["point"],
        )
        for m, e in rows:
            d = e["delta"]["cvar25"]
            flag = "SIG" if d["p_holm"] < 0.05 else ""
            print(
                f"    {m:14} cvar25={e['point']['cvar25']:.4f} "
                f"delta={d['point']:+.4f} CI[{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}] "
                f"p_holm={d['p_holm']:.3f} {flag}"
            )

    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

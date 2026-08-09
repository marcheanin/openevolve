"""Headroom audit: where the remaining error actually lives, and what could move it.

Answers four questions from one saved eval dir (needs `worker_predictions.npy`,
so the dir must come from ``eval_prompt_on_test.py`` or ``_eval_split``):

1. How much is reachable by better **aggregation** alone (oracle over workers)
   versus how much needs a genuine prompt change (all workers wrong)?
2. Which **gold class** carries the error — i.e. is this an ordinal threshold
   problem rather than a sentiment problem?
3. How much of the between-**cluster** accuracy spread is explained by nothing
   but each cluster's label mix? A high R^2 means the "group" axis is a label-mix
   proxy and group-robust optimisation degenerates into class calibration.
4. How many test users a given effect size needs at 80% power (paired design).

Usage:
  python scripts/headroom_audit.py --eval results/<run>/evals/initial_prompt \
      [--compare results/<run>/evals/final_selected]
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def _load(d: Path) -> Dict[str, np.ndarray]:
    out = {
        "ens": np.load(d / "ensemble_predictions.npy"),
        "y": np.load(d / "labels.npy"),
        "uid": np.load(d / "user_ids.npy", allow_pickle=True),
        "cid": np.load(d / "cluster_ids.npy"),
    }
    wp = d / "worker_predictions.npy"
    if wp.exists():
        out["wp"] = np.load(wp)
    return out


def _user_acc(correct: np.ndarray, uid: np.ndarray) -> np.ndarray:
    acc: Dict = defaultdict(list)
    for u, ok in zip(uid.tolist(), correct.tolist()):
        acc[u].append(ok)
    return np.array([float(np.mean(acc[u])) for u in sorted(acc)])


def aggregation_headroom(d: Dict[str, np.ndarray], label: str) -> None:
    ens, y = d["ens"], d["y"]
    ens_acc = float((ens == y).mean())
    print(f"\n[{label}] aggregation headroom")
    if "wp" not in d:
        print(f"  ensemble={ens_acc:.4f} (no worker_predictions.npy — cannot compute oracle)")
        return
    wp = d["wp"]
    per_worker = [float((wp[w] == y).mean()) for w in range(wp.shape[0])]
    hit_any = np.any(wp == y[None, :], axis=0)
    err = ens != y
    print(f"  per-worker: {[f'{a:.4f}' for a in per_worker]} best_single={max(per_worker):.4f}")
    print(f"  ensemble={ens_acc:.4f}  oracle_over_workers={hit_any.mean():.4f}")
    print(f"  reachable by better aggregation alone = {hit_any.mean() - ens_acc:+.4f}")
    print(
        f"  ensemble errors={int(err.sum())}: some worker right={int((err & hit_any).sum())} "
        f"({(err & hit_any).sum() / max(1, err.sum()):.0%}), all workers wrong="
        f"{int((err & ~hit_any).sum())} ({(err & ~hit_any).sum() / max(1, err.sum()):.0%})"
    )


def per_class(a: Dict[str, np.ndarray], b: Optional[Dict[str, np.ndarray]]) -> None:
    y = a["y"]
    print("\nper-gold-class accuracy (the ordinal threshold view)")
    header = "  gold   n     base" + ("    compare     delta" if b else "")
    print(header)
    for g in sorted(set(y.tolist())):
        m = y == g
        base = float((a["ens"][m] == g).mean())
        line = f"  {g}    {int(m.sum()):4d}   {base:.3f}"
        if b:
            comp = float((b["ens"][m] == g).mean())
            line += f"    {comp:.3f}     {comp - base:+.3f}"
        print(line)


def cluster_vs_class_mix(d: Dict[str, np.ndarray]) -> None:
    """Is the group axis anything more than each group's label mix?"""
    y, cid, uid = d["y"], d["cid"], d["uid"]
    ok = (d["ens"] == y).astype(float)
    class_acc = {g: float(ok[y == g].mean()) for g in sorted(set(y.tolist()))}
    print("\ncluster accuracy vs class-mix-only prediction")
    print(f"  global per-class acc: {({g: round(v, 3) for g, v in class_acc.items()})}")
    print("  cluster  users  observed  predicted  residual")
    obs, pred = [], []
    for k in sorted(set(cid.tolist())):
        m = cid == k
        o = float(ok[m].mean())
        p = float(np.mean([class_acc[g] for g in y[m]]))
        obs.append(o)
        pred.append(p)
        print(
            f"  c{k}       {len(set(uid[m].tolist())):4d}   {o:.4f}    {p:.4f}     {o - p:+.4f}"
        )
    obs_a, pred_a = np.array(obs), np.array(pred)
    denom = float(((obs_a - obs_a.mean()) ** 2).sum())
    r2 = 1 - float(((obs_a - pred_a) ** 2).sum()) / denom if denom else float("nan")
    resid = obs_a - pred_a
    print(f"  R^2 of class-mix-only model = {r2:.3f}")
    print(
        f"  spread: observed={obs_a.max() - obs_a.min():.4f} "
        f"explained_by_class_mix={pred_a.max() - pred_a.min():.4f} "
        f"residual_group_effect={resid.max() - resid.min():.4f}"
    )


def power(a: Dict[str, np.ndarray], b: Optional[Dict[str, np.ndarray]]) -> None:
    y, uid = a["y"], a["uid"]
    acc_a = _user_acc((a["ens"] == y).astype(float), uid)
    reviews = Counter(uid.tolist())
    print(
        f"\nresolution: {min(reviews.values())}-{max(reviews.values())} reviews/user "
        f"-> per-user accuracy step 1/{min(reviews.values())}"
    )
    if b is None:
        sd = float(acc_a.std(ddof=1)) * np.sqrt(2)
        note = "unpaired approximation (no --compare given)"
    else:
        acc_b = _user_acc((b["ens"] == y).astype(float), uid)
        sd = float((acc_a - acc_b).std(ddof=1))
        note = "paired delta SD from the two eval dirs"
    print(f"users needed for 80% power on R_global ({note}, SD={sd:.4f}):")
    for eff in (0.01, 0.02, 0.03, 0.05):
        print(f"  effect {eff:.2f} -> ~{(2.8 * sd / eff) ** 2:,.0f} users")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, help="baseline eval dir")
    ap.add_argument("--compare", help="optional second eval dir on the same slice")
    args = ap.parse_args()

    a = _load(Path(args.eval))
    b = _load(Path(args.compare)) if args.compare else None
    if b is not None and not np.array_equal(a["y"], b["y"]):
        raise SystemExit("eval dirs are not aligned: labels differ")

    print(f"examples={len(a['y'])} users={len(set(a['uid'].tolist()))}")
    aggregation_headroom(a, Path(args.eval).name)
    if b is not None:
        aggregation_headroom(b, Path(args.compare).name)
    per_class(a, b)
    cluster_vs_class_mix(a)
    power(a, b)


if __name__ == "__main__":
    main()

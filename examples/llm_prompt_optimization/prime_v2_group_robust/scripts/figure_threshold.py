#!/usr/bin/env python
"""The figure for the mechanism: every prompt is one point on one curve per group.

Each panel is a group. Each dot is one of the pool's prompts at its operating point
(false-alarm rate on the horizontal axis, catch rate on the vertical). Under the threshold
hypothesis all the dots of a panel lie on that group's ROC curve and their left-to-right
order is the same in every panel, because a prompt sets one threshold for all groups at once.
A prompt that discriminated better would sit visibly above the others' curve; none does.

The seed and the pool's best prompt are marked so a reader can see that first place is a
position on the shared curve, not a curve of its own.

Usage:
  python scripts/figure_threshold.py                       # gemma, the main matrix
  S11_PREDS_DIR=results/S11_protocol_matrix/scorer2_gpt4omini/preds \
      python scripts/figure_threshold.py --tag gpt4omini   # the second scorer
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_s11 import IDS, load_preds, load_set, metric, valid_rows  # noqa: E402
from dataset_config import cfg  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="gemma", help="суффикс имени файла")
    ap.add_argument("--target", default="cvar25")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y, c, _ = load_set(cfg.test_set)
    P = load_preds(cfg.test_set, len(y))
    names = sorted(n for n in P if not n.startswith("CONTROL:"))
    keep = valid_rows(P, len(y), names)
    truth = {n: metric(args.target, P[n], y, c, rows=keep) for n in names}
    best = max(truth, key=truth.get)
    gnames = cfg.group_names()

    groups = list(IDS)
    ncol = 4
    nrow = int(np.ceil(len(groups) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.0 * nrow), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for j, g in enumerate(groups):
        ax = axes[j]
        pos, neg = (c == g) & (y == 1) & keep, (c == g) & (y == 0) & keep
        fpr = np.array([(P[n][neg] == 1).mean() for n in names])
        tpr = np.array([(P[n][pos] == 1).mean() for n in names])
        o = np.argsort(fpr)
        ax.plot(fpr[o], tpr[o], "-", color="0.75", lw=1.2, zorder=1)
        kind = np.array([0 if n == "seed" else (1 if n.startswith("r15:") else 2) for n in names])
        for k, (col, mk, lab) in enumerate([("#c0392b", "*", "стартовый промпт"),
                                            ("#2980b9", "s", "однострочные правки"),
                                            ("#7f8c8d", "o", "финалы оптимизаторов")]):
            m = kind == k
            ax.scatter(fpr[m], tpr[m], s=64 if k == 0 else 22, c=col, marker=mk,
                       zorder=3 if k == 0 else 2, label=lab if j == 0 else None,
                       edgecolors="white", linewidths=0.4)
        ib = names.index(best)
        ax.scatter([fpr[ib]], [tpr[ib]], s=95, facecolors="none", edgecolors="#27ae60",
                   linewidths=1.8, zorder=4, label="лучший по цели" if j == 0 else None)
        ax.set_title(f"{gnames.get(g, g)}", fontsize=10)
        ax.grid(alpha=0.25, lw=0.5)
    for j in range(len(groups), len(axes)):
        axes[j].axis("off")

    fig.supxlabel("доля ложных тревог (FPR)", y=0.075, fontsize=11)
    fig.supylabel("доля пойманного (TPR)", fontsize=11)
    fig.suptitle("Каждый промпт — точка на одной кривой своей группы; порядок точек одинаков во всех группах",
                 fontsize=11)
    fig.tight_layout(rect=(0.02, 0.11, 1, 0.95))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.005))

    out = args.out or (cfg.outputs / "figures" / f"threshold_{args.tag}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=190)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"{len(names)} промптов, {len(groups)} групп -> {out} и {out.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

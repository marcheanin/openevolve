"""
Standalone batch composition chart (Hard vs Anchor per AL cycle).

Paper styling: title 16pt, axis labels / ticks / legend 14pt.
No expansion or consolidation event lines.

Usage:
    python plot_batch_composition.py
    python plot_batch_composition.py --results-dir results_all_categories_evolve_subsample
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent

FONT_TITLE = 16
FONT_LABEL = 14
FONT_TICK = 14
FONT_LEGEND = 14

COLOR_HARD = "#d62728"
COLOR_ANCHOR = "#1f77b4"


def _load_log(results_dir: Path) -> list:
    path = results_dir / "active_loop_log.json"
    if not path.exists():
        raise SystemExit(f"Missing {path}")
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    if not entries:
        raise SystemExit("active_loop_log.json is empty.")
    entries.sort(key=lambda e: e.get("al_iter", 0))
    return entries


def plot_batch_composition(
    results_dir: Path,
    *,
    out_path: Path | None = None,
    title: str = "Batch composition (Hard / Anchor)",
) -> Path:
    entries = _load_log(results_dir)
    cycles = [int(e.get("al_iter", i)) for i, e in enumerate(entries)]
    batch_hard = [int(e.get("batch_n_hard", 0)) for e in entries]
    batch_anchor = [int(e.get("batch_n_anchor", 0)) for e in entries]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    width = 0.36
    ax.bar(
        [c - width / 2 for c in cycles],
        batch_hard,
        width,
        color=COLOR_HARD,
        label="Hard",
        edgecolor="#333333",
        linewidth=0.6,
    )
    ax.bar(
        [c + width / 2 for c in cycles],
        batch_anchor,
        width,
        color=COLOR_ANCHOR,
        label="Anchor",
        edgecolor="#333333",
        linewidth=0.6,
    )

    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold", pad=12)
    ax.set_xlabel("AL cycle", fontsize=FONT_LABEL)
    ax.set_ylabel("Examples in batch", fontsize=FONT_LABEL)
    ax.set_xticks(cycles)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.legend(fontsize=FONT_LEGEND, loc="upper left", frameon=True)
    ax.set_ylim(0, max(max(batch_hard), max(batch_anchor)) * 1.08)

    plt.tight_layout()
    if out_path is None:
        out_path = results_dir / "plots" / "batch_composition.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Batch Hard/Anchor composition per AL cycle.")
    p.add_argument(
        "--results-dir",
        type=str,
        default="results_all_categories_evolve_subsample",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PNG (default: <results-dir>/plots/batch_composition.png)",
    )
    p.add_argument("--title", type=str, default="Batch composition (Hard / Anchor)")
    args = p.parse_args()

    rd = (SCRIPT_DIR / args.results_dir).resolve()
    if not rd.is_dir():
        raise SystemExit(f"Not a directory: {rd}")

    out = Path(args.out).resolve() if args.out else None
    plot_batch_composition(rd, out_path=out, title=args.title)


if __name__ == "__main__":
    main()

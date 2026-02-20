"""
Visualization for Active Prompt Evolution results.

Generates plots: Hard/Anchor/Full accuracy, error cluster evolution,
prompt length over time, comparison with Exp3b baseline.
"""

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# Exp3b baseline (from plan)
EXP3B_R_GLOBAL = 0.6827
EXP3B_COMBINED = 0.736


def load_log() -> list:
    """Load active_loop_log.json."""
    log_path = SCRIPT_DIR / "results" / "active_loop_log.json"
    if not log_path.exists():
        return []
    with open(log_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_results(save_dir: Path | None = None) -> None:
    """
    Generate visualization plots.
    Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        return

    entries = load_log()
    if not entries:
        print("No log data. Run active_loop.py first.")
        return

    entries.sort(key=lambda e: e.get("al_iter", 0))
    iters = [e.get("al_iter", i) for i, e in enumerate(entries)]
    r_global = [e.get("val_R_global", e.get("R_global", 0)) for e in entries]
    combined = [e.get("val_combined_score", 0) for e in entries]
    n_hard = [e.get("batch_n_hard", e.get("n_hard", 0)) for e in entries]
    n_anchor = [e.get("batch_n_anchor", e.get("n_anchor", 0)) for e in entries]
    mae = [e.get("val_mae", e.get("mae", 0)) for e in entries]

    expansion_iters = [e["al_iter"] for e in entries if e.get("expanded")]
    consolidation_iters = [e["al_iter"] for e in entries if e.get("consolidated")]

    def mark_events(ax):
        for i, xi in enumerate(expansion_iters):
            ax.axvline(
                x=xi, color="orange", linestyle="--", alpha=0.8, linewidth=1.5,
                label="Expansion" if i == 0 else None,
            )
        for i, xi in enumerate(consolidation_iters):
            ax.axvline(
                x=xi, color="green", linestyle=":", alpha=0.7, linewidth=1.5,
                label="Consolidation" if i == 0 else None,
            )

    out = save_dir or (SCRIPT_DIR / "results" / "plots")
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # 1. R_global over iterations
    ax = axes[0, 0]
    ax.plot(iters, r_global, "b-o", label="R_global")
    mark_events(ax)
    ax.axhline(EXP3B_R_GLOBAL, color="gray", linestyle=":", label=f"Exp3b ({EXP3B_R_GLOBAL:.1%})")
    ax.set_xlabel("AL Iteration")
    ax.set_ylabel("R_global")
    ax.set_title("Global Accuracy")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 2. Combined score over iterations
    ax = axes[0, 1]
    ax.plot(iters, combined, "g-o", label="Combined Score")
    mark_events(ax)
    ax.axhline(EXP3B_COMBINED, color="gray", linestyle=":", label=f"Exp3b ({EXP3B_COMBINED:.3f})")
    ax.set_xlabel("AL Iteration")
    ax.set_ylabel("Combined Score")
    ax.set_title("Validation Combined Score")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 3. Hard vs Anchor set sizes (batch)
    ax = axes[1, 0]
    ax.plot(iters, n_hard, "r-o", label="Hard")
    ax.plot(iters, n_anchor, "b-o", label="Anchor")
    mark_events(ax)
    ax.set_xlabel("AL Iteration")
    ax.set_ylabel("Count")
    ax.set_title("Hard vs Anchor (batch)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 4. MAE
    ax = axes[1, 1]
    ax.plot(iters, mae, "m-o", label="MAE")
    mark_events(ax)
    ax.set_xlabel("AL Iteration")
    ax.set_ylabel("MAE")
    ax.set_title("Mean Absolute Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out / "active_evolution_curves.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    plot_results()


if __name__ == "__main__":
    main()

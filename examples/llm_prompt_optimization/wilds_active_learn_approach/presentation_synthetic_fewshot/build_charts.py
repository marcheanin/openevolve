"""Generate matplotlib charts for the synthetic few-shot presentation.

Outputs:
- assets/bar_rworst_pairs.png        -- 3-act narrative R_worst bar chart
- assets/bar_full_summary.png        -- all runs Acc_Hard / R_worst side by side
- assets/cycle_curves_8x20.png       -- test R_worst over AL cycles for 8x20 ablation pair
- assets/cycle_curves_acc_hard.png   -- test Acc_Hard over AL cycles for 8x20 ablation pair
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Cyrillic labels in charts (Windows-friendly)
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
EXP_ROOT = ROOT.parent
ASSETS = ROOT / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _final(run_dir: str) -> dict:
    return _load_json(EXP_ROOT / run_dir / "final_test_metrics.json")


def _log(run_dir: str) -> list:
    return _load_json(EXP_ROOT / run_dir / "active_loop_log.json")


def chart_three_acts() -> None:
    """Two-act narrative for R_worst gains. 8x20 controlled ablation is shown separately
    on the cycle-curves chart with Acc_Hard as the highlight."""
    acts = [
        (
            "Этап 1: одна категория\n(replace \u2192 inject_as_hint)",
            ["без синт.\n(v11)", "с синт. + hint\n(v14_gemini)"],
            [0.503, 0.569],
        ),
        (
            "Этап 2: все 15 категорий\n(test)",
            ["без синт.\n(uncapped_train)", "с синт. + hint\n(evolve_subsample)"],
            [0.533, 0.667],
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    bar_color_no = "#9aa0a6"
    bar_color_yes = "#1f77b4"

    for ax, (title, labels, values) in zip(axes, acts):
        delta = values[1] - values[0]
        colors = [bar_color_no, bar_color_yes]
        bars = ax.bar(labels, values, color=colors, width=0.55, edgecolor="#333", linewidth=0.8)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=13, fontweight="bold")
        sign = "+" if delta >= 0 else ""
        ax.set_title(f"{title}\n\u0394 R_worst = {sign}{delta*100:.1f} п.п.", fontsize=11)
        ax.set_ylim(0.0, 0.78)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("R_worst (test)", fontsize=12)
    fig.suptitle(
        "Синтетические boundary few-shot: прирост R_worst в двух постановках",
        fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(ASSETS / "bar_rworst_pairs.png", dpi=170)
    plt.close(fig)


def chart_full_summary() -> None:
    """Two-panel: R_worst and Acc_Hard across the most relevant runs."""
    runs = [
        ("v10",                       0.5167, 0.2368, "no synth"),
        ("v11",                       0.5029, 0.2727, "no synth"),
        ("v12_synth (replace)",       0.5020, 0.2464, "synth replace"),
        ("v13_from_v12 (replace)",    0.4222, 0.3704, "synth replace"),
        ("v14_gemini (hint)",         0.5686, 0.3500, "synth hint"),
        ("ablation_with_synth_8x20",  0.4667, 0.3810, "synth hint"),
        ("ablation_without_synth_8x20", 0.5853, 0.2687, "no synth"),
        ("all_cat_uncapped_train",    0.5333, 0.2834, "no synth"),
        ("all_cat_evolve_subsample",  0.6667, 0.3529, "synth hint"),
    ]
    color_map = {
        "no synth": "#9aa0a6",
        "synth replace": "#ff7f0e",
        "synth hint": "#1f77b4",
    }
    names = [r[0] for r in runs]
    rworst = [r[1] for r in runs]
    acchard = [r[2] for r in runs]
    colors = [color_map[r[3]] for r in runs]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    for ax, vals, title in zip(axes, [rworst, acchard], ["R_worst", "Acc_Hard"]):
        bars = ax.barh(names, vals, color=colors, edgecolor="#333", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=9)
        ax.set_xlabel(title, fontsize=11)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.25, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in color_map.values()]
    leg_ru = {
        "no synth": "без синт.",
        "synth replace": "синт. replace",
        "synth hint": "синт. hint",
    }
    fig.legend(handles, [leg_ru[k] for k in color_map.keys()], loc="lower center", ncol=3, frameon=False, fontsize=10)
    fig.suptitle("Итоговые метрики по прогонам (test)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(ASSETS / "bar_full_summary.png", dpi=170)
    plt.close(fig)


def chart_cycle_curves_8x20() -> None:
    """Test R_worst and Acc_Hard over AL cycles for the controlled 8x20 pair."""
    log_with = _log("results_ablation_with_synth_8x20")
    log_without = _log("results_ablation_without_synth_8x20")

    cycles = list(range(len(log_with)))
    rw_with = [c["test_R_worst"] for c in log_with]
    rw_without = [c["test_R_worst"] for c in log_without]
    ah_with = [c["test_Acc_Hard"] for c in log_with]
    ah_without = [c["test_Acc_Hard"] for c in log_without]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    ax = axes[0]
    ax.plot(cycles, rw_without, "o-", color="#9aa0a6", linewidth=2, label="без синт.")
    ax.plot(cycles, rw_with, "s-", color="#1f77b4", linewidth=2, label="с синт. (hint)")
    ax.set_title("test R_worst по циклам AL (8\u00d720)", fontsize=11)
    ax.set_xlabel("Цикл AL")
    ax.set_ylabel("test R_worst")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.plot(cycles, ah_without, "o-", color="#9aa0a6", linewidth=2, label="без синт.")
    ax.plot(cycles, ah_with, "s-", color="#1f77b4", linewidth=2, label="с синт. (hint)")
    ax.set_title("test Acc_Hard по циклам AL (8\u00d720)", fontsize=11)
    ax.set_xlabel("Цикл AL")
    ax.set_ylabel("test Acc_Hard")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Эффект синт. few-shot по циклам AL (8×20, контроль)",
        fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(ASSETS / "cycle_curves_8x20.png", dpi=170)
    plt.close(fig)


def chart_pipeline_arrows() -> None:
    """Schematic block diagram of the active loop with synthetic few-shot insertion.

    Layout (single horizontal chain) avoids label overlap.
    """
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.4)
    ax.axis("off")

    def block(x, y, w, h, text, color, bold=True, fg="#111"):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#333", linewidth=1.4)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=10, fontweight=("bold" if bold else "normal"), color=fg)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#333", lw=1.5))

    # Top-row chain (main pipeline)
    y_top = 3.4
    block(0.2,  y_top, 1.9, 1.1, "\u041f\u0443\u043b \u0432\u0438\u0434\u0435\u043d\u043d\u044b\u0445\n(Hard / Anchor)", "#cfe8ff")
    block(2.5,  y_top, 1.9, 1.1, "\u0410\u043d\u0441\u0430\u043c\u0431\u043b\u044c\n3 LLM", "#cfe8ff")
    block(4.8,  y_top, 2.0, 1.1, "Confusion matrix\n+ \u043a\u043e\u043d\u0442\u0435\u043a\u0441\u0442 \u043e\u0448\u0438\u0431\u043e\u043a", "#ffe2b5")
    block(7.2,  y_top, 2.6, 1.1, "\u0413\u0435\u043d\u0435\u0440\u0430\u0442\u043e\u0440\nsynthetic few-shot\n(twin-pairs)", "#1f77b4", fg="white")
    block(10.2, y_top, 2.0, 1.1, "MAP-Elites\n\u043c\u0443\u0442\u0430\u0442\u043e\u0440", "#cfe8ff")
    block(12.6, y_top, 1.2, 1.1, "Val\n\u043e\u0446\u0435\u043d\u043a\u0430", "#cfe8ff")

    for x1, x2 in [(2.1, 2.5), (4.4, 4.8), (6.8, 7.2), (9.8, 10.2), (12.2, 12.6)]:
        arrow(x1, y_top + 0.55, x2, y_top + 0.55)

    # Below the generator block: indicate replace / inject_as_hint modes
    block(7.2, 1.8, 2.6, 0.9,
          "<FewShotExamples>\nreplace | inject_as_hint",
          "#fff5d6", bold=False)
    arrow(8.5, 3.4, 8.5, 2.7)
    arrow(8.5, 1.8, 8.5, 1.4)
    arrow(8.5, 1.4, 10.6, 1.4)
    arrow(10.6, 1.4, 10.6, 3.4)

    # Feedback loop arrow at bottom
    arrow(13.2, 3.4, 13.2, 0.6)
    arrow(13.2, 0.6, 1.15, 0.6)
    arrow(1.15, 0.6, 1.15, 3.4)
    ax.text(
        7.0, 0.3,
        "\u041e\u0431\u0440\u0430\u0442\u043d\u0430\u044f \u0441\u0432\u044f\u0437\u044c: val \u2192 \u043e\u0431\u043d\u043e\u0432\u043b\u0435\u043d\u0438\u0435 Hard/Anchor",
        ha="center", fontsize=9, style="italic", color="#444")

    fig.suptitle(
        "Active Learning + \u0441\u0438\u043d\u0442\u0435\u0442\u0438\u0447\u0435\u0441\u043a\u0438\u0435 boundary few-shot",
        fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(ASSETS / "pipeline_diagram.png", dpi=170)
    plt.close(fig)


def main() -> None:
    chart_three_acts()
    chart_full_summary()
    chart_cycle_curves_8x20()
    chart_pipeline_arrows()
    print("Charts written to:", ASSETS)


if __name__ == "__main__":
    main()

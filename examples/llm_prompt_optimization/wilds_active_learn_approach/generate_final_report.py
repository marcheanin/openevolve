"""
Generate final report comparing Active Learning results with Exp3b baseline.
Includes test-set evaluation: baseline (initial prompt) vs final (after AL).
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

EXP3B = {"R_global": 0.6827, "R_worst": 0.4769, "mae": 0.338, "combined": 0.736}
EMBEDDING_BASELINE = {"val_accuracy": 0.703}


def main():
    log_path = SCRIPT_DIR / "results" / "active_loop_log.json"
    if not log_path.exists():
        print("No active_loop_log.json. Run active_loop.py first.")
        return

    with open(log_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    entries.sort(key=lambda e: e.get("al_iter", 0))

    baseline_path = SCRIPT_DIR / "results" / "baseline" / "metrics.json"
    baseline_metrics = {}
    baseline_test = {}
    if baseline_path.exists():
        with open(baseline_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        baseline_metrics = data.get("validation", {})
        baseline_test = data.get("test", {})

    # Test metrics: before AL (from active_loop) or from baseline run
    baseline_test_path = SCRIPT_DIR / "results" / "baseline_test_metrics.json"
    if baseline_test_path.exists():
        with open(baseline_test_path, "r", encoding="utf-8") as f:
            baseline_test = json.load(f)
    elif baseline_test:
        pass
    else:
        baseline_test = {}

    final_test_path = SCRIPT_DIR / "results" / "final_test_metrics.json"
    final_test = {}
    if final_test_path.exists():
        with open(final_test_path, "r", encoding="utf-8") as f:
            final_test = json.load(f)

    if not entries:
        print("Log is empty.")
        return

    # Best by val Acc_Hard (primary) or fallback to val_R_global
    best = max(entries, key=lambda e: e.get("val_Acc_Hard", e.get("val_combined_score", 0)))
    last = entries[-1]

    report = []
    report.append("# Active Prompt Evolution - Final Report")
    report.append("")

    # Validation: Hard/Anchor metrics from AL log
    report.append("## Validation: Hard/Anchor Metrics Across AL Iterations")
    report.append("")
    report.append("| Metric | Best Iteration | Last Iteration |")
    report.append("|--------|----------------|----------------|")
    for key, label in [
        ("val_Acc_Hard", "Acc_Hard"),
        ("val_Acc_Anchor", "Acc_Anchor"),
        ("val_R_global", "R_global"),
        ("val_mae", "MAE"),
        ("val_n_hard", "Val Hard count"),
        ("val_n_anchor", "Val Anchor count"),
    ]:
        b_val = best.get(key, best.get(key.replace("val_", ""), 0))
        l_val = last.get(key, last.get(key.replace("val_", ""), 0))
        if isinstance(b_val, float) and b_val <= 1.0 and key != "val_mae":
            report.append(f"| {label} | {b_val:.2%} | {l_val:.2%} |")
        else:
            report.append(f"| {label} | {b_val} | {l_val} |")
    report.append(f"| Best at AL iter | {best.get('al_iter')} | {last.get('al_iter')} |")
    report.append("")

    # Comparison with Exp3b (R_global for compatibility)
    report.append("## Comparison with Baselines (R_global)")
    report.append("")
    report.append("| Metric | Exp3b (Yandex) | AL Best | AL Last | Embedding Baseline |")
    report.append("|--------|----------------|---------|---------|-------------------|")
    r_gl = best.get("val_R_global", best.get("R_global", 0))
    r_gl_l = last.get("val_R_global", last.get("R_global", 0))
    emb_acc = baseline_metrics.get("R_global") or EMBEDDING_BASELINE.get("val_accuracy", 0)
    emb_str = f"{emb_acc:.2%}" if isinstance(emb_acc, (int, float)) else str(emb_acc)
    report.append(f"| R_global | {EXP3B['R_global']:.2%} | {r_gl:.2%} | {r_gl_l:.2%} | {emb_str} |")
    report.append("")

    # Test: before vs after
    report.append("## Test Set: Before vs After AL")
    report.append("")
    report.append("| Metric | Baseline (initial prompt) | Final (after AL) | Delta |")
    report.append("|--------|---------------------------|------------------|-------|")
    bt_r = baseline_test.get("R_global", 0)
    bt_w = baseline_test.get("R_worst", 0)
    bt_mae = baseline_test.get("mae", 0)
    bt_comb = baseline_test.get("combined_score", 0)
    ft_r = final_test.get("R_global", 0)
    ft_w = final_test.get("R_worst", 0)
    ft_mae = final_test.get("mae", 0)
    ft_comb = final_test.get("combined_score", 0)
    report.append(f"| R_global | {bt_r:.2%} | {ft_r:.2%} | {ft_r - bt_r:+.2%} |")
    report.append(f"| R_worst | {bt_w:.2%} | {ft_w:.2%} | {ft_w - bt_w:+.2%} |")
    report.append(f"| MAE | {bt_mae:.3f} | {ft_mae:.3f} | {ft_mae - bt_mae:+.3f} |")
    report.append(f"| Combined | {bt_comb:.4f} | {ft_comb:.4f} | {ft_comb - bt_comb:+.4f} |")
    report.append("")

    out_path = SCRIPT_DIR / "results" / "final_report" / "REPORT.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"Report saved: {out_path}")


if __name__ == "__main__":
    main()

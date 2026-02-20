"""
Active Learning Loop Controller for Active Prompt Evolution.

Two-level evolution:
  Level 1 (inner, fast): Evolve <DynamicRules> + <FewShotExamples> via OpenEvolve
      with per-evaluation error artifacts fed back to the mutator.
  Level 2 (outer, between cycles): Consolidation step — analyze top-K prompts
      from the MAP-Elites archive, allow modification of ALL sections including
      <BaseGuidelines>. Produces the seed prompt for the next AL cycle.

Workflow per cycle:
  1. Build active batch from Seen pool (Hard + Anchor).
  2. Run OpenEvolve evolution on the batch.
  3. Re-evaluate and reclassify batch.
  4. Consolidation: top-K prompts → LLM → improved seed (may change BaseGuidelines).
  5. Expand pool from Unseen if Hard examples exhausted.
"""

import json
import os
import sys
import time as _time
from pathlib import Path
from typing import List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
WILDS_EXPERIMENT = SCRIPT_DIR.parent / "wilds_experiment"
EXPERIMENTS_ROOT = WILDS_EXPERIMENT / "experiments"
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent

script_dir_str = str(SCRIPT_DIR)
if script_dir_str in sys.path:
    sys.path.remove(script_dir_str)
sys.path.insert(0, script_dir_str)
for p in (PROJECT_ROOT, WILDS_EXPERIMENT, EXPERIMENTS_ROOT):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.append(p_str)

import numpy as np
import yaml
from collections import Counter
from data_manager import DataManager, disagreement_score
from error_analyzer import ErrorAnalyzer
from evaluator import _build_workers, _load_split_data


class MajorityVoteAggregator:
    def aggregate(self, worker_predictions):
        return Counter(worker_predictions).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Consolidation helpers (Level 2)
# ---------------------------------------------------------------------------

def _validate_prompt_structure(prompt_text: str) -> Tuple[bool, str]:
    """Validate that consolidated prompt preserves required XML structure."""
    errors = []
    if "<BaseGuidelines>" not in prompt_text:
        errors.append("Missing <BaseGuidelines>")
    if "<DynamicRules>" not in prompt_text:
        errors.append("Missing <DynamicRules>")
    if "<FewShotExamples>" not in prompt_text:
        errors.append("Missing <FewShotExamples>")
    if "<Task>" not in prompt_text:
        errors.append("Missing <Task>")
    if "{review}" not in prompt_text:
        errors.append("Missing {review} placeholder in <Task>")
    if "Rating:" not in prompt_text:
        errors.append("Missing 'Rating:' line in <Task>")
    has_format = any(p in prompt_text.lower() for p in [
        "single number", "only a single number", "only a number",
        "1, 2, 3, 4, or 5", "1-5",
    ])
    if not has_format:
        errors.append("Missing output format instruction (e.g. 'ONLY a single number: 1, 2, 3, 4, or 5')")
    return (len(errors) == 0, "; ".join(errors))


def _load_top_programs_from_db(db_dir: str, n: int = 3) -> list:
    """Read top-N programs (by combined_score) from saved database directory."""
    programs_dir = Path(db_dir) / "programs"
    if not programs_dir.exists():
        return []
    programs = []
    for pf in programs_dir.glob("*.json"):
        try:
            with open(pf, "r", encoding="utf-8") as f:
                data = json.load(f)
            code = data.get("code", "")
            # Strip EVOLVE-BLOCK markers added by OpenEvolve
            code = code.replace("# EVOLVE-BLOCK-START\n", "").replace("\n# EVOLVE-BLOCK-END", "")
            code = code.strip()
            metrics = data.get("metrics", {})
            if code and isinstance(metrics.get("combined_score"), (int, float)):
                programs.append({"code": code, "metrics": metrics})
        except Exception:
            continue
    programs.sort(key=lambda p: p["metrics"].get("combined_score", 0), reverse=True)
    return programs[:n]


def _call_consolidation_llm(system_msg: str, user_msg: str, config: dict) -> str:
    """Single LLM call for the consolidation step (uses evolution model)."""
    from openai import OpenAI

    llm_cfg = config.get("llm", {})
    api_base = llm_cfg.get("api_base", "https://openrouter.ai/api/v1")
    models = llm_cfg.get("models", [])
    model_name = models[0]["name"] if models else "deepseek/deepseek-r1"

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    client = OpenAI(base_url=api_base, api_key=api_key)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.7,
        max_tokens=8192,
        timeout=300,
    )

    usage = getattr(response, "usage", None)
    if usage is not None:
        try:
            from token_usage import get_tracker
            inp = getattr(usage, "prompt_tokens", 0) or 0
            out = getattr(usage, "completion_tokens", 0) or 0
            total = getattr(usage, "total_tokens", None)
            get_tracker().record("consolidation/" + model_name, inp, out, total)
        except Exception:
            pass

    content = getattr(response.choices[0].message, "content", None)
    if content is None:
        reasoning = getattr(response.choices[0].message, "reasoning_content", None)
        if reasoning:
            return reasoning.strip()
        raise RuntimeError("Consolidation LLM returned empty response")
    return content.strip()


CONSOLIDATION_SYSTEM = """\
You are an expert prompt engineer performing a consolidation step between \
active learning cycles.

You will receive the top-performing prompts from an evolution cycle with their \
evaluation metrics. These prompts classify Amazon Home & Kitchen product reviews \
(1-5 stars) via a 3-model voting ensemble (DeepSeek V3, Gemma3-27B, GPT-4o-mini).

Create a single improved prompt by analyzing patterns across the top performers.

ALLOWED modifications:
- <BaseGuidelines>: Promote consistently successful rules here. Keep the output \
format instruction ("ONLY a single number: 1, 2, 3, 4, or 5") and the star \
rating scale definition.
- <DynamicRules>: Refine, merge, or remove rules based on what worked.
- <FewShotExamples>: Update examples to better cover all 5 rating categories, \
especially commonly confused pairs (4 vs 5, 2 vs 3). Include 5-7 diverse examples.

FORBIDDEN:
- Do NOT change the <Task> section. It must remain EXACTLY:
    <Task>
    Review: {review}
    Rating:
    </Task>
- Do NOT remove the <Role> tag inside <System>.
- Preserve the XML structure: <System>(<Role>, <BaseGuidelines>, <DynamicRules>)\
</System>, <FewShotExamples>, <Task>.

Guidelines:
1. Rules appearing in ALL top prompts -> promote to <BaseGuidelines>.
2. Rules in only one prompt that didn't improve Acc_Hard -> remove.
3. Contradictory rules -> resolve using the best-scoring prompt's version.
4. FewShotExamples must cover all 5 rating levels (1, 2, 3, 4, 5).
5. Keep the total prompt concise (under ~2000 estimated tokens).

Output ONLY the complete prompt text from <System> to </Task>.
No explanation, no markdown code fences.\
"""


def _consolidate_prompt(
    top_programs: list,
    al_iter: int,
    config: dict,
    max_retries: int = 2,
) -> Optional[str]:
    """
    Consolidation step: analyze top prompts and create an improved seed.
    May modify ALL sections including BaseGuidelines.
    Returns consolidated prompt text or None on failure.
    """
    parts = [f"Here are the top {len(top_programs)} prompts from AL cycle {al_iter + 1}:\n"]
    for i, prog in enumerate(top_programs):
        m = prog["metrics"]
        parts.append(
            f"=== Prompt {i+1} (combined_score: {m.get('combined_score', 0):.4f}, "
            f"Acc_Hard: {m.get('Acc_Hard', 0):.2%}, "
            f"Acc_Anchor: {m.get('Acc_Anchor', 0):.2%}) ==="
        )
        parts.append(prog["code"])
        parts.append("")
    parts.append("Create the consolidated prompt:")
    user_msg = "\n".join(parts)

    for attempt in range(max_retries + 1):
        try:
            result = _call_consolidation_llm(CONSOLIDATION_SYSTEM, user_msg, config)

            # Strip markdown fences if the LLM wrapped its output
            if result.startswith("```"):
                lines = result.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                result = "\n".join(lines)

            valid, errs = _validate_prompt_structure(result)
            if valid:
                return result

            print(f"    Consolidation attempt {attempt+1}: validation failed: {errs}")
            if attempt < max_retries:
                user_msg += (
                    f"\n\nPREVIOUS ATTEMPT FAILED VALIDATION: {errs}\n"
                    "Please fix these issues and output the complete prompt again:"
                )
        except Exception as e:
            print(f"    Consolidation attempt {attempt+1} error: {e}")

    return None


def _evaluate_batch(
    prompt_template: str,
    batch_indices: List[int],
    data_manager: DataManager,
    config: dict,
) -> dict:
    """Evaluate prompt on a specific list of indices. Returns per-example results."""
    workers = _build_workers(config)
    aggregator = MajorityVoteAggregator()

    predictions = []
    worker_preds = [[] for _ in workers]
    for idx in batch_indices:
        text = data_manager.texts[idx]
        sample_preds = []
        for w in workers:
            p = w.predict(text, prompt_template)
            sample_preds.append(p)
        for k, p in enumerate(sample_preds):
            worker_preds[k].append(p)
        predictions.append(aggregator.aggregate(sample_preds))

    gold_labels = [data_manager.labels[idx] for idx in batch_indices]
    user_ids = [data_manager.user_ids[idx] for idx in batch_indices]

    return {
        "predictions": predictions,
        "gold_labels": gold_labels,
        "user_ids": user_ids,
        "worker_predictions": worker_preds,
    }


def _evaluate_split_full(
    prompt_template: str,
    config: dict,
    split_name: str = "validation",
) -> dict:
    """
    Single-pass evaluation on a split: runs inference once, returns both
    traditional metrics (R_global, R_worst, MAE, combined_score, mean_kappa)
    and Hard/Anchor classification (Acc_Hard, Acc_Anchor, n_hard, n_anchor).
    """
    from experiments.metrics import compute_metrics, compute_combined_score_unified

    texts, labels, user_ids = _load_split_data(config, split_name)
    workers = _build_workers(config)
    aggregator = MajorityVoteAggregator()
    uncertainty_threshold = config.get("active_learning", {}).get(
        "uncertainty_threshold", 0.0
    )

    predictions = []
    worker_preds = [[] for _ in workers]
    for text in texts:
        sample_preds = []
        for w in workers:
            p = w.predict(text, prompt_template)
            sample_preds.append(p)
        for k, p in enumerate(sample_preds):
            worker_preds[k].append(p)
        predictions.append(aggregator.aggregate(sample_preds))

    pred_arr = np.array(predictions)
    labels_arr = np.asarray(labels)
    user_arr = np.asarray(user_ids)
    wp_arr = [np.array(wp) for wp in worker_preds]

    # Traditional metrics
    metrics = compute_metrics(pred_arr, labels_arr, user_arr, worker_predictions=wp_arr)
    combined = compute_combined_score_unified(metrics, is_ensemble=True)

    # Hard/Anchor classification
    labels_list = labels_arr.tolist()
    n = len(predictions)
    hard_correct, hard_total = 0, 0
    anchor_correct, anchor_total = 0, 0

    for i in range(n):
        pred = predictions[i]
        gold = labels_list[i]
        wp = [int(worker_preds[k][i]) for k in range(len(workers))]
        correct = pred == gold
        d_score = disagreement_score(wp, rating_min=1, rating_max=5)
        is_hard = (not correct) or (d_score > uncertainty_threshold)
        if is_hard:
            hard_total += 1
            if correct:
                hard_correct += 1
        else:
            anchor_total += 1
            if correct:
                anchor_correct += 1

    return {
        "R_global": metrics.get("R_global", 0),
        "R_worst": metrics.get("R_worst", 0),
        "mae": metrics.get("mae", 0),
        "mean_kappa": metrics.get("mean_kappa", 0),
        "combined_score": combined,
        "Acc_Hard": hard_correct / hard_total if hard_total > 0 else 0.0,
        "Acc_Anchor": anchor_correct / anchor_total if anchor_total > 0 else 1.0,
        "n_hard": hard_total,
        "n_anchor": anchor_total,
        "n_total": n,
    }


def _write_active_batch(
    indices: list,
    hard_indices: list,
    anchor_indices: list,
    output_path: Path,
) -> None:
    data = {
        "indices": indices,
        "hard_indices": hard_indices,
        "anchor_indices": anchor_indices,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run_active_loop(
    n_al_iterations: int = 4,
    n_evolve_iterations: int = 15,
    initial_prompt_path: Optional[Path] = None,
) -> None:
    """
    Run the Active Prompt Evolution loop with pool expansion.

    Scheme C: few AL cycles (3-5), moderate evolve iterations (15-20) each.
    Total ~60-80 OpenEvolve iterations. MAP-Elites archive resets per cycle
    to keep fitness scores clean after batch changes.

    Args:
        n_al_iterations: Max number of outer AL cycles (each = evolve + maybe expand).
        n_evolve_iterations: OpenEvolve iterations per cycle.
        initial_prompt_path: Path to initial prompt (default: initial_prompt.txt).
    """
    from openevolve.api import run_evolution
    from openevolve.config import load_config

    config_path = SCRIPT_DIR / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    config = load_config(str(config_path))
    prompt_path = initial_prompt_path or (SCRIPT_DIR / "initial_prompt.txt")
    results_dir = SCRIPT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    al_cfg = cfg_dict.get("active_learning", {})
    batch_size = al_cfg.get("batch_size", 80)
    hard_ratio = al_cfg.get("hard_ratio", 0.7)
    expansion_trigger = al_cfg.get("expansion_trigger", 5)

    with open(prompt_path, "r", encoding="utf-8") as f:
        current_prompt = f.read()

    data_manager = DataManager(cfg_dict)
    error_analyzer = ErrorAnalyzer(data_manager)
    evaluator_path = str(SCRIPT_DIR / "evaluator.py")
    active_batch_path = SCRIPT_DIR / "active_batch.json"
    error_context_path = SCRIPT_DIR / "error_context.txt"

    log_entries = []

    # ======================================================================
    # Step 0: One-time full pool evaluation to initialize Seen/Unseen
    # ======================================================================
    print("=" * 60)
    print("Initialization: evaluating initial prompt on full train pool")
    print("=" * 60)
    full_results = _evaluate_batch(
        current_prompt, list(range(data_manager.n_total)), data_manager, cfg_dict
    )
    batch_indices = data_manager.initialize_from_evaluation(
        full_results["predictions"],
        full_results["gold_labels"],
        full_results["worker_predictions"],
        batch_size=batch_size,
        hard_ratio=hard_ratio,
        seed=42,
    )
    print(f"  Total pool: {data_manager.n_total}")
    print(f"  Initial batch: {len(batch_indices)} (Hard: {data_manager.n_hard}, Anchor: {data_manager.n_anchor})")
    print(f"  Unseen: {data_manager.n_unseen}")

    # Baseline test evaluation (initial prompt on test set) — single inference pass
    print("  Evaluating initial prompt on test set...")
    baseline_test_metrics = _evaluate_split_full(current_prompt, cfg_dict, split_name="test")
    with open(results_dir / "baseline_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(baseline_test_metrics, f, indent=2)
    print(f"  Baseline test: R_global={baseline_test_metrics['R_global']:.2%}  Acc_Hard={baseline_test_metrics['Acc_Hard']:.2%}  Acc_Anchor={baseline_test_metrics['Acc_Anchor']:.2%}")

    # ======================================================================
    # Main AL loop
    # ======================================================================
    for al_iter in range(n_al_iterations):
        print(f"\n{'=' * 60}")
        print(f"AL Cycle {al_iter + 1}/{n_al_iterations}  |  Hard: {data_manager.n_hard}  Anchor: {data_manager.n_anchor}  Unseen: {data_manager.n_unseen}")
        print("=" * 60)

        # Save current prompt
        prompt_file = results_dir / f"al_iter_{al_iter}_prompt.txt"
        prompt_file.write_text(current_prompt, encoding="utf-8")

        # Build active batch from current Hard + Anchor
        batch_indices, hard_in_batch, anchor_in_batch = data_manager.build_active_batch(
            batch_size=batch_size,
            hard_ratio=hard_ratio,
            seed=42 + al_iter,
        )
        print(f"  Active batch: {len(batch_indices)} (Hard: {len(hard_in_batch)}, Anchor: {len(anchor_in_batch)})")

        # Analyze errors for the evolution system message
        error_analysis = error_analyzer.analyze_errors(
            data_manager.hard_indices,
            k_clusters=min(6, max(1, data_manager.n_hard // 2)),
            seed=42,
        )
        error_context = error_analyzer.format_for_evolution(error_analysis)
        error_context_path.write_text(error_context, encoding="utf-8")

        # Write active_batch.json for the evaluator
        _write_active_batch(batch_indices, hard_in_batch, anchor_in_batch, active_batch_path)

        # Augment system message with static error summary (complements per-eval artifacts)
        base_system = cfg_dict.get("prompt", {}).get("system_message", "")
        augmented_system = base_system + "\n\n## Error Pattern Summary (from batch analysis):\n" + error_context
        evolve_config = load_config(str(config_path))
        if evolve_config.prompt:
            evolve_config.prompt.system_message = augmented_system

        # Persist evolution database for consolidation step
        out_dir = results_dir / f"al_iter_{al_iter}"
        out_dir.mkdir(parents=True, exist_ok=True)
        db_dir = str(out_dir / "database")
        evolve_config.database.db_path = db_dir

        # Run OpenEvolve
        print(f"  Running evolution ({n_evolve_iterations} iterations)...")
        (out_dir / "start_prompt.txt").write_text(current_prompt, encoding="utf-8")

        try:
            result = run_evolution(
                initial_program=current_prompt,
                evaluator=evaluator_path,
                config=evolve_config,
                iterations=n_evolve_iterations,
                output_dir=str(out_dir / "openevolve_output"),
                cleanup=False,
            )
            best_code = current_prompt
            if result:
                if result.best_program and hasattr(result.best_program, "code"):
                    best_code = result.best_program.code
                elif result.best_code:
                    best_code = result.best_code
            current_prompt = best_code
            (out_dir / "best_prompt.txt").write_text(current_prompt, encoding="utf-8")
        except Exception as e:
            print(f"  Evolution failed: {e}")
            import traceback
            traceback.print_exc()

        # Re-evaluate only the batch with the evolved prompt
        print("  Re-evaluating batch...")
        batch_results = _evaluate_batch(current_prompt, batch_indices, data_manager, cfg_dict)

        # Reclassify batch
        hard_now, anchor_now = data_manager.reclassify_batch(
            batch_indices,
            batch_results["predictions"],
            batch_results["gold_labels"],
            batch_results["worker_predictions"],
        )
        print(f"  After evolution: Hard: {len(hard_now)}, Anchor: {len(anchor_now)}")

        # Validation: single-pass eval with Hard/Anchor + traditional metrics
        print("  Validation...")
        val_ha = _evaluate_split_full(current_prompt, cfg_dict, split_name="validation")

        # --- Consolidation (Level 2): top-K prompts → LLM → improved seed ---
        consolidated = False
        top_progs = _load_top_programs_from_db(db_dir, n=3)
        if len(top_progs) >= 2:
            print(f"  Consolidation: analyzing top {len(top_progs)} prompts (may update BaseGuidelines)...")
            cons_prompt = _consolidate_prompt(top_progs, al_iter, cfg_dict)
            if cons_prompt is not None:
                current_prompt = cons_prompt
                (out_dir / "consolidated_prompt.txt").write_text(current_prompt, encoding="utf-8")
                print("  Consolidated prompt accepted as seed for next cycle.")
                consolidated = True
            else:
                print("  Consolidation failed; keeping best evolution prompt as seed.")
        else:
            print("  Skipping consolidation (fewer than 2 programs in archive).")

        entry = {
            "al_iter": al_iter,
            "val_Acc_Hard": val_ha["Acc_Hard"],
            "val_Acc_Anchor": val_ha["Acc_Anchor"],
            "val_R_global": val_ha["R_global"],
            "val_R_worst": val_ha["R_worst"],
            "val_mae": val_ha["mae"],
            "val_mean_kappa": val_ha["mean_kappa"],
            "val_combined_score": val_ha["combined_score"],
            "val_n_hard": val_ha["n_hard"],
            "val_n_anchor": val_ha["n_anchor"],
            "batch_n_hard": len(hard_now),
            "batch_n_anchor": len(anchor_now),
            "n_seen": data_manager.n_seen,
            "n_unseen": data_manager.n_unseen,
            "expanded": False,
            "consolidated": consolidated,
        }

        # Check if expansion is needed
        if data_manager.needs_expansion(threshold=expansion_trigger):
            n_hard_needed = int(batch_size * hard_ratio) - data_manager.n_hard
            n_hard_needed = max(n_hard_needed, expansion_trigger)
            print(f"  Expansion triggered: Hard={data_manager.n_hard} <= {expansion_trigger}. Adding {n_hard_needed} new examples from Unseen...")

            new_indices = data_manager.expand_pool(n_new=n_hard_needed, seed=42 + al_iter)
            data_manager.hard_indices.extend(new_indices)

            print(f"  After expansion: Hard: {data_manager.n_hard}, Anchor: {data_manager.n_anchor}, Unseen: {data_manager.n_unseen}")
            entry["expanded"] = True
            entry["n_expanded"] = len(new_indices)

        print(f"  Val: R_global={val_ha['R_global']:.2%}  Acc_Hard={val_ha['Acc_Hard']:.2%}  Acc_Anchor={val_ha['Acc_Anchor']:.2%}  (Hard:{val_ha['n_hard']}/Anchor:{val_ha['n_anchor']})")
        log_entries.append(entry)

        log_path = results_dir / "active_loop_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_entries, f, indent=2)

        if data_manager.n_unseen == 0 and data_manager.n_hard <= expansion_trigger:
            print(f"\n  Pool exhausted and Hard={data_manager.n_hard}. Stopping.")
            break

    # Cleanup
    if active_batch_path.exists():
        active_batch_path.unlink()

    # Final test evaluation (evolved prompt on test set) — single inference pass
    print("\n" + "=" * 60)
    print("Final test evaluation")
    print("=" * 60)
    final_test_metrics = _evaluate_split_full(current_prompt, cfg_dict, split_name="test")
    with open(results_dir / "final_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_test_metrics, f, indent=2)
    print(f"Baseline test: R_global={baseline_test_metrics['R_global']:.2%}  R_worst={baseline_test_metrics['R_worst']:.2%}  combined={baseline_test_metrics['combined_score']:.4f}")
    print(f"Final test:    R_global={final_test_metrics['R_global']:.2%}  R_worst={final_test_metrics['R_worst']:.2%}  combined={final_test_metrics['combined_score']:.4f}")
    print(f"Final test:    Acc_Hard={final_test_metrics['Acc_Hard']:.2%}  Acc_Anchor={final_test_metrics['Acc_Anchor']:.2%}  (Hard:{final_test_metrics['n_hard']}/Anchor:{final_test_metrics['n_anchor']})")
    delta_r = final_test_metrics["R_global"] - baseline_test_metrics["R_global"]
    delta_comb = final_test_metrics["combined_score"] - baseline_test_metrics["combined_score"]
    print(f"Delta:        R_global={delta_r:+.2%}  combined={delta_comb:+.4f}")
    print("=" * 60)

    # Сохранить учёт токенов и сформировать отчёт
    try:
        from token_usage import get_tracker
        usage_path = results_dir / "token_usage.json"
        report_path = results_dir / "token_usage_report.md"
        tracker = get_tracker()
        tracker.save_json(usage_path)
        tracker.write_report(report_path, title="Отчёт об использовании токенов (после полной эволюции)")
        u = tracker.get_usage()
        print(f"Token usage: total={u['total_tokens']:,} (report: {report_path})")
    except Exception as e:
        print(f"Token usage save skipped: {e}")

    print(f"\nDone. Log: {results_dir / 'active_loop_log.json'}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-al", type=int, default=4)
    parser.add_argument("--n-evolve", type=int, default=15)
    parser.add_argument("--prompt", type=str, default=None)
    args = parser.parse_args()
    run_active_loop(
        n_al_iterations=args.n_al,
        n_evolve_iterations=args.n_evolve,
        initial_prompt_path=Path(args.prompt) if args.prompt else None,
    )


if __name__ == "__main__":
    main()

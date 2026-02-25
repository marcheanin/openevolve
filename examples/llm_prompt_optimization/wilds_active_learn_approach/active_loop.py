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
from evaluator import _build_workers, _load_split_data, _parallel_predict, DEFAULT_MAX_PARALLEL


class MajorityVoteAggregator:
    def aggregate(self, worker_predictions):
        return Counter(worker_predictions).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Diagnostic JSONL logger — writes one JSON object per event to a file.
# ---------------------------------------------------------------------------

class DiagnosticLogger:
    """Append-only JSONL logger for detailed debugging of the AL pipeline."""

    def __init__(self, path: Path):
        self._path = path
        self._fh = open(path, "a", encoding="utf-8")

    def log(self, event: str, **kwargs):
        entry = {"ts": _time.time(), "event": event, **kwargs}
        self._fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()

    def close(self):
        self._fh.close()


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


def _load_top_programs_from_db(db_dir, n: int = 3) -> list:
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
            code = code.replace("# EVOLVE-BLOCK-START\n", "").replace("\n# EVOLVE-BLOCK-END", "")
            code = code.strip()
            metrics = data.get("metrics", {})
            artifacts = data.get("artifacts_json", None)
            if code and isinstance(metrics.get("combined_score"), (int, float)):
                entry = {"code": code, "metrics": metrics}
                if artifacts:
                    try:
                        entry["artifacts"] = json.loads(artifacts) if isinstance(artifacts, str) else artifacts
                    except (json.JSONDecodeError, TypeError):
                        pass
                programs.append(entry)
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

You will receive:
1. The BEST prompt from the evolution cycle (Prompt 1) — use it as the BASE.
2. Other top-performing prompts for reference.
3. Error artifacts: concrete examples where the best prompt still fails.

Your task: take the best prompt as-is and CAREFULLY IMPROVE it by:
- Adding rules from other top prompts that address error patterns
- Adding NEW rules to fix the specific errors shown in the artifacts
- Optionally promoting stable, proven rules to <BaseGuidelines>

CRITICAL PRINCIPLES:
- The best prompt is your STARTING POINT. Do NOT rewrite it from scratch.
- NEVER DELETE existing rules — they were evolved to solve specific problems.
- ADD new rules that address the error patterns shown in the artifacts.
- Ensure <DynamicRules> covers ALL 5 rating levels (1, 2, 3, 4, 5) with
  at least 2 rules per level.
- Ensure <FewShotExamples> has at least one example per rating level (1-5),
  aim for 7-10 examples total.

ALLOWED modifications:
- <BaseGuidelines>: Promote consistently successful rules here. Keep the output \
format instruction ("ONLY a single number: 1, 2, 3, 4, or 5") and the star \
rating scale definition.
- <DynamicRules>: ADD new rules, REFINE wording, MERGE near-duplicates.
  NEVER remove a rule unless it directly contradicts a better-scoring rule.
- <FewShotExamples>: Add or replace examples to improve category coverage. \
Never remove an example if it is the only one for its rating level.

FORBIDDEN:
- Do NOT change the <Task> section. It must remain EXACTLY:
    <Task>
    Review: {review}
    Rating:
    </Task>
- Do NOT remove the <Role> tag inside <System>.
- Preserve the XML structure: <System>(<Role>, <BaseGuidelines>, <DynamicRules>)\
</System>, <FewShotExamples>, <Task>.
- Do NOT delete rules to make the prompt shorter.

Output ONLY the complete prompt text from <System> to </Task>.
No explanation, no markdown code fences.\
"""


def _consolidate_prompt(
    top_programs: list,
    al_iter: int,
    config: dict,
    max_retries: int = 2,
    error_artifacts: Optional[dict] = None,
) -> Optional[str]:
    """
    Consolidation step: take the best prompt as base, enrich with rules from
    other top prompts and fix errors shown in artifacts.
    Returns consolidated prompt text or None on failure.
    """
    parts = [f"Here are the top {len(top_programs)} prompts from AL cycle {al_iter + 1}:\n"]
    for i, prog in enumerate(top_programs):
        m = prog["metrics"]
        label = " [THIS IS THE BASE — start from this prompt]" if i == 0 else ""
        parts.append(
            f"=== Prompt {i+1}{label} (combined_score: {m.get('combined_score', 0):.4f}, "
            f"Acc_Hard: {m.get('Acc_Hard', 0):.2%}, "
            f"Acc_Anchor: {m.get('Acc_Anchor', 0):.2%}) ==="
        )
        parts.append(prog["code"])
        parts.append("")

    if error_artifacts:
        error_text = error_artifacts.get("error_examples", "")
        if error_text:
            parts.append("=== ERROR ARTIFACTS (from the best prompt's last evaluation) ===")
            parts.append("These are concrete examples where Prompt 1 STILL FAILS.")
            parts.append("Add rules to fix these patterns:\n")
            parts.append(error_text)
            parts.append("")

    parts.append("Create the improved prompt (start from Prompt 1 as base, add improvements):")
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
    max_parallel = config.get("worker_defaults", {}).get("max_parallel", DEFAULT_MAX_PARALLEL)

    texts = [data_manager.texts[idx] for idx in batch_indices]
    predictions, worker_preds = _parallel_predict(workers, texts, prompt_template, max_parallel)

    gold_labels = [data_manager.labels[idx] for idx in batch_indices]
    user_ids = [data_manager.user_ids[idx] for idx in batch_indices]

    return {
        "predictions": predictions,
        "gold_labels": gold_labels,
        "user_ids": user_ids,
        "worker_predictions": [list(wp) for wp in worker_preds],
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
    max_parallel = config.get("worker_defaults", {}).get("max_parallel", DEFAULT_MAX_PARALLEL)
    uncertainty_threshold = config.get("active_learning", {}).get(
        "uncertainty_threshold", 0.0
    )

    predictions, worker_preds = _parallel_predict(workers, list(texts), prompt_template, max_parallel)

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
        wp = [int(wp_arr[k][i]) for k in range(len(wp_arr))]
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


def _quick_batch_score(
    prompt_template: str,
    batch_indices: List[int],
    hard_indices: List[int],
    anchor_indices: List[int],
    data_manager: DataManager,
    config: dict,
) -> float:
    """Evaluate prompt on the current active batch and return combined_score."""
    from evaluator import _compute_weighted_fitness

    workers = _build_workers(config)
    max_parallel = config.get("worker_defaults", {}).get("max_parallel", DEFAULT_MAX_PARALLEL)
    hard_set = set(hard_indices)
    anchor_set = set(anchor_indices)

    texts = [data_manager.texts[idx] for idx in batch_indices]
    predictions, worker_preds = _parallel_predict(workers, texts, prompt_template, max_parallel)

    hard_correct, hard_total = 0, 0
    anchor_correct, anchor_total = 0, 0
    for j, idx in enumerate(batch_indices):
        correct = predictions[j] == data_manager.labels[idx]
        if idx in hard_set:
            hard_total += 1
            if correct:
                hard_correct += 1
        elif idx in anchor_set:
            anchor_total += 1
            if correct:
                anchor_correct += 1

    acc_hard = hard_correct / hard_total if hard_total > 0 else 0.0
    acc_anchor = anchor_correct / anchor_total if anchor_total > 0 else 1.0

    kappa_hard = 0.0
    try:
        from sklearn.metrics import cohen_kappa_score
        hard_j = [j for j, idx in enumerate(batch_indices) if idx in hard_set]
        if len(hard_j) >= 2 and len(worker_preds) > 1:
            hw = [np.array([worker_preds[k][j] for j in hard_j]) for k in range(len(worker_preds))]
            kappas = []
            for i in range(len(hw)):
                for j2 in range(i + 1, len(hw)):
                    kappas.append(cohen_kappa_score(hw[i], hw[j2], weights="quadratic"))
            kappa_hard = float(np.mean(kappas)) if kappas else 0.0
    except ImportError:
        pass

    return _compute_weighted_fitness(acc_hard, acc_anchor, kappa_hard, prompt_template)


def run_active_loop(
    n_al_iterations: int = 4,
    n_evolve_iterations: int = 15,
    initial_prompt_path: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    smoke_mode: bool = False,
) -> None:
    """
    Run the Active Prompt Evolution loop with pool expansion.

    Scheme C: few AL cycles (3-5), moderate evolve iterations (15-20) each.
    Total ~60-80 OpenEvolve iterations. MAP-Elites archive resets per cycle
    to keep fitness scores clean after batch changes.
    """
    from openevolve.api import run_evolution
    from openevolve.config import load_config

    config_path = SCRIPT_DIR / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    config = load_config(str(config_path))
    prompt_path = initial_prompt_path or (SCRIPT_DIR / "initial_prompt.txt")
    results_dir = Path(results_dir) if results_dir else (SCRIPT_DIR / "results")
    results_dir = results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    log_entries = []
    diag = DiagnosticLogger(results_dir / "debug_trace.jsonl")
    smoke_timings = []  # (stage, duration_s) when smoke_mode

    al_cfg = cfg_dict.get("active_learning", {})
    batch_size = al_cfg.get("batch_size", 80)
    hard_ratio = al_cfg.get("hard_ratio", 0.7)
    expansion_trigger = al_cfg.get("expansion_trigger", 5)

    with open(prompt_path, "r", encoding="utf-8") as f:
        current_prompt = f.read()

    # --- smoke: load dataset ---
    if smoke_mode:
        t0 = _time.time()
    data_manager = DataManager(cfg_dict)
    error_analyzer = ErrorAnalyzer(data_manager)
    if smoke_mode:
        dur = _time.time() - t0
        print(f"  [smoke] load_dataset: {dur:.1f}s")
        smoke_timings.append(("load_dataset", round(dur, 2)))
        diag.log("smoke_timing", stage="load_dataset", duration_s=round(dur, 2))

    evaluator_path = str(SCRIPT_DIR / "evaluator.py")
    active_batch_path = SCRIPT_DIR / "active_batch.json"
    error_context_path = SCRIPT_DIR / "error_context.txt"

    # ======================================================================
    # Step 0: One-time full pool evaluation to initialize Seen/Unseen
    # ======================================================================
    print("=" * 60)
    print("Initialization: evaluating initial prompt on full train pool")
    print("=" * 60)
    if smoke_mode:
        t0 = _time.time()
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
    if smoke_mode:
        dur = _time.time() - t0
        print(f"  [smoke] init_full_eval (train pool n={data_manager.n_total}): {dur:.1f}s")
        smoke_timings.append(("init_full_eval", round(dur, 2)))
        diag.log("smoke_timing", stage="init_full_eval", duration_s=round(dur, 2), n_samples=data_manager.n_total)
    print(f"  Total pool: {data_manager.n_total}")
    print(f"  Initial batch: {len(batch_indices)} (Hard: {data_manager.n_hard}, Anchor: {data_manager.n_anchor})")
    print(f"  Unseen: {data_manager.n_unseen}")
    diag.log("init", total_pool=data_manager.n_total, batch=len(batch_indices),
             hard=data_manager.n_hard, anchor=data_manager.n_anchor, unseen=data_manager.n_unseen)

    # Baseline test evaluation (initial prompt on test set)
    print("  Evaluating initial prompt on test set...")
    if smoke_mode:
        t0 = _time.time()
    baseline_test_metrics = _evaluate_split_full(current_prompt, cfg_dict, split_name="test")
    if smoke_mode:
        dur = _time.time() - t0
        print(f"  [smoke] baseline_test: {dur:.1f}s")
        smoke_timings.append(("baseline_test", round(dur, 2)))
        diag.log("smoke_timing", stage="baseline_test", duration_s=round(dur, 2))
    with open(results_dir / "baseline_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(baseline_test_metrics, f, indent=2)
    print(f"  Baseline test: R_global={baseline_test_metrics['R_global']:.2%}  "
          f"Acc_Hard={baseline_test_metrics['Acc_Hard']:.2%}  Acc_Anchor={baseline_test_metrics['Acc_Anchor']:.2%}")
    diag.log("baseline_test", **{k: v for k, v in baseline_test_metrics.items() if isinstance(v, (int, float))})

    # ======================================================================
    # Main AL loop
    # ======================================================================
    for al_iter in range(n_al_iterations):
        t_cycle_start = _time.time()
        print(f"\n{'=' * 60}")
        print(f"AL Cycle {al_iter + 1}/{n_al_iterations}  |  Hard: {data_manager.n_hard}  "
              f"Anchor: {data_manager.n_anchor}  Unseen: {data_manager.n_unseen}")
        print("=" * 60)

        # Save current prompt
        prompt_file = results_dir / f"al_iter_{al_iter}_prompt.txt"
        prompt_file.write_text(current_prompt, encoding="utf-8")

        # Build active batch
        if smoke_mode:
            t0_batch = _time.time()
        batch_indices, hard_in_batch, anchor_in_batch = data_manager.build_active_batch(
            batch_size=batch_size, hard_ratio=hard_ratio, seed=42 + al_iter,
        )
        print(f"  Active batch: {len(batch_indices)} (Hard: {len(hard_in_batch)}, Anchor: {len(anchor_in_batch)})")
        diag.log("cycle_start", al_iter=al_iter, batch_size=len(batch_indices),
                 n_hard_in_batch=len(hard_in_batch), n_anchor_in_batch=len(anchor_in_batch),
                 n_hard_total=data_manager.n_hard, n_anchor_total=data_manager.n_anchor,
                 n_unseen=data_manager.n_unseen, seed_prompt_len=len(current_prompt))

        # Error analysis for system message
        error_analysis = error_analyzer.analyze_errors(
            data_manager.hard_indices,
            k_clusters=min(6, max(1, data_manager.n_hard // 2)),
            seed=42,
        )
        error_context = error_analyzer.format_for_evolution(error_analysis)
        error_context_path.write_text(error_context, encoding="utf-8")

        # Write active_batch.json for the evaluator
        _write_active_batch(batch_indices, hard_in_batch, anchor_in_batch, active_batch_path)
        if smoke_mode:
            dur = _time.time() - t0_batch
            st = f"cycle_{al_iter}_build_batch"
            print(f"  [smoke] {st}: {dur:.1f}s")
            smoke_timings.append((st, round(dur, 2)))
            diag.log("smoke_timing", stage=st, duration_s=round(dur, 2), al_iter=al_iter)

        # Build evolution config for this cycle
        base_system = cfg_dict.get("prompt", {}).get("system_message", "")
        augmented_system = base_system + "\n\n## Error Pattern Summary (from batch analysis):\n" + error_context
        evolve_config = load_config(str(config_path))
        if evolve_config.prompt:
            evolve_config.prompt.system_message = augmented_system

        out_dir = results_dir / f"al_iter_{al_iter}"
        out_dir.mkdir(parents=True, exist_ok=True)
        db_dir = out_dir / "database"
        if db_dir.exists():
            import shutil
            shutil.rmtree(db_dir)
        evolve_config.database.db_path = str(db_dir)

        # Override evolution trace path so it lands in the per-cycle output
        oe_output_dir = str(out_dir / "openevolve_output")
        if evolve_config.evolution_trace and evolve_config.evolution_trace.enabled:
            evolve_config.evolution_trace.output_path = ""

        # Run OpenEvolve
        print(f"  Running evolution ({n_evolve_iterations} iterations)...")
        (out_dir / "start_prompt.txt").write_text(current_prompt, encoding="utf-8")
        diag.log("evolution_start", al_iter=al_iter, n_iterations=n_evolve_iterations)

        best_evolution_score = -1.0
        if smoke_mode:
            t0_evo = _time.time()
        try:
            result = run_evolution(
                initial_program=current_prompt,
                evaluator=evaluator_path,
                config=evolve_config,
                iterations=n_evolve_iterations,
                output_dir=oe_output_dir,
                cleanup=False,
            )
            best_code = current_prompt
            if result:
                if result.best_program and hasattr(result.best_program, "code"):
                    best_code = result.best_program.code
                    best_evolution_score = result.best_score
                elif result.best_code:
                    best_code = result.best_code
                    best_evolution_score = result.best_score
            current_prompt = best_code
            (out_dir / "best_prompt.txt").write_text(current_prompt, encoding="utf-8")
            diag.log("evolution_done", al_iter=al_iter, best_score=best_evolution_score,
                     best_prompt_len=len(current_prompt))
        except Exception as e:
            print(f"  Evolution failed: {e}")
            import traceback
            traceback.print_exc()
            diag.log("evolution_error", al_iter=al_iter, error=str(e))
        if smoke_mode:
            dur = _time.time() - t0_evo
            st = f"cycle_{al_iter}_evolution"
            print(f"  [smoke] {st} ({n_evolve_iterations} iter): {dur:.1f}s")
            smoke_timings.append((st, round(dur, 2)))
            diag.log("smoke_timing", stage=st, duration_s=round(dur, 2), al_iter=al_iter)

        # Log programs in archive
        top_progs = _load_top_programs_from_db(db_dir, n=10)
        diag.log("archive_snapshot", al_iter=al_iter, n_programs=len(top_progs),
                 programs=[{"score": p["metrics"].get("combined_score"),
                            "Acc_Hard": p["metrics"].get("Acc_Hard"),
                            "Acc_Anchor": p["metrics"].get("Acc_Anchor"),
                            "prompt_snippet": p["code"][:200]} for p in top_progs[:5]])

        # Re-evaluate batch
        print("  Re-evaluating batch...")
        if smoke_mode:
            t0 = _time.time()
        batch_results = _evaluate_batch(current_prompt, batch_indices, data_manager, cfg_dict)
        hard_now, anchor_now = data_manager.reclassify_batch(
            batch_indices, batch_results["predictions"],
            batch_results["gold_labels"], batch_results["worker_predictions"],
        )
        if smoke_mode:
            dur = _time.time() - t0
            st = f"cycle_{al_iter}_reeval_batch"
            print(f"  [smoke] {st} (n={len(batch_indices)}): {dur:.1f}s")
            smoke_timings.append((st, round(dur, 2)))
            diag.log("smoke_timing", stage=st, duration_s=round(dur, 2), al_iter=al_iter)
        print(f"  After evolution: Hard: {len(hard_now)}, Anchor: {len(anchor_now)}")
        diag.log("reclassify", al_iter=al_iter, hard=len(hard_now), anchor=len(anchor_now))

        # Validation
        print("  Validation...")
        if smoke_mode:
            t0 = _time.time()
        val_ha = _evaluate_split_full(current_prompt, cfg_dict, split_name="validation")
        if smoke_mode:
            dur = _time.time() - t0
            st = f"cycle_{al_iter}_validation"
            print(f"  [smoke] {st}: {dur:.1f}s")
            smoke_timings.append((st, round(dur, 2)))
            diag.log("smoke_timing", stage=st, duration_s=round(dur, 2), al_iter=al_iter)
        diag.log("validation", al_iter=al_iter,
                 **{k: v for k, v in val_ha.items() if isinstance(v, (int, float))})

        # --- Consolidation (Level 2) — always accept if valid ---
        consolidated = False
        cons_top = _load_top_programs_from_db(db_dir, n=3)
        if len(cons_top) >= 2:
            print(f"  Consolidation: analyzing top {len(cons_top)} prompts...")
            diag.log("consolidation_start", al_iter=al_iter, n_top=len(cons_top),
                     top_scores=[p["metrics"].get("combined_score") for p in cons_top])
            if smoke_mode:
                t0_cons = _time.time()

            best_artifacts = cons_top[0].get("artifacts", None)
            cons_prompt = _consolidate_prompt(cons_top, al_iter, cfg_dict,
                                              error_artifacts=best_artifacts)
            if cons_prompt is not None:
                (out_dir / "consolidated_prompt.txt").write_text(cons_prompt, encoding="utf-8")

                print("  Evaluating consolidated prompt on batch...")
                cons_score = _quick_batch_score(
                    cons_prompt, batch_indices, hard_in_batch, anchor_in_batch,
                    data_manager, cfg_dict,
                )
                if smoke_mode:
                    dur = _time.time() - t0_cons
                    st = f"cycle_{al_iter}_consolidation"
                    print(f"  [smoke] {st}: {dur:.1f}s")
                    smoke_timings.append((st, round(dur, 2)))
                    diag.log("smoke_timing", stage=st, duration_s=round(dur, 2), al_iter=al_iter)
                print(f"  Consolidated batch score: {cons_score:.4f}  (evolution best: {best_evolution_score:.4f})")
                diag.log("consolidation_eval", al_iter=al_iter,
                         cons_score=cons_score, evo_best_score=best_evolution_score)

                current_prompt = cons_prompt
                consolidated = True
                print("  Consolidated prompt ACCEPTED (always-accept policy).")
                diag.log("consolidation_decision", al_iter=al_iter,
                         accepted=True, cons_score=cons_score, evo_score=best_evolution_score)
            else:
                if smoke_mode:
                    dur = _time.time() - t0_cons
                    st = f"cycle_{al_iter}_consolidation"
                    smoke_timings.append((st, round(dur, 2)))
                    diag.log("smoke_timing", stage=st, duration_s=round(dur, 2), al_iter=al_iter)
                print("  Consolidation LLM failed; keeping best evolution prompt.")
                diag.log("consolidation_failed", al_iter=al_iter)
        else:
            print("  Skipping consolidation (fewer than 2 programs in archive).")
            diag.log("consolidation_skip", al_iter=al_iter, reason="too_few_programs")

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
            "evo_best_score": best_evolution_score,
        }

        # Expansion check
        if data_manager.needs_expansion(threshold=expansion_trigger):
            n_hard_needed = int(batch_size * hard_ratio) - data_manager.n_hard
            n_hard_needed = max(n_hard_needed, expansion_trigger)
            print(f"  Expansion triggered: Hard={data_manager.n_hard} <= {expansion_trigger}. "
                  f"Adding {n_hard_needed} from Unseen...")
            new_indices = data_manager.expand_pool(n_new=n_hard_needed, seed=42 + al_iter)
            data_manager.hard_indices.extend(new_indices)
            print(f"  After expansion: Hard: {data_manager.n_hard}, Anchor: {data_manager.n_anchor}, "
                  f"Unseen: {data_manager.n_unseen}")
            entry["expanded"] = True
            entry["n_expanded"] = len(new_indices)
            diag.log("expansion", al_iter=al_iter, added=len(new_indices),
                     hard=data_manager.n_hard, anchor=data_manager.n_anchor, unseen=data_manager.n_unseen)

        cycle_time = _time.time() - t_cycle_start
        print(f"  Val: R_global={val_ha['R_global']:.2%}  Acc_Hard={val_ha['Acc_Hard']:.2%}  "
              f"Acc_Anchor={val_ha['Acc_Anchor']:.2%}  (Hard:{val_ha['n_hard']}/Anchor:{val_ha['n_anchor']})  "
              f"[{cycle_time:.0f}s]")
        entry["cycle_time_s"] = round(cycle_time, 1)
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

    # Final test evaluation
    print("\n" + "=" * 60)
    print("Final test evaluation")
    print("=" * 60)
    if smoke_mode:
        t0 = _time.time()
    final_test_metrics = _evaluate_split_full(current_prompt, cfg_dict, split_name="test")
    if smoke_mode:
        dur = _time.time() - t0
        print(f"  [smoke] final_test: {dur:.1f}s")
        smoke_timings.append(("final_test", round(dur, 2)))
        diag.log("smoke_timing", stage="final_test", duration_s=round(dur, 2))
    with open(results_dir / "final_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_test_metrics, f, indent=2)
    (results_dir / "final_prompt.txt").write_text(current_prompt, encoding="utf-8")

    print(f"Baseline test: R_global={baseline_test_metrics['R_global']:.2%}  "
          f"R_worst={baseline_test_metrics['R_worst']:.2%}  combined={baseline_test_metrics['combined_score']:.4f}")
    print(f"Final test:    R_global={final_test_metrics['R_global']:.2%}  "
          f"R_worst={final_test_metrics['R_worst']:.2%}  combined={final_test_metrics['combined_score']:.4f}")
    print(f"Final test:    Acc_Hard={final_test_metrics['Acc_Hard']:.2%}  "
          f"Acc_Anchor={final_test_metrics['Acc_Anchor']:.2%}  "
          f"(Hard:{final_test_metrics['n_hard']}/Anchor:{final_test_metrics['n_anchor']})")
    delta_r = final_test_metrics["R_global"] - baseline_test_metrics["R_global"]
    delta_comb = final_test_metrics["combined_score"] - baseline_test_metrics["combined_score"]
    print(f"Delta:        R_global={delta_r:+.2%}  combined={delta_comb:+.4f}")
    print("=" * 60)

    diag.log("final_test", **{k: v for k, v in final_test_metrics.items() if isinstance(v, (int, float))},
             delta_R_global=delta_r, delta_combined=delta_comb)

    # Token report
    try:
        from token_usage import get_tracker
        usage_path = results_dir / "token_usage.json"
        report_path = results_dir / "token_usage_report.md"
        tracker = get_tracker()
        tracker.save_json(usage_path)
        tracker.write_report(report_path, title="Token usage report")
        u = tracker.get_usage()
        print(f"Token usage: total={u['total_tokens']:,} (report: {report_path})")
    except Exception as e:
        print(f"Token usage save skipped: {e}")

    diag.close()

    if smoke_mode and smoke_timings:
        total_s = sum(d for _, d in smoke_timings)
        print("\n" + "=" * 60)
        print("  [smoke] Timing summary")
        print("=" * 60)
        for stage, dur in smoke_timings:
            print(f"    {stage}: {dur:.1f}s")
        print(f"    TOTAL: {total_s:.1f}s")
        print("=" * 60)

    print(f"\nDone. Log: {results_dir / 'active_loop_log.json'}")
    print(f"Debug trace: {results_dir / 'debug_trace.jsonl'}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-al", type=int, default=4)
    parser.add_argument("--n-evolve", type=int, default=15)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Output directory (default: results). Use results_smoke for smoke run.")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke run: 2 AL cycles, 4 evolve iter, results_smoke/.")
    args = parser.parse_args()

    if args.smoke:
        n_al, n_evolve = 2, 4
        out_dir = SCRIPT_DIR / "results_smoke"
        print("Smoke run: --n-al 2 --n-evolve 4 --results-dir results_smoke")
    else:
        n_al, n_evolve = args.n_al, args.n_evolve
        out_dir = Path(args.results_dir) if args.results_dir else None

    run_active_loop(
        n_al_iterations=n_al,
        n_evolve_iterations=n_evolve,
        initial_prompt_path=Path(args.prompt) if args.prompt else None,
        results_dir=out_dir,
        smoke_mode=args.smoke,
    )


if __name__ == "__main__":
    main()

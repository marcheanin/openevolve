"""
Active Learning Loop Controller for Active Prompt Evolution.

Two-level evolution:
  Level 1 (inner, fast): Evolve <DynamicRules> + <FewShotExamples> via OpenEvolve
      (optional synthetic few-shot replace or draft-in-system; mutator lock is configurable)
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

Holdout test: baseline + final run, plus per-cycle test metrics written to the log for
graphs only (not used for consolidation, expansion triggers, or prompt selection).
"""

import json
import os
import re
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
from synthetic_fewshot_generator import (
    SyntheticFewShotGenerator,
    save_synthetic_manifest,
)


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
    
    # Validation updated for CoT structure
    # if "Rating:" not in prompt_text:
    #    errors.append("Missing 'Rating:' line in <Task>")
    
    # Check for CoT related keywords instead of strict single number
    # has_format = any(p in prompt_text.lower() for p in [
    #    "single number", "only a single number", "only a number",
    #    "1, 2, 3, 4, or 5", "1-5",
    # ])
    # if not has_format:
    #    errors.append("Missing output format instruction (e.g. 'ONLY a single number: 1, 2, 3, 4, or 5')")
    
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


def _extract_mutation_log(text: str) -> Optional[str]:
    """
    Extract <mutation_log>...</mutation_log> block from prompt text.
    Returns stripped content or None if the block is absent.
    """
    if not text:
        return None
    m = re.search(r"<mutation_log>\s*(.*?)\s*</mutation_log>", text, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    block = m.group(1).strip()
    return block or None


def _load_mutation_logs_from_trace(trace_path: Path, top_n: int = 10) -> list:
    """
    Read mutation logs from evolution_trace.jsonl (llm_response field).
    Returns top entries by combined_score.
    """
    if not trace_path.exists():
        return []
    entries = []
    try:
        with open(trace_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                llm_response = row.get("llm_response", "") or ""
                mutation_log = _extract_mutation_log(llm_response)
                child_metrics = row.get("child_metrics", {}) or {}
                if not isinstance(child_metrics, dict):
                    child_metrics = {}
                score = child_metrics.get("combined_score", None)
                if not isinstance(score, (int, float)):
                    continue
                child_code = row.get("child_code", "") or ""
                entries.append(
                    {
                        "score": float(score),
                        "mutation_log": mutation_log,
                        "has_mutation_log": bool(mutation_log),
                        "prompt_snippet": child_code[:160],
                    }
                )
    except Exception:
        return []
    entries.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return entries[:top_n]


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

Your PRIMARY goal is to produce a prompt that is **SHORTER or equal in length** \
to Prompt 1, while preserving or improving classification accuracy. \
Prompt length directly hurts fitness through length_penalty and wastes API budget.

CONSOLIDATION STRATEGY (in this priority order):
1. **MERGE truly near-identical rules.** Scan <DynamicRules> for pairs that \
describe the same signal with different wording. Combine each such pair into \
one concise rule. Only merge rules that are genuinely redundant in meaning — \
do NOT aggressively prune rules that cover distinct concepts.
2. **TIGHTEN wording.** Shorten verbose rule descriptions without losing meaning.
3. **IMPORT targeted fixes** from other top prompts ONLY if they address a \
specific error pattern from the artifacts AND no existing rule already covers it.
4. **Promote** consistently successful DynamicRules to <BaseGuidelines> if they \
are general enough, then remove them from DynamicRules.

SOFT BUDGET: aim for 30-50 distinct rules/bullets in <DynamicRules>. If the input has >50 \
rules, merge only truly redundant pairs — do NOT merge distinct boundary rules into one vague line.

CRITICAL CONSTRAINTS:
- The best prompt is your STARTING POINT. Do NOT rewrite it from scratch.
- Ensure <DynamicRules> covers ALL 5 rating levels (1-5) with at least 2 rules each.
- Ensure <FewShotExamples> has at least one example per rating level (1-5), \
aim for 7-10 examples total, do NOT exceed 12.
- Keep the output format instruction (single rating 1-5) in <BaseGuidelines>.

ALLOWED modifications:
- <BaseGuidelines>: Promote general rules here. Keep output format and scale.
- <DynamicRules>: MERGE near-duplicates, REFINE wording, ADD only if needed.
- <FewShotExamples>: Replace weak examples with error cases from artifacts. \
Each example must end with a single line: Rating: [1-5]. \
Never remove an example if it is the only one for its rating level.

FORBIDDEN:
- Do NOT change the <Task> section. It must remain EXACTLY as in Prompt 1.
- Do NOT remove the <Role> tag inside <System>.
- Preserve the XML structure: <System>(<Role>, <BaseGuidelines>, <DynamicRules>)\
</System>, <FewShotExamples>, <Task>.

OUTPUT FORMAT:
First, output a brief <consolidation_log> block (3-7 lines) listing what you \
merged, removed, added, or refined. Then output the complete prompt text.

Example:
<consolidation_log>
- Merged 3 pairs of near-duplicate 4-star rules (45→39 rules).
- Tightened wording on 5 rules (-120 chars).
- Added 1 boundary rule for gold=3→pred=4 from error artifacts.
- Replaced few-shot #7 with misclassified review (gold=2).
</consolidation_log>
<System>
...complete prompt...
</Task>\
"""


def _consolidate_prompt(
    top_programs: list,
    al_iter: int,
    config: dict,
    max_retries: int = 2,
    error_artifacts: Optional[dict] = None,
    metrics_summary: Optional[dict] = None,
) -> Optional[str]:
    """
    Consolidation step: take the best prompt as base, enrich with rules from
    other top prompts and fix errors shown in artifacts.
    metrics_summary: optional validation-only metrics (never use holdout test here).
    Returns consolidated prompt text or None on failure.
    """
    parts = [f"Here are the top {len(top_programs)} prompts from AL cycle {al_iter + 1}:\n"]

    if metrics_summary:
        parts.append(
            "Validation metrics for the current best prompt (holdout test is NOT used in training):"
        )
        parts.append(
            "  combined_score={val_combined:.4f}, R_global={val_R_global:.2%}, "
            "R_worst={val_R_worst:.2%}".format(
                val_combined=metrics_summary.get("val_combined", 0.0),
                val_R_global=metrics_summary.get("val_R_global", 0.0),
                val_R_worst=metrics_summary.get("val_R_worst", 0.0),
            )
        )
        parts.append("")

    def _extract_section(text: str, tag: str) -> str:
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start = text.find(start_tag)
        end = text.find(end_tag)
        if start == -1 or end == -1 or end <= start:
            return ""
        return text[start + len(start_tag) : end]

    base_prompt_text = top_programs[0]["code"]
    base_dyn = _extract_section(base_prompt_text, "DynamicRules")
    base_few = _extract_section(base_prompt_text, "FewShotExamples")

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

    extra_rules_blocks = []
    extra_fewshot_blocks = []
    for i, prog in enumerate(top_programs[1:], start=2):
        code_i = prog["code"]
        dyn_i = _extract_section(code_i, "DynamicRules")
        few_i = _extract_section(code_i, "FewShotExamples")

        if dyn_i:
            base_lines = {ln.strip() for ln in base_dyn.splitlines() if ln.strip()}
            dyn_lines = [
                ln for ln in dyn_i.splitlines()
                if ln.strip() and ln.strip() not in base_lines
            ]
            if dyn_lines:
                extra_rules_blocks.append(
                    f"Additional DynamicRules from Prompt {i} that are NOT in Prompt 1:\n"
                    + "\n".join(f"  - {ln.strip()}" for ln in dyn_lines[:20])
                )

        if few_i:
            base_examples = {ln.strip() for ln in base_few.splitlines() if ln.strip()}
            few_lines = [
                ln for ln in few_i.splitlines()
                if ln.strip() and ln.strip() not in base_examples
            ]
            if few_lines:
                extra_fewshot_blocks.append(
                    f"Additional FewShotExamples fragments from Prompt {i} that are NOT in Prompt 1:\n"
                    + "\n".join(f"  - {ln.strip()}" for ln in few_lines[:20])
                )

    if extra_rules_blocks or extra_fewshot_blocks:
        parts.append("=== DIFFERENTIAL VIEW (import only what fixes real errors) ===")
        if extra_rules_blocks:
            parts.append("DynamicRules candidates from other top prompts:")
            parts.extend(extra_rules_blocks)
        if extra_fewshot_blocks:
            parts.append("")
            parts.append("Few-shot fragments from other top prompts:")
            parts.extend(extra_fewshot_blocks)
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

            # Strip <consolidation_log>...</consolidation_log> block if present
            import re as _re
            result = _re.sub(
                r"<consolidation_log>.*?</consolidation_log>\s*",
                "",
                result,
                flags=_re.DOTALL,
            )

            # Strip markdown fences if the LLM wrapped its output
            if result.startswith("```"):
                lines = result.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                result = "\n".join(lines)

            result = result.strip()

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


def _sample_hard_examples(
    data_manager: DataManager,
    max_n: int = 8,
) -> List[Tuple[str, int]]:
    """Return a small sample of current hard examples: (text, gold_label)."""
    out: List[Tuple[str, int]] = []
    for idx in data_manager.hard_indices[:max_n]:
        out.append((data_manager.texts[idx], int(data_manager.labels[idx])))
    return out


def _top_confusion_pairs_from_batch(
    data_manager: DataManager,
    batch_indices: Optional[List[int]],
    predictions: Optional[List[int]],
    top_k: int = 4,
) -> List[Tuple[int, int, int]]:
    """Return top confusion pairs (gold, pred, count) from previous batch eval."""
    if not batch_indices or not predictions:
        return []
    if len(batch_indices) != len(predictions):
        return []

    from collections import Counter

    confusion = Counter()
    for j, idx in enumerate(batch_indices):
        gold = int(data_manager.labels[idx])
        pred = int(predictions[j])
        if gold != pred:
            confusion[(gold, pred)] += 1
    return [(g, p, c) for ((g, p), c) in confusion.most_common(top_k)]


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


def _is_api_error(exc: BaseException) -> bool:
    """True if exception looks like API/auth failure (401, timeout, etc.)."""
    msg = str(exc).lower()
    return (
        "401" in msg or "user not found" in msg or "llm call failed" in msg
        or "authentication" in msg or "api key" in msg
    )


def _exit_on_api_error(
    exc: BaseException,
    results_dir: Path,
    last_completed_cycle: int,
    prompt_path: Optional[Path],
    n_al: int,
    n_evolve: int,
) -> None:
    """Write resume state and exit with instructions."""
    state = {
        "last_completed_cycle": last_completed_cycle,
        "next_cycle": last_completed_cycle + 1,
        "prompt_path": str(prompt_path) if prompt_path else None,
        "n_al_iterations": n_al,
        "n_evolve_iterations": n_evolve,
    }
    state_path = results_dir / "resume_state.json"
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    print("\n" + "!" * 60)
    print("API error (e.g. 401). Check OPENROUTER_API_KEY in .env")
    print(str(exc))
    print(f"Resume state saved to {state_path}")
    print("Fix the key and run with: --resume-from-dir", results_dir)
    print("!" * 60 + "\n")
    sys.exit(1)


def run_active_loop(
    n_al_iterations: int = 4,
    n_evolve_iterations: int = 15,
    initial_prompt_path: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    smoke_mode: bool = False,
    resume_start_cycle: Optional[int] = None,
    config_path: Optional[Path] = None,
    no_evolve_early_stop: bool = False,
    no_al_early_stop: bool = False,
) -> None:
    """
    Run the Active Prompt Evolution loop with pool expansion.

    Scheme C: few AL cycles (3-5), moderate evolve iterations (15-20) each.
    Total ~60-80 OpenEvolve iterations. MAP-Elites archive resets per cycle
    to keep fitness scores clean after batch changes.
    """
    from openevolve.api import run_evolution
    from openevolve.config import load_config

    config_path = Path(config_path) if config_path else (SCRIPT_DIR / "config.yaml")
    config_path = config_path.resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    if no_al_early_stop:
        al_sec = cfg_dict.setdefault("active_learning", {})
        al_sec["al_early_stopping_patience"] = 0
    config = load_config(str(config_path))
    prompt_path = initial_prompt_path or (SCRIPT_DIR / "initial_prompt.txt")
    results_dir = Path(results_dir) if results_dir else (SCRIPT_DIR / "results")
    results_dir = results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    log_entries = []
    log_path = results_dir / "active_loop_log.json"
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            log_entries = json.load(f)

    if resume_start_cycle is not None:
        if resume_start_cycle <= 0:
            raise ValueError("resume_start_cycle must be >= 1")
        resume_prompt_path = results_dir / f"al_iter_{resume_start_cycle}" / "best_prompt.txt"
        if not resume_prompt_path.exists():
            raise FileNotFoundError(
                f"Resume prompt not found: {resume_prompt_path}. "
                "Ensure evolution for that cycle completed (best_prompt.txt exists)."
            )
        prompt_path = resume_prompt_path
        print(f"Resuming from cycle {resume_start_cycle + 1}, prompt: {prompt_path}")

    diag = DiagnosticLogger(results_dir / "debug_trace.jsonl")
    smoke_timings = []  # (stage, duration_s) when smoke_mode

    al_cfg = cfg_dict.get("active_learning", {})
    batch_size = al_cfg.get("batch_size", 80)
    hard_ratio = al_cfg.get("hard_ratio", 0.7)
    expansion_trigger = al_cfg.get("expansion_trigger", 5)
    refresh_per_cycle = al_cfg.get("refresh_per_cycle", 0)
    consolidation_gate_delta = al_cfg.get("consolidation_gate_delta", 0.02)
    consolidation_gate_vs_best = al_cfg.get("consolidation_gate_vs_best_delta", 0.04)
    soft_skip_near_best = al_cfg.get("soft_expansion_skip_near_best", 0.005)
    al_stop_patience = (cfg_dict.get("active_learning") or {}).get("al_early_stopping_patience", 0)
    al_stop_min_cycles = al_cfg.get("al_early_stopping_min_cycles", 2)
    min_hard_batch_ratio = al_cfg.get("min_hard_batch_ratio", 0.55)
    gap_warn_threshold = float(al_cfg.get("generalization_gap_warn_threshold", 0.03))
    gap_stop_threshold = float(al_cfg.get("generalization_gap_stop_threshold", 0.08))
    gap_stop_patience = int(al_cfg.get("generalization_gap_stop_patience", 2))
    synthetic_gen = SyntheticFewShotGenerator.from_config_dict(cfg_dict)
    sf_cfg_loop = al_cfg.get("synthetic_fewshot") or {}
    mutator_lock_fewshot = bool(sf_cfg_loop.get("mutator_lock_fewshot", False))

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
    try:
        full_results = _evaluate_batch(
            current_prompt, list(range(data_manager.n_total)), data_manager, cfg_dict
        )
    except RuntimeError as e:
        if _is_api_error(e):
            _exit_on_api_error(e, results_dir, 0, None, n_al_iterations, n_evolve_iterations)
        raise
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

    # Baseline test evaluation (initial prompt on test set); skip when resuming
    baseline_path = results_dir / "baseline_test_metrics.json"
    if resume_start_cycle is None:
        print("  Evaluating initial prompt on test set...")
        if smoke_mode:
            t0 = _time.time()
        try:
            baseline_test_metrics = _evaluate_split_full(current_prompt, cfg_dict, split_name="test")
        except RuntimeError as e:
            if _is_api_error(e):
                _exit_on_api_error(e, results_dir, 0, None, n_al_iterations, n_evolve_iterations)
            raise
        if smoke_mode:
            dur = _time.time() - t0
            print(f"  [smoke] baseline_test: {dur:.1f}s")
            smoke_timings.append(("baseline_test", round(dur, 2)))
            diag.log("smoke_timing", stage="baseline_test", duration_s=round(dur, 2))
        with open(baseline_path, "w", encoding="utf-8") as f:
            json.dump(baseline_test_metrics, f, indent=2)
        print(f"  Baseline test: R_global={baseline_test_metrics['R_global']:.2%}  "
              f"Acc_Hard={baseline_test_metrics['Acc_Hard']:.2%}  Acc_Anchor={baseline_test_metrics['Acc_Anchor']:.2%}")
        diag.log("baseline_test", **{k: v for k, v in baseline_test_metrics.items() if isinstance(v, (int, float))})
    else:
        if not baseline_path.exists():
            raise FileNotFoundError(f"Resume requires {baseline_path} from initial run.")
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline_test_metrics = json.load(f)
        print(f"  Baseline test (from file): R_global={baseline_test_metrics['R_global']:.2%}  "
              f"Acc_Hard={baseline_test_metrics['Acc_Hard']:.2%}  Acc_Anchor={baseline_test_metrics['Acc_Anchor']:.2%}")

    # ======================================================================
    # Main AL loop
    # ======================================================================
    if resume_start_cycle is not None:
        # Восстановление незавершённого цикла (смещение на конкретный al_iter)
        al_start = resume_start_cycle
    elif log_entries:
        # Расширение существующего эксперимента: начинаем с следующего цикла
        last_iter = max(e.get("al_iter", -1) for e in log_entries)
        al_start = last_iter + 1
    else:
        al_start = 0

    # History of val_combined_score for soft expansion trigger
    val_scores_history = []
    gap_stop_bad_streak = 0
    if log_entries:
        val_scores_history = [
            e.get("seed_val_score", e.get("val_combined_score", -1.0))
            for e in log_entries
        ]

    best_val_score = -1.0
    best_val_prompt = current_prompt
    best_val_cycle = -1
    if log_entries:
        for e in log_entries:
            sv = e.get("seed_val_score", e.get("val_combined_score", -1.0))
            ai = e.get("al_iter", -1)
            if sv > best_val_score:
                best_val_score = sv
                best_val_cycle = ai
        best_val_prompt_path = results_dir / "best_val_prompt.txt"
        if best_val_prompt_path.exists():
            best_val_prompt = best_val_prompt_path.read_text(encoding="utf-8")

    last_cycle_predictions: Optional[List[int]] = None
    last_cycle_worker_preds: Optional[List[List[int]]] = None
    last_cycle_batch_indices: Optional[List[int]] = None
    last_cycle_seed_val_score: Optional[float] = None

    for al_iter in range(al_start, n_al_iterations):
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
            batch_size=batch_size,
            hard_ratio=hard_ratio,
            seed=42 + al_iter,
            min_hard_ratio=min_hard_batch_ratio,
        )
        hb_frac = len(hard_in_batch) / len(batch_indices) if batch_indices else 0.0
        print(
            f"  Active batch: {len(batch_indices)} (Hard: {len(hard_in_batch)}, Anchor: {len(anchor_in_batch)}; "
            f"hard_frac={hb_frac:.2%})"
        )
        diag.log("cycle_start", al_iter=al_iter, batch_size=len(batch_indices),
                 n_hard_in_batch=len(hard_in_batch), n_anchor_in_batch=len(anchor_in_batch),
                 n_hard_total=data_manager.n_hard, n_anchor_total=data_manager.n_anchor,
                 n_unseen=data_manager.n_unseen, seed_prompt_len=len(current_prompt))

        out_dir = results_dir / f"al_iter_{al_iter}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Error analysis for system message
        error_analysis = error_analyzer.analyze_errors(
            hard_in_batch,
            k_clusters=min(6, max(1, len(hard_in_batch) // 2)),
            seed=42,
        )
        batch_stats = {
            "batch_size": len(batch_indices),
            "batch_hard": len(hard_in_batch),
            "batch_anchor": len(anchor_in_batch),
            "n_seen": data_manager.n_seen,
            "n_unseen": data_manager.n_unseen,
            "n_hard_total": data_manager.n_hard,
            "n_anchor_total": data_manager.n_anchor,
        }
        error_context = error_analyzer.format_for_evolution(
            error_analysis,
            predictions=last_cycle_predictions,
            worker_predictions=last_cycle_worker_preds,
            batch_indices=last_cycle_batch_indices,
            batch_stats=batch_stats,
        )
        error_context_path.write_text(error_context, encoding="utf-8")

        # Synthetic FewShotExamples: optional replace in prompt, or inject_as_hint in system message.
        synth_raw_path = out_dir / "synthetic_fewshot_examples.txt"
        synth_manifest_path = out_dir / "synthetic_fewshot_manifest.json"
        synthetic_fewshot_system_hint = ""
        if synthetic_gen.cfg.enabled:
            hard_samples = _sample_hard_examples(data_manager, max_n=8)
            prompt_after_synth, synth_err, synth_system_hint = synthetic_gen.generate(
                current_prompt=current_prompt,
                error_context=error_context,
                hard_examples=hard_samples,
                confusion_pairs=_top_confusion_pairs_from_batch(
                    data_manager,
                    last_cycle_batch_indices,
                    last_cycle_predictions,
                    top_k=max(3, int(synthetic_gen.cfg.n_boundary_pairs)),
                ),
                save_raw_to=synth_raw_path,
            )
            if synth_system_hint:
                synthetic_fewshot_system_hint = synth_system_hint
            if prompt_after_synth is not None:
                current_prompt = prompt_after_synth
                save_synthetic_manifest(
                    synth_manifest_path,
                    enabled=True,
                    status="ok",
                    meta={
                        "mode": synthetic_gen.cfg.mode,
                        "mutator_lock_fewshot": mutator_lock_fewshot,
                        "hint_in_system_message": bool(synth_system_hint),
                        "n_hard_samples": len(hard_samples),
                        "n_examples_target": synthetic_gen.cfg.n_examples,
                        "n_boundary_pairs_target": synthetic_gen.cfg.n_boundary_pairs,
                    },
                )
                diag.log(
                    "synthetic_fewshot",
                    al_iter=al_iter,
                    status="ok",
                    mode=synthetic_gen.cfg.mode,
                    hint_in_system=bool(synth_system_hint),
                    n_hard_samples=len(hard_samples),
                )
                if synth_system_hint:
                    print("  Synthetic FewShotExamples: draft appended to evolution system message (inject_as_hint).")
                else:
                    print("  Synthetic FewShotExamples: generated and replaced current block.")
            else:
                save_synthetic_manifest(
                    synth_manifest_path,
                    enabled=True,
                    status="failed",
                    reason=synth_err or "unknown",
                    meta={
                        "mode": synthetic_gen.cfg.mode,
                        "mutator_lock_fewshot": mutator_lock_fewshot,
                    },
                )
                diag.log(
                    "synthetic_fewshot",
                    al_iter=al_iter,
                    status="failed",
                    reason=synth_err or "unknown",
                )
                print(f"  Synthetic FewShotExamples: generation failed ({synth_err}); using previous block.")
        else:
            save_synthetic_manifest(
                synth_manifest_path,
                enabled=False,
                status="disabled",
                meta={"mutator_lock_fewshot": mutator_lock_fewshot},
            )

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
        synthetic_lock = ""
        if mutator_lock_fewshot:
            synthetic_lock = (
                "\n\n## FewShotExamples lock for this cycle:\n"
                "Do NOT modify <FewShotExamples> in this mutation cycle. "
                "Focus changes on <DynamicRules> and, if useful, <BaseGuidelines>. "
                "(Few-shot may come from a synthetic step, consolidation, or the seed prompt.)"
            )
        hint_section = ""
        if synthetic_fewshot_system_hint:
            hint_section = (
                "\n\n## Suggested synthetic FewShot (draft)\n"
                + synthetic_fewshot_system_hint
            )
        cycle_context_lines = [
            "\n\n## Cycle Context",
            (
                f"AL cycle: {al_iter + 1}/{n_al_iterations}. Batch: "
                f"{len(hard_in_batch)} Hard + {len(anchor_in_batch)} Anchor = {len(batch_indices)}."
            ),
            (
                f"Pool snapshot: Seen={data_manager.n_seen}, Unseen={data_manager.n_unseen}, "
                f"Hard={data_manager.n_hard}, Anchor={data_manager.n_anchor}."
            ),
            (
                f"Global best seed_val so far: {best_val_score:.4f}"
                if best_val_score >= 0
                else "Global best seed_val so far: n/a (first cycle)."
            ),
            (
                f"Previous cycle seed_val: {last_cycle_seed_val_score:.4f}"
                if last_cycle_seed_val_score is not None
                else "Previous cycle seed_val: n/a."
            ),
            "Prompt budget hint: length penalty grows above ~2000 tokens; be concise unless accuracy clearly improves.",
        ]
        cycle_context = "\n".join(cycle_context_lines)
        augmented_system = (
            base_system
            + synthetic_lock
            + hint_section
            + cycle_context
            + "\n\n## Error Pattern Summary (from batch analysis):\n"
            + error_context
        )
        evolve_config = load_config(str(config_path))
        if no_evolve_early_stop:
            evolve_config.early_stopping_patience = None
        if evolve_config.prompt:
            evolve_config.prompt.system_message = augmented_system

        db_dir = out_dir / "database"
        is_resume_cycle = (resume_start_cycle is not None and al_iter == resume_start_cycle)
        if not is_resume_cycle and db_dir.exists():
            import shutil
            shutil.rmtree(db_dir)
        evolve_config.database.db_path = str(db_dir)

        # Override evolution trace path so it lands in the per-cycle output
        oe_output_dir = str(out_dir / "openevolve_output")
        if evolve_config.evolution_trace and evolve_config.evolution_trace.enabled:
            evolve_config.evolution_trace.output_path = ""

        best_evolution_score = -1.0
        if is_resume_cycle:
            # Skip evolution; keep current_prompt (already loaded from best_prompt.txt)
            (out_dir / "start_prompt.txt").write_text(current_prompt, encoding="utf-8")
            best_info_path = out_dir / "openevolve_output" / "best" / "best_program_info.json"
            if best_info_path.exists():
                with open(best_info_path, "r", encoding="utf-8") as f:
                    best_info = json.load(f)
                best_evolution_score = float(best_info.get("metrics", {}).get("combined_score", -1.0))
            print(f"  Resuming: skipping evolution, best_evolution_score={best_evolution_score:.4f}")
            diag.log("evolution_skipped_resume", al_iter=al_iter, best_score=best_evolution_score)
        else:
            # Run OpenEvolve
            print(f"  Running evolution ({n_evolve_iterations} iterations)...")
            (out_dir / "start_prompt.txt").write_text(current_prompt, encoding="utf-8")
            diag.log("evolution_start", al_iter=al_iter, n_iterations=n_evolve_iterations)

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
                diag.log(
                    "evolution_done",
                    al_iter=al_iter,
                    best_score=best_evolution_score,
                    best_prompt_len=len(current_prompt),
                )
            except Exception as e:
                print(f"  Evolution failed: {e}")
                import traceback
                traceback.print_exc()
                diag.log("evolution_error", al_iter=al_iter, error=str(e))
        if smoke_mode and not is_resume_cycle:
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
        trace_top = _load_mutation_logs_from_trace(
            Path(oe_output_dir) / "evolution_trace.jsonl",
            top_n=20,
        )
        trace_by_score = {}
        for t in trace_top:
            key = round(float(t.get("score", 0.0)), 10)
            trace_by_score.setdefault(key, []).append(t)

        mutation_logs_snapshot = []
        for rank, p in enumerate(top_progs[:10], start=1):
            score = float(p.get("metrics", {}).get("combined_score", 0.0))
            key = round(score, 10)
            matched = None
            bucket = trace_by_score.get(key) or []
            if bucket:
                matched = bucket.pop(0)
            mlog = matched.get("mutation_log") if matched else None
            mutation_logs_snapshot.append(
                {
                    "rank": rank,
                    "score": score,
                    "has_mutation_log": bool(mlog),
                    "mutation_log": mlog,
                    "prompt_snippet": (matched.get("prompt_snippet") if matched else p.get("code", "")[:160]),
                }
            )
        mutation_logs_payload = {
            "al_iter": al_iter,
            "n_programs": len(top_progs),
            "entries": mutation_logs_snapshot,
        }
        (out_dir / "mutation_logs_snapshot.json").write_text(
            json.dumps(mutation_logs_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        diag.log(
            "mutation_logs_snapshot",
            al_iter=al_iter,
            n_programs=len(top_progs),
            with_mutation_log=sum(1 for x in mutation_logs_snapshot if x["has_mutation_log"]),
            entries=[
                {
                    "rank": x["rank"],
                    "score": x["score"],
                    "has_mutation_log": x["has_mutation_log"],
                    "mutation_log": x["mutation_log"],
                }
                for x in mutation_logs_snapshot[:5]
            ],
        )

        # Re-evaluate batch
        print("  Re-evaluating batch...")
        if smoke_mode:
            t0 = _time.time()
        try:
            batch_results = _evaluate_batch(current_prompt, batch_indices, data_manager, cfg_dict)
        except RuntimeError as e:
            if _is_api_error(e):
                _exit_on_api_error(
                    e, results_dir, al_iter - 1,
                    results_dir / f"al_iter_{al_iter}" / "best_prompt.txt",
                    n_al_iterations, n_evolve_iterations,
                )
            raise
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

        last_cycle_predictions = list(batch_results["predictions"])
        last_cycle_worker_preds = [list(wp) for wp in batch_results["worker_predictions"]]
        last_cycle_batch_indices = list(batch_indices)

        # Validation
        print("  Validation...")
        if smoke_mode:
            t0 = _time.time()
        try:
            val_ha = _evaluate_split_full(current_prompt, cfg_dict, split_name="validation")
        except RuntimeError as e:
            if _is_api_error(e):
                _exit_on_api_error(
                    e, results_dir, al_iter,
                    results_dir / f"al_iter_{al_iter}" / "best_prompt.txt",
                    n_al_iterations, n_evolve_iterations,
                )
            raise
        if smoke_mode:
            dur = _time.time() - t0
            st = f"cycle_{al_iter}_validation"
            print(f"  [smoke] {st}: {dur:.1f}s")
            smoke_timings.append((st, round(dur, 2)))
            diag.log("smoke_timing", stage=st, duration_s=round(dur, 2), al_iter=al_iter)
        diag.log("validation", al_iter=al_iter,
                 **{k: v for k, v in val_ha.items() if isinstance(v, (int, float))})

        # Текущая оценка промпта на validation (для трекинга лучшего по val)
        seed_val_score = float(val_ha.get("combined_score", 0.0))
        last_cycle_seed_val_score = seed_val_score

        # --- Consolidation (Level 2) — accept only if not much worse than best ---
        consolidated = False
        cons_top = _load_top_programs_from_db(db_dir, n=3)
        if len(cons_top) >= 2:
            print(f"  Consolidation: analyzing top {len(cons_top)} prompts...")
            diag.log("consolidation_start", al_iter=al_iter, n_top=len(cons_top),
                     top_scores=[p["metrics"].get("combined_score") for p in cons_top])
            if smoke_mode:
                t0_cons = _time.time()

            best_artifacts = cons_top[0].get("artifacts", None)
            metrics_summary = {
                "val_combined": float(val_ha.get("combined_score", 0.0)),
                "val_R_global": float(val_ha.get("R_global", 0.0)),
                "val_R_worst": float(val_ha.get("R_worst", 0.0)),
            }
            cons_prompt = _consolidate_prompt(
                cons_top,
                al_iter,
                cfg_dict,
                max_retries=2,
                error_artifacts=best_artifacts,
                metrics_summary=metrics_summary,
            )
            cons_val = None
            if cons_prompt is not None:
                (out_dir / "consolidated_prompt.txt").write_text(cons_prompt, encoding="utf-8")

                print("  Evaluating consolidated prompt on validation split...")
                try:
                    cons_val = _evaluate_split_full(cons_prompt, cfg_dict, split_name="validation")
                except RuntimeError as e:
                    if _is_api_error(e):
                        _exit_on_api_error(
                            e, results_dir, al_iter,
                            results_dir / f"al_iter_{al_iter}" / "best_prompt.txt",
                            n_al_iterations, n_evolve_iterations,
                        )
                    raise
                if smoke_mode:
                    dur = _time.time() - t0_cons
                    st = f"cycle_{al_iter}_consolidation"
                    print(f"  [smoke] {st}: {dur:.1f}s")
                    smoke_timings.append((st, round(dur, 2)))
                    diag.log("smoke_timing", stage=st, duration_s=round(dur, 2), al_iter=al_iter)

                if cons_val is not None:
                    cons_score = float(cons_val.get("combined_score", 0.0))
                    evo_val_score = float(val_ha.get("combined_score", 0.0))
                    print(
                        "  Consolidated validation score: "
                        f"{cons_score:.4f}  (evolution val: {evo_val_score:.4f})"
                    )
                    delta = evo_val_score - cons_score
                    diag.log(
                        "consolidation_eval",
                        al_iter=al_iter,
                        cons_score=cons_score,
                        evo_val_score=evo_val_score,
                        delta=delta,
                    )

                    # Gate 1: vs evolution validation this cycle
                    reject_vs_evo = cons_score + consolidation_gate_delta < evo_val_score
                    # Gate 2: vs best val we know (previous cycles ∪ this cycle's evolution val)
                    ref_best_val = max(best_val_score, evo_val_score)
                    reject_vs_best = ref_best_val > 0 and (
                        cons_score + consolidation_gate_vs_best < ref_best_val
                    )
                    if reject_vs_evo:
                        print(
                            f"  Consolidated prompt REJECTED (vs evolution: Δ_val > {consolidation_gate_delta}, "
                            "keep evolution best)."
                        )
                        diag.log(
                            "consolidation_decision",
                            al_iter=al_iter,
                            accepted=False,
                            cons_score=cons_score,
                            evo_score=best_evolution_score,
                            delta=delta,
                            reason="vs_evolution",
                        )
                    elif reject_vs_best:
                        print(
                            f"  Consolidated prompt REJECTED (vs ref best val: "
                            f"cons {cons_score:.4f} < ref_best {ref_best_val:.4f} − {consolidation_gate_vs_best})."
                        )
                        diag.log(
                            "consolidation_decision",
                            al_iter=al_iter,
                            accepted=False,
                            cons_score=cons_score,
                            evo_score=best_evolution_score,
                            delta=delta,
                            reason="vs_ref_best",
                            ref_best_val=ref_best_val,
                        )
                    else:
                        current_prompt = cons_prompt
                        consolidated = True
                        seed_val_score = cons_score
                        print(
                            f"  Consolidated prompt ACCEPTED (Δ_val ≤ {consolidation_gate_delta} "
                            f"and vs global best OK)."
                        )
                        diag.log(
                            "consolidation_decision",
                            al_iter=al_iter,
                            accepted=True,
                            cons_score=cons_score,
                            evo_score=best_evolution_score,
                            delta=delta,
                        )
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

        # Обновляем глобально лучший по validation промпт (для анализа и финального теста)
        if seed_val_score > best_val_score:
            best_val_score = seed_val_score
            best_val_prompt = current_prompt
            best_val_cycle = al_iter
            (results_dir / "best_val_prompt.txt").write_text(best_val_prompt, encoding="utf-8")

        current_mutation_log = None
        if mutation_logs_snapshot:
            current_mutation_log = mutation_logs_snapshot[0].get("mutation_log")
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
            "seed_val_score": seed_val_score,
            "batch_actual_size": len(batch_indices),
            "prompt_tokens": len(current_prompt) // 4,
            "mutation_log_present": bool(current_mutation_log),
            "mutation_log_chars": len(current_mutation_log) if current_mutation_log else 0,
        }
        if current_mutation_log:
            (out_dir / "best_prompt_mutation_log.txt").write_text(
                current_mutation_log,
                encoding="utf-8",
            )

        # =================================================================
        # Periodic Seen-pool re-evaluation (every N cycles)
        # =================================================================
        seen_reeval_interval = al_cfg.get("seen_reeval_interval", 3)
        if seen_reeval_interval > 0 and (al_iter + 1) % seen_reeval_interval == 0:
            seen_list = sorted(data_manager.seen_indices)
            if seen_list:
                print(f"  [SEEN RE-EVAL] Re-evaluating all {len(seen_list)} Seen examples...")
                try:
                    seen_results = _evaluate_batch(
                        current_prompt, seen_list, data_manager, cfg_dict
                    )
                    old_hard, old_anchor = data_manager.n_hard, data_manager.n_anchor
                    data_manager.reclassify_seen(
                        seen_results["predictions"],
                        seen_results["gold_labels"],
                        seen_results["worker_predictions"],
                        seen_list,
                    )
                    print(f"  [SEEN RE-EVAL] Done. Hard: {old_hard}->{data_manager.n_hard}, "
                          f"Anchor: {old_anchor}->{data_manager.n_anchor}")
                    diag.log("seen_reeval", al_iter=al_iter,
                             n_seen=len(seen_list),
                             hard_before=old_hard, hard_after=data_manager.n_hard,
                             anchor_before=old_anchor, anchor_after=data_manager.n_anchor)
                    entry["seen_reeval"] = {
                        "n_seen": len(seen_list),
                        "hard_before": old_hard, "hard_after": data_manager.n_hard,
                        "anchor_before": old_anchor, "anchor_after": data_manager.n_anchor,
                    }
                except RuntimeError as e:
                    if _is_api_error(e):
                        _exit_on_api_error(
                            e, results_dir, al_iter,
                            results_dir / f"al_iter_{al_iter}" / "best_prompt.txt",
                            n_al_iterations, n_evolve_iterations,
                        )
                    raise

        # =================================================================
        # Expansion / refresh check
        # =================================================================
        # Use val_combined_score for soft expansion (stable across cycles)
        val_scores_history.append(seed_val_score)

        soft_patience = al_cfg.get("soft_expansion_patience", 5)
        soft_delta = al_cfg.get("soft_expansion_min_delta", 0.02)

        needs_soft_expansion = False
        if len(val_scores_history) >= soft_patience:
            window_scores = val_scores_history[-soft_patience:]
            window_max = max(window_scores)
            window_start = window_scores[0]
            if (window_max - window_start) < soft_delta:
                needs_soft_expansion = True
                print(f"  Soft expansion condition met: val improvement {window_max - window_start:.4f} < {soft_delta} over {soft_patience} cycles.")

        needs_hard_expansion = data_manager.needs_expansion(threshold=expansion_trigger)

        # After best_val update, best_val_score is the global max including this cycle's seed_val_score.
        at_or_near_best_val = seed_val_score >= (best_val_score - soft_skip_near_best)
        if needs_soft_expansion and not needs_hard_expansion and at_or_near_best_val:
            print(
                f"  Soft expansion SKIPPED: seed_val {seed_val_score:.4f} at/near best val "
                f"{best_val_score:.4f} (threshold ±{soft_skip_near_best})."
            )
            diag.log(
                "soft_expansion_skipped_near_best",
                al_iter=al_iter,
                seed_val_score=seed_val_score,
                best_val_score=best_val_score,
            )
            needs_soft_expansion = False

        if (needs_hard_expansion or needs_soft_expansion) and data_manager.n_unseen > 0:
            if needs_hard_expansion:
                reason = f"Hard={data_manager.n_hard} <= {expansion_trigger}"
            else:
                reason = "Stagnation detected (val_combined_score plateau)"

            n_hard_needed = int(batch_size * hard_ratio) - data_manager.n_hard
            min_add = expansion_trigger if needs_hard_expansion else 15
            n_hard_needed = max(n_hard_needed, min_add)

            print(f"  [POOL EXPANSION] Triggered. Reason: {reason}. Selecting {n_hard_needed} from Unseen...")
            new_indices = data_manager.expand_pool(n_new=n_hard_needed, seed=42 + al_iter)

            # Evaluate expanded examples before classifying as Hard/Anchor
            if new_indices:
                print(f"  [POOL EXPANSION] Evaluating {len(new_indices)} new examples...")
                try:
                    exp_results = _evaluate_batch(
                        current_prompt, new_indices, data_manager, cfg_dict
                    )
                    exp_hard, exp_anchor = [], []
                    uncertainty_threshold = al_cfg.get("uncertainty_threshold", 0.0)
                    for j, idx in enumerate(new_indices):
                        pred = exp_results["predictions"][j]
                        gold = exp_results["gold_labels"][j]
                        wp = [int(w[j]) for w in exp_results["worker_predictions"]]
                        correct = pred == gold
                        d_score = disagreement_score(wp, rating_min=1, rating_max=5)
                        if (not correct) or (d_score > uncertainty_threshold):
                            exp_hard.append(idx)
                        else:
                            exp_anchor.append(idx)
                    data_manager.hard_indices.extend(exp_hard)
                    data_manager.anchor_indices.extend(exp_anchor)
                    print(f"  [POOL EXPANSION] Classified: {len(exp_hard)} Hard, {len(exp_anchor)} Anchor")
                except RuntimeError as e:
                    if _is_api_error(e):
                        _exit_on_api_error(
                            e, results_dir, al_iter,
                            results_dir / f"al_iter_{al_iter}" / "best_prompt.txt",
                            n_al_iterations, n_evolve_iterations,
                        )
                    # Fallback: add all as Hard if evaluation fails
                    data_manager.hard_indices.extend(new_indices)
                    exp_hard = new_indices
                    exp_anchor = []
                    print(f"  [POOL EXPANSION] Eval failed, added all {len(new_indices)} as Hard (fallback)")

            print(f"  [POOL EXPANSION] Done. Hard: {data_manager.n_hard}, Anchor: {data_manager.n_anchor}, Unseen: {data_manager.n_unseen}")
            entry["expanded"] = True
            entry["n_expanded"] = len(new_indices)
            entry["n_expanded_hard"] = len(exp_hard)
            entry["n_expanded_anchor"] = len(exp_anchor)
            entry["expansion_reason"] = reason
            entry["expansion_event"] = "pool_expansion"
            diag.log(
                "expansion",
                event_description="pool_expansion",
                reason=reason,
                al_iter=al_iter,
                added=len(new_indices),
                added_hard=len(exp_hard),
                added_anchor=len(exp_anchor),
                hard=data_manager.n_hard,
                anchor=data_manager.n_anchor,
                unseen=data_manager.n_unseen,
            )

        # Pool refresh: promote Unseen → Seen, then label Hard vs Anchor (same logic as expansion)
        if refresh_per_cycle > 0 and data_manager.n_unseen > 0:
            n_refresh = min(refresh_per_cycle, data_manager.n_unseen)
            print(f"  [POOL REFRESH] Adding {n_refresh} Unseen examples (will classify Hard/Anchor)...")
            refresh_indices = data_manager.expand_pool(n_new=n_refresh, seed=42 + al_iter + 1000)
            refresh_hard, refresh_anchor = [], []
            if refresh_indices:
                try:
                    print(f"  [POOL REFRESH] Evaluating {len(refresh_indices)} new examples...")
                    refresh_results = _evaluate_batch(
                        current_prompt, refresh_indices, data_manager, cfg_dict
                    )
                    uncertainty_threshold = al_cfg.get("uncertainty_threshold", 0.0)
                    for j, idx in enumerate(refresh_indices):
                        pred = refresh_results["predictions"][j]
                        gold = refresh_results["gold_labels"][j]
                        wp = [int(w[j]) for w in refresh_results["worker_predictions"]]
                        correct = pred == gold
                        d_score = disagreement_score(wp, rating_min=1, rating_max=5)
                        if (not correct) or (d_score > uncertainty_threshold):
                            refresh_hard.append(idx)
                        else:
                            refresh_anchor.append(idx)
                    data_manager.hard_indices.extend(refresh_hard)
                    data_manager.anchor_indices.extend(refresh_anchor)
                    print(
                        f"  [POOL REFRESH] Classified: {len(refresh_hard)} Hard, "
                        f"{len(refresh_anchor)} Anchor"
                    )
                except RuntimeError as e:
                    if _is_api_error(e):
                        _exit_on_api_error(
                            e,
                            results_dir,
                            al_iter,
                            results_dir / f"al_iter_{al_iter}" / "best_prompt.txt",
                            n_al_iterations,
                            n_evolve_iterations,
                        )
                    # Non-API failure: same fallback as pool expansion
                    data_manager.hard_indices.extend(refresh_indices)
                    refresh_hard = list(refresh_indices)
                    refresh_anchor = []
                    print(
                        f"  [POOL REFRESH] Eval failed, added all {len(refresh_indices)} as Hard (fallback)"
                    )

                entry["refresh_event"] = {
                    "added": len(refresh_indices),
                    "added_hard": len(refresh_hard),
                    "added_anchor": len(refresh_anchor),
                    "hard": data_manager.n_hard,
                    "anchor": data_manager.n_anchor,
                    "unseen": data_manager.n_unseen,
                }
                diag.log(
                    "refresh",
                    event_description="cycle_refresh",
                    al_iter=al_iter,
                    added=len(refresh_indices),
                    added_hard=len(refresh_hard),
                    added_anchor=len(refresh_anchor),
                    hard=data_manager.n_hard,
                    anchor=data_manager.n_anchor,
                    unseen=data_manager.n_unseen,
                )
                print(
                    f"  [POOL REFRESH] Done. Hard: {data_manager.n_hard}, "
                    f"Anchor: {data_manager.n_anchor}, Unseen: {data_manager.n_unseen}"
                )

        # Per-cycle test: diagnostics / visualize.py only (no training decisions use these metrics).
        print("  Test evaluation (per-cycle, for plots — not used in AL decisions)...")
        if smoke_mode:
            t0_test = _time.time()
        try:
            test_metrics = _evaluate_split_full(current_prompt, cfg_dict, split_name="test")
        except RuntimeError as e:
            if _is_api_error(e):
                _exit_on_api_error(
                    e, results_dir, al_iter,
                    results_dir / f"al_iter_{al_iter}" / "best_prompt.txt",
                    n_al_iterations, n_evolve_iterations,
                )
            raise
        if smoke_mode:
            dur = _time.time() - t0_test
            st = f"cycle_{al_iter}_test"
            print(f"  [smoke] {st}: {dur:.1f}s")
            smoke_timings.append((st, round(dur, 2)))
            diag.log("smoke_timing", stage=st, duration_s=round(dur, 2), al_iter=al_iter)
        entry["test_R_global"] = test_metrics["R_global"]
        entry["test_R_worst"] = test_metrics["R_worst"]
        entry["test_combined_score"] = test_metrics["combined_score"]
        entry["test_Acc_Hard"] = test_metrics["Acc_Hard"]
        entry["test_Acc_Anchor"] = test_metrics["Acc_Anchor"]
        entry["test_mae"] = test_metrics["mae"]
        entry["test_mean_kappa"] = test_metrics["mean_kappa"]
        entry["test_n_hard"] = test_metrics["n_hard"]
        entry["test_n_anchor"] = test_metrics["n_anchor"]
        val_test_gap = float(seed_val_score - test_metrics["combined_score"])
        entry["val_test_gap"] = val_test_gap
        if val_test_gap >= gap_warn_threshold:
            print(
                f"  [warn] Val-Test combined gap={val_test_gap:+.4f} "
                f"(val={seed_val_score:.4f}, test={test_metrics['combined_score']:.4f})"
            )
        if gap_stop_patience > 0 and val_test_gap >= gap_stop_threshold:
            gap_stop_bad_streak += 1
        else:
            gap_stop_bad_streak = 0
        print(f"  Test: R_global={test_metrics['R_global']:.2%}  R_worst={test_metrics['R_worst']:.2%}  "
              f"combined={test_metrics['combined_score']:.4f}  Acc_Hard={test_metrics['Acc_Hard']:.2%}")
        diag.log("cycle_test", al_iter=al_iter, **{k: v for k, v in test_metrics.items() if isinstance(v, (int, float))})
        diag.log(
            "generalization_gap",
            al_iter=al_iter,
            val_combined=seed_val_score,
            test_combined=test_metrics["combined_score"],
            gap=val_test_gap,
            warn_threshold=gap_warn_threshold,
            stop_threshold=gap_stop_threshold,
            stop_streak=gap_stop_bad_streak,
        )

        cycle_time = _time.time() - t_cycle_start
        print(f"  Val: R_global={val_ha['R_global']:.2%}  Acc_Hard={val_ha['Acc_Hard']:.2%}  "
              f"Acc_Anchor={val_ha['Acc_Anchor']:.2%}  (Hard:{val_ha['n_hard']}/Anchor:{val_ha['n_anchor']})  "
              f"[{cycle_time:.0f}s]")
        entry["cycle_time_s"] = round(cycle_time, 1)
        entry["best_val_cycle"] = best_val_cycle
        entry["global_best_val_score"] = best_val_score
        log_entries.append(entry)

        log_path = results_dir / "active_loop_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_entries, f, indent=2)

        if data_manager.n_unseen == 0 and data_manager.n_hard <= expansion_trigger:
            print(f"\n  Pool exhausted and Hard={data_manager.n_hard}. Stopping.")
            break

        if (
            gap_stop_patience > 0
            and gap_stop_bad_streak >= gap_stop_patience
            and (al_iter + 1) >= al_stop_min_cycles
        ):
            log_entries[-1]["generalization_gap_early_stopped"] = True
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(log_entries, f, indent=2)
            print(
                f"\n  AL early stopping (generalization gap): gap >= {gap_stop_threshold:.4f} "
                f"for {gap_stop_bad_streak} consecutive cycle(s)."
            )
            diag.log(
                "al_gap_early_stop",
                al_iter=al_iter,
                gap=val_test_gap,
                gap_stop_threshold=gap_stop_threshold,
                gap_stop_patience=gap_stop_patience,
                stop_streak=gap_stop_bad_streak,
            )
            break

        if (
            al_stop_patience > 0
            and best_val_cycle >= 0
            and (al_iter - best_val_cycle) >= al_stop_patience
            and (al_iter + 1) >= al_stop_min_cycles
        ):
            log_entries[-1]["al_early_stopped"] = True
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(log_entries, f, indent=2)
            print(
                f"\n  AL early stopping: no new best val for {al_stop_patience} cycle(s) "
                f"(best at AL cycle index {best_val_cycle}, patience={al_stop_patience})."
            )
            diag.log(
                "al_early_stop",
                al_iter=al_iter,
                best_val_cycle=best_val_cycle,
                patience=al_stop_patience,
                best_val_score=best_val_score,
            )
            break

    # Cleanup
    if active_batch_path.exists():
        active_batch_path.unlink()

    # Single holdout test evaluation (train/val used during AL; test only here for reporting).
    # Use global best-by-val prompt so a weak last AL cycle does not define the reported test score.
    print("\n" + "=" * 60)
    print("Final test evaluation (holdout; best_val_prompt — best seed_val_score on validation)")
    print("=" * 60)
    print(
        f"  best_val_score={best_val_score:.4f}  (AL cycle index {best_val_cycle}; "
        f"last_cycle len={len(current_prompt)} chars, best_val len={len(best_val_prompt)} chars)"
    )
    if smoke_mode:
        t0 = _time.time()
    try:
        final_test_metrics = _evaluate_split_full(best_val_prompt, cfg_dict, split_name="test")
    except RuntimeError as e:
        if _is_api_error(e):
            last_prompt = results_dir / "final_prompt.txt"
            if not last_prompt.exists():
                last_prompt = results_dir / f"al_iter_{n_al_iterations - 1}" / "best_prompt.txt"
            _exit_on_api_error(
                e, results_dir, n_al_iterations - 1,
                last_prompt if last_prompt.exists() else None,
                n_al_iterations, n_evolve_iterations,
            )
        raise
    if smoke_mode:
        dur = _time.time() - t0
        print(f"  [smoke] final_test: {dur:.1f}s")
        smoke_timings.append(("final_test", round(dur, 2)))
        diag.log("smoke_timing", stage="final_test", duration_s=round(dur, 2))
    with open(results_dir / "final_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_test_metrics, f, indent=2)
    (results_dir / "final_prompt.txt").write_text(best_val_prompt, encoding="utf-8")
    (results_dir / "last_cycle_prompt.txt").write_text(current_prompt, encoding="utf-8")
    with open(results_dir / "best_val_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_val_score": best_val_score,
                "best_val_cycle": best_val_cycle,
                "last_cycle_chars": len(current_prompt),
                "best_val_chars": len(best_val_prompt),
            },
            f,
            indent=2,
        )

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

    diag.log(
        "final_test",
        **{k: v for k, v in final_test_metrics.items() if isinstance(v, (int, float))},
        delta_R_global=delta_r,
        delta_combined=delta_comb,
        prompt_source="best_val_prompt",
        best_val_cycle=best_val_cycle,
        best_val_score=best_val_score,
    )

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
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (default: config.yaml next to active_loop.py).",
    )
    parser.add_argument(
        "--no-evolve-early-stop",
        action="store_true",
        help="Disable OpenEvolve early stopping: run the full --n-evolve iterations each AL cycle.",
    )
    parser.add_argument(
        "--no-al-early-stop",
        action="store_true",
        help="Disable AL early stopping (sets al_early_stopping_patience to 0 for this run).",
    )
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke run: 2 AL cycles, 4 evolve iter, results_smoke/.")
    parser.add_argument(
        "--resume-from-dir",
        type=str,
        default=None,
        help=(
            "Continue or recover a previous run in this results dir. "
            "If active_loop_log.json exists, uses the last completed cycle's best_prompt "
            "as the seed and appends new AL cycles."
        ),
    )
    args = parser.parse_args()

    resume_start_cycle = None
    initial_prompt_path = Path(args.prompt) if args.prompt else None
    out_dir = None

    if args.resume_from_dir:
        results_dir = Path(args.resume_from_dir).resolve()
        if not results_dir.is_dir():
            raise SystemExit(f"Not a directory: {results_dir}")
        out_dir = results_dir
        log_path = results_dir / "active_loop_log.json"
        if log_path.exists():
            with open(log_path, "r", encoding="utf-8") as f:
                log_entries = json.load(f)
            if log_entries:
                # Расширяем существующий эксперимент: берём последний завершённый цикл как seed
                last_iter = max(e.get("al_iter", -1) for e in log_entries)
                prompt_path = results_dir / f"al_iter_{last_iter}" / "best_prompt.txt"
                if not prompt_path.exists():
                    raise SystemExit(
                        f"Resume seed prompt not found: {prompt_path}. "
                        "Run without --resume-from-dir first."
                    )
                initial_prompt_path = prompt_path
                resume_start_cycle = None  # al_start будет last_iter+1 внутри run_active_loop
                n_al, n_evolve = args.n_al, args.n_evolve
                print(
                    f"Resume (extend): last completed cycle {last_iter}, "
                    f"next cycle index will be {last_iter + 1}, seed {initial_prompt_path}"
                )
            else:
                # Лог пустой, но директория существует — ведём себя как свежий запуск
                resume_start_cycle = None
                n_al, n_evolve = args.n_al, args.n_evolve
        else:
            # Нет логов — свежий запуск в уже существующей директории
            resume_start_cycle = None
            n_al, n_evolve = args.n_al, args.n_evolve
    elif args.smoke:
        n_al, n_evolve = 2, 4
        out_dir = SCRIPT_DIR / "results_smoke"
        print("Smoke run: --n-al 2 --n-evolve 4 --results-dir results_smoke")
    else:
        n_al, n_evolve = args.n_al, args.n_evolve
        out_dir = Path(args.results_dir) if args.results_dir else None

    run_active_loop(
        n_al_iterations=n_al,
        n_evolve_iterations=n_evolve,
        initial_prompt_path=initial_prompt_path,
        results_dir=out_dir,
        smoke_mode=args.smoke,
        resume_start_cycle=resume_start_cycle,
        config_path=Path(args.config).resolve() if args.config else None,
        no_evolve_early_stop=args.no_evolve_early_stop,
        no_al_early_stop=args.no_al_early_stop,
    )


if __name__ == "__main__":
    main()

"""Thin OpenEvolve inner-loop wrapper and candidate evaluation.

SPEC v3 (Р6): fitness / selection decisions come exclusively from D_select
(heldout sources); the active batch only feeds mutator error artifacts.
Worker predictions are cached by (sha256(prompt), example_id) — workers run at
T=0, so repeated evaluations are free (SPEC v3 §4.3).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from prime.config import PrimeConfig
from prime.evolution.artifacts import (
    compose_mutator_artifacts,
    format_dselect_error_sample,
    format_error_artifacts,
    format_gba_dashboard,
)
from prime.evolution.prompt_blocks import strip_evolve_markers
from prime.evolution.qd_features import merge_qd_into_eval_metrics
from prime.fitness.objective import compute_fitness
from prime.workers.ensemble import (
    build_aggregator,
    build_workers,
    mock_predict,
    parallel_predict,
)

try:
    from openevolve.evaluation_result import EvaluationResult
except ImportError:  # pragma: no cover
    EvaluationResult = None  # type: ignore


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:24]


class PredictionCache:
    """File-backed cache: {cache_dir}/{prompt_hash}.json -> {example_id: [worker votes]}."""

    def __init__(self, cache_dir: Optional[Path]) -> None:
        self.cache_dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, phash: str) -> Optional[Path]:
        return None if self.cache_dir is None else self.cache_dir / f"{phash}.json"

    def load(self, phash: str) -> Dict[int, List[int]]:
        path = self._path(phash)
        if path is None or not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return {int(k): [int(x) for x in v] for k, v in data.items()}
        except (json.JSONDecodeError, OSError):
            return {}

    def store(self, phash: str, votes: Dict[int, List[int]]) -> None:
        path = self._path(phash)
        if path is None:
            return
        merged = self.load(phash)
        merged.update(votes)
        path.write_text(
            json.dumps({str(k): v for k, v in merged.items()}), encoding="utf-8"
        )


class CandidateEvaluator:
    """
    Evaluate prompt candidates.

    - `select_data` (D_select arrays) -> unbiased fitness (SPEC v3 §4.3);
    - `active_batch` over pool arrays -> mutator error artifacts only.
    Fallback: without select_data, evaluates on batch/pool (legacy v2 path,
    kept for smoke until Phase 2 completes the rewiring).
    """

    def __init__(
        self,
        cfg: PrimeConfig,
        pool_texts: List[str],
        pool_labels: List[int],
        pool_user_ids: List[int],
        pool_cluster_ids: Optional[List[int]] = None,
        active_batch: Optional[Dict[str, List[int]]] = None,
        use_mock: bool = False,
        actual_n_clusters: Optional[int] = None,
        select_data: Optional[Dict[str, list]] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.cfg = cfg
        self.pool_texts = pool_texts
        self.pool_labels = pool_labels
        self.pool_user_ids = pool_user_ids
        self.pool_cluster_ids = pool_cluster_ids
        self.active_batch = active_batch
        self.use_mock = use_mock
        self.actual_n_clusters = (
            int(actual_n_clusters)
            if actual_n_clusters is not None
            else int(cfg.clusters.n_clusters)
        )
        self.select_data = select_data
        self.cache = PredictionCache(cache_dir)
        self.workers = build_workers(cfg.ensemble)

    def evaluate_prompt(self, prompt_template: str) -> Dict[str, Any]:
        """Route fitness eval by mode.

        - ``v1_weighted``: Hard/Anchor batch (authentic v1 objective; needs hard_mask).
        - otherwise with data roles: D_select (SPEC v3 Р6).
        - fallback: active batch / full train.
        """
        from prime.evolution.prompt_contract import check_prompt_contract
        from prime.fitness.objective import REJECT_FITNESS

        contract = check_prompt_contract(
            prompt_template,
            label_space=self.cfg.dataset.label_space,
            prompt_len_limit=self.cfg.fitness.prompt_len_limit,
        )
        if not contract.ok:
            return {
                "fitness": REJECT_FITNESS,
                "combined_score": REJECT_FITNESS,
                "base_score": REJECT_FITNESS,
                "length_penalty": 0.0,
                "reject_reason": f"contract:{contract.reason}",
                "R_global": 0.0,
                "n_eval_examples": 0,
            }

        use_v1_batch = (
            self.cfg.fitness.mode == "v1_weighted"
            and self.active_batch
            and self.active_batch.get("indices")
        )
        if use_v1_batch:
            return self._evaluate_active(prompt_template)
        if self.select_data and self.select_data.get("texts"):
            return self.evaluate_select(prompt_template)
        if self.active_batch and self.active_batch.get("indices"):
            return self._evaluate_active(prompt_template)
        return self._evaluate_full_train(prompt_template)

    # ---------- D_select (v3) ----------

    def evaluate_select(self, prompt_template: str) -> Dict[str, Any]:
        data = self.select_data or {}
        texts: List[str] = data["texts"]
        labels: List[int] = data["labels"]
        example_ids: List[int] = data.get("example_ids", list(range(len(texts))))

        ensemble, wp = self._predict_cached(prompt_template, texts, labels, example_ids)

        result = compute_fitness(
            np.array(ensemble),
            np.array(labels),
            np.array(data["user_ids"]),
            prompt_template,
            self.cfg.fitness,
            worker_predictions=[np.array(w) for w in wp],
            cluster_ids=np.array(data["cluster_ids"]) if data.get("cluster_ids") else None,
        )
        result["eval_set"] = "d_select"
        result["n_eval_examples"] = len(texts)
        return result

    def evaluate_anchor(
        self, prompt_template: str, anchor_data: Dict[str, list]
    ) -> Dict[str, Any]:
        """Accuracy on D_anchor plus per-example correctness for the paired gate."""
        texts: List[str] = anchor_data["texts"]
        labels: List[int] = anchor_data["labels"]
        example_ids: List[int] = anchor_data.get("example_ids", list(range(len(texts))))
        if not texts:
            return {"anchor_accuracy": 1.0, "n_anchor": 0, "correct": [], "predictions": []}
        ensemble, _wp = self._predict_cached(prompt_template, texts, labels, example_ids)
        correct = [bool(int(p) == int(g)) for p, g in zip(ensemble, labels)]
        acc = float(np.mean(np.array(ensemble) == np.array(labels)))
        return {
            "anchor_accuracy": acc,
            "n_anchor": len(texts),
            "correct": correct,
            "predictions": [int(p) for p in ensemble],
        }

    def _predict_cached(
        self,
        prompt_template: str,
        texts: List[str],
        labels: List[int],
        example_ids: List[int],
    ) -> tuple[List[int], List[List[int]]]:
        """Per-worker votes via cache; aggregate according to config."""
        n_workers = len(self.workers) or len(self.cfg.ensemble.workers)
        phash = prompt_hash(prompt_template)
        cached = self.cache.load(phash)
        missing = [i for i, eid in enumerate(example_ids) if eid not in cached]

        if missing:
            m_texts = [texts[i] for i in missing]
            m_labels = [labels[i] for i in missing]
            if self.use_mock:
                _e, m_wp = mock_predict(
                    m_texts,
                    m_labels,
                    n_workers,
                    seed=self.cfg.active_learning.seed,
                    aggregation=self.cfg.ensemble.aggregation,
                    label_space=self.cfg.dataset.label_space,
                )
            else:
                _e, m_wp = parallel_predict(
                    self.workers,
                    m_texts,
                    prompt_template,
                    max_parallel=self.cfg.ensemble.max_parallel,
                    tie_break=self.cfg.ensemble.tie_break,
                    aggregation=self.cfg.ensemble.aggregation,
                    label_space=self.cfg.dataset.label_space,
                    fail_closed=bool(getattr(self.cfg.ensemble, "fail_closed", False)),
                )
            new_votes = {
                example_ids[i]: [int(m_wp[w][j]) for w in range(n_workers)]
                for j, i in enumerate(missing)
            }
            self.cache.store(phash, new_votes)
            cached.update(new_votes)

        from prime.workers.ensemble import INVALID, default_label

        fail_default = (
            INVALID
            if getattr(self.cfg.ensemble, "fail_closed", False)
            else default_label(self.cfg.dataset.label_space)
        )
        aggregator = build_aggregator(
            self.cfg.ensemble.aggregation,
            self.cfg.ensemble.tie_break,
            empty_default=fail_default,
        )
        wp: List[List[int]] = [[0] * len(texts) for _ in range(n_workers)]
        ensemble: List[int] = []
        for i, eid in enumerate(example_ids):
            votes = cached.get(eid) or [fail_default] * n_workers
            for w in range(n_workers):
                wp[w][i] = int(votes[w]) if w < len(votes) else fail_default
            ensemble.append(aggregator.aggregate([int(v) for v in votes]))
        return ensemble, wp

    # ---------- legacy batch paths (mutator artifacts / smoke fallback) ----------

    def _evaluate_active(self, prompt_template: str) -> Dict[str, Any]:
        indices = self.active_batch["indices"]  # type: ignore[index]
        hard_set = set(self.active_batch.get("hard_indices", []))
        texts = [self.pool_texts[i] for i in indices]
        labels = [self.pool_labels[i] for i in indices]
        user_ids = np.array([self.pool_user_ids[i] for i in indices])
        cluster_ids = (
            np.array([self.pool_cluster_ids[i] for i in indices])
            if self.pool_cluster_ids
            else None
        )
        # Namespace pool indices away from heldout example_ids in the shared cache
        # (both are small ints and would otherwise collide).
        cache_ids = [10_000_000 + int(i) for i in indices]

        if self.cache.cache_dir is not None:
            ensemble, wp = self._predict_cached(
                prompt_template, texts, labels, cache_ids
            )
        elif self.use_mock:
            ensemble, wp = mock_predict(
                texts,
                labels,
                len(self.workers) or len(self.cfg.ensemble.workers),
                self.cfg.active_learning.seed,
                aggregation=self.cfg.ensemble.aggregation,
                label_space=self.cfg.dataset.label_space,
            )
        else:
            ensemble, wp = parallel_predict(
                self.workers,
                texts,
                prompt_template,
                max_parallel=self.cfg.ensemble.max_parallel,
                tie_break=self.cfg.ensemble.tie_break,
                aggregation=self.cfg.ensemble.aggregation,
                label_space=self.cfg.dataset.label_space,
                fail_closed=bool(getattr(self.cfg.ensemble, "fail_closed", False)),
            )

        labels_arr = np.array(labels)
        hard_mask = np.array([idx in hard_set for idx in indices])
        result = compute_fitness(
            np.array(ensemble),
            labels_arr,
            user_ids,
            prompt_template,
            self.cfg.fitness,
            worker_predictions=[np.array(w) for w in wp],
            cluster_ids=cluster_ids,
            hard_mask=hard_mask,
        )
        result["indices"] = indices
        result["predictions"] = ensemble
        result["gold_labels"] = labels
        result["worker_predictions"] = wp
        result["eval_set"] = "active_batch_v1" if self.cfg.fitness.mode == "v1_weighted" else "active_batch"
        result["n_eval_examples"] = len(indices)
        result["n_hard"] = int(hard_mask.sum())
        result["n_anchor"] = int((~hard_mask).sum())
        return result

    def _evaluate_full_train(self, prompt_template: str) -> Dict[str, Any]:
        texts = self.pool_texts
        labels = np.array(self.pool_labels)
        user_ids = np.array(self.pool_user_ids)
        cluster_ids = np.array(self.pool_cluster_ids) if self.pool_cluster_ids else None
        if self.use_mock:
            ensemble, wp = mock_predict(
                texts,
                labels.tolist(),
                len(self.workers),
                self.cfg.active_learning.seed,
                aggregation=self.cfg.ensemble.aggregation,
                label_space=self.cfg.dataset.label_space,
            )
        else:
            ensemble, wp = parallel_predict(
                self.workers,
                texts,
                prompt_template,
                max_parallel=self.cfg.ensemble.max_parallel,
                tie_break=self.cfg.ensemble.tie_break,
                aggregation=self.cfg.ensemble.aggregation,
                label_space=self.cfg.dataset.label_space,
                fail_closed=bool(getattr(self.cfg.ensemble, "fail_closed", False)),
            )
        return compute_fitness(
            np.array(ensemble),
            labels,
            user_ids,
            prompt_template,
            self.cfg.fitness,
            worker_predictions=[np.array(w) for w in wp],
            cluster_ids=cluster_ids,
        )


def write_active_batch(path: Path, batch: Dict[str, List[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(batch, indent=2), encoding="utf-8")


def run_evolution(
    cfg: PrimeConfig,
    initial_prompt: str,
    output_dir: Path,
    config_snapshot_path: Path,
    active_batch_path: Path,
    use_mock: bool = False,
    cycle: int = 1,
    openevolve_config_path: Optional[Path] = None,
    seed_checkpoint: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run OpenEvolve inner loop when available; otherwise return initial prompt eval.
    In mock/smoke mode, simulates a mutation and writes error artifacts for validation.

    When `seed_checkpoint` points at a checkpoint written by
    `prime.consolidation.pareto.write_seed_checkpoint`, OpenEvolve resumes from
    that population instead of starting from a single program, so the previous
    cycle's Pareto front is available to the mutator as top / diverse inspiration
    (OBSERVATIONS O11). Any failure falls back to the plain fresh start.
    """
    os.environ["PRIME_CONFIG_PATH"] = str(config_snapshot_path.resolve())
    os.environ["PRIME_ACTIVE_BATCH_PATH"] = str(active_batch_path.resolve())
    os.environ["PRIME_USE_MOCK"] = "1" if use_mock else "0"

    (output_dir / "initial_prompt.txt").write_text(initial_prompt, encoding="utf-8")

    if use_mock:
        evaluator = _build_evaluator_from_env()
        result = evaluator.evaluate_prompt(initial_prompt)
        marker = f"\n<!-- PRIME_SMOKE_EVOLVED cycle={cycle} -->"
        mutated = initial_prompt if marker.strip() in initial_prompt else initial_prompt + marker
        best_path = output_dir / "best_prompt.txt"
        best_path.write_text(mutated, encoding="utf-8")

        artifacts_path = output_dir / "error_artifacts_evo.txt"
        if result.get("predictions") and evaluator.active_batch:
            indices = result.get("indices", [])
            texts = [evaluator.pool_texts[i] for i in indices]
            cluster_ids = (
                [evaluator.pool_cluster_ids[i] for i in indices]
                if evaluator.pool_cluster_ids
                else None
            )
            err_text = format_error_artifacts(
                result["predictions"],
                result["gold_labels"],
                result["worker_predictions"],
                texts,
                cluster_ids=cluster_ids,
                hard_indices=evaluator.active_batch.get("hard_indices"),
                anchor_indices=evaluator.active_batch.get("anchor_indices"),
                pool_indices=indices,
                label_space=cfg.dataset.label_space,
                contrastive_pairs=bool(getattr(cfg.evolution, "contrastive_pairs", True)),
                contrastive_pair_limit=int(
                    getattr(cfg.evolution, "contrastive_pair_limit", 4)
                ),
            )
            if err_text:
                artifacts_path.write_text(err_text, encoding="utf-8")

        info = {
            "best_score": result.get("fitness", 0.0),
            "mock": True,
            "cycle": cycle,
            "prompt_changed": mutated != initial_prompt,
        }
        (output_dir / "best_program_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        return {
            "best_prompt": mutated,
            "best_score": result.get("fitness", 0.0),
            "mock": True,
            "prompt_changed": mutated != initial_prompt,
            "artifacts_path": str(artifacts_path) if artifacts_path.is_file() else None,
        }

    try:
        from openevolve.api import run_evolution as oe_run
        from openevolve.config import load_config as oe_load_config
    except ImportError:
        evaluator = _build_evaluator_from_env()
        result = evaluator.evaluate_prompt(initial_prompt)
        return {"best_prompt": initial_prompt, "best_score": result.get("fitness", 0.0), "openevolve_missing": True}

    oe_cfg_path = openevolve_config_path or cfg.openevolve_config_path
    if not oe_cfg_path:
        evaluator = _build_evaluator_from_env()
        result = evaluator.evaluate_prompt(initial_prompt)
        best_path = output_dir / "best_prompt.txt"
        best_path.write_text(initial_prompt, encoding="utf-8")
        info = {
            "best_score": result.get("fitness", 0.0),
            "mode": "eval_only_stub",
            "note": "Set openevolve_config_path for real inner-loop mutations",
        }
        (output_dir / "best_program_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        return {
            "best_prompt": initial_prompt,
            "best_score": result.get("fitness", 0.0),
            "eval_only_stub": True,
            "prompt_changed": False,
        }

    oe_path = Path(oe_cfg_path)
    if not oe_path.is_absolute():
        oe_path = Path.cwd() / oe_path
    oe_config = oe_load_config(str(oe_path))
    # Keep OE loop length in sync with Prime evolution.n_evolve_iterations
    # (YAML max_iterations can drift; E1 configs set both to 5).
    n_iters = int(getattr(cfg.evolution, "n_evolve_iterations", 0) or 0)
    if n_iters > 0 and hasattr(oe_config, "max_iterations"):
        oe_config.max_iterations = n_iters
    # Experiment YAML mutator_* overrides the OE llm.models list (was previously
    # dead config — OE always used whatever was hard-coded in the OE yaml).
    mutator = str(getattr(cfg.evolution, "mutator_model", "") or "").strip()
    if mutator and getattr(oe_config, "llm", None) is not None:
        from openevolve.config import LLMModelConfig

        temp = float(getattr(cfg.evolution, "mutator_temperature", 0.6) or 0.6)
        max_tok = int(getattr(cfg.evolution, "mutator_max_tokens", 8192) or 8192)
        oe_config.llm.models = [LLMModelConfig(name=mutator, weight=1.0)]
        oe_config.llm.temperature = temp
        oe_config.llm.max_tokens = max_tok
        print(
            f"[openevolve] mutator override -> {mutator} (T={temp}, max_tokens={max_tok})",
            flush=True,
        )
    # Nested results/.../al_iter_N/openevolve_output/checkpoints/... exceeds
    # Windows MAX_PATH (260). Keep a short staging dir and mirror key artifacts.
    token = hashlib.sha1(str(output_dir.resolve()).encode("utf-8")).hexdigest()[:12]
    oe_staging = (Path.cwd() / "results" / "_oe" / token).resolve()
    oe_staging.mkdir(parents=True, exist_ok=True)
    oe_output = str(oe_staging)
    (output_dir / "openevolve_output_path.txt").write_text(oe_output + "\n", encoding="utf-8")
    # OpenEvolve api._prepare_evaluator expects a filesystem path or callable,
    # not a "module:function" string (that is treated as source code).
    evaluator_path = Path(__file__).resolve().parent / "evaluator_entry.py"

    seed_mode = "fresh"
    result = None
    if seed_checkpoint is not None and Path(seed_checkpoint).is_dir():
        try:
            result = _oe_run_from_checkpoint(
                initial_prompt=initial_prompt,
                evaluator_path=evaluator_path,
                oe_config=oe_config,
                output_dir=oe_output,
                checkpoint_path=Path(seed_checkpoint),
                iterations=n_iters,
            )
            seed_mode = "pareto_checkpoint"
        except Exception as exc:  # noqa: BLE001 - never lose a cycle to seeding
            print(
                f"[GRAPE WARNING] Pareto seed checkpoint failed ({exc}); "
                "falling back to fresh OpenEvolve start",
                flush=True,
            )
            result = None
            seed_mode = "fresh_after_seed_error"

    if result is None:
        result = oe_run(
            initial_program=initial_prompt,
            evaluator=str(evaluator_path),
            config=oe_config,
            output_dir=oe_output,
            cleanup=False,
        )

    best_prompt = initial_prompt
    best_score = 0.0
    if result is not None:
        best_score = float(getattr(result, "best_score", 0.0) or 0.0)
        code = getattr(result, "best_code", None) or ""
        prog = getattr(result, "best_program", None)
        if code:
            best_prompt = code
        elif prog is not None and getattr(prog, "code", None):
            best_prompt = prog.code
    # OE wraps marker-less programs in "# EVOLVE-BLOCK-START/END" and hands them
    # back inside best_code; unstripped they reach the rating workers verbatim.
    best_prompt = strip_evolve_markers(best_prompt)
    # Persist into the long cycle dir (best_prompt alone is well under MAX_PATH).
    (output_dir / "best_prompt.txt").write_text(best_prompt, encoding="utf-8")
    (output_dir / "best_program_info.json").write_text(
        json.dumps(
            {
                "best_score": best_score,
                "prompt_changed": best_prompt != initial_prompt,
                "openevolve_output": oe_output,
                "seed_mode": seed_mode,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    # Link cycle dir → short staging (avoids Windows MAX_PATH on nested checkpoints).
    link_mode = _attach_oe_output(oe_staging, output_dir)
    return {
        "best_prompt": best_prompt,
        "best_score": best_score,
        "prompt_changed": best_prompt != initial_prompt,
        "openevolve_output": oe_output,
        "openevolve_staging": str(oe_staging),
        "openevolve_link": link_mode,
        "seed_mode": seed_mode,
    }


def _oe_run_from_checkpoint(
    *,
    initial_prompt: str,
    evaluator_path: Path,
    oe_config: Any,
    output_dir: str,
    checkpoint_path: Path,
    iterations: int,
):
    """
    Drive OpenEvolve's controller directly so we can resume a seeded population.

    `openevolve.api.run_evolution` has no checkpoint parameter, but the controller
    does. The seed checkpoint is written with `last_iteration = 0`, so the resumed
    run spends its full `iterations` budget on new mutations.
    """
    import asyncio
    import tempfile

    from openevolve.controller import OpenEvolve

    code = initial_prompt
    if "EVOLVE-BLOCK-START" not in code:
        code = f"# EVOLVE-BLOCK-START\n{code}\n# EVOLVE-BLOCK-END"
    tmp = Path(tempfile.gettempdir()) / f"prime_seed_{os.getpid()}_{hashlib.sha1(code.encode()).hexdigest()[:8]}.txt"
    tmp.write_text(code, encoding="utf-8")

    controller = OpenEvolve(
        initial_program_path=str(tmp),
        evaluation_file=str(evaluator_path),
        config=oe_config,
        output_dir=output_dir,
    )
    best = asyncio.run(
        controller.run(iterations=iterations, checkpoint_path=str(checkpoint_path))
    )
    if best is None:
        raise RuntimeError("resumed OpenEvolve run returned no best program")

    class _Result:
        best_program = best
        best_code = getattr(best, "code", "") or ""
        best_score = float((getattr(best, "metrics", {}) or {}).get("combined_score", 0.0))

    return _Result()


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.exists():
        if path.is_dir() and not path.is_symlink():
            # Junctions on Windows often report as dirs; try unlink first.
            try:
                path.unlink()
                return
            except OSError:
                pass
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


def _attach_oe_output(oe_staging: Path, output_dir: Path) -> str:
    """
    Expose OE artifacts next to the AL cycle without copying deep trees.

    Nested ``al_iter_N/openevolve_output/checkpoints/.../programs/<uuid>.json``
    exceeds Windows MAX_PATH (~260). Staging under ``results/_oe/<token>/`` is
    short enough; we junction/symlink that directory into the cycle folder.
    """
    mirror = output_dir / "openevolve_output"
    _remove_path(mirror)

    # 1) Windows directory junction (no admin required).
    if os.name == "nt":
        try:
            completed = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(mirror), str(oe_staging)],
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode == 0 and mirror.exists():
                print(f"[GRAPE] OE output junction: {mirror} -> {oe_staging}", flush=True)
                return "junction"
            err = (completed.stderr or completed.stdout or "").strip()
            print(f"[GRAPE WARNING] mklink /J failed ({err}); trying symlink", flush=True)
        except OSError as exc:
            print(f"[GRAPE WARNING] mklink /J unavailable ({exc}); trying symlink", flush=True)

    # 2) Symlink (POSIX; Windows needs Developer Mode / admin).
    try:
        mirror.symlink_to(oe_staging, target_is_directory=True)
        print(f"[GRAPE] OE output symlink: {mirror} -> {oe_staging}", flush=True)
        return "symlink"
    except OSError as exc:
        print(f"[GRAPE WARNING] symlink failed ({exc}); shallow-copying best+logs", flush=True)

    # 3) Shallow copy of short paths only (best/ + logs/), never checkpoints/programs.
    mirror.mkdir(parents=True, exist_ok=True)
    copied = 0
    for sub in ("best", "logs"):
        src = oe_staging / sub
        if not src.is_dir():
            continue
        dst = mirror / sub
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.rglob("*"):
            if not f.is_file():
                continue
            rel = f.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
            copied += 1
    (mirror / "STAGED_AT.txt").write_text(str(oe_staging) + "\n", encoding="utf-8")
    print(
        f"[GRAPE] OE shallow copy ({copied} files); full tree at {oe_staging}",
        flush=True,
    )
    return "shallow_copy"


def _build_evaluator_from_env() -> CandidateEvaluator:
    cfg_path = Path(os.environ["PRIME_CONFIG_PATH"])
    from prime.config import load_config

    cfg = load_config(cfg_path)
    batch_path = Path(os.environ.get("PRIME_ACTIVE_BATCH_PATH", ""))
    active = json.loads(batch_path.read_text(encoding="utf-8")) if batch_path.is_file() else None
    use_mock = os.environ.get("PRIME_USE_MOCK", "0") == "1"
    # Pool arrays stored alongside batch by controller via env PRIME_POOL_PATH
    pool_path = Path(os.environ.get("PRIME_POOL_PATH", ""))
    pool = json.loads(pool_path.read_text(encoding="utf-8")) if pool_path.is_file() else {
        "texts": [], "labels": [], "user_ids": [], "cluster_ids": [],
    }
    select_path = Path(os.environ.get("PRIME_DSELECT_PATH", ""))
    select_data = (
        json.loads(select_path.read_text(encoding="utf-8")) if select_path.is_file() else None
    )
    cache_env = os.environ.get("PRIME_PRED_CACHE_DIR", "")
    cache_dir = Path(cache_env) if cache_env else None
    actual_k = os.environ.get("PRIME_N_CLUSTERS")
    return CandidateEvaluator(
        cfg,
        pool["texts"],
        pool["labels"],
        pool["user_ids"],
        pool.get("cluster_ids"),
        active_batch=active,
        use_mock=use_mock,
        actual_n_clusters=int(actual_k) if actual_k else None,
        select_data=select_data,
        cache_dir=cache_dir,
    )


def _qd_exclude_clusters(cfg: PrimeConfig) -> List[int]:
    """Omit forever-zero ``none`` axis when GBA excludes it (E5)."""
    if (
        getattr(cfg.fitness, "group_acc", "") == "balanced_within"
        and bool(getattr(cfg.fitness, "gba_exclude_none", True))
    ):
        return [0]
    return []


def _oracle_group_names() -> Dict[int, str]:
    try:
        from prime.data.civilcomments_loader import ORACLE_GROUP_NAMES

        return {i: name for i, name in enumerate(ORACLE_GROUP_NAMES)}
    except Exception:  # pragma: no cover
        return {}


def _load_frozen_artifacts() -> Optional[str]:
    path = Path(os.environ.get("PRIME_FROZEN_ARTIFACTS_PATH", ""))
    if path.is_file():
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return None
    return None


def _load_batch_baseline_preds(pool_indices: List[int]) -> Optional[List[int]]:
    """Map cycle-entry batch predictions onto the current batch row order."""
    path = Path(os.environ.get("PRIME_BATCH_BASELINE_PREDS_PATH", ""))
    if not path.is_file() or not pool_indices:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    by_pool = {int(k): int(v) for k, v in (data.get("by_pool_index") or {}).items()}
    return [by_pool.get(int(i), -1) for i in pool_indices]


def evaluate_for_openevolve(prompt_path: Optional[str] = None) -> Union[Dict[str, Any], Any]:
    """
    Entry point referenced by OpenEvolve evaluator string.

    SPEC v3 (Р6): fitness/QD metrics from D_select (when provided by controller);
    mutator error artifacts from the active batch (fit sources, adversarial view),
    always merged with the cycle's frozen pre-inject report when present.
    """
    prompt_file = Path(prompt_path) if prompt_path else Path("initial_prompt.txt")
    prompt_template = prompt_file.read_text(encoding="utf-8")
    evaluator = _build_evaluator_from_env()
    result = evaluator.evaluate_prompt(prompt_template)

    metrics = {
        "combined_score": result.get("fitness", 0.0),
        "fitness": result.get("fitness", 0.0),
        "R_global": result.get("R_global", 0.0),
        "R_worst": result.get("R_worst", 0.0),
        "R_worst_group": result.get("R_worst_group", 0.0),
        "R_worst_gba": result.get("R_worst_gba", 0.0),
        "R_gba_mean": result.get("R_gba_mean", 0.0),
        "R_soft_min_group": result.get("R_soft_min_group", 0.0),
        "R_soft_min_gba": result.get("R_soft_min_gba", 0.0),
        "toxic_recall": result.get("toxic_recall", 0.0),
        "specificity": result.get("specificity", 0.0),
        "invalid_rate": result.get("invalid_rate", 0.0),
        "pred_pos_rate": result.get("pred_pos_rate", 0.0),
        "CVaR_cluster": result.get("CVaR_cluster", 0.0),
        "CVaR_cluster_shrunk": result.get("CVaR_cluster_shrunk", 0.0),
        "CVaR_cluster_balanced_shrunk": result.get("CVaR_cluster_balanced_shrunk", 0.0),
        "R_macro": result.get("R_macro", 0.0),
        "mae": result.get("mae", 0.0),
        "mean_kappa": max(0.0, result.get("mean_kappa", 0.0)),  # diagnostic only
    }
    # QD descriptors: prefer within-group GBA when that is the objective axis.
    cluster_accs = (
        result.get("cluster_gba_shrunk")
        or result.get("cluster_gba")
        or result.get("cluster_accuracies_balanced_shrunk")
        or result.get("cluster_accuracies_shrunk")
        or result.get("cluster_accuracies")
    )
    exclude = _qd_exclude_clusters(evaluator.cfg)
    # Always emit configured cluster_acc_* so MAP-Elites never crashes on
    # contract/fail-closed rejects that omit per-group vectors.
    metrics = merge_qd_into_eval_metrics(
        metrics,
        {int(k): float(v) for k, v in (cluster_accs or {}).items()},
        prompt_template,
        n_clusters=evaluator.actual_n_clusters or 9,
        exclude_clusters=exclude,
    )

    group_names = _oracle_group_names() if evaluator.cfg.dataset.label_space == "binary" else None
    gba_src = result.get("cluster_gba_shrunk") or result.get("cluster_gba") or {}
    gba_dashboard = format_gba_dashboard(
        cluster_gba={int(k): float(v) for k, v in gba_src.items()} if gba_src else None,
        softmin=result.get("R_soft_min_gba") or result.get("R_soft_min_group"),
        worst_gba=result.get("R_worst_gba") or result.get("R_worst_group"),
        gba_mean=result.get("R_gba_mean"),
        toxic_recall=result.get("toxic_recall"),
        specificity=result.get("specificity"),
        pred_pos_rate=result.get("pred_pos_rate"),
        group_names=group_names,
        source="D_select",
    )

    d_select_sample = None
    if (
        evaluator.cfg.dataset.label_space == "binary"
        and result.get("predictions") is not None
        and result.get("gold_labels") is not None
        and evaluator.select_data
    ):
        texts = list(evaluator.select_data.get("texts") or [])
        preds = list(result["predictions"])
        golds = list(result["gold_labels"])
        if texts and len(texts) == len(preds):
            d_select_sample = format_dselect_error_sample(
                preds, golds, texts, limit=3, max_text_len=180
            )

    # Live mutator artifacts from the batch view (+ baseline damage).
    # Fitness usually comes from D_select, so always re-score the active batch here.
    live_text = None
    if evaluator.active_batch and evaluator.active_batch.get("indices"):
        batch_result = evaluator._evaluate_active(prompt_template)
        indices = list(batch_result.get("indices", []))
        texts = [evaluator.pool_texts[i] for i in indices]
        cluster_ids = (
            [evaluator.pool_cluster_ids[i] for i in indices]
            if evaluator.pool_cluster_ids
            else None
        )
        live_text = format_error_artifacts(
            batch_result["predictions"],
            batch_result["gold_labels"],
            batch_result["worker_predictions"],
            texts,
            cluster_ids=cluster_ids,
            hard_indices=evaluator.active_batch.get("hard_indices"),
            anchor_indices=evaluator.active_batch.get("anchor_indices"),
            pool_indices=indices,
            prev_predictions=_load_batch_baseline_preds(indices),
            label_space=evaluator.cfg.dataset.label_space,
            contrastive_pairs=bool(getattr(evaluator.cfg.evolution, "contrastive_pairs", True)),
            contrastive_pair_limit=int(
                getattr(evaluator.cfg.evolution, "contrastive_pair_limit", 4)
            ),
            group_names=group_names,
        )

    frozen_text = _load_frozen_artifacts()
    error_text = compose_mutator_artifacts(
        frozen_text=frozen_text,
        live_text=live_text,
        gba_dashboard=gba_dashboard or None,
        d_select_sample=d_select_sample,
    )
    if EvaluationResult is not None and error_text:
        return EvaluationResult(metrics=metrics, artifacts={"error_examples": error_text})
    return metrics

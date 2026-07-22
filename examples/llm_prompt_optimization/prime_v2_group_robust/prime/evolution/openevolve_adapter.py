"""Thin OpenEvolve inner-loop wrapper and candidate evaluation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from prime.config import PrimeConfig
from prime.evolution.artifacts import format_error_artifacts
from prime.evolution.qd_features import merge_qd_into_eval_metrics
from prime.fitness.objective import compute_fitness
from prime.workers.ensemble import build_workers, mock_predict, parallel_predict

try:
    from openevolve.evaluation_result import EvaluationResult
except ImportError:  # pragma: no cover
    EvaluationResult = None  # type: ignore


class CandidateEvaluator:
    """Evaluate prompt candidates on active batch or validation split."""

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
        self.workers = build_workers(cfg.ensemble)

    def evaluate_prompt(self, prompt_template: str) -> Dict[str, Any]:
        if self.active_batch and self.active_batch.get("indices"):
            return self._evaluate_active(prompt_template)
        return self._evaluate_full_train(prompt_template)

    def _run_predict(
        self,
        texts: List[str],
        labels: List[int],
    ) -> tuple[List[int], List[List[int]]]:
        if self.use_mock:
            return mock_predict(texts, labels, len(self.workers), seed=self.cfg.active_learning.seed)
        return parallel_predict(
            self.workers,
            texts,
            prompt_template="",
            max_parallel=self.cfg.ensemble.max_parallel,
            tie_break=self.cfg.ensemble.tie_break,
        )

    def _evaluate_active(self, prompt_template: str) -> Dict[str, Any]:
        indices = self.active_batch["indices"]  # type: ignore[index]
        hard_set = set(self.active_batch.get("hard_indices", []))
        texts = [self.pool_texts[i] for i in indices]
        labels = np.array([self.pool_labels[i] for i in indices])
        user_ids = np.array([self.pool_user_ids[i] for i in indices])
        cluster_ids = (
            np.array([self.pool_cluster_ids[i] for i in indices])
            if self.pool_cluster_ids
            else None
        )

        if self.use_mock:
            ensemble, wp = mock_predict(texts, labels.tolist(), len(self.workers), self.cfg.active_learning.seed)
        else:
            ensemble, wp = parallel_predict(
                self.workers,
                texts,
                prompt_template,
                max_parallel=self.cfg.ensemble.max_parallel,
                tie_break=self.cfg.ensemble.tie_break,
            )

        hard_mask = np.array([idx in hard_set for idx in indices])
        result = compute_fitness(
            np.array(ensemble),
            labels,
            user_ids,
            prompt_template,
            self.cfg.fitness,
            worker_predictions=[np.array(w) for w in wp],
            cluster_ids=cluster_ids,
            hard_mask=hard_mask,
        )
        result["indices"] = indices
        result["predictions"] = ensemble
        result["gold_labels"] = labels.tolist()
        result["worker_predictions"] = wp
        return result

    def _evaluate_full_train(self, prompt_template: str) -> Dict[str, Any]:
        texts = self.pool_texts
        labels = np.array(self.pool_labels)
        user_ids = np.array(self.pool_user_ids)
        cluster_ids = np.array(self.pool_cluster_ids) if self.pool_cluster_ids else None
        if self.use_mock:
            ensemble, wp = mock_predict(texts, labels.tolist(), len(self.workers), self.cfg.active_learning.seed)
        else:
            ensemble, wp = parallel_predict(
                self.workers,
                texts,
                prompt_template,
                max_parallel=self.cfg.ensemble.max_parallel,
                tie_break=self.cfg.ensemble.tie_break,
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
) -> Dict[str, Any]:
    """
    Run OpenEvolve inner loop when available; otherwise return initial prompt eval.
    In mock/smoke mode, simulates a mutation and writes error artifacts for validation.
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
    oe_output = str(output_dir / "openevolve_output")
    best = oe_run(
        initial_program=initial_prompt,
        evaluator="prime.evolution.evaluator_entry:evaluate",
        config=oe_config,
        output_dir=oe_output,
    )
    return {"best_prompt": best.get("best_program", initial_prompt), "best_score": best.get("best_score", 0.0)}


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
    )


def evaluate_for_openevolve(prompt_path: Optional[str] = None) -> Union[Dict[str, Any], Any]:
    """Entry point referenced by OpenEvolve evaluator string."""
    prompt_file = Path(prompt_path) if prompt_path else Path("initial_prompt.txt")
    prompt_template = prompt_file.read_text(encoding="utf-8")
    evaluator = _build_evaluator_from_env()
    result = evaluator.evaluate_prompt(prompt_template)

    metrics = {
        "combined_score": result.get("fitness", 0.0),
        "fitness": result.get("fitness", 0.0),
        "R_global": result.get("R_global", 0.0),
        "R_worst": result.get("R_worst", 0.0),
        "CVaR_cluster": result.get("CVaR_cluster", 0.0),
        "mae": result.get("mae", 0.0),
        "mean_kappa": max(0.0, result.get("mean_kappa", 0.0)),
    }
    cluster_accs = result.get("cluster_accuracies")
    if isinstance(cluster_accs, dict):
        metrics = merge_qd_into_eval_metrics(
            metrics,
            {int(k): float(v) for k, v in cluster_accs.items()},
            prompt_template,
            n_clusters=evaluator.actual_n_clusters,
        )

    if "predictions" in result and evaluator.active_batch:
        indices = result.get("indices", [])
        texts = [evaluator.pool_texts[i] for i in indices]
        cluster_ids = (
            [evaluator.pool_cluster_ids[i] for i in indices]
            if evaluator.pool_cluster_ids
            else None
        )
        error_text = format_error_artifacts(
            result["predictions"],
            result["gold_labels"],
            result["worker_predictions"],
            texts,
            cluster_ids=cluster_ids,
            hard_indices=evaluator.active_batch.get("hard_indices"),
            anchor_indices=evaluator.active_batch.get("anchor_indices"),
        )
        if EvaluationResult is not None and error_text:
            return EvaluationResult(metrics=metrics, artifacts={"error_examples": error_text})
    return metrics

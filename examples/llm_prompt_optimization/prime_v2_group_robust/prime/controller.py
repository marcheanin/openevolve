"""Thin orchestration for PRIME v2 active learning loop."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from prime.acquisition.batch_builder import build_active_batch
from prime.acquisition.expansion import expand_pool_disagreement
from prime.acquisition.pool import AcquisitionPool
from prime.config import PrimeConfig, load_config
from prime.consolidation.base_consolidator import run_base_consolidation
from prime.consolidation.pool_carryover import (
    PromptRecord,
    merge_with_consolidated,
    select_carryover,
)
from prime.data.clustering import ClusterArtifacts, assign_users_to_clusters, attach_example_clusters
from prime.data.wilds_loader import ReviewSplit, load_amazon_splits, subsample_split
from prime.evolution.artifacts import format_error_artifacts
from prime.evolution.openevolve_adapter import CandidateEvaluator, run_evolution, write_active_batch
from prime.evolution.openevolve_config_patch import patch_openevolve_feature_dimensions
from prime.evolution.prompt_blocks import extract_block
from prime.experiment.logging import JsonlLogger
from prime.experiment.proxy_validation import proxy_validation_report
from prime.experiment.run_context import RunContext
from prime.experiment.stage_trace import StageTracer
from prime.workers.ensemble import disagreement_score, mock_predict, parallel_predict


class PrimeController:
    """External AL loop: batch -> evolve -> reclassify -> consolidate -> select -> expand."""

    def __init__(self, cfg: PrimeConfig, project_root: Path, config_path: Path) -> None:
        self.cfg = cfg
        self.project_root = project_root
        self.config_path = config_path
        self.ctx = RunContext(cfg, config_path, project_root)
        self.logger = JsonlLogger(self.ctx.run_dir / "events.jsonl")
        self.trace = StageTracer(
            self.ctx.run_dir,
            enabled=cfg.experiment.smoke_validate,
            verbose=cfg.experiment.verbose or cfg.experiment.smoke_validate,
        )
        from prime.workers.ensemble import load_dotenv_if_present

        loaded_env = load_dotenv_if_present()
        self._loaded_env_path = loaded_env
        has_api = bool(os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"))
        self.use_mock = cfg.experiment.force_mock or not has_api
        self._best_selection_key: tuple = (-1.0, -1.0)
        self._selected_prompt: Optional[str] = None
        self._proxy_val_history: List[tuple[float, float]] = []
        self._actual_n_clusters: int = cfg.clusters.n_clusters
        self._openevolve_config_used: Optional[Path] = None
        self._last_cluster_accs: Dict[int, float] = {}

    def run(self) -> Dict[str, Any]:
        self.trace.stage(
            "01_config",
            "Config loaded",
            mode=self.cfg.fitness.mode,
            acquisition_policy=self.cfg.acquisition.policy,
            smoke=self.cfg.experiment.smoke,
            use_mock=self.use_mock,
            inference_backend="mock" if self.use_mock else "openrouter",
            openrouter_api_base=self.cfg.ensemble.api_base,
            n_workers=len(self.cfg.ensemble.workers),
            worker_models=[w.name for w in self.cfg.ensemble.workers],
            seed=self.cfg.active_learning.seed,
        )
        if not self.use_mock and self.cfg.experiment.verbose:
            print(
                "[LIVE] OpenRouter ensemble inference ENABLED "
                f"({len(self.cfg.ensemble.workers)} workers, parallel={self.cfg.ensemble.max_parallel})",
                flush=True,
            )
            if self._loaded_env_path:
                print(f"[ENV] API key loaded from {self._loaded_env_path}", flush=True)
            else:
                print("[ENV] API key from environment variable", flush=True)
        elif self.cfg.experiment.verbose and self.use_mock:
            print("[MOCK] No API key or force_mock=true — using deterministic mock ensemble", flush=True)

        snap = self.ctx.snapshot()
        meta_path = self.ctx.run_dir / "run_metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.trace.stage(
            "02_run_context",
            "Run context snapshot",
            config_snapshot=str(snap),
            git_hash=meta.get("git_hash"),
            run_dir=str(self.ctx.run_dir),
        )

        splits = self._load_data()
        train = splits["train"]
        val = splits["validation"]
        data_source = "wilds" if not getattr(self, "_used_synthetic", False) else "synthetic_fallback"
        self.trace.stage(
            "03_data_load",
            "Official OOD splits loaded",
            source=data_source,
            train_examples=len(train),
            train_users=len(set(train.user_ids)),
            val_examples=len(val),
            val_users=len(set(val.user_ids)),
            test_examples=len(splits["test"]),
            train_val_disjoint=train.user_disjoint_check(val),
            train_test_disjoint=train.user_disjoint_check(splits["test"]),
        )

        cluster_art = self._load_or_fit_clusters(train, val, splits["test"])
        train = attach_example_clusters(train, cluster_art.user_to_cluster)
        val = self._attach_split_clusters(val, cluster_art)
        test = self._attach_split_clusters(splits["test"], cluster_art)
        cluster_counts: Dict[int, int] = {}
        for cid in train.example_cluster_ids or []:
            cluster_counts[cid] = cluster_counts.get(cid, 0) + 1
        self.trace.stage(
            "04_clustering",
            "Style clusters fitted",
            n_clusters=cluster_art.n_clusters,
            n_train_users=len(cluster_art.user_to_cluster),
            embedding_model=cluster_art.embedding_model,
            cluster_example_counts=cluster_counts,
            train_projection_agreement=round(cluster_art.train_projection_agreement, 4),
            is_synthetic=cluster_art.is_synthetic,
            artifact_path=str(self.ctx.run_dir / "clusters.json"),
        )
        if cluster_art.is_synthetic:
            msg = (
                "[GRAPE WARNING] Cluster artifacts are SYNTHETIC — "
                "group-aware metrics are not meaningful for research conclusions."
            )
            print(msg, flush=True)
        elif cluster_art.train_projection_agreement < 0.5:
            print(
                f"[GRAPE WARNING] Low train projection agreement "
                f"({cluster_art.train_projection_agreement:.2f}): "
                "full vs label-free cluster assignment diverges on train users.",
                flush=True,
            )

        self._actual_n_clusters = int(cluster_art.n_clusters)
        os.environ["PRIME_N_CLUSTERS"] = str(self._actual_n_clusters)
        oe_src = None
        if self.cfg.openevolve_config_path:
            oe_src = Path(self.cfg.openevolve_config_path)
            if not oe_src.is_absolute():
                oe_src = self.project_root / oe_src
        self._openevolve_config_used = patch_openevolve_feature_dimensions(
            oe_src if oe_src and oe_src.is_file() else None,
            self.ctx.run_dir / "openevolve_config_used.yaml",
            self._actual_n_clusters,
        )
        self.trace.stage(
            "04b_qd_dims",
            "OpenEvolve QD feature_dimensions patched",
            actual_n_clusters=self._actual_n_clusters,
            openevolve_config=str(self._openevolve_config_used),
        )

        prompt = self._read_prompt()
        pool = AcquisitionPool(train, self.cfg.dataset, self.cfg.acquisition, self.cfg.active_learning.seed)
        initial_seen = pool.initialize_seen(self.cfg.acquisition.batch_size)
        self.trace.stage(
            "05_pool_init",
            "Acquisition pool initialized",
            pool_size=len(pool.pool_indices),
            seen=len(pool.seen),
            unseen=len(pool.unseen),
            initial_seen=len(initial_seen),
        )

        archive: List[PromptRecord] = []
        n_cycles = (
            self.cfg.experiment.smoke_n_cycles
            if self.cfg.experiment.smoke
            else self.cfg.active_learning.n_cycles
        )
        cycle_summaries: List[Dict[str, Any]] = []

        for cycle in range(1, n_cycles + 1):
            if cycle > 1 and self._selected_prompt:
                prompt = self._selected_prompt
                if self.cfg.experiment.verbose:
                    print(f"[CYCLE {cycle}] carryover prompt selected (len={len(prompt)})", flush=True)

            cycle_dir = self.ctx.cycle_dir(cycle)
            batch, eval_result, artifacts_path = self._cycle_step(pool, prompt, train, cycle_dir, cycle)

            archive.append(
                PromptRecord(
                    prompt=prompt,
                    fitness=float(eval_result.get("fitness", 0.0)),
                    metrics={
                        "R_global": float(eval_result.get("R_global", 0.0)),
                        "CVaR_cluster": float(eval_result.get("CVaR_cluster", 0.0)),
                    },
                    cluster_scores=eval_result.get("cluster_accuracies"),
                )
            )

            evo = run_evolution(
                self.cfg,
                prompt,
                cycle_dir,
                self.ctx.run_dir / "config_used.yaml",
                cycle_dir / "active_batch.json",
                use_mock=self.use_mock,
                cycle=cycle,
                openevolve_config_path=self._openevolve_config_used,
            )
            prompt = evo.get("best_prompt", prompt)
            (cycle_dir / "best_prompt.txt").write_text(prompt, encoding="utf-8")
            self.trace.stage(
                "10_evolution",
                f"Inner-loop evolution (cycle {cycle})",
                cycle=cycle,
                mock=evo.get("mock", False),
                best_score=evo.get("best_score"),
                prompt_changed=evo.get("prompt_changed", False),
                artifacts_file=evo.get("artifacts_path"),
            )

            carryover = select_carryover(
                archive,
                self.cfg.consolidation.pool_carryover_k,
                cluster_pareto=self.cfg.consolidation.cluster_pareto,
            )
            best_rec = PromptRecord(
                prompt=prompt,
                fitness=float(evo.get("best_score", 0.0)),
                metrics={"source": "evolved"},
                cluster_scores=eval_result.get("cluster_accuracies"),
            )
            gate_passed = False
            base_before = extract_block(prompt, "BaseGuidelines") or ""
            consolidated_rec = best_rec
            if (
                self.cfg.consolidation.enabled
                and self.cfg.consolidation.scope == "base_guidelines"
                and cycle % self.cfg.consolidation.every_n_cycles == 0
            ):
                candidate_prompt, consolidated_rec, _ = run_base_consolidation(
                    prompt,
                    carryover,
                    best_rec,
                    use_mock=self.use_mock,
                    gate_delta=self.cfg.consolidation.gate_delta,
                    consolidated_fitness=float(evo.get("best_score", 0.0)),
                    model=self.cfg.evolution.mutator_model,
                    api_base=self.cfg.ensemble.api_base,
                )
                # Live: re-score consolidated prompt on val for a meaningful gate
                if not self.use_mock and candidate_prompt != prompt:
                    cons_val = self._eval_split(candidate_prompt, val)
                    cons_fit = float(cons_val.get("CVaR_cluster", cons_val.get("R_global", 0.0)))
                    consolidated_rec = PromptRecord(
                        prompt=candidate_prompt,
                        fitness=cons_fit,
                        metrics={
                            "source": "base_guidelines_consolidation",
                            "R_global": float(cons_val.get("R_global", 0.0)),
                            "CVaR_cluster": float(cons_val.get("CVaR_cluster", 0.0)),
                        },
                        cluster_scores=cons_val.get("cluster_accuracies"),
                    )
                    from prime.consolidation.pool_carryover import gate_consolidation

                    gate_passed = gate_consolidation(
                        consolidated_rec, best_rec, self.cfg.consolidation.gate_delta
                    )
                    if gate_passed:
                        prompt = candidate_prompt
                else:
                    from prime.consolidation.pool_carryover import gate_consolidation

                    gate_passed = gate_consolidation(
                        consolidated_rec, best_rec, self.cfg.consolidation.gate_delta
                    )
                    if gate_passed:
                        prompt = candidate_prompt
                (cycle_dir / "best_prompt.txt").write_text(prompt, encoding="utf-8")
            carryover_prompts = merge_with_consolidated(carryover, consolidated_rec)
            base_after = extract_block(prompt, "BaseGuidelines") or ""
            self.trace.stage(
                "11_consolidation",
                f"Pool carryover (cycle {cycle})",
                cycle=cycle,
                archive_size=len(archive),
                carryover_k=len(carryover),
                carryover_fitnesses=[r.fitness for r in carryover],
                merged_population_size=len(carryover_prompts),
            )
            self.trace.stage(
                "11b_base_consolidation",
                f"BaseGuidelines consolidation (cycle {cycle})",
                cycle=cycle,
                enabled=self.cfg.consolidation.enabled,
                scope=self.cfg.consolidation.scope,
                gate_passed=gate_passed,
                base_len_before=len(base_before),
                base_len_after=len(base_after),
            )
            (cycle_dir / "carryover_prompts.json").write_text(
                json.dumps(
                    [{"fitness": r.fitness, "prompt_len": len(r.prompt)} for r in carryover],
                    indent=2,
                ),
                encoding="utf-8",
            )

            val_metrics = self._eval_split(prompt, val)
            selection_key = self._selection_key(val_metrics)
            if selection_key >= self._best_selection_key:
                self._best_selection_key = selection_key
                self._selected_prompt = prompt

            self._proxy_val_history.append(
                (float(val_metrics.get("CVaR_cluster", 0.0)), float(val_metrics.get("R_worst", 0.0)))
            )
            proxy = proxy_validation_report(
                self._proxy_val_history,
                min_cycles=self.cfg.active_learning.proxy_validation_min_cycles,
            )
            self.trace.stage(
                "12_selection_val",
                f"Val selection + proxy validation (cycle {cycle})",
                cycle=cycle,
                selection_mode=self.cfg.active_learning.selection_mode,
                selection_key=list(selection_key),
                best_so_far=list(self._best_selection_key),
                CVaR_cluster=val_metrics.get("CVaR_cluster"),
                R_global=val_metrics.get("R_global"),
                R_worst=val_metrics.get("R_worst"),
                proxy_cvar_rworst_corr=proxy.get("correlation"),
                proxy_ready=proxy.get("ready"),
            )
            self.logger.log(
                "val_metrics",
                cycle=cycle,
                selection_key=list(selection_key),
                proxy_validation=proxy,
                **val_metrics,
            )

            n_anchor_before = len(pool.anchor)
            added = self._expand_pool(pool, prompt, cycle)
            self.trace.stage(
                "13_pool_expand",
                f"Pool expansion (cycle {cycle})",
                cycle=cycle,
                expansion_policy=self.cfg.acquisition.expansion_policy,
                anchor_before=n_anchor_before,
                anchor_after=len(pool.anchor),
                added=len(added),
                seen_after=len(pool.seen),
                unseen_after=len(pool.unseen),
            )

            self.logger.log(
                "cycle_done",
                cycle=cycle,
                fitness=eval_result.get("fitness"),
                best_evo_score=evo.get("best_score"),
                carryover_k=len(carryover),
                batch_hard=len(batch.get("hard_indices", [])),
                batch_anchor=len(batch.get("anchor_indices", [])),
            )
            cycle_summaries.append(
                {
                    "cycle": cycle,
                    "fitness": eval_result.get("fitness"),
                    "val_selection_key": list(selection_key),
                    "proxy_cvar_rworst_corr": proxy.get("correlation"),
                    "artifacts": str(artifacts_path) if artifacts_path else None,
                }
            )

        final_prompt = self._selected_prompt or prompt
        final_test = self._eval_split(final_prompt, test)
        self.trace.stage(
            "14_final_test",
            "Final OOD test evaluation",
            R_global=final_test.get("R_global"),
            R_worst=final_test.get("R_worst"),
            CVaR_cluster=final_test.get("CVaR_cluster"),
            mae=final_test.get("mae"),
            num_users=final_test.get("num_users"),
        )

        summary = {
            "final_test": final_test,
            "run_dir": str(self.ctx.run_dir),
            "use_mock": self.use_mock,
            "inference_backend": "mock" if self.use_mock else "openrouter",
            "best_selection_key": list(self._best_selection_key),
            "proxy_validation_final": proxy_validation_report(
                self._proxy_val_history,
                min_cycles=self.cfg.active_learning.proxy_validation_min_cycles,
            ),
            "cycles": cycle_summaries,
        }
        if not self.use_mock:
            from prime.workers.ensemble import get_token_tracker

            summary["token_usage"] = get_token_tracker().snapshot()
        self.ctx.write_json("summary.json", summary)
        self.trace.stage("15_summary", "Summary written", path=str(self.ctx.run_dir / "summary.json"))
        checklist = self.trace.write_checklist()
        summary["smoke_checklist"] = str(checklist)
        summary["all_stages_passed"] = json.loads(checklist.read_text(encoding="utf-8"))["all_stages_passed"]
        return summary

    def _load_data(self) -> Dict[str, ReviewSplit]:
        self._used_synthetic = False
        if self.cfg.experiment.smoke:
            try:
                if self.cfg.dataset.name == "amazon":
                    splits = load_amazon_splits(self.cfg.dataset, seed=self.cfg.active_learning.seed)
                else:
                    raise ValueError(f"Unsupported dataset: {self.cfg.dataset.name}")
            except Exception as exc:
                if self.cfg.experiment.verbose:
                    print(f"[data] WILDS unavailable ({exc}), using synthetic splits", flush=True)
                splits = self._synthetic_splits()
                self._used_synthetic = True
            cap = self.cfg.experiment.smoke_max_examples
            splits = {
                k: subsample_split(v, cap, self.cfg.active_learning.seed + i)
                for i, (k, v) in enumerate(splits.items())
            }
            return splits
        if self.cfg.dataset.name == "amazon":
            return load_amazon_splits(self.cfg.dataset, seed=self.cfg.active_learning.seed)
        raise ValueError(f"Unsupported dataset: {self.cfg.dataset.name}")

    def _synthetic_splits(self) -> Dict[str, ReviewSplit]:
        rng = np.random.RandomState(self.cfg.active_learning.seed)
        splits: Dict[str, ReviewSplit] = {}
        offset = 0
        for name, n_users, n_per_user in (("train", 20, 8), ("validation", 10, 6), ("test", 10, 6)):
            texts, labels, users = [], [], []
            for u in range(offset, offset + n_users):
                for _ in range(n_per_user):
                    rating = int(rng.randint(1, 6))
                    texts.append(f"user{u} review with sentiment level {rating}")
                    labels.append(rating)
                    users.append(u)
            splits[name] = ReviewSplit(name=name, texts=texts, labels=labels, user_ids=users)
            offset += n_users
        return splits

    def _load_or_fit_clusters(
        self, train: ReviewSplit, val: ReviewSplit, test: ReviewSplit
    ) -> ClusterArtifacts:
        art_path = self.cfg.dataset.cluster_artifact
        if art_path:
            return ClusterArtifacts.load(Path(art_path))
        from prime.data.clustering import fit_style_clusters

        try:
            art = fit_style_clusters(train, self.cfg.clusters, self.cfg.dataset)
        except Exception as exc:
            if not self.cfg.clusters.allow_synthetic_fallback:
                raise RuntimeError(
                    "Style cluster fitting failed and clusters.allow_synthetic_fallback=false. "
                    f"Original error: {exc}"
                ) from exc
            print(
                f"[GRAPE WARNING] Cluster fitting failed ({exc}); "
                "using SYNTHETIC random embeddings. Set clusters.allow_synthetic_fallback=true "
                "to silence this error — results are NOT valid for research.",
                flush=True,
            )
            art = self._synthetic_clusters(train)
        out = self.ctx.run_dir / "clusters.json"
        art.save(out)
        return art

    def _synthetic_clusters(self, train: ReviewSplit) -> ClusterArtifacts:
        from prime.data.clustering import _compute_label_free_centroids
        from prime.data.profiles import (
            build_user_profiles,
            fit_profile_pipeline,
            profile_feature_matrix,
        )
        from sklearn.cluster import KMeans

        n = len(train.texts)
        emb = np.random.RandomState(self.cfg.clusters.seed).randn(n, 8).astype(np.float32)
        profiles = build_user_profiles(train.texts, train.labels, train.user_ids, emb)
        pipeline = fit_profile_pipeline(profiles, n_pca=min(4, self.cfg.clusters.pca_components))
        X, users = profile_feature_matrix(profiles, pipeline=pipeline, mode="full")
        k = min(self.cfg.clusters.n_clusters, max(2, len(users) // 3))
        km = KMeans(n_clusters=k, random_state=self.cfg.clusters.seed, n_init=10)
        labels = km.fit_predict(X)
        user_to_cluster = {u: int(l) for u, l in zip(users, labels)}
        lf_centroids = _compute_label_free_centroids(profiles, user_to_cluster, pipeline, k)
        from prime.data.profiles import projection_agreement

        agreement = projection_agreement(profiles, user_to_cluster, pipeline, lf_centroids)
        return ClusterArtifacts(
            n_clusters=k,
            user_to_cluster=user_to_cluster,
            cluster_centroids=km.cluster_centers_.astype(np.float32),
            cluster_centroids_label_free=lf_centroids,
            embedding_model="synthetic",
            seed=self.cfg.clusters.seed,
            pipeline=pipeline,
            train_projection_agreement=agreement,
            is_synthetic=True,
        )

    def _attach_split_clusters(self, split: ReviewSplit, art: ClusterArtifacts) -> ReviewSplit:
        try:
            mapping = assign_users_to_clusters(split, art, self.cfg.clusters, self.cfg.dataset)
        except Exception as exc:
            if not self.cfg.clusters.allow_synthetic_fallback:
                raise RuntimeError(
                    f"Cluster assignment failed for split '{split.name}' and "
                    "clusters.allow_synthetic_fallback=false. "
                    f"Original error: {exc}"
                ) from exc
            print(
                f"[GRAPE WARNING] Cluster assignment failed for '{split.name}' ({exc}); "
                "using user_id mod n_clusters fallback.",
                flush=True,
            )
            mapping = {u: int(u) % max(1, art.n_clusters) for u in set(split.user_ids)}
        return attach_example_clusters(split, mapping)

    def _selection_key(self, val_metrics: Dict[str, Any]) -> tuple:
        cvar = float(val_metrics.get("CVaR_cluster", 0.0))
        global_acc = float(val_metrics.get("R_global", 0.0))
        if self.cfg.active_learning.selection_mode == "lexicographic":
            return (cvar, global_acc)
        score = (
            self.cfg.active_learning.selection_w_cvar * cvar
            + self.cfg.active_learning.selection_w_global * global_acc
        )
        return (score,)

    def _expand_pool(self, pool: AcquisitionPool, prompt: str, cycle: int) -> List[int]:
        """Expand seen pool: disagreement + group coverage (default) or farthest fallback."""
        if len(pool.anchor) < self.cfg.acquisition.expansion_trigger or not pool.unseen:
            return []
        n_add = min(self.cfg.acquisition.expansion_batch, len(pool.unseen))
        if self.cfg.acquisition.expansion_policy == "farthest":
            return pool.maybe_expand()

        # Sample candidate subset from unseen (oversample) for one cheap ensemble pass
        candidates = sorted(pool.unseen)
        cap = min(len(candidates), max(n_add, int(n_add * self.cfg.acquisition.oversample_factor)))
        rng = np.random.RandomState(self.cfg.active_learning.seed + cycle + 17)
        if len(candidates) > cap:
            pick = sorted(int(i) for i in rng.choice(candidates, size=cap, replace=False))
        else:
            pick = candidates
        texts = [pool.texts[i] for i in pick]
        labels = [pool.labels[i] for i in pick]
        if self.use_mock:
            _ens, wp = mock_predict(
                texts, labels, len(self.cfg.ensemble.workers), self.cfg.active_learning.seed + cycle
            )
        else:
            from prime.workers.ensemble import build_workers

            workers = build_workers(self.cfg.ensemble)
            _ens, wp = parallel_predict(
                workers,
                texts,
                prompt,
                max_parallel=self.cfg.ensemble.max_parallel,
                tie_break=self.cfg.ensemble.tie_break,
            )
        d_map: Dict[int, float] = {}
        for j, idx in enumerate(pick):
            d_map[idx] = disagreement_score([wp[w][j] for w in range(len(wp))])
        return expand_pool_disagreement(
            pool,
            d_map,
            n_add=n_add,
            cluster_accuracies=self._last_cluster_accs or None,
            seed=self.cfg.active_learning.seed + cycle,
        )

    def _read_prompt(self) -> str:
        p = Path(self.cfg.prompt_path)
        if not p.is_absolute():
            p = self.project_root / p
        return p.read_text(encoding="utf-8")

    def _cycle_step(
        self,
        pool: AcquisitionPool,
        prompt: str,
        train: ReviewSplit,
        cycle_dir: Path,
        cycle: int,
    ) -> tuple[Dict[str, List[int]], Dict[str, Any], Optional[Path]]:
        seen_idx, texts, labels, user_ids, cluster_ids = pool.seen_arrays()
        if self.use_mock:
            ensemble, wp = mock_predict(
                texts, labels, len(self.cfg.ensemble.workers), self.cfg.active_learning.seed + cycle
            )
            inference_mode = "mock"
        else:
            from prime.workers.ensemble import build_workers

            workers = build_workers(self.cfg.ensemble)
            ensemble, wp = parallel_predict(
                workers,
                texts,
                prompt,
                max_parallel=self.cfg.ensemble.max_parallel,
                tie_break=self.cfg.ensemble.tie_break,
            )
            inference_mode = "live_llm"

        n_correct = sum(int(p == g) for p, g in zip(ensemble, labels))
        self.trace.stage(
            "06_inference_seen",
            f"Ensemble inference on seen pool (cycle {cycle})",
            cycle=cycle,
            mode=inference_mode,
            n_seen=len(texts),
            n_workers=len(self.cfg.ensemble.workers),
            accuracy_seen=round(n_correct / max(1, len(texts)), 4),
            tie_break=self.cfg.ensemble.tie_break,
        )

        disagreements = [
            disagreement_score([wp[w][i] for w in range(len(wp))]) for i in range(len(texts))
        ]
        from prime.fitness.metrics import cluster_accuracies as compute_cluster_accs

        cluster_accs = compute_cluster_accs(
            np.array(ensemble), np.array(labels), np.array(cluster_ids)
        )
        self._last_cluster_accs = {int(k): float(v) for k, v in cluster_accs.items()}
        batch = build_active_batch(
            seen_idx,
            ensemble,
            labels,
            disagreements,
            cluster_ids,
            None,
            self.cfg.acquisition,
            seed=self.cfg.active_learning.seed + cycle,
            cluster_accuracies=cluster_accs,
        )
        write_active_batch(cycle_dir / "active_batch.json", batch)
        self.trace.stage(
            "07_acquisition",
            f"Active batch built (cycle {cycle})",
            cycle=cycle,
            policy=self.cfg.acquisition.policy,
            batch_size=len(batch["indices"]),
            hard=len(batch.get("hard_indices", [])),
            anchor=len(batch.get("anchor_indices", [])),
            group_hard_quotas=batch.get("group_hard_quotas"),
            cluster_accuracies={str(k): round(v, 4) for k, v in cluster_accs.items()},
            top_disagreement=round(max(disagreements) if disagreements else 0.0, 4),
        )

        pool_data = {
            "texts": train.texts,
            "labels": train.labels,
            "user_ids": train.user_ids,
            "cluster_ids": train.example_cluster_ids,
        }
        (cycle_dir / "pool_data.json").write_text(json.dumps(pool_data), encoding="utf-8")
        os.environ["PRIME_POOL_PATH"] = str((cycle_dir / "pool_data.json").resolve())
        os.environ["PRIME_CONFIG_PATH"] = str((self.ctx.run_dir / "config_used.yaml").resolve())
        os.environ["PRIME_ACTIVE_BATCH_PATH"] = str((cycle_dir / "active_batch.json").resolve())
        os.environ["PRIME_USE_MOCK"] = "1" if self.use_mock else "0"
        os.environ["PRIME_N_CLUSTERS"] = str(self._actual_n_clusters)

        evaluator = CandidateEvaluator(
            self.cfg,
            train.texts,
            train.labels,
            train.user_ids,
            train.example_cluster_ids,
            active_batch=batch,
            use_mock=self.use_mock,
            actual_n_clusters=self._actual_n_clusters,
        )
        result = evaluator.evaluate_prompt(prompt)
        self.trace.stage(
            "08_fitness",
            f"Fitness computed (cycle {cycle})",
            cycle=cycle,
            mode=self.cfg.fitness.mode,
            fitness=round(float(result.get("fitness", 0.0)), 4),
            R_global=round(float(result.get("R_global", 0.0)), 4),
            CVaR_cluster=round(float(result.get("CVaR_cluster", 0.0)), 4),
            R_worst=round(float(result.get("R_worst", 0.0)), 4),
            mean_kappa=round(float(result.get("mean_kappa", 0.0)), 4),
        )

        artifacts_path: Optional[Path] = None
        if result.get("predictions") and batch.get("indices"):
            batch_texts = [train.texts[i] for i in batch["indices"]]
            cids = [train.example_cluster_ids[i] for i in batch["indices"]] if train.example_cluster_ids else None
            err_text = format_error_artifacts(
                result["predictions"],
                result["gold_labels"],
                result["worker_predictions"],
                batch_texts,
                cluster_ids=cids,
                hard_indices=batch.get("hard_indices"),
                anchor_indices=batch.get("anchor_indices"),
            )
            if err_text:
                artifacts_path = cycle_dir / "error_artifacts.txt"
                artifacts_path.write_text(err_text, encoding="utf-8")
                self.trace.stage(
                    "09_error_artifacts",
                    f"Mutator error report (cycle {cycle})",
                    cycle=cycle,
                    path=str(artifacts_path),
                    chars=len(err_text),
                    preview=err_text[:200].replace("\n", " | "),
                )

        pool.update_hard_anchor(ensemble, wp)
        return batch, result, artifacts_path

    def _eval_split(self, prompt: str, split: ReviewSplit) -> Dict[str, Any]:
        if self.use_mock:
            ensemble, wp = mock_predict(
                split.texts,
                split.labels,
                len(self.cfg.ensemble.workers),
                self.cfg.active_learning.seed,
            )
        else:
            from prime.workers.ensemble import build_workers

            workers = build_workers(self.cfg.ensemble)
            ensemble, wp = parallel_predict(
                workers,
                split.texts,
                prompt,
                max_parallel=self.cfg.ensemble.max_parallel,
                tie_break=self.cfg.ensemble.tie_break,
            )
        from prime.fitness.metrics import compute_metrics

        cluster_ids = np.array(split.example_cluster_ids) if split.example_cluster_ids else None
        return compute_metrics(
            np.array(ensemble),
            np.array(split.labels),
            np.array(split.user_ids),
            worker_predictions=[np.array(w) for w in wp],
            cluster_ids=cluster_ids,
            cvar_quantile=self.cfg.fitness.cvar_quantile,
        )


def run_from_config(config_path: Path) -> Dict[str, Any]:
    project_root = config_path.resolve().parents[1]
    if config_path.parent.name == "experiments":
        project_root = config_path.resolve().parents[2]
    cfg = load_config(config_path)
    return PrimeController(cfg, project_root, config_path).run()

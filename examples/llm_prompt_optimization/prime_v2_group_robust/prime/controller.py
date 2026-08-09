"""Thin orchestration for PRIME v2 active learning loop."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from prime.acquisition.batch_builder import build_active_batch
from prime.acquisition.batch_diagnostics import build_batch_diagnostics
from prime.acquisition.eval_sets import (
    EvalSets,
    build_d_anchor,
    build_d_audit,
    build_d_select,
    power_rule_max_k,
    rotate_d_select,
    split_arrays,
)
from prime.acquisition.expansion import expand_pool_disagreement
from prime.acquisition.pool import AcquisitionPool
from prime.config import PrimeConfig, load_config
from prime.consolidation.base_consolidator import build_consolidated_prompt, check_not_lossy
from prime.consolidation.pareto import (
    Candidate,
    champion_archive,
    front_summary,
    read_oe_candidates,
    reconcile_oe_with_carried,
    sync_seed_checkpoint_metrics,
    write_seed_checkpoint,
)
from prime.experiment.anchor_gate import evaluate_anchor_gate, summarize_gate_history
from prime.experiment.budget import BudgetExhausted, TokenTracker
from prime.experiment.dev_gate import (
    evaluate_dev_gate,
    generalization_gap,
    softmin_from_metrics,
)
from prime.consolidation.pool_carryover import PromptRecord, select_carryover
from prime.data.clustering import ClusterArtifacts, assign_users_to_clusters, attach_example_clusters
from prime.data.fixed_sets import (
    FixedSet,
    build_d_dev,
    build_test_fixed,
    materialize_split,
)
from prime.data.splits import split_fit_heldout
from prime.data.wilds_loader import ReviewSplit, load_amazon_splits, subsample_split
from prime.data.civilcomments_loader import (
    build_oracle_cluster_artifacts,
    load_civilcomments_splits,
)
from prime.evolution.artifacts import (
    extract_systematic_fewshot_examples,
    format_error_artifacts,
    lint_dynamic_rules_for_triggers,
)
from prime.evolution.openevolve_adapter import (
    CandidateEvaluator,
    prompt_hash,
    run_evolution,
    write_active_batch,
)
from prime.evolution.openevolve_config_patch import patch_openevolve_feature_dimensions
from prime.evolution.prompt_blocks import extract_block, inject_verbatim_fewshot, strip_evolve_markers
from prime.experiment.logging import JsonlLogger
from prime.experiment.proxy_validation import proxy_validation_report
from prime.experiment.run_context import RunContext
from prime.experiment.stage_trace import StageTracer
from prime.workers.ensemble import disagreement_score, mock_predict, parallel_predict


def _archive_cluster_scores(eval_result: Dict[str, Any]) -> Dict[int, float]:
    """
    Per-cluster axis for the champion archive, in the same estimator the objective
    uses. Preference order matches `openevolve_adapter`'s `cluster_acc_*`, otherwise
    controller-built candidates and OpenEvolve programs would be compared on
    different quantities.
    """
    accs = (
        eval_result.get("cluster_accuracies_balanced_shrunk")
        or eval_result.get("cluster_accuracies_shrunk")
        or eval_result.get("cluster_accuracies")
        or {}
    )
    return {int(k): float(v) for k, v in accs.items()}


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
        self._anchor_gate_history: List[Dict[str, Any]] = []
        self._dev_gate_history: List[Dict[str, Any]] = []
        self._generalization_gaps: List[Dict[str, Any]] = []
        self._actual_n_clusters: int = cfg.clusters.n_clusters
        self._openevolve_config_used: Optional[Path] = None
        self._last_cluster_accs: Dict[int, float] = {}
        # Champion archive carried between cycles + the OE checkpoint that seeds it.
        self._carried_front: List[Candidate] = []
        # cluster_id -> prompt of the incumbent champion (P9 anti-churn margin).
        self._champion_slots: Dict[int, str] = {}
        self._seed_checkpoint: Optional[Path] = None
        # val/test predictions keyed by (prompt hash, split name): the consolidation
        # path and the selection path used to pay for the same val eval twice
        # (OBSERVATIONS O12).
        self._split_metric_cache: Dict[tuple, Dict[str, Any]] = {}
        # train example index -> previous cycle's ensemble prediction, for the
        # mutator's damage report.
        self._prev_example_preds: Dict[int, int] = {}
        # Last accepted consolidated BaseGuidelines: the slow layer only accumulates
        # if it is fed back in (OBSERVATIONS O21).
        self._last_consolidated_base: Optional[str] = None
        self._dev_split: Optional[ReviewSplit] = None
        self._test_fixed_split: Optional[ReviewSplit] = None
        self._fixed_set_meta: Dict[str, Any] = {}
        self._call_budget = TokenTracker.from_cfg(cfg.budget)

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
        seed_cache = os.environ.get("PRIME_SEED_PRED_CACHE", "").strip()
        if seed_cache:
            src = Path(seed_cache)
            if not src.is_absolute():
                src = Path.cwd() / src
            if src.is_dir():
                dst = self.ctx.run_dir / "pred_cache"
                dst.mkdir(parents=True, exist_ok=True)
                n_copied = 0
                for f in src.iterdir():
                    if f.is_file():
                        target = dst / f.name
                        if not target.exists():
                            shutil.copy2(f, target)
                            n_copied += 1
                print(
                    f"[GRAPE] seeded pred_cache from {src} ({n_copied} files)",
                    flush=True,
                )
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

        # SPEC v3 Р7: user-level fit/heldout source split BEFORE cluster fitting,
        # so cluster geometry never sees selection sources.
        roles = self.cfg.data_roles
        if roles.enabled:
            source_split = split_fit_heldout(
                train,
                roles.fit_fraction,
                seed=self.cfg.active_learning.seed,
                stratify=bool(getattr(roles, "stratify", True)),
            )
            train_fit = source_split.fit
            train_heldout: Optional[ReviewSplit] = source_split.heldout
            source_split.save(self.ctx.run_dir / "source_split.json")
            self.trace.stage(
                "03b_source_split",
                "Train sources split into fit/heldout (SPEC v3 4.3)",
                fit_fraction=roles.fit_fraction,
                stratify=bool(getattr(roles, "stratify", True)),
                fit_users=len(source_split.fit_users),
                heldout_users=len(source_split.heldout_users),
                fit_examples=len(train_fit),
                heldout_examples=len(train_heldout),
                strata_balance=source_split.strata_balance,
                user_disjoint=True,
            )
        else:
            train_fit, train_heldout = train, None
        self._train_heldout = train_heldout

        prompt = self._read_prompt()
        (self.ctx.run_dir / "initial_prompt.txt").write_text(prompt, encoding="utf-8")

        cluster_art = self._load_or_fit_clusters(train_fit, prompt=prompt)
        train_fit = attach_example_clusters(train_fit, cluster_art.user_to_cluster)
        if train_heldout is not None:
            train_heldout = self._attach_split_clusters(train_heldout, cluster_art, prompt=prompt)
        val = self._attach_split_clusters(val, cluster_art, prompt=prompt)
        test = self._attach_split_clusters(splits["test"], cluster_art, prompt=prompt)
        cluster_counts: Dict[int, int] = {}
        for cid in train_fit.example_cluster_ids or []:
            cluster_counts[cid] = cluster_counts.get(cid, 0) + 1
        self.trace.stage(
            "04_clustering",
            "Clusters fitted",
            n_clusters=cluster_art.n_clusters,
            fit_mode=cluster_art.fit_mode,
            geometry=getattr(self.cfg.clusters, "geometry", "style"),
            n_train_users=len(cluster_art.user_to_cluster),
            embedding_model=cluster_art.embedding_model,
            cluster_example_counts=cluster_counts,
            train_projection_agreement=round(cluster_art.train_projection_agreement, 4),
            is_synthetic=cluster_art.is_synthetic,
            diagnostics=cluster_art.diagnostics,
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

        # SPEC v3 §4.3: fixed evaluation sets from heldout sources. All candidate
        # decisions come from D_select; D_anchor is the regression gate (Р6, Р9).
        self._select_data: Optional[Dict[str, list]] = None
        self._anchor_data: Optional[Dict[str, list]] = None
        self._pred_cache_dir = self.ctx.run_dir / "pred_cache"
        if roles.enabled and train_heldout is not None and len(train_heldout) > 0:
            eval_sets = self._build_eval_sets(train_heldout, prompt)
            self._eval_sets = eval_sets
            self._select_data = split_arrays(train_heldout, eval_sets.d_select)
            self._anchor_data = split_arrays(train_heldout, eval_sets.d_anchor)
            select_path = self.ctx.run_dir / "d_select_data.json"
            select_path.write_text(json.dumps(self._select_data), encoding="utf-8")
            os.environ["PRIME_DSELECT_PATH"] = str(select_path.resolve())
            os.environ["PRIME_PRED_CACHE_DIR"] = str(self._pred_cache_dir.resolve())
            self.trace.stage(
                "04c_eval_sets",
                "D_select / D_anchor / D_audit built from heldout sources",
                d_select=len(eval_sets.d_select),
                d_anchor=len(eval_sets.d_anchor),
                d_audit=len(eval_sets.d_audit),
                content_hash=eval_sets.content_hash,
                disjoint=not (
                    set(eval_sets.d_select) & set(eval_sets.d_anchor)
                    or set(eval_sets.d_select) & set(eval_sets.d_audit)
                    or set(eval_sets.d_anchor) & set(eval_sets.d_audit)
                ),
                path=str(self.ctx.run_dir / "eval_sets.json"),
            )
        else:
            os.environ.pop("PRIME_DSELECT_PATH", None)
            os.environ.pop("PRIME_PRED_CACHE_DIR", None)

        self._build_fixed_eval_sets(val, test)

        # Acquisition pool over fit sources only: it feeds the mutator (D_mut
        # role, adversarial bias allowed), never selection (Р6).
        pool = AcquisitionPool(train_fit, self.cfg.dataset, self.cfg.acquisition, self.cfg.active_learning.seed)
        initial_seen = pool.initialize_seen(self.cfg.acquisition.batch_size)
        self.trace.stage(
            "05_pool_init",
            "Acquisition pool initialized (fit sources)",
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
            if (
                cycle > 1
                and roles.enabled
                and getattr(roles, "d_select_rotate", False)
                and self._train_heldout is not None
                and getattr(self, "_eval_sets", None) is not None
            ):
                self._rotate_d_select_for_cycle(self._train_heldout, cycle)

            if cycle > 1 and self._selected_prompt:
                prompt = self._selected_prompt
                if self.cfg.experiment.verbose:
                    print(f"[CYCLE {cycle}] carryover prompt selected (len={len(prompt)})", flush=True)

            cycle_start_prompt = prompt
            cycle_dir = self.ctx.cycle_dir(cycle)
            batch, eval_result, artifacts_path = self._cycle_step(pool, prompt, train_fit, cycle_dir, cycle)

            # O22: inject verbatim few-shots into the parent prompt before OE mutates.
            if (
                getattr(self.cfg.evolution, "inject_verbatim_fewshot", False)
                and artifacts_path
                and artifacts_path.is_file()
            ):
                examples = extract_systematic_fewshot_examples(
                    artifacts_path.read_text(encoding="utf-8"),
                    limit=int(getattr(self.cfg.evolution, "fewshot_inject_limit", 4)),
                )
                if examples:
                    before = prompt
                    prompt = inject_verbatim_fewshot(
                        prompt,
                        examples,
                        label_space=self.cfg.dataset.label_space,
                    )
                    if prompt != before:
                        (cycle_dir / "prompt_with_injected_fewshot.txt").write_text(
                            prompt, encoding="utf-8"
                        )
                        self.trace.stage(
                            "09b_fewshot_inject",
                            "Verbatim few-shot injection (O22)",
                            cycle=cycle,
                            n_examples=len(examples),
                            prompt_changed=True,
                        )

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

            cfg_for_oe = self.ctx.run_dir / "config_resolved.yaml"
            if not cfg_for_oe.is_file():
                cfg_for_oe = self.ctx.run_dir / "config_used.yaml"
            evo = run_evolution(
                self.cfg,
                prompt,
                cycle_dir,
                cfg_for_oe,
                cycle_dir / "active_batch.json",
                use_mock=self.use_mock,
                cycle=cycle,
                openevolve_config_path=self._openevolve_config_used,
                seed_checkpoint=self._seed_checkpoint,
            )
            evolved_prompt = evo.get("best_prompt", prompt)
            evolved_prompt = strip_evolve_markers(evolved_prompt)

            # O13: drop DynamicRules lines that name clusters/groups.
            if getattr(self.cfg.evolution, "enforce_text_triggers", False):
                violations = lint_dynamic_rules_for_triggers(evolved_prompt)
                if violations:
                    evolved_prompt = self._strip_cluster_addressed_rules(evolved_prompt)
                    (cycle_dir / "o13_trigger_violations.json").write_text(
                        json.dumps({"violations": violations}, indent=2),
                        encoding="utf-8",
                    )
                    self.trace.stage(
                        "10b_o13_lint",
                        "Stripped cluster-addressed DynamicRules (O13)",
                        cycle=cycle,
                        n_violations=len(violations),
                    )

            # O22: re-assert verbatim few-shots after OE (mutator may invent fiction).
            if (
                getattr(self.cfg.evolution, "inject_verbatim_fewshot", False)
                and artifacts_path
                and artifacts_path.is_file()
            ):
                examples = extract_systematic_fewshot_examples(
                    artifacts_path.read_text(encoding="utf-8"),
                    limit=int(getattr(self.cfg.evolution, "fewshot_inject_limit", 4)),
                )
                if examples:
                    evolved_prompt = inject_verbatim_fewshot(
                        evolved_prompt,
                        examples,
                        label_space=self.cfg.dataset.label_space,
                    )

            self.trace.stage(
                "10_evolution",
                f"Inner-loop evolution (cycle {cycle})",
                cycle=cycle,
                mock=evo.get("mock", False),
                best_score=evo.get("best_score"),
                prompt_changed=evo.get("prompt_changed", False),
                artifacts_file=evo.get("artifacts_path"),
                seed_mode=evo.get("seed_mode", "fresh"),
            )

            # ---- Pareto front over per-cluster accuracy on D_select ----------
            # Every member is measured on the same fixed D_select, so building the
            # front costs no API calls beyond scoring the consolidation candidate.
            candidates: List[Candidate] = list(self._carried_front)
            candidates.append(
                Candidate(
                    prompt=cycle_start_prompt,
                    fitness=float(eval_result.get("fitness", 0.0)),
                    cluster_scores=_archive_cluster_scores(eval_result),
                    metrics={
                        "R_global": float(eval_result.get("R_global", 0.0)),
                        "CVaR_cluster": float(eval_result.get("CVaR_cluster", 0.0)),
                    },
                    source="cycle_entry",
                    cycle=cycle,
                )
            )
            oe_candidates = read_oe_candidates(
                Path(evo["openevolve_staging"]), cycle, strip_evolve_markers
            ) if evo.get("openevolve_staging") else []
            if not oe_candidates and evolved_prompt != cycle_start_prompt:
                # mock / eval-only paths write no checkpoint; keep the mutation.
                oe_candidates = [
                    Candidate(
                        prompt=evolved_prompt,
                        fitness=float(evo.get("best_score", 0.0)),
                        cluster_scores=_archive_cluster_scores(eval_result),
                        source="oe",
                        cycle=cycle,
                    )
                ]
            # M27: OE seed JSONs may still carry previous-cycle combined_score;
            # prefer O26-rescored carried fitness for identical prompts.
            oe_candidates = reconcile_oe_with_carried(self._carried_front or [], oe_candidates)
            candidates.extend(oe_candidates)

            # ---- Consolidation as a competitor, not an heir ------------------
            carryover = select_carryover(
                archive,
                self.cfg.consolidation.pool_carryover_k,
                cluster_pareto=self.cfg.consolidation.cluster_pareto,
            )
            consolidation_info: Dict[str, Any] = {"enabled": False}
            base_before = extract_block(evolved_prompt, "BaseGuidelines") or ""
            # Frequency: consolidating a freshly-mutated prompt every cycle just
            # churns — the slow layer only has something to absorb once rules have
            # accumulated. The last cycle always runs so the final prompt gets one
            # tidy-up pass (OBSERVATIONS O21).
            is_last_cycle = cycle >= self.cfg.active_learning.n_cycles
            due = (cycle % self.cfg.consolidation.every_n_cycles == 0) or (
                is_last_cycle and self.cfg.consolidation.run_on_last_cycle
            )
            if (
                self.cfg.consolidation.enabled
                and self.cfg.consolidation.scope == "base_guidelines"
                and due
            ):
                cons_prompt = build_consolidated_prompt(
                    evolved_prompt,
                    carryover,
                    use_mock=self.use_mock,
                    model=self.cfg.consolidation.model,
                    api_base=self.cfg.ensemble.api_base,
                    prev_base=self._last_consolidated_base,
                )
                not_lossy, guard_reason = check_not_lossy(evolved_prompt, cons_prompt)
                consolidation_info = {
                    "enabled": True,
                    "changed": cons_prompt != evolved_prompt,
                    "stateful_prev_base_len": len(self._last_consolidated_base or ""),
                    "guard_passed": not_lossy,
                    "guard_reason": guard_reason,
                    "base_len_before": len(base_before),
                    "base_len_after": len(extract_block(cons_prompt, "BaseGuidelines") or ""),
                    "dynamic_len_before": len(extract_block(evolved_prompt, "DynamicRules") or ""),
                    "dynamic_len_after": len(extract_block(cons_prompt, "DynamicRules") or ""),
                }
                if not not_lossy:
                    print(
                        f"[GRAPE] consolidation discarded ({guard_reason})",
                        flush=True,
                    )
                else:
                    self._last_consolidated_base = (
                        extract_block(cons_prompt, "BaseGuidelines") or ""
                    )
                if (
                    not_lossy
                    and cons_prompt != evolved_prompt
                    and self.cfg.pareto.score_consolidated_on_select
                ):
                    cons_eval = self._score_on_select(cons_prompt)
                    consolidation_info["select_fitness"] = float(cons_eval.get("fitness", 0.0))
                    consolidation_info["evolved_select_fitness"] = float(
                        evo.get("best_score", 0.0)
                    )
                    consolidation_info["delta_vs_evolved"] = round(
                        consolidation_info["select_fitness"]
                        - consolidation_info["evolved_select_fitness"],
                        6,
                    )
                    candidates.append(
                        Candidate(
                            prompt=cons_prompt,
                            fitness=float(cons_eval.get("fitness", 0.0)),
                            cluster_scores=_archive_cluster_scores(cons_eval),
                            metrics={
                                "R_global": float(cons_eval.get("R_global", 0.0)),
                                "CVaR_cluster": float(cons_eval.get("CVaR_cluster", 0.0)),
                            },
                            source="consolidated",
                            cycle=cycle,
                        )
                    )
                (cycle_dir / "consolidated_prompt.txt").write_text(cons_prompt, encoding="utf-8")
            (cycle_dir / "consolidation.json").write_text(
                json.dumps(consolidation_info, indent=2), encoding="utf-8"
            )

            front, self._champion_slots = champion_archive(
                candidates,
                prev_champions=self._champion_slots,
                margin=self.cfg.pareto.champion_margin,
            )
            prompt, heir_gate = self._pick_heir(front, cycle_start_prompt, cycle, cycle_dir)
            heir_select_fit = next(
                (c.fitness for c in front if c.prompt == prompt),
                eval_result.get("fitness"),
            )
            prompt, dev_gate = self._apply_dev_gate(
                prompt,
                cycle_start_prompt,
                cycle,
                cycle_dir,
                select_fitness=heir_select_fit,
            )
            self._carried_front = self._reorder_front_for_heir(front, prompt)
            (cycle_dir / "pareto_front.json").write_text(
                json.dumps(
                    {
                        "n_candidates": len(candidates),
                        "candidate_sources": sorted({c.source for c in candidates}),
                        "heir_source": next(
                            (c.source for c in front if c.prompt == prompt), "cycle_entry"
                        ),
                        "champion_slots": {
                            str(cid): next(
                                (
                                    {"source": c.source, "cycle": c.cycle,
                                     "fitness": round(c.fitness, 6)}
                                    for c in front
                                    if c.prompt == p
                                ),
                                None,
                            )
                            for cid, p in sorted(self._champion_slots.items())
                        },
                        **front_summary(front),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            (cycle_dir / "best_prompt.txt").write_text(prompt, encoding="utf-8")

            base_after = extract_block(prompt, "BaseGuidelines") or ""
            self.trace.stage(
                "11_consolidation",
                f"Pareto front + carryover (cycle {cycle})",
                cycle=cycle,
                archive_size=len(archive),
                n_candidates=len(candidates),
                front_size=len(front),
                front_sources=sorted({c.source for c in front}),
                heir_source=next((c.source for c in front if c.prompt == prompt), "cycle_entry"),
                heir_select_fitness=next(
                    (round(c.fitness, 6) for c in front if c.prompt == prompt), None
                ),
                carryover_k=len(carryover),
            )
            self.trace.stage(
                "11b_base_consolidation",
                f"BaseGuidelines consolidation (cycle {cycle})",
                cycle=cycle,
                enabled=consolidation_info.get("enabled", False),
                changed=consolidation_info.get("changed", False),
                select_fitness=consolidation_info.get("select_fitness"),
                evolved_select_fitness=consolidation_info.get("evolved_select_fitness"),
                delta_vs_evolved=consolidation_info.get("delta_vs_evolved"),
                won_front=next(
                    (c.source for c in front if c.prompt == prompt), "cycle_entry"
                ) == "consolidated",
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
            if self.cfg.pareto.seed_openevolve_from_front and evo.get("openevolve_staging"):
                # Keep seed checkpoints under this run_dir — NOT results/_oe/seed_cN,
                # which collides when two experiments run in parallel (shared parent).
                self._seed_checkpoint = write_seed_checkpoint(
                    self._carried_front,
                    Path(evo["openevolve_staging"]),
                    self.ctx.run_dir / f"seed_c{cycle + 1}",
                )

            selection_split = str(
                getattr(self.cfg.active_learning, "selection_split", "validation") or "validation"
            )
            if selection_split == "d_dev" and self._dev_split is not None:
                sel_metrics = self._eval_split(prompt, self._dev_split)
                sel_name = "d_dev"
            else:
                sel_metrics = self._eval_split(prompt, val)
                sel_name = "validation"
            # Never score full WILDS val here: at max_val_users=50k that alone
            # exhausts the 60k scorer budget. Proxy series uses selection metrics.
            val_metrics = sel_metrics

            selection_key = self._selection_key(sel_metrics)
            if selection_key >= self._best_selection_key:
                self._best_selection_key = selection_key
                self._selected_prompt = prompt

            tail_key = self.cfg.active_learning.proxy_tail_metric
            self._proxy_val_history.append(
                (float(val_metrics.get("CVaR_cluster", 0.0)), float(val_metrics.get(tail_key, 0.0)))
            )
            proxy = proxy_validation_report(
                self._proxy_val_history,
                min_cycles=self.cfg.active_learning.proxy_validation_min_cycles,
            )
            self.trace.stage(
                "12_selection_val",
                f"Selection on {sel_name} + proxy (cycle {cycle})",
                cycle=cycle,
                selection_split=sel_name,
                selection_mode=self.cfg.active_learning.selection_mode,
                selection_key=list(selection_key),
                best_so_far=list(self._best_selection_key),
                R_soft_min_gba=sel_metrics.get("R_soft_min_gba"),
                R_worst_gba=sel_metrics.get("R_worst_gba"),
                R_gba_mean=sel_metrics.get("R_gba_mean"),
                CVaR_cluster=val_metrics.get("CVaR_cluster"),
                R_global=sel_metrics.get("R_global"),
                R_worst=sel_metrics.get("R_worst"),
                R_tail=sel_metrics.get("R_tail"),
                proxy_tail_metric=tail_key,
                proxy_cvar_tail_corr=proxy.get("correlation"),
                proxy_ready=proxy.get("ready"),
                dev_gate=dev_gate,
            )
            self.logger.log(
                "val_metrics",
                cycle=cycle,
                selection_split=sel_name,
                selection_key=list(selection_key),
                proxy_validation=proxy,
                dev_gate=dev_gate,
                **sel_metrics,
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

            heir_source = next(
                (c.source for c in front if c.prompt == prompt), "cycle_entry"
            )
            self.logger.log(
                "cycle_done",
                cycle=cycle,
                fitness=eval_result.get("fitness"),
                best_evo_score=evo.get("best_score"),
                carryover_k=len(carryover),
                front_size=len(front),
                heir_source=heir_source,
                batch_hard=len(batch.get("hard_indices", [])),
                batch_anchor=len(batch.get("anchor_indices", [])),
            )
            gate_accepted = [
                g
                for g in self._anchor_gate_history
                if g.get("cycle") == cycle and g.get("accepted")
            ]
            gate_this_cycle = gate_accepted[0] if gate_accepted else next(
                (g for g in self._anchor_gate_history if g.get("cycle") == cycle), None
            )
            cycle_summaries.append(
                {
                    "cycle": cycle,
                    "fitness": eval_result.get("fitness"),
                    "val_selection_key": list(selection_key),
                    "proxy_cvar_tail_corr": proxy.get("correlation"),
                    "artifacts": str(artifacts_path) if artifacts_path else None,
                    "best_evo_score": evo.get("best_score"),
                    "oe_seed_mode": evo.get("seed_mode"),
                    "pareto": {
                        "n_candidates": len(candidates),
                        "front_size": len(front),
                        "front_sources": sorted({c.source for c in front}),
                        "heir_source": heir_source,
                        "heir_select_fitness": next(
                            (c.fitness for c in front if c.prompt == prompt), None
                        ),
                    },
                    "consolidation": consolidation_info,
                    "anchor_gate": (
                        {
                            "accepted": gate_this_cycle["accepted"],
                            "reason": gate_this_cycle["reason"],
                            "drop": gate_this_cycle["drop"],
                            "mcnemar_p_one_sided": gate_this_cycle["mcnemar_p_one_sided"],
                            "candidate_source": gate_this_cycle.get("candidate_source"),
                            "front_rank": gate_this_cycle.get("front_rank"),
                        }
                        if gate_this_cycle
                        else None
                    ),
                }
            )

        final_prompt = self._selected_prompt or prompt
        if self._test_fixed_split is not None:
            final_test = self._eval_split(final_prompt, self._test_fixed_split)
            final_tag = "test_fixed"
        else:
            final_test = self._eval_split(final_prompt, test)
            final_tag = "test"
        self.trace.stage(
            "14_final_test",
            f"Final OOD evaluation ({final_tag})",
            split=final_tag,
            fingerprint=self._fixed_set_meta.get("test_fixed", {}).get("fingerprint"),
            R_global=final_test.get("R_global"),
            R_worst_gba=final_test.get("R_worst_gba"),
            R_gba_mean=final_test.get("R_gba_mean"),
            R_worst=final_test.get("R_worst"),
            R_tail=final_test.get("R_tail"),
            CVaR_cluster=final_test.get("CVaR_cluster"),
            mae=final_test.get("mae"),
            num_users=final_test.get("num_users"),
        )

        summary = {
            "final_test": final_test,
            "final_test_split": final_tag,
            "run_dir": str(self.ctx.run_dir),
            "use_mock": self.use_mock,
            "inference_backend": "mock" if self.use_mock else "openrouter",
            "cluster_geometry": getattr(self.cfg.clusters, "geometry", "style"),
            "actual_n_clusters": self._actual_n_clusters,
            "fitness_mode": self.cfg.fitness.mode,
            "best_selection_key": list(self._best_selection_key),
            "selection_split": getattr(self.cfg.active_learning, "selection_split", "validation"),
            "fixed_sets": self._fixed_set_meta,
            "generalization_gap": self._generalization_gaps,
            "proxy_validation_final": proxy_validation_report(
                self._proxy_val_history,
                min_cycles=self.cfg.active_learning.proxy_validation_min_cycles,
            ),
            "cycles": cycle_summaries,
            "anchor_gate": summarize_gate_history(self._anchor_gate_history),
            "dev_gate": summarize_gate_history(self._dev_gate_history),
            "pareto": {
                "enabled": self.cfg.pareto.enabled,
                "mode": "champion_archive",
                "champion_margin": self.cfg.pareto.champion_margin,
                "seed_openevolve_from_front": self.cfg.pareto.seed_openevolve_from_front,
                "final_front": front_summary(self._carried_front),
                "final_champion_slots": {
                    str(cid): next(
                        (c.source for c in self._carried_front if c.prompt == p), None
                    )
                    for cid, p in sorted(self._champion_slots.items())
                },
                "heir_sources": [c.get("pareto", {}).get("heir_source") for c in cycle_summaries],
                "consolidation_deltas": [
                    c.get("consolidation", {}).get("delta_vs_evolved") for c in cycle_summaries
                ],
            },
        }
        budget_snap = dict(self._call_budget.snapshot())
        if not self.use_mock:
            from prime.workers.ensemble import get_token_tracker

            token_snap = get_token_tracker().snapshot()
            summary["token_usage"] = token_snap
            budget_snap["token_usage"] = token_snap
            # F10: scorer/optimizer call counts are the comparison currency.
            budget_snap["scorer_tokens"] = int(
                sum(int(v) for v in (token_snap.get("by_model") or {}).values())
                if isinstance(token_snap.get("by_model"), dict)
                else token_snap.get("total_tokens", 0) or 0
            )
        summary["budget"] = budget_snap
        (self.ctx.run_dir / "budget_report.json").write_text(
            json.dumps(budget_snap, indent=2),
            encoding="utf-8",
        )
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
                elif self.cfg.dataset.name == "civilcomments":
                    splits = load_civilcomments_splits(
                        self.cfg.dataset, seed=self.cfg.active_learning.seed
                    )
                else:
                    raise ValueError(f"Unsupported dataset: {self.cfg.dataset.name}")
            except Exception as exc:
                if self.cfg.experiment.verbose:
                    print(f"[data] dataset unavailable ({exc}), using synthetic splits", flush=True)
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
        if self.cfg.dataset.name == "civilcomments":
            return load_civilcomments_splits(self.cfg.dataset, seed=self.cfg.active_learning.seed)
        raise ValueError(f"Unsupported dataset: {self.cfg.dataset.name}")

    def _synthetic_splits(self) -> Dict[str, ReviewSplit]:
        rng = np.random.RandomState(self.cfg.active_learning.seed)
        splits: Dict[str, ReviewSplit] = {}
        offset = 0
        binary = self.cfg.dataset.label_space == "binary"
        for name, n_users, n_per_user in (("train", 20, 8), ("validation", 10, 6), ("test", 10, 6)):
            texts, labels, users, groups = [], [], [], []
            for u in range(offset, offset + n_users):
                for j in range(n_per_user):
                    if binary:
                        label = int(rng.randint(0, 2))
                        texts.append(f"user{u} comment about topic {j} toxic={label}")
                        labels.append(label)
                        groups.append(int(u % 9))
                    else:
                        rating = int(rng.randint(1, 6))
                        texts.append(f"user{u} review with sentiment level {rating}")
                        labels.append(rating)
                        groups.append(None)
                    users.append(u)
            splits[name] = ReviewSplit(
                name=name,
                texts=texts,
                labels=labels,
                user_ids=users,
                example_cluster_ids=groups if binary else None,
            )
            offset += n_users
        return splits

    def _load_or_fit_clusters(
        self, train: ReviewSplit, prompt: Optional[str] = None
    ) -> ClusterArtifacts:
        """Fit clusters on fit sources (style label-free or pred_profile)."""
        art_path = self.cfg.dataset.cluster_artifact
        if art_path:
            src = Path(art_path)
            if not src.is_absolute():
                src = Path.cwd() / src
            art = ClusterArtifacts.load(src)
            # Keep a copy next to this run for audit (do not mutate source).
            art.save(self.ctx.run_dir / "clusters.json")
            print(
                f"[GRAPE] loaded cluster_artifact: {src} "
                f"(K={art.n_clusters} fit_mode={art.fit_mode})",
                flush=True,
            )
            return art

        roles = self.cfg.data_roles
        max_k = (
            power_rule_max_k(roles.d_select_size, roles.n_min_per_group)
            if roles.enabled
            else None
        )
        geometry = getattr(self.cfg.clusters, "geometry", "style") or "style"

        if geometry == "oracle":
            if train.example_cluster_ids is None:
                raise RuntimeError(
                    "oracle geometry requires example_cluster_ids on the fit split "
                    "(CivilComments loader attaches them at load time)"
                )
            art = build_oracle_cluster_artifacts(train)
            # Honour configured n_clusters as the declared group count when larger.
            if self.cfg.clusters.n_clusters > art.n_clusters:
                art = ClusterArtifacts(
                    n_clusters=self.cfg.clusters.n_clusters,
                    user_to_cluster=art.user_to_cluster,
                    cluster_centroids=np.zeros(
                        (self.cfg.clusters.n_clusters, 1), dtype=np.float32
                    ),
                    cluster_centroids_label_free=np.zeros(
                        (self.cfg.clusters.n_clusters, 1), dtype=np.float32
                    ),
                    embedding_model=art.embedding_model,
                    seed=art.seed,
                    fit_mode="oracle",
                    diagnostics=art.diagnostics,
                )
        elif geometry == "pred_profile":
            if not prompt:
                raise RuntimeError("pred_profile clustering requires the start prompt")
            art = self._fit_pred_profile_clusters(train, prompt, max_k=max_k)
        else:
            from prime.data.clustering import fit_style_clusters

            try:
                art = fit_style_clusters(
                    train,
                    self.cfg.clusters,
                    self.cfg.dataset,
                    fit_mode="label_free",
                    min_reviews_for_fit=roles.min_reviews_for_fit,
                    max_k=max_k,
                )
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

    def _fit_pred_profile_clusters(
        self,
        train: ReviewSplit,
        prompt: str,
        max_k: Optional[int] = None,
    ) -> ClusterArtifacts:
        """Seed-prompt ensemble on fit sources → pred_profile k-means (OBSERVATIONS C6)."""
        from prime.data.pred_profile_clusters import fit_pred_profile_as_cluster_artifacts

        ens, wp = self._ensemble_predict_split(train, prompt, cache_tag="cluster_fit_train")
        wp_arr = np.asarray(wp, dtype=np.int16) if wp is not None else None
        np.save(self.ctx.run_dir / "cluster_fit_predictions.npy", ens)
        if wp_arr is not None:
            np.save(self.ctx.run_dir / "cluster_fit_worker_predictions.npy", wp_arr)
        min_users = int(getattr(self.cfg.clusters, "min_users_per_cluster", 0) or 0)
        art = fit_pred_profile_as_cluster_artifacts(
            np.asarray(train.user_ids),
            ens,
            n_clusters=self.cfg.clusters.n_clusters,
            seed=self.cfg.clusters.seed,
            worker_preds=wp_arr,
            max_k=max_k,
            min_users_per_cluster=min_users,
        )
        merges = (art.diagnostics or {}).get("merges") or []
        sizes = (art.diagnostics or {}).get("cluster_sizes") or {}
        print(
            f"[GRAPE] pred_profile fit: K={art.n_clusters} users={len(art.user_to_cluster)} "
            f"examples={len(train)} min_users={min_users} sizes={sizes} "
            f"merges={len(merges)} (seed-prompt ensemble)",
            flush=True,
        )
        return art

    def _ensemble_predict_split(
        self, split: ReviewSplit, prompt: str, cache_tag: str
    ) -> tuple:
        """Run ensemble on a split; cache ens + worker votes under pred_cache/."""
        cache_dir = self.ctx.run_dir / "pred_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        ens_path = cache_dir / f"{cache_tag}_ens.npy"
        wp_path = cache_dir / f"{cache_tag}_workers.npy"
        if ens_path.is_file() and wp_path.is_file():
            ens = np.load(ens_path)
            wp = np.load(wp_path)
            if len(ens) == len(split):
                return ens, wp

        if self.use_mock:
            ens_list, wp = mock_predict(
                list(split.texts),
                list(split.labels),
                len(self.cfg.ensemble.workers),
                self.cfg.active_learning.seed,
                aggregation=self.cfg.ensemble.aggregation,
                label_space=self.cfg.dataset.label_space,
            )
            ens = np.asarray(ens_list)
            wp_arr = np.asarray(wp, dtype=np.int16)
        else:
            from prime.workers.ensemble import build_workers

            workers = build_workers(self.cfg.ensemble)
            ens_list, wp = parallel_predict(
                workers,
                list(split.texts),
                prompt,
                max_parallel=self.cfg.ensemble.max_parallel,
                tie_break=self.cfg.ensemble.tie_break,
                aggregation=self.cfg.ensemble.aggregation,
                label_space=self.cfg.dataset.label_space,
            )
            ens = np.asarray(ens_list)
            wp_arr = np.asarray(wp, dtype=np.int16)

        np.save(ens_path, ens)
        np.save(wp_path, wp_arr)
        return ens, wp_arr

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

    def _attach_split_clusters(
        self,
        split: ReviewSplit,
        art: ClusterArtifacts,
        prompt: Optional[str] = None,
    ) -> ReviewSplit:
        if art.fit_mode == "oracle" or getattr(self.cfg.clusters, "geometry", "") == "oracle":
            # Groups are per-comment and already attached at load time.
            if split.example_cluster_ids is None:
                raise RuntimeError(
                    f"oracle geometry: split '{split.name}' missing example_cluster_ids"
                )
            assign_path = self.ctx.run_dir / f"cluster_assign_{split.name}.json"
            mapping = {
                int(u): int(g) for u, g in zip(split.user_ids, split.example_cluster_ids)
            }
            assign_path.write_text(
                json.dumps({str(k): int(v) for k, v in mapping.items()}, indent=2),
                encoding="utf-8",
            )
            return split

        if art.fit_mode == "pred_profile":
            if not prompt:
                raise RuntimeError("pred_profile assignment requires the start prompt")
            if art.scaler_mean is None or art.scaler_scale is None:
                raise RuntimeError("pred_profile artifacts missing scaler")
            from prime.data.pred_profile_clusters import assign_pred_profile_clusters

            ens, wp = self._ensemble_predict_split(
                split, prompt, cache_tag=f"cluster_assign_{split.name}"
            )
            wp_arr = np.asarray(wp, dtype=np.int16) if wp is not None else None
            mapping = assign_pred_profile_clusters(
                np.asarray(split.user_ids),
                ens,
                art.cluster_centroids_label_free,
                art.scaler_mean,
                art.scaler_scale,
                worker_preds=wp_arr,
            )
            # Persist OOD assignments for audit
            assign_path = self.ctx.run_dir / f"cluster_assign_{split.name}.json"
            assign_path.write_text(
                json.dumps({str(k): int(v) for k, v in mapping.items()}, indent=2),
                encoding="utf-8",
            )
            return attach_example_clusters(split, mapping)

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

    def _fixed_sets_dir(self) -> Optional[Path]:
        raw = getattr(self.cfg.data_roles, "fixed_sets_dir", None)
        if not raw:
            return None
        p = Path(raw)
        if not p.is_absolute():
            p = self.project_root / p
        return p

    @staticmethod
    def _review_split_from_materialized(name: str, mat: Dict[str, Any]) -> ReviewSplit:
        return ReviewSplit(
            name=name,
            texts=list(mat["texts"]),
            labels=[int(x) for x in mat["labels"]],
            user_ids=list(mat["user_ids"]),
            example_cluster_ids=(
                [int(x) for x in mat["cluster_ids"]] if mat.get("cluster_ids") is not None else None
            ),
        )

    def _build_fixed_eval_sets(self, val: ReviewSplit, test: ReviewSplit) -> None:
        """Build / load D_dev (validation) and test_fixed (test) — ROADMAP §4.3."""
        roles = self.cfg.data_roles
        seed = self.cfg.active_learning.seed
        out_dir = self._fixed_sets_dir()
        self._dev_split = None
        self._test_fixed_split = None
        self._fixed_set_meta = {}

        if int(getattr(roles, "d_dev_size", 0) or 0) > 0 and bool(
            getattr(roles, "d_dev_from_validation", True)
        ):
            try:
                fs: Optional[FixedSet] = None
                if out_dir and (out_dir / "d_dev.json").is_file():
                    fs = FixedSet.load(out_dir / "d_dev.json")
                else:
                    fs = build_d_dev(val, seed=seed)
                    if out_dir:
                        out_dir.mkdir(parents=True, exist_ok=True)
                        fs.save(out_dir / "d_dev.json")
                mat = materialize_split(val, fs.indices)
                self._dev_split = self._review_split_from_materialized("d_dev", mat)
                self._fixed_set_meta["d_dev"] = fs.to_dict()
                (self.ctx.run_dir / "d_dev.json").write_text(
                    json.dumps(fs.to_dict(), indent=2), encoding="utf-8"
                )
            except Exception as exc:
                print(f"[GRAPE] D_dev build failed, selection falls back to val: {exc}", flush=True)

        if bool(getattr(roles, "build_test_fixed", True)) and self.cfg.dataset.name == "civilcomments":
            try:
                fs_t: Optional[FixedSet] = None
                if out_dir and (out_dir / "test_fixed.json").is_file():
                    fs_t = FixedSet.load(out_dir / "test_fixed.json")
                else:
                    fs_t = build_test_fixed(test, seed=seed)
                    if out_dir:
                        out_dir.mkdir(parents=True, exist_ok=True)
                        fs_t.save(out_dir / "test_fixed.json")
                mat_t = materialize_split(test, fs_t.indices)
                self._test_fixed_split = self._review_split_from_materialized("test_fixed", mat_t)
                self._fixed_set_meta["test_fixed"] = fs_t.to_dict()
                (self.ctx.run_dir / "test_fixed.json").write_text(
                    json.dumps(fs_t.to_dict(), indent=2), encoding="utf-8"
                )
            except Exception as exc:
                print(f"[GRAPE] test_fixed build failed, final uses full test: {exc}", flush=True)

        self.trace.stage(
            "04e_fixed_sets",
            "D_dev / test_fixed ready",
            d_dev=len(self._dev_split) if self._dev_split else 0,
            test_fixed=len(self._test_fixed_split) if self._test_fixed_split else 0,
            d_dev_fp=(self._fixed_set_meta.get("d_dev") or {}).get("fingerprint"),
            test_fixed_fp=(self._fixed_set_meta.get("test_fixed") or {}).get("fingerprint"),
        )

    def _build_eval_sets(self, heldout: ReviewSplit, prompt: str) -> EvalSets:
        """D_select (stratified, unbiased) + D_anchor (confidently solved cells)."""
        roles = self.cfg.data_roles
        seed = self.cfg.active_learning.seed

        # Reuse a pinned split when asked. D_select is deterministic, but D_anchor
        # depends on a live ensemble pass over heldout candidates, so two runs with
        # the same seed can still land on different sets (same root cause as the
        # cluster divergence in OBSERVATIONS O14). Indices are positions in the
        # heldout arrays, which are byte-identical across runs at equal caps/seed,
        # so pinning is safe — and it is required for a comparable pair.
        if roles.eval_sets_artifact:
            src = Path(roles.eval_sets_artifact)
            if not src.is_absolute():
                src = Path.cwd() / src
            pinned = EvalSets.load(src)
            if int(pinned.meta.get("heldout_examples", len(heldout))) != len(heldout):
                raise RuntimeError(
                    f"pinned eval_sets {src} was built on "
                    f"{pinned.meta.get('heldout_examples')} heldout examples but this "
                    f"run has {len(heldout)}; indices would not line up"
                )
            pinned.meta = {**pinned.meta, "pinned_from": str(src)}
            pinned.save(self.ctx.run_dir / "eval_sets.json")
            print(
                f"[GRAPE] loaded eval_sets_artifact: {src} "
                f"(hash={pinned.content_hash}, select={len(pinned.d_select)})",
                flush=True,
            )
            return pinned

        d_select = build_d_select(
            heldout,
            roles.d_select_size,
            seed=seed,
            stratify_by_label=bool(getattr(roles, "d_select_stratify_label", True)),
        )
        if bool(getattr(roles, "balanced_cells", False)):
            from prime.data.balanced_cells import sample_balanced_from_split
            from prime.data.civilcomments_loader import IDENTITY_VARS

            per_cell = int(getattr(roles, "balanced_per_cell", 45) or 45)
            include_none = bool(getattr(roles, "balanced_include_none", False))
            groups = list(range(0 if include_none else 1, 1 + len(IDENTITY_VARS)))
            try:
                d_select = sample_balanced_from_split(
                    heldout,
                    groups=groups,
                    per_cell=per_cell,
                    seed=seed,
                )
            except ValueError as exc:
                print(f"[GRAPE] balanced_cells fallback to stratified: {exc}", flush=True)

        n = len(heldout)
        excluded = set(d_select)
        candidates = [i for i in range(n) if i not in excluded]
        rng = np.random.RandomState(seed + 7)
        cap = min(len(candidates), roles.d_anchor_size * 3)
        if len(candidates) > cap:
            candidates = sorted(int(i) for i in rng.choice(candidates, size=cap, replace=False))

        # One ensemble pass with the initial prompt to find confidently solved examples.
        predictions: List[int] = [-1] * n
        disagreements: List[float] = [1.0] * n
        if candidates and roles.d_anchor_size > 0:
            texts = [heldout.texts[i] for i in candidates]
            labels = [heldout.labels[i] for i in candidates]
            if self.use_mock:
                ens, wp = mock_predict(
                    texts,
                    labels,
                    len(self.cfg.ensemble.workers),
                    seed,
                    aggregation=self.cfg.ensemble.aggregation,
                label_space=self.cfg.dataset.label_space,
                )
            else:
                from prime.workers.ensemble import build_workers

                workers = build_workers(self.cfg.ensemble)
                ens, wp = parallel_predict(
                    workers,
                    texts,
                    prompt,
                    max_parallel=self.cfg.ensemble.max_parallel,
                    tie_break=self.cfg.ensemble.tie_break,
                    aggregation=self.cfg.ensemble.aggregation,
                label_space=self.cfg.dataset.label_space,
                    fail_closed=bool(getattr(self.cfg.ensemble, "fail_closed", False)),
                )
            for j, i in enumerate(candidates):
                predictions[i] = int(ens[j])
                disagreements[i] = disagreement_score(
                    [wp[w][j] for w in range(len(wp))],
                    label_space=self.cfg.dataset.label_space,
                )

        max_d = float(roles.anchor_max_disagreement)
        if getattr(roles, "anchor_max_uncertainty", None) is not None:
            # Single-scorer path: disagreement is always 0 with 1 worker; uncertainty
            # threshold 0.0 means "require agreement" once self-consistency is wired.
            max_d = float(roles.anchor_max_uncertainty)
        d_anchor = build_d_anchor(
            heldout,
            d_select,
            predictions,
            disagreements,
            roles.d_anchor_size,
            seed=seed,
            max_disagreement=max_d,
        )
        d_audit = build_d_audit(
            heldout,
            excluded=list(d_select) + list(d_anchor),
            size=int(getattr(roles, "d_audit_size", 0) or 0),
            seed=seed,
            stratify_by_label=bool(getattr(roles, "d_select_stratify_label", True)),
        )
        d_dev: List[int] = []
        if int(getattr(roles, "d_dev_size", 0) or 0) > 0:
            d_dev = build_d_audit(
                heldout,
                excluded=list(d_select) + list(d_anchor) + list(d_audit),
                size=int(roles.d_dev_size),
                seed=seed + 99,
                stratify_by_label=True,
            )
        eval_sets = EvalSets(
            d_select=d_select,
            d_anchor=d_anchor,
            d_audit=d_audit,
            meta={
                "heldout_examples": n,
                "heldout_users": len(set(heldout.user_ids)),
                "anchor_candidates_evaluated": len(candidates),
                "seed": seed,
                "d_dev": d_dev,
                "d_dev_size": len(d_dev),
                "dev_gate_mode": getattr(roles, "dev_gate_mode", "off"),
                "stratify_by_label": bool(
                    getattr(roles, "d_select_stratify_label", True)
                ),
                "d_select_rotate": bool(getattr(roles, "d_select_rotate", False)),
            },
        )
        eval_sets.validate()
        eval_sets.save(self.ctx.run_dir / "eval_sets.json")
        return eval_sets

    def _rotate_d_select_for_cycle(self, heldout: ReviewSplit, cycle: int) -> None:
        """
        M17: re-sample D_select for this cycle; O26: re-score carried champions.
        D_anchor is frozen. Refreshes d_select_data.json and env path.
        """
        roles = self.cfg.data_roles
        prev_hash = self._eval_sets.content_hash
        self._eval_sets = rotate_d_select(
            heldout,
            self._eval_sets,
            d_select_size=roles.d_select_size,
            d_audit_size=int(getattr(roles, "d_audit_size", 0) or 0),
            seed=self.cfg.active_learning.seed,
            stratify_by_label=bool(getattr(roles, "d_select_stratify_label", True)),
            cycle=cycle,
        )
        self._select_data = split_arrays(heldout, self._eval_sets.d_select)
        select_path = self.ctx.run_dir / "d_select_data.json"
        select_path.write_text(json.dumps(self._select_data), encoding="utf-8")
        os.environ["PRIME_DSELECT_PATH"] = str(select_path.resolve())
        cycle_dir = self.ctx.cycle_dir(cycle)
        cycle_dir.mkdir(parents=True, exist_ok=True)
        self._eval_sets.save(cycle_dir / "eval_sets_rotated.json")
        self._eval_sets.save(self.ctx.run_dir / "eval_sets.json")

        # O26/M27: refresh fitness on carried front AND rewrite seed-checkpoint
        # metrics on disk. Memory-only rescore is not enough — OE resumes from
        # seed_c{N} combined_score and never re-evals seeded programs.
        rescored = 0
        for cand in list(self._carried_front or []):
            scored = self._score_on_select(cand.prompt)
            cand.fitness = float(scored.get("fitness", cand.fitness))
            cand.cluster_scores = _archive_cluster_scores(scored)
            cand.metrics = {
                **(cand.metrics or {}),
                "R_global": float(scored.get("R_global", 0.0)),
                "CVaR_cluster": float(scored.get("CVaR_cluster", 0.0)),
                "rescored_cycle": cycle,
            }
            rescored += 1

        seed_synced = 0
        if self._seed_checkpoint is not None and Path(self._seed_checkpoint).is_dir():
            seed_synced = sync_seed_checkpoint_metrics(
                Path(self._seed_checkpoint), self._carried_front or []
            )

        self.trace.stage(
            "04d_d_select_rotate",
            f"D_select rotated for cycle {cycle} (M17/O26/M27)",
            cycle=cycle,
            prev_hash=prev_hash,
            new_hash=self._eval_sets.content_hash,
            d_select=len(self._eval_sets.d_select),
            d_anchor=len(self._eval_sets.d_anchor),
            champions_rescored=rescored,
            seed_metrics_synced=seed_synced,
            seed_checkpoint=str(self._seed_checkpoint) if self._seed_checkpoint else None,
        )

    def _strip_cluster_addressed_rules(self, prompt: str) -> str:
        """Remove DynamicRules lines that violate O13 trigger discipline."""
        from prime.evolution.artifacts import _CLUSTER_RULE_RE
        from prime.evolution.prompt_blocks import replace_block

        body = extract_block(prompt, "DynamicRules")
        if body is None:
            return prompt
        kept: List[str] = []
        for line in body.splitlines():
            if _CLUSTER_RULE_RE.search(line.strip()):
                continue
            kept.append(line)
        new_inner = "\n".join(kept)
        if not new_inner.endswith("\n"):
            new_inner += "\n"
        return replace_block(prompt, "DynamicRules", new_inner)

    def _make_eval_only_evaluator(self) -> CandidateEvaluator:
        """Evaluator bound to D_select/D_anchor + prediction cache (no batch)."""
        return CandidateEvaluator(
            self.cfg,
            [],
            [],
            [],
            None,
            active_batch=None,
            use_mock=self.use_mock,
            actual_n_clusters=self._actual_n_clusters,
            select_data=self._select_data,
            cache_dir=self._pred_cache_dir,
        )

    def _score_on_select(self, prompt: str) -> Dict[str, Any]:
        """Fitness + per-cluster accuracy on D_select, via the prediction cache."""
        evaluator = self._make_eval_only_evaluator()
        return evaluator.evaluate_prompt(prompt)

    def _selection_key(self, val_metrics: Dict[str, Any]) -> tuple:
        """Cross-cycle heir key aligned with fitness.mode (Phase 2c / M26).

        Historical default was always (CVaR, global), which silently ignored
        fitness.mode=global. Arms now match their objective:
          global / macro     → (R_global|R_macro,)
          min_group_lex      → (R_worst_group, R_global|R_macro)
          soft_min_lex       → (R_soft_min_group|R_worst_group, R_global|R_macro)
          cvar_* / default   → (CVaR_*, R_global|R_macro)
        """
        global_acc = float(
            val_metrics.get("R_macro", 0.0)
            if self.cfg.fitness.class_balanced
            else val_metrics.get("R_global", 0.0)
        )
        mode = self.cfg.fitness.mode

        if mode in ("global", "macro"):
            if self.cfg.active_learning.selection_mode == "lexicographic":
                return (global_acc,)
            return (global_acc,)

        if mode == "min_group_lex":
            r_wg = float(val_metrics.get("R_worst_group", 0.0))
            if r_wg <= 0.0:
                # Fallback if metrics omitted R_worst_group
                for key in (
                    "cluster_accuracies_balanced_shrunk",
                    "cluster_accuracies_shrunk",
                    "cluster_accuracies",
                ):
                    accs = val_metrics.get(key) or {}
                    if isinstance(accs, dict) and accs:
                        r_wg = float(min(accs.values()))
                        break
            if self.cfg.active_learning.selection_mode == "lexicographic":
                return (r_wg, global_acc)
            score = (
                self.cfg.active_learning.selection_w_cvar * r_wg
                + self.cfg.active_learning.selection_w_global * global_acc
            )
            return (score,)

        if mode == "soft_min_lex":
            from prime.fitness.objective import soft_min_accuracies, _group_accs_for_lex

            r_soft = val_metrics.get("R_soft_min_gba", val_metrics.get("R_soft_min_group"))
            if r_soft is None:
                accs = _group_accs_for_lex(val_metrics, self.cfg.fitness)
                r_soft = soft_min_accuracies(accs, self.cfg.fitness.soft_min_tau) if accs else 0.0
            r_soft = float(r_soft)
            if getattr(self.cfg.fitness, "group_acc", "") == "balanced_within":
                global_acc = float(
                    val_metrics.get(
                        "R_gba_mean_shrunk",
                        val_metrics.get("R_gba_mean", global_acc),
                    )
                )
            if self.cfg.active_learning.selection_mode == "lexicographic":
                return (r_soft, global_acc)
            score = (
                self.cfg.active_learning.selection_w_cvar * r_soft
                + self.cfg.active_learning.selection_w_global * global_acc
            )
            return (score,)

        # Prefer the shrunk (and, under class balancing, class-balanced) cluster tail:
        # on the pair run's test split the raw CVaR had a bootstrap SD of 0.029 against
        # 0.022 for the shrunk one, and the raw version's argmin hops between clusters
        # on noise (OBSERVATIONS C8/M14).
        cvar = float(val_metrics.get("CVaR_cluster", 0.0))
        if self.cfg.active_learning.selection_use_shrunk_cvar:
            for key in ("CVaR_cluster_balanced_shrunk", "CVaR_cluster_shrunk"):
                if key in val_metrics:
                    cvar = float(val_metrics[key])
                    break
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
        # Sample candidate subset from unseen (oversample)
        candidates = sorted(pool.unseen)
        cap = min(len(candidates), max(n_add, int(n_add * self.cfg.acquisition.oversample_factor)))
        # F6: keep self-consistency affordable via pool_score_subsample.
        if self.cfg.acquisition.expansion_policy == "uncertainty":
            cap = min(cap, int(getattr(self.cfg.acquisition, "pool_score_subsample", cap) or cap))
        rng = np.random.RandomState(self.cfg.active_learning.seed + cycle + 17)
        if len(candidates) > cap:
            pick = sorted(int(i) for i in rng.choice(candidates, size=cap, replace=False))
        else:
            pick = candidates
        texts = [pool.texts[i] for i in pick]
        labels = [pool.labels[i] for i in pick]
        d_map: Dict[int, float] = {}
        if (
            self.cfg.acquisition.expansion_policy == "uncertainty"
            and getattr(self.cfg.ensemble, "mode", "ensemble") == "single"
        ):
            from prime.workers.scorer import Scorer

            scorer = Scorer(
                self.cfg.ensemble,
                label_space=self.cfg.dataset.label_space,
                use_mock=self.use_mock,
            )
            k = int(getattr(self.cfg.acquisition, "uncertainty_k", 3) or 3)
            temp = float(getattr(self.cfg.acquisition, "uncertainty_temperature", 0.7) or 0.7)
            unc, n_calls = scorer.self_consistency_parallel(
                texts, prompt, k=k, temperature=temp, max_parallel=self.cfg.ensemble.max_parallel
            )
            self._charge_scorer(n_calls, level="other", note=f"expand_uncertainty_c{cycle}")
            for j, idx in enumerate(pick):
                d_map[idx] = float(unc[j])
        else:
            if self.use_mock:
                _ens, wp = mock_predict(
                    texts,
                    labels,
                    len(self.cfg.ensemble.workers),
                    self.cfg.active_learning.seed + cycle,
                    aggregation=self.cfg.ensemble.aggregation,
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
                    aggregation=self.cfg.ensemble.aggregation,
                    label_space=self.cfg.dataset.label_space,
                )
            self._charge_scorer(len(texts) * max(1, len(self.cfg.ensemble.workers)), level="other")
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
                texts,
                labels,
                len(self.cfg.ensemble.workers),
                self.cfg.active_learning.seed + cycle,
                aggregation=self.cfg.ensemble.aggregation,
                label_space=self.cfg.dataset.label_space,
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
                aggregation=self.cfg.ensemble.aggregation,
                label_space=self.cfg.dataset.label_space,
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
        self._charge_scorer(
            len(texts) * max(1, len(self.cfg.ensemble.workers)),
            level="other",
            note=f"seen_infer_c{cycle}",
        )
        # F6: under single-scorer, ensemble disagreement is identically 0 — replace
        # with self-consistency uncertainty on a subsample of the seen pool.
        if (
            getattr(self.cfg.ensemble, "mode", "ensemble") == "single"
            and int(getattr(self.cfg.acquisition, "uncertainty_k", 0) or 0) > 1
        ):
            from prime.workers.scorer import Scorer

            scorer = Scorer(
                self.cfg.ensemble,
                label_space=self.cfg.dataset.label_space,
                use_mock=self.use_mock,
            )
            k = int(self.cfg.acquisition.uncertainty_k)
            temp = float(getattr(self.cfg.acquisition, "uncertainty_temperature", 0.7) or 0.7)
            sub_n = min(
                len(texts),
                int(getattr(self.cfg.acquisition, "pool_score_subsample", 1200) or 1200),
            )
            rng = np.random.RandomState(self.cfg.active_learning.seed + cycle + 31)
            if sub_n < len(texts):
                sub_pos = sorted(int(i) for i in rng.choice(len(texts), size=sub_n, replace=False))
            else:
                sub_pos = list(range(len(texts)))
            sub_texts = [texts[i] for i in sub_pos]
            unc, n_calls = scorer.self_consistency_parallel(
                sub_texts, prompt, k=k, temperature=temp, max_parallel=self.cfg.ensemble.max_parallel
            )
            self._charge_scorer(n_calls, level="other", note=f"seen_uncertainty_c{cycle}")
            disagreements = [0.0] * len(texts)
            for local_i, pos in enumerate(sub_pos):
                disagreements[pos] = float(unc[local_i])
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

        d_select_idxs = (
            list(self._eval_sets.d_select) if getattr(self, "_eval_sets", None) else None
        )
        d_anchor_idxs = (
            list(self._eval_sets.d_anchor) if getattr(self, "_eval_sets", None) else None
        )
        diag = build_batch_diagnostics(
            cycle=cycle,
            batch=batch,
            seen_indices=seen_idx,
            unseen_count=len(pool.unseen),
            pool_hard=sorted(pool.hard),
            pool_anchor=sorted(pool.anchor),
            predictions=ensemble,
            gold=labels,
            disagreements=disagreements,
            cluster_ids=cluster_ids,
            cluster_accuracies=cluster_accs,
            policy=self.cfg.acquisition.policy,
            d_select=d_select_idxs,
            d_anchor=d_anchor_idxs,
        )
        (cycle_dir / "batch_diagnostics.json").write_text(
            json.dumps(diag, indent=2), encoding="utf-8"
        )
        warn_bits = [
            k
            for k, v in (diag.get("representativeness") or {}).items()
            if k.startswith("warn_") and v
        ]
        print(
            f"[GRAPE] batch diag C{cycle}: seen={diag['pool']['n_seen']} "
            f"unseen={diag['pool']['n_unseen']} hard={diag['batch']['n_hard']} "
            f"anchor={diag['batch']['n_anchor']} "
            f"weak_slots={diag['representativeness']['weak_cluster_hard_slots']} "
            f"quotas={diag['group_hard_quotas'] or '{}'} "
            f"leak_Dsel={diag['role_integrity']['n_leak_d_select']} "
            f"warns={warn_bits or 'none'}",
            flush=True,
        )
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
            batch_diagnostics_path=str(cycle_dir / "batch_diagnostics.json"),
            warn_bits=warn_bits,
        )

        pool_data = {
            "texts": train.texts,
            "labels": train.labels,
            "user_ids": train.user_ids,
            "cluster_ids": train.example_cluster_ids,
        }
        (cycle_dir / "pool_data.json").write_text(json.dumps(pool_data), encoding="utf-8")
        os.environ["PRIME_POOL_PATH"] = str((cycle_dir / "pool_data.json").resolve())
        # Prefer fully-resolved snapshot (includes already merged). Raw
        # config_used.yaml breaks includes when loaded from run_dir (O28).
        resolved = self.ctx.run_dir / "config_resolved.yaml"
        cfg_for_oe = resolved if resolved.is_file() else self.ctx.run_dir / "config_used.yaml"
        os.environ["PRIME_CONFIG_PATH"] = str(cfg_for_oe.resolve())
        os.environ["PRIME_ACTIVE_BATCH_PATH"] = str((cycle_dir / "active_batch.json").resolve())
        os.environ["PRIME_USE_MOCK"] = "1" if self.use_mock else "0"
        os.environ["PRIME_N_CLUSTERS"] = str(self._actual_n_clusters)

        # SPEC v3 (Р6): default fitness from D_select when data roles are on.
        # Ablation `v1_weighted` routes to the Hard/Anchor active batch instead
        # (see CandidateEvaluator.evaluate_prompt) — authentic v1 objective.
        evaluator = CandidateEvaluator(
            self.cfg,
            train.texts,
            train.labels,
            train.user_ids,
            train.example_cluster_ids,
            active_batch=batch,
            use_mock=self.use_mock,
            actual_n_clusters=self._actual_n_clusters,
            select_data=self._select_data,
            cache_dir=self._pred_cache_dir,
        )
        result = evaluator.evaluate_prompt(prompt)
        self.trace.stage(
            "08_fitness",
            f"Fitness computed (cycle {cycle})",
            cycle=cycle,
            mode=self.cfg.fitness.mode,
            eval_set=result.get("eval_set", "batch_legacy"),
            fitness=round(float(result.get("fitness", 0.0)), 4),
            R_global=round(float(result.get("R_global", 0.0)), 4),
            Acc_Hard=round(float(result["Acc_Hard"]), 4) if "Acc_Hard" in result else None,
            Acc_Anchor=round(float(result["Acc_Anchor"]), 4) if "Acc_Anchor" in result else None,
            CVaR_cluster=round(float(result.get("CVaR_cluster", 0.0)), 4),
            CVaR_cluster_shrunk=round(float(result.get("CVaR_cluster_shrunk", 0.0)), 4),
            R_worst=round(float(result.get("R_worst", 0.0)), 4),
            mean_kappa=round(float(result.get("mean_kappa", 0.0)), 4),
            n_hard=result.get("n_hard"),
            n_anchor=result.get("n_anchor"),
        )

        # Mutator error artifacts from the batch view, reusing the seen-pool
        # ensemble pass (no extra API calls).
        artifacts_path: Optional[Path] = None
        if batch.get("indices"):
            pos_of = {idx: i for i, idx in enumerate(seen_idx)}
            b_pos = [pos_of[i] for i in batch["indices"] if i in pos_of]
            batch_texts = [train.texts[i] for i in batch["indices"] if i in pos_of]
            cids = (
                [train.example_cluster_ids[i] for i in batch["indices"] if i in pos_of]
                if train.example_cluster_ids
                else None
            )
            batch_pool_idx = [i for i in batch["indices"] if i in pos_of]
            err_text = format_error_artifacts(
                [ensemble[p] for p in b_pos],
                [labels[p] for p in b_pos],
                [[wp[w][p] for p in b_pos] for w in range(len(wp))],
                batch_texts,
                cluster_ids=cids,
                hard_indices=batch.get("hard_indices"),
                anchor_indices=batch.get("anchor_indices"),
                pool_indices=batch_pool_idx,
                prev_predictions=[
                    self._prev_example_preds.get(int(i), -1) for i in batch_pool_idx
                ],
                label_space=self.cfg.dataset.label_space,
                contrastive_pairs=bool(
                    getattr(self.cfg.evolution, "contrastive_pairs", True)
                ),
                contrastive_pair_limit=int(
                    getattr(self.cfg.evolution, "contrastive_pair_limit", 4)
                ),
            )
            # Remember this cycle's verdict per example so the next cycle can tell the
            # mutator what its last change broke (OBSERVATIONS C9: without attribution
            # it keeps stacking rules on top of a regression it cannot see).
            for idx, p in zip(seen_idx, ensemble):
                self._prev_example_preds[int(idx)] = int(p)
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

    def _charge_scorer(self, n_calls: int, *, level: str = "val", note: str = "") -> None:
        if n_calls <= 0:
            return
        try:
            self._call_budget.charge(level, n_calls=int(n_calls), kind="scorer", note=note)
        except BudgetExhausted:
            raise

    def _apply_dev_gate(
        self,
        candidate_prompt: str,
        cycle_start_prompt: str,
        cycle: int,
        cycle_dir: Path,
        *,
        select_fitness: Optional[float],
    ) -> tuple[str, Optional[Dict[str, Any]]]:
        """F8: reject heir if D_dev softmin drops more than delta vs champion."""
        mode = str(getattr(self.cfg.data_roles, "dev_gate_mode", "off") or "off")
        if mode == "off" or self._dev_split is None:
            return candidate_prompt, None
        if candidate_prompt == cycle_start_prompt and self._selected_prompt is None:
            # First-cycle incumbent: still score for gap reporting.
            champ_prompt = candidate_prompt
        else:
            champ_prompt = self._selected_prompt or cycle_start_prompt

        if candidate_prompt == champ_prompt:
            champ_m = self._eval_split(champ_prompt, self._dev_split)
            soft = softmin_from_metrics(champ_m, tau=self.cfg.fitness.soft_min_tau)
            gap = generalization_gap(select_fitness, soft)
            rec = {
                "cycle": cycle,
                "accepted": True,
                "reason": "candidate_is_champion",
                "mode": mode,
                "softmin_champion": soft,
                "softmin_candidate": soft,
                "drop": 0.0,
                "generalization_gap": gap,
            }
            self._dev_gate_history.append(rec)
            self._generalization_gaps.append(
                {"cycle": cycle, "select_fitness": select_fitness, "dev_softmin": soft, "gap": gap}
            )
            (cycle_dir / "dev_gate.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
            return candidate_prompt, rec

        champ_m = self._eval_split(champ_prompt, self._dev_split)
        cand_m = self._eval_split(candidate_prompt, self._dev_split)
        gate = evaluate_dev_gate(
            champion_metrics=champ_m,
            candidate_metrics=cand_m,
            delta=float(getattr(self.cfg.data_roles, "dev_gate_delta", 0.01)),
            tau=float(self.cfg.fitness.soft_min_tau),
            mode=mode,
        )
        gate["cycle"] = cycle
        soft_c = float(gate["softmin_candidate"])
        gap = generalization_gap(select_fitness, soft_c)
        gate["generalization_gap"] = gap
        self._generalization_gaps.append(
            {
                "cycle": cycle,
                "select_fitness": select_fitness,
                "dev_softmin": soft_c,
                "gap": gap,
                "accepted": gate["accepted"],
            }
        )
        self._dev_gate_history.append(gate)
        (cycle_dir / "dev_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
        self.trace.stage(
            "10c_dev_gate",
            f"D_dev generalization gate (cycle {cycle})",
            cycle=cycle,
            accepted=gate["accepted"],
            reason=gate["reason"],
            drop=gate["drop"],
            delta=gate["delta"],
            softmin_champion=gate["softmin_champion"],
            softmin_candidate=gate["softmin_candidate"],
            generalization_gap=gap,
        )
        if gate["accepted"]:
            return candidate_prompt, gate
        (cycle_dir / "rejected_by_dev_gate.txt").write_text(candidate_prompt, encoding="utf-8")
        return champ_prompt, gate

    def _pick_heir(
        self,
        front: List[Candidate],
        cycle_start_prompt: str,
        cycle: int,
        cycle_dir: Path,
    ) -> tuple[str, Optional[Dict[str, Any]]]:
        """
        Choose the prompt that leaves this cycle, best-first down the front.

        The heir is the scalar-best front member — group specialists are preserved
        by staying *in* the carried front, not by being promoted past a better
        prompt. Ties break toward the newer candidate, otherwise a stale front
        member could displace an equally good fresh one. Reaching the incumbent
        (the cycle-entry prompt) stops the walk: anything below it would be a
        regression on the objective.

        The anchor gate stays a hard constraint (P6), but a rejection no longer
        throws the cycle away — the next front member is tried first.
        """
        rejected: List[Dict[str, Any]] = []
        ranked = sorted(front, key=lambda c: (round(c.fitness, 6), c.cycle), reverse=True)
        order: List[Candidate] = []
        for cand in ranked:
            if cand.prompt == cycle_start_prompt:
                break
            order.append(cand)
        if not self.cfg.pareto.gate_fallback_down_front:
            order = order[:1]

        # In monitor mode the gate is still evaluated and logged but never blocks.
        # It has no operating point at the flip counts it actually sees: with 4
        # discordant anchor pairs the smallest achievable one-sided exact p is
        # 2^-4 = 0.0625 > alpha, so it mathematically cannot reject, and reading
        # `accepted: true` as "no regression" is a mistake (OBSERVATIONS M12).
        monitor_only = self.cfg.data_roles.anchor_gate_mode == "monitor"
        for rank, cand in enumerate(order):
            gate = self._anchor_gate_for(cand.prompt, cycle_start_prompt, cycle, rank, cand.source)
            if gate is not None:
                gate["mode"] = self.cfg.data_roles.anchor_gate_mode
            if gate is None or gate["accepted"] or monitor_only:
                if gate is not None:
                    self._anchor_gate_history.append(gate)
                    (cycle_dir / "anchor_gate.json").write_text(
                        json.dumps(
                            {**gate, "rejected_before_accept": rejected}, indent=2
                        ),
                        encoding="utf-8",
                    )
                    self._trace_anchor_gate(gate, cycle, cand.source, len(rejected))
                return cand.prompt, gate
            rejected.append(
                {
                    "source": cand.source,
                    "front_rank": rank,
                    "drop": gate["drop"],
                    "mcnemar_p_one_sided": gate["mcnemar_p_one_sided"],
                    "reason": gate["reason"],
                }
            )
            # Keep the rejected candidate: the anchor loss may later read as a
            # real tail trade-off rather than damage (OBSERVATIONS M8).
            (cycle_dir / f"rejected_by_anchor_gate_{rank}.txt").write_text(
                cand.prompt, encoding="utf-8"
            )
            self._anchor_gate_history.append(gate)

        if rejected:
            (cycle_dir / "anchor_gate.json").write_text(
                json.dumps({"all_rejected": rejected, "cycle": cycle}, indent=2),
                encoding="utf-8",
            )
            self.trace.stage(
                "10b_anchor_gate",
                f"D_anchor regression gate (cycle {cycle})",
                cycle=cycle,
                accepted=False,
                n_front_candidates_rejected=len(rejected),
                reason="all_better_candidates_rejected",
            )
        elif not order:
            self.trace.stage(
                "10b_anchor_gate",
                f"D_anchor regression gate (cycle {cycle})",
                cycle=cycle,
                accepted=True,
                reason="incumbent_is_front_best_no_gate_needed",
            )
        return cycle_start_prompt, None

    def _anchor_gate_for(
        self,
        candidate_prompt: str,
        reference_prompt: str,
        cycle: int,
        front_rank: int,
        source: str,
    ) -> Optional[Dict[str, Any]]:
        if not (self._anchor_data and self._anchor_data.get("texts")):
            return None
        anchor_eval = self._make_eval_only_evaluator()
        before = anchor_eval.evaluate_anchor(reference_prompt, self._anchor_data)
        after = anchor_eval.evaluate_anchor(candidate_prompt, self._anchor_data)
        gate = evaluate_anchor_gate(
            before_correct=before.get("correct", []),
            after_correct=after.get("correct", []),
            labels=self._anchor_data.get("labels", []),
            cluster_ids=self._anchor_data.get("cluster_ids"),
            delta=self.cfg.data_roles.anchor_gate_delta,
            alpha=self.cfg.data_roles.anchor_gate_alpha,
        )
        gate["cycle"] = cycle
        gate["front_rank"] = front_rank
        gate["candidate_source"] = source
        gate["rejected_prompt_len"] = len(candidate_prompt) if not gate["accepted"] else None
        return gate

    def _trace_anchor_gate(
        self, gate: Dict[str, Any], cycle: int, source: str, n_rejected_before: int
    ) -> None:
        self.trace.stage(
            "10b_anchor_gate",
            f"D_anchor regression gate (cycle {cycle})",
            cycle=cycle,
            candidate_source=source,
            front_rank=gate.get("front_rank"),
            n_rejected_before_accept=n_rejected_before,
            anchor_acc_before=gate["acc_before"],
            anchor_acc_after=gate["acc_after"],
            gate_delta=gate["delta_config"],
            delta_effective=gate["delta_effective"],
            n_improved=gate["n_improved"],
            n_worsened=gate["n_worsened"],
            mcnemar_p=gate["mcnemar_p_one_sided"],
            min_detectable_worsened=gate["min_detectable_worsened"],
            regression_concentration=gate["regression_concentration"],
            top_regressed_cell=gate["top_regressed_cell"],
            reason=gate["reason"],
            accepted=gate["accepted"],
        )

    @staticmethod
    def _reorder_front_for_heir(front: List[Candidate], heir_prompt: str) -> List[Candidate]:
        """Put the accepted heir first so it seeds the next OE population."""
        heir = [c for c in front if c.prompt == heir_prompt]
        rest = [c for c in front if c.prompt != heir_prompt]
        return heir + rest

    def _eval_split(self, prompt: str, split: ReviewSplit) -> Dict[str, Any]:
        cache_key = (prompt_hash(prompt), split.name, len(split.texts))
        cached = self._split_metric_cache.get(cache_key)
        if cached is not None:
            return cached
        metrics = self._eval_split_uncached(prompt, split)
        self._split_metric_cache[cache_key] = metrics
        return metrics

    def _eval_split_uncached(self, prompt: str, split: ReviewSplit) -> Dict[str, Any]:
        if self.use_mock:
            ensemble, wp = mock_predict(
                split.texts,
                split.labels,
                len(self.cfg.ensemble.workers),
                self.cfg.active_learning.seed,
                aggregation=self.cfg.ensemble.aggregation,
                label_space=self.cfg.dataset.label_space,
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
                aggregation=self.cfg.ensemble.aggregation,
                label_space=self.cfg.dataset.label_space,
                fail_closed=bool(getattr(self.cfg.ensemble, "fail_closed", False)),
            )
        level = {
            "d_dev": "val",
            "validation": "val",
            "test_fixed": "audit",
            "test": "audit",
        }.get(split.name, "val")
        self._charge_scorer(
            len(split.texts) * max(1, len(self.cfg.ensemble.workers)),
            level=level,
            note=f"eval_{split.name}",
        )
        from prime.fitness.metrics import compute_metrics
        from prime.fitness.objective import soft_min_accuracies, _group_accs_for_lex

        cluster_ids = np.array(split.example_cluster_ids) if split.example_cluster_ids else None
        metrics = compute_metrics(
            np.array(ensemble),
            np.array(split.labels),
            np.array(split.user_ids),
            worker_predictions=[np.array(w) for w in wp],
            cluster_ids=cluster_ids,
            cvar_quantile=self.cfg.fitness.cvar_quantile,
            beta_a=self.cfg.fitness.beta_a,
            beta_b=self.cfg.fitness.beta_b,
            tail_quantile=self.cfg.active_learning.tail_quantile,
            shrink_prior_weight=self.cfg.fitness.shrink_prior_weight,
            class_balanced=self.cfg.fitness.class_balanced,
            gba_min_pos=int(getattr(self.cfg.fitness, "gba_min_pos", 10)),
            gba_min_neg=int(getattr(self.cfg.fitness, "gba_min_neg", 10)),
            gba_exclude_none=bool(getattr(self.cfg.fitness, "gba_exclude_none", True)),
        )
        if getattr(self.cfg.fitness, "group_acc", "") == "balanced_within":
            # Softmin for selection may use shrunk GBA; never overwrite raw R_worst_gba
            # (M34: summary was publishing shrunk min as raw).
            accs = _group_accs_for_lex(metrics, self.cfg.fitness)
            if accs:
                metrics["R_soft_min_gba"] = float(
                    soft_min_accuracies(accs, self.cfg.fitness.soft_min_tau)
                )
                if "R_worst_gba" not in metrics:
                    metrics["R_worst_gba"] = float(min(accs.values()))
        # Persist raw votes for OOD splits so per-worker metrics can be recomputed
        # offline (OBSERVATIONS O3: without worker votes, disagreement/kappa die).
        if split.name in ("test", "validation", "test_fixed", "d_dev") and not self.use_mock:
            tag = f"eval_{split.name}"
            out = self.ctx.run_dir / "evals" / tag
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / "ensemble_predictions.npy", np.asarray(ensemble, dtype=np.int16))
            np.save(out / "worker_predictions.npy", np.asarray(wp, dtype=np.int16))
            np.save(out / "labels.npy", np.asarray(split.labels, dtype=np.int16))
            np.save(out / "user_ids.npy", np.asarray(split.user_ids))
            if cluster_ids is not None:
                np.save(out / "cluster_ids.npy", np.asarray(cluster_ids, dtype=np.int16))
            (out / "metrics.json").write_text(
                json.dumps(
                    {
                        "split": split.name,
                        "n_examples": len(split.texts),
                        "ensemble": {
                            k: metrics.get(k)
                            for k in [
                                "R_global",
                                "R_macro",
                                "R_worst",
                                "R_tail",
                                "R_worst_gba",
                                "R_gba_mean",
                                "R_soft_min_gba",
                                "toxic_recall",
                                "specificity",
                                "invalid_rate",
                                "pred_pos_rate",
                                "gba_eligible_groups",
                                "worst_gba_group",
                                "cluster_gba",
                                "cluster_gba_shrunk",
                                "CVaR_cluster",
                                "CVaR_cluster_shrunk",
                                "CVaR_cluster_balanced_shrunk",
                                "mae",
                                "cluster_accuracies",
                                "accuracy_per_class",
                                "mean_kappa",
                            ]
                        },
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
        return metrics


def run_from_config(config_path: Path) -> Dict[str, Any]:
    project_root = config_path.resolve().parents[1]
    if config_path.parent.name == "experiments":
        project_root = config_path.resolve().parents[2]
    cfg = load_config(config_path)
    return PrimeController(cfg, project_root, config_path).run()

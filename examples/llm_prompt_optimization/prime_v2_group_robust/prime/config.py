"""Typed configuration: YAML -> dataclasses with validation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class WorkerSpec:
    name: str
    temperature: float = 0.0
    max_tokens: int = 128
    # OpenRouter reasoning.effort for rating workers; "none" avoids empty
    # content when CoT burns the completion budget (DeepSeek V4 / GLM-5).
    reasoning_effort: Optional[str] = "none"


@dataclass
class EnsembleCfg:
    api_base: str = "https://openrouter.ai/api/v1"
    workers: List[WorkerSpec] = field(default_factory=list)
    max_parallel: int = 8
    timeout: int = 75
    max_retries: int = 4
    # SPEC v3 (Р8): median is consistent for ordinal scale, no tie-break needed.
    aggregation: str = "median"  # median | majority
    tie_break: str = "lowest_rating"  # only used when aggregation=majority
    # Synced from dataset.label_space in load_config (ordinal5 | binary).
    label_space: str = "ordinal5"
    # Phase3 F5: "single" = one scorer; "ensemble" = legacy multi-worker.
    mode: str = "ensemble"  # ensemble | single
    # Phase3 F1: API/parse failures → INVALID (-1), never silent majority class.
    fail_closed: bool = False


@dataclass
class DatasetCfg:
    name: str = "amazon"
    data_root: str = "./data"
    category_id: Optional[int] = None  # None = all categories (all splits, legacy)
    # Category-shift (E10): train on one category; eval excludes it (or more).
    # When set, overrides `category_id` for that role. Books = 0 in WILDS Amazon.
    train_category_id: Optional[int] = None
    eval_exclude_category_ids: Optional[List[int]] = None
    min_user_reviews: int = 1
    max_train_users: Optional[int] = None
    max_val_users: Optional[int] = None
    max_test_users: Optional[int] = None
    max_reviews_per_user: Optional[int] = None
    al_candidate_pool_size: int = 5000
    cluster_artifact: Optional[str] = None  # path to serialized cluster mapping
    cache_dir: Optional[str] = None  # default: {data_root}/.prime_cache
    use_cache: bool = True
    # ordinal5 = Amazon 1–5; binary = CivilComments toxicity {0, 1}
    label_space: str = "ordinal5"


@dataclass
class ClusterCfg:
    n_clusters: int = 8
    embedding_model: str = "all-MiniLM-L6-v2"
    seed: int = 42
    pca_components: int = 32
    allow_synthetic_fallback: bool = False
    # style = label-free text features (legacy E0 default)
    # pred_profile = k-means on ensemble prediction mix (E0 large-cap winner)
    # oracle = identity / metadata groups attached at load (CivilComments E4)
    geometry: str = "style"  # style | pred_profile | oracle
    # After k-means, merge clusters with fewer than this many fit users into the
    # nearest surviving centroid (OBSERVATIONS C7). 0 disables.
    min_users_per_cluster: int = 0
    # SPEC Р4 / Р13: type descriptors + random-clusters ablation control.
    control: str = "none"  # none | shuffle
    descriptors_enabled: bool = True
    descriptors_top_n: int = 5
    descriptors_llm_editor: bool = False


@dataclass
class BudgetCfg:
    total_calls: int = 60_000
    on_exhausted: str = "stop"  # stop | degrade | warn
    # Phase3 F10: explicit scorer/optimizer caps (calls, not tokens).
    scorer_calls: Optional[int] = None  # None → use total_calls
    optimizer_calls: Optional[int] = None


@dataclass
class FitnessCfg:
    # SPEC v3 §4.4: cvar_lex = CVaR_shrunk + eps*global - len penalty (main mode).
    # cvar (weighted mix, no kappa) and global / v1_weighted remain as ablation arms.
    # global_tail_mix = w_global_mix*R_global + w_tail*R_tail - length (E2 default).
    # min_group_lex = min Acc_g (+shrunk) + eps*global - length (Phase 2c arm B).
    # soft_min_lex = log-sum-exp softmin Acc_g (+shrunk) + eps*global - length;
    #   lower variance than hard min so D_select rotation is usable again (M28).
    mode: str = "cvar_lex"  # cvar_lex | cvar | global | global_tail_mix | macro | v1_weighted | min_group_lex | soft_min_lex
    epsilon_global: float = 0.01  # lexicographic tie-break weight in cvar_lex / min_group_lex / soft_min_lex
    beta_a: float = 1.0  # legacy Beta-prior smoothing (superseded by shrink_prior_weight)
    beta_b: float = 1.0
    w_cvar: float = 0.5  # legacy weighted mode only
    w_global: float = 0.3  # legacy weighted mode only
    w_kappa: float = 0.2  # v1_weighted ablation only (SPEC v3 §4.2: kappa out of fitness)
    w_global_mix: float = 0.5  # global_tail_mix weight on R_global
    w_tail: float = 0.5  # global_tail_mix weight on R_tail
    tail_quantile: float = 0.2  # worst-user fraction for R_tail in fitness
    cvar_quantile: float = 0.33  # worst third of clusters
    # Softmin temperature for soft_min_lex (accuracies in [0,1]). tau→0 → hard min;
    # tau→∞ → mean. ~0.08 keeps pressure on the floor without single-group hops.
    soft_min_tau: float = 0.08
    # Shrink per-cluster accuracy toward the grand mean with a prior worth this many
    # pseudo-examples. Beta(1,1) moved a 48-example cluster by ~4%, i.e. the "shrunk"
    # metric was not shrunk at all; measured on the pair run, a real prior cuts the
    # tail statistic's bootstrap SD by 39% (OBSERVATIONS M14). 0 restores the legacy
    # Beta path for reproducing historical runs.
    shrink_prior_weight: float = 60.0
    # Weight examples so gold classes are equal before computing accuracy / CVaR.
    # Blocks the ordinal-threshold exploit of OBSERVATIONS C11: shifting 4/5 calls
    # scores under raw accuracy but is neutral here, so the search has to find real
    # discrimination.
    class_balanced: bool = True
    # Phase3 F3/F4: which group accuracy feeds min/soft_min lex.
    # raw | balanced_global (legacy class weights) | balanced_within (GBA).
    group_acc: str = "balanced_global"
    gba_min_pos: int = 10
    gba_min_neg: int = 10
    gba_exclude_none: bool = True
    # Phase3 F1/F2 guards.
    fail_closed: bool = False
    max_invalid_rate: float = 0.02
    min_pred_pos_rate: float = 0.02
    prompt_len_limit: int = 8000
    # Word-count length penalty (estimate_tokens). Defaults were effectively off for
    # CivilComments (seed≈211 words, evolved≈390); post-M28 forms use start≈250.
    len_penalty_per_100: float = 0.01
    len_penalty_start: int = 2000


@dataclass
class DataRolesCfg:
    """SPEC v3 §4.3 data roles: fit/heldout sources, D_select, D_anchor, D_audit."""

    enabled: bool = True
    fit_fraction: float = 0.7  # Р7: 70% fit / 30% heldout by user_id
    stratify: bool = True  # Р15: stratify fit/heldout by length (+ optional PCA-1)
    d_select_size: int = 400  # Р9 full run; capped E1 overrides to 200
    d_anchor_size: int = 100  # Р9 full run; capped E1 overrides to 50
    d_audit_size: int = 100  # Р15 nested-validation slice; capped E1: 50
    # Phase3 F8: heldout generalization gate (disjoint from D_select/anchor/test).
    d_dev_size: int = 0
    n_min_per_group: int = 40  # power rule: K <= |D_select| / n_min (Р9)
    min_reviews_for_fit: int = 3  # users below threshold are assigned, not fitted (Р9)
    anchor_max_disagreement: float = 0.2  # "confidently solved": err=0 and d <= this
    # Phase3 F6: when using self-consistency, max uncertainty for anchor membership.
    # 0.0 means all k votes agree. None → fall back to anchor_max_disagreement.
    anchor_max_uncertainty: Optional[float] = None
    # Р15 gate is a CI/δ *tolerance*, not strict no-regression: δ is floored at
    # 2 examples and a drop beyond it must also be significant at alpha
    # (exact one-sided McNemar) to reject. See prime/experiment/anchor_gate.py.
    # "monitor" evaluates and logs the gate without blocking; "reject" restores the
    # hard constraint. Monitor is the default because the gate cannot reach alpha at
    # our anchor flip counts (OBSERVATIONS M12) — the anchor *trend* is still the
    # most informative in-run regression signal we have, so it keeps being computed.
    anchor_gate_mode: str = "monitor"  # monitor | reject
    anchor_gate_delta: float = 0.02
    anchor_gate_alpha: float = 0.05
    # Dev-gate (Phase3 F8): promote heir only if D_dev softmin does not drop > delta.
    dev_gate_mode: str = "off"  # off | monitor | reject
    dev_gate_delta: float = 0.01
    # When True, build D_dev from WILDS validation via balanced cells (§4.3),
    # not from train heldout. Cross-cycle selection uses D_dev softmin.
    d_dev_from_validation: bool = True
    d_dev_per_cell: int = 50
    d_dev_include_none: bool = True
    # Final OOD headline set (never used for selection).
    build_test_fixed: bool = True
    test_fixed_per_cell: int = 100
    test_fixed_include_none: bool = True
    fixed_sets_dir: Optional[str] = None  # load/save fingerprints if set
    # Path to an existing eval_sets.json to reuse instead of rebuilding. D_anchor
    # construction needs a live ensemble pass, so two runs at the same seed can
    # still get different sets; both arms of a pair must share one (O14 analogue).
    eval_sets_artifact: Optional[str] = None
    # OBSERVATIONS M17: re-sample D_select each AL cycle (group×class stratified)
    # so repeated selection cannot overfit a fixed bandit set. D_anchor stays fixed.
    # When True, O26/M27 require re-scoring carried champions AND rewriting seed
    # checkpoint metrics on the new D_select before OE resume.
    # For hard min_group_lex on CivilComments prefer False (M28: rotation noise ≫
    # between-candidate signal). soft_min_lex / cvar_lex can keep True (M27 sync
    # required) — anti-overfit via rotated D_select + gate + val.
    # Phase3 E5 default: False + D_dev gate (M28 vs M30).
    d_select_rotate: bool = True
    d_select_stratify_label: bool = True
    # Phase3 F9: when True, build D_select via equal cells (group×label).
    balanced_cells: bool = False
    balanced_per_cell: int = 45
    balanced_include_none: bool = False


@dataclass
class AcquisitionCfg:
    policy: str = "lexicographic"  # random|qbc_d|hardest|lexicographic|group_aware
    batch_size: int = 80
    hard_ratio: float = 0.7
    oversample_factor: float = 3.0
    n_diversity_clusters: int = 8
    expansion_trigger: int = 25
    expansion_batch: int = 40
    expansion_policy: str = "disagreement"  # disagreement | farthest | uncertainty
    group_quota_enabled: bool = True
    # Phase3 F6 self-consistency uncertainty (single-scorer replacement for QBC).
    uncertainty_k: int = 3
    uncertainty_temperature: float = 0.7
    pool_score_subsample: int = 1200


@dataclass
class EvolutionCfg:
    n_evolve_iterations: int = 15
    population_size: int = 30
    archive_size: int = 500
    mutator_model: str = "z-ai/glm-5"
    mutator_temperature: float = 0.6
    mutator_max_tokens: int = 8192
    structured_prompt: bool = True
    synthetic_fewshot: bool = False
    # OBSERVATIONS O22: mechanically overwrite <FewShotExamples> with verbatim
    # failing examples from the cycle error report (do not trust the LLM to quote).
    inject_verbatim_fewshot: bool = True
    fewshot_inject_limit: int = 4
    # OBSERVATIONS O13: reject DynamicRules that name cluster/group identifiers.
    enforce_text_triggers: bool = True
    # Contrastive FAIL vs OK same-group pairs in mutator artifacts.
    contrastive_pairs: bool = True
    contrastive_pair_limit: int = 4


@dataclass
class ConsolidationCfg:
    enabled: bool = True
    # Strong non-reasoning-heavy model; consolidation is called once per cycle so
    # its cost is negligible. glm-5 (the mutator model) consistently returned
    # empty replies here and degraded to the stub (OBSERVATIONS O16/O17).
    model: str = "google/gemini-3.1-pro-preview"
    top_k_archive: int = 3
    pool_carryover_k: int = 3
    cluster_pareto: bool = False
    gate_delta: float = 0.01
    # Consolidation is an unguided rewrite of a prompt that search has already tuned,
    # so its expected fitness delta is negative and it lost 6/6 times in the pair run.
    # It stays enabled because as a *competitor* it cannot damage the heir, but it
    # only makes sense once rules have accumulated — hence every other cycle plus a
    # guaranteed final pass (OBSERVATIONS O21).
    every_n_cycles: int = 2
    run_on_last_cycle: bool = True
    scope: str = "base_guidelines"  # base_guidelines | full_prompt


@dataclass
class ParetoCfg:
    """
    Cross-cycle Pareto front over per-cluster accuracy on D_select.

    OBSERVATIONS O9/O10/O11: with a single heir the consolidated prompt inherited
    unconditionally and cost 0.015 / 0.066 of D_select fitness on the completed
    cvar_lex run. Here every candidate — cycle-entry prompt, each OE program, the
    consolidation variant — is scored on the same D_select and competes.
    """

    enabled: bool = True
    front_k: int = 4  # legacy cap for the dominance front; unused by champion_archive
    # P9: champion archive with fixed slots (1 scalar-best + one per cluster).
    # An incumbent cluster champion keeps its slot unless a challenger beats its
    # shrunk score by more than this margin (anti-churn; scores are Beta-shrunk).
    champion_margin: float = 0.01
    # Score the consolidation candidate on D_select and let it compete instead of
    # gating it against a val number in a different metric (the old O10 defect).
    score_consolidated_on_select: bool = True
    # Resume OpenEvolve from the carried front so the mutator sees the whole
    # population, not just the heir. Falls back to a fresh start on any error.
    seed_openevolve_from_front: bool = True
    # If the anchor gate rejects the heir, try the next front member before
    # falling back to the cycle-entry prompt.
    gate_fallback_down_front: bool = True


@dataclass
class ActiveLearningCfg:
    n_cycles: int = 6
    seed: int = 42
    selection_mode: str = "lexicographic"  # lexicographic | weighted
    selection_w_cvar: float = 0.7
    selection_w_global: float = 0.3
    proxy_validation_min_cycles: int = 2
    # Which tail statistic to track against CVaR in proxy validation.
    # R_worst (10th percentile of per-user accuracy) is pinned to the 1/8 lattice at
    # 8 reviews/user and stayed at exactly 0.500 for three cycles, which also makes
    # Spearman undefined (M9). R_tail moves continuously but is the *worst*
    # discriminator we measured (|effect|/SD = 0.48 vs 1.49 for the shrunk cluster
    # tail — M14). Default is the shrunk cluster tail for that reason.
    proxy_tail_metric: str = "CVaR_cluster_shrunk"  # CVaR_cluster_shrunk | R_tail | R_worst
    tail_quantile: float = 0.2
    # Selection on val uses the shrunk (and, when fitness.class_balanced, balanced)
    # cluster tail rather than the raw CVaR: same reason as above.
    selection_use_shrunk_cvar: bool = True
    # Phase3 §6.3: Top-1 final selection on D_dev softmin (not full WILDS val).
    # "d_dev" | "validation" (legacy full val).
    selection_split: str = "d_dev"


@dataclass
class ExperimentCfg:
    name: str = "run"
    results_dir: Optional[str] = None
    smoke: bool = False
    smoke_n_cycles: int = 1
    smoke_n_evolve: int = 2
    smoke_max_examples: int = 200
    verbose: bool = False
    smoke_validate: bool = False  # write stage checklist + extra demos
    force_mock: bool = False  # skip LLM calls even if API key is set
    # After final test_fixed eval: re-score seed + final this many times and write
    # evals/stable_test_fixed/ (mean±SD, McNemar, bootstrap). 0 = skip.
    # E5 default 3 — single-shot GBA headlines are not variance-aware (M38).
    stable_final_repeats: int = 0


@dataclass
class PrimeConfig:
    dataset: DatasetCfg = field(default_factory=DatasetCfg)
    clusters: ClusterCfg = field(default_factory=ClusterCfg)
    ensemble: EnsembleCfg = field(default_factory=EnsembleCfg)
    data_roles: DataRolesCfg = field(default_factory=DataRolesCfg)
    fitness: FitnessCfg = field(default_factory=FitnessCfg)
    acquisition: AcquisitionCfg = field(default_factory=AcquisitionCfg)
    evolution: EvolutionCfg = field(default_factory=EvolutionCfg)
    consolidation: ConsolidationCfg = field(default_factory=ConsolidationCfg)
    pareto: ParetoCfg = field(default_factory=ParetoCfg)
    active_learning: ActiveLearningCfg = field(default_factory=ActiveLearningCfg)
    experiment: ExperimentCfg = field(default_factory=ExperimentCfg)
    budget: BudgetCfg = field(default_factory=BudgetCfg)
    prompt_path: str = "prompts/initial_prompt.txt"
    openevolve_config_path: Optional[str] = None

    def validate(self) -> None:
        if self.acquisition.hard_ratio < 0 or self.acquisition.hard_ratio > 1:
            raise ValueError("acquisition.hard_ratio must be in [0, 1]")
        if self.fitness.mode not in (
            "cvar_lex",
            "cvar",
            "global",
            "global_tail_mix",
            "macro",
            "v1_weighted",
            "min_group_lex",
            "soft_min_lex",
        ):
            raise ValueError(f"Unknown fitness.mode: {self.fitness.mode}")
        if self.fitness.shrink_prior_weight < 0:
            raise ValueError("fitness.shrink_prior_weight must be >= 0")
        if self.fitness.soft_min_tau < 0:
            raise ValueError("fitness.soft_min_tau must be >= 0")
        if self.fitness.group_acc not in ("raw", "balanced_global", "balanced_within"):
            raise ValueError("fitness.group_acc must be raw|balanced_global|balanced_within")
        if self.fitness.group_acc == "balanced_within" and self.dataset.label_space != "binary":
            raise ValueError("fitness.group_acc=balanced_within requires dataset.label_space=binary")
        if not (0.0 < self.fitness.tail_quantile <= 1.0):
            raise ValueError("fitness.tail_quantile must be in (0, 1]")
        if self.fitness.mode == "global_tail_mix":
            w_sum = float(self.fitness.w_global_mix) + float(self.fitness.w_tail)
            if w_sum <= 0:
                raise ValueError("global_tail_mix weights must sum to > 0")
        # A tail quantile that selects a single cluster is not a CVaR, it is an argmin,
        # and the argmin hops between clusters on noise (OBSERVATIONS C8/M14).
        n_tail_clusters = math.ceil(self.fitness.cvar_quantile * self.clusters.n_clusters)
        if self.fitness.mode in ("cvar_lex", "cvar") and n_tail_clusters < 2:
            raise ValueError(
                f"fitness.cvar_quantile={self.fitness.cvar_quantile} with "
                f"clusters.n_clusters={self.clusters.n_clusters} averages "
                f"{n_tail_clusters} cluster(s); CVaR needs >= 2. Raise cvar_quantile "
                f"to at least {2 / self.clusters.n_clusters:.2f}."
            )
        if self.ensemble.mode not in ("ensemble", "single"):
            raise ValueError("ensemble.mode must be ensemble or single")
        if self.ensemble.mode == "single" and len(self.ensemble.workers) != 1:
            raise ValueError("ensemble.mode=single requires exactly one worker")
        if self.ensemble.mode == "single" and self.fitness.mode == "v1_weighted":
            raise ValueError("v1_weighted requires multi-worker kappa; incompatible with single scorer")
        if self.ensemble.aggregation not in ("median", "majority"):
            raise ValueError("ensemble.aggregation must be median or majority")
        if not (0.0 < self.data_roles.fit_fraction < 1.0):
            raise ValueError("data_roles.fit_fraction must be in (0, 1)")
        if self.data_roles.d_select_size < 1 or self.data_roles.d_anchor_size < 0:
            raise ValueError("data_roles sizes must be positive")
        if self.data_roles.d_audit_size < 0:
            raise ValueError("data_roles.d_audit_size must be >= 0")
        if self.data_roles.d_dev_size < 0:
            raise ValueError("data_roles.d_dev_size must be >= 0")
        if self.data_roles.n_min_per_group < 1:
            raise ValueError("data_roles.n_min_per_group must be >= 1")
        if self.data_roles.anchor_gate_mode not in ("monitor", "reject"):
            raise ValueError("data_roles.anchor_gate_mode must be monitor or reject")
        if self.data_roles.dev_gate_mode not in ("off", "monitor", "reject"):
            raise ValueError("data_roles.dev_gate_mode must be off|monitor|reject")
        policies = ("random", "qbc_d", "hardest", "lexicographic", "group_aware")
        if self.acquisition.policy not in policies:
            raise ValueError(f"acquisition.policy must be one of {policies}")
        if self.acquisition.expansion_policy not in ("disagreement", "farthest", "uncertainty"):
            raise ValueError("acquisition.expansion_policy must be disagreement|farthest|uncertainty")
        if self.acquisition.policy == "qbc_d" and self.ensemble.mode == "single":
            raise ValueError("acquisition.policy=qbc_d requires ensemble.mode=ensemble")
        if self.active_learning.selection_mode not in ("lexicographic", "weighted"):
            raise ValueError("active_learning.selection_mode must be lexicographic or weighted")
        if self.clusters.pca_components < 1:
            raise ValueError("clusters.pca_components must be >= 1")
        if self.clusters.control not in ("none", "shuffle"):
            raise ValueError("clusters.control must be none or shuffle")
        if self.clusters.geometry not in ("style", "pred_profile", "oracle"):
            raise ValueError("clusters.geometry must be style, pred_profile, or oracle")
        if self.dataset.label_space not in ("ordinal5", "binary"):
            raise ValueError("dataset.label_space must be ordinal5 or binary")
        if self.dataset.name == "civilcomments" and self.dataset.label_space != "binary":
            raise ValueError("civilcomments requires dataset.label_space=binary")
        if self.clusters.geometry == "oracle" and self.fitness.mode in ("cvar_lex", "cvar"):
            # Oracle CivilComments has 9 primary groups (none + 8 identities).
            n_tail = math.ceil(self.fitness.cvar_quantile * self.clusters.n_clusters)
            if n_tail < 2:
                raise ValueError(
                    "oracle CVaR needs clusters.n_clusters large enough that "
                    f"cvar_quantile selects >= 2 groups (got {n_tail})"
                )
        if self.consolidation.every_n_cycles < 1:
            raise ValueError("consolidation.every_n_cycles must be >= 1")
        if self.consolidation.scope not in ("base_guidelines", "full_prompt"):
            raise ValueError("consolidation.scope must be base_guidelines or full_prompt")
        if self.pareto.front_k < 1:
            raise ValueError("pareto.front_k must be >= 1")
        tail_metrics = ("R_tail", "R_worst", "CVaR_cluster_shrunk")
        if self.active_learning.proxy_tail_metric not in tail_metrics:
            raise ValueError(f"active_learning.proxy_tail_metric must be one of {tail_metrics}")
        if not (0.0 < self.active_learning.tail_quantile <= 1.0):
            raise ValueError("active_learning.tail_quantile must be in (0, 1]")
        if self.budget.total_calls < 1:
            raise ValueError("budget.total_calls must be >= 1")
        if self.budget.on_exhausted not in ("stop", "degrade", "warn"):
            raise ValueError("budget.on_exhausted must be stop|degrade|warn")


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k == "includes":
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml_tree(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    includes = raw.pop("includes", []) or []
    merged: Dict[str, Any] = {}
    for inc in includes:
        inc_path = (path.parent / str(inc)).resolve()
        if not inc_path.is_file():
            inc_path = (path.parent.parent / "configs" / str(inc)).resolve()
        child = _load_yaml_tree(inc_path) if inc_path.is_file() else {}
        merged = _merge_dict(merged, child)
    return _merge_dict(merged, raw)


def _dict_to_worker_spec(d: Dict[str, Any]) -> WorkerSpec:
    effort = d.get("reasoning_effort", "none")
    if effort is not None:
        effort = str(effort)
    return WorkerSpec(
        name=str(d["name"]),
        temperature=float(d.get("temperature", 0.0)),
        max_tokens=int(d.get("max_tokens", 128)),
        reasoning_effort=effort,
    )


def load_config(path: Path, overrides: Optional[Dict[str, Any]] = None) -> PrimeConfig:
    """Load YAML config; optional dict overrides merged recursively."""
    raw = _load_yaml_tree(path)
    if overrides:
        raw = _merge_dict(raw, overrides)

    ds = raw.get("dataset", {})
    cl = raw.get("clusters", {})
    ens = raw.get("ensemble", {})
    roles = raw.get("data_roles", {})
    fit = raw.get("fitness", {})
    acq = raw.get("acquisition", {})
    evo = raw.get("evolution", {})
    cons = raw.get("consolidation", {})
    par = raw.get("pareto", {})
    al = raw.get("active_learning", {})
    exp = raw.get("experiment", {})
    bud = raw.get("budget", {})

    workers_raw = ens.get("workers", [])
    workers = [_dict_to_worker_spec(w) for w in workers_raw]
    if not workers:
        workers = [
            WorkerSpec("deepseek/deepseek-v4-pro"),
            WorkerSpec("moonshotai/kimi-k2.5"),
            WorkerSpec("qwen/qwen3-235b-a22b-2507"),
        ]

    cfg = PrimeConfig(
        dataset=DatasetCfg(**{k: ds[k] for k in DatasetCfg.__dataclass_fields__ if k in ds}),
        clusters=ClusterCfg(**{k: cl[k] for k in ClusterCfg.__dataclass_fields__ if k in cl}),
        ensemble=EnsembleCfg(
            api_base=ens.get("api_base", "https://openrouter.ai/api/v1"),
            workers=workers,
            max_parallel=int(ens.get("max_parallel", 8)),
            timeout=int(ens.get("timeout", 75)),
            max_retries=int(ens.get("max_retries", 4)),
            aggregation=str(ens.get("aggregation", "median")),
            tie_break=str(ens.get("tie_break", "lowest_rating")),
            label_space=str(ds.get("label_space", ens.get("label_space", "ordinal5"))),
            mode=str(ens.get("mode", "ensemble")),
            fail_closed=bool(ens.get("fail_closed", False)),
        ),
        data_roles=DataRolesCfg(
            **{k: roles[k] for k in DataRolesCfg.__dataclass_fields__ if k in roles}
        ),
        fitness=FitnessCfg(**{k: fit[k] for k in FitnessCfg.__dataclass_fields__ if k in fit}),
        acquisition=AcquisitionCfg(
            **{k: acq[k] for k in AcquisitionCfg.__dataclass_fields__ if k in acq}
        ),
        evolution=EvolutionCfg(**{k: evo[k] for k in EvolutionCfg.__dataclass_fields__ if k in evo}),
        consolidation=ConsolidationCfg(
            **{k: cons[k] for k in ConsolidationCfg.__dataclass_fields__ if k in cons}
        ),
        pareto=ParetoCfg(**{k: par[k] for k in ParetoCfg.__dataclass_fields__ if k in par}),
        active_learning=ActiveLearningCfg(
            **{k: al[k] for k in ActiveLearningCfg.__dataclass_fields__ if k in al}
        ),
        experiment=ExperimentCfg(**{k: exp[k] for k in ExperimentCfg.__dataclass_fields__ if k in exp}),
        budget=BudgetCfg(**{k: bud[k] for k in BudgetCfg.__dataclass_fields__ if k in bud}),
        prompt_path=str(raw.get("prompt_path", "prompts/initial_prompt.txt")),
        openevolve_config_path=raw.get("openevolve_config_path"),
    )
    # Keep ensemble parsers in sync with the dataset label space.
    cfg.ensemble.label_space = cfg.dataset.label_space
    # Sync fail_closed: ensemble OR fitness flag enables both.
    if cfg.fitness.fail_closed or cfg.ensemble.fail_closed:
        cfg.ensemble.fail_closed = True
        cfg.fitness.fail_closed = True
    cfg.validate()
    return cfg

"""Typed configuration: YAML -> dataclasses with validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class WorkerSpec:
    name: str
    temperature: float = 0.0
    max_tokens: int = 64


@dataclass
class EnsembleCfg:
    api_base: str = "https://openrouter.ai/api/v1"
    workers: List[WorkerSpec] = field(default_factory=list)
    max_parallel: int = 8
    timeout: int = 75
    max_retries: int = 4
    tie_break: str = "lowest_rating"  # lowest_rating | highest_rating | first_worker


@dataclass
class DatasetCfg:
    name: str = "amazon"
    data_root: str = "./data"
    category_id: Optional[int] = None  # None = all categories
    min_user_reviews: int = 1
    max_train_users: Optional[int] = None
    max_val_users: Optional[int] = None
    max_test_users: Optional[int] = None
    max_reviews_per_user: Optional[int] = None
    al_candidate_pool_size: int = 5000
    cluster_artifact: Optional[str] = None  # path to serialized cluster mapping
    cache_dir: Optional[str] = None  # default: {data_root}/.prime_cache
    use_cache: bool = True


@dataclass
class ClusterCfg:
    n_clusters: int = 8
    embedding_model: str = "all-MiniLM-L6-v2"
    seed: int = 42
    pca_components: int = 32
    allow_synthetic_fallback: bool = False


@dataclass
class FitnessCfg:
    mode: str = "cvar"  # cvar | global | v1_weighted
    w_cvar: float = 0.5
    w_global: float = 0.3
    w_kappa: float = 0.2
    cvar_quantile: float = 0.33  # worst third of clusters
    prompt_len_limit: int = 8000
    len_penalty_per_100: float = 0.01
    len_penalty_start: int = 2000


@dataclass
class AcquisitionCfg:
    policy: str = "lexicographic"  # random|qbc_d|hardest|lexicographic|group_aware
    batch_size: int = 80
    hard_ratio: float = 0.7
    oversample_factor: float = 3.0
    n_diversity_clusters: int = 8
    expansion_trigger: int = 25
    expansion_batch: int = 40
    expansion_policy: str = "disagreement"  # disagreement | farthest
    group_quota_enabled: bool = True


@dataclass
class EvolutionCfg:
    n_evolve_iterations: int = 15
    population_size: int = 30
    archive_size: int = 500
    mutator_model: str = "google/gemini-2.5-pro"
    mutator_temperature: float = 0.6
    mutator_max_tokens: int = 8192
    structured_prompt: bool = True
    synthetic_fewshot: bool = False


@dataclass
class ConsolidationCfg:
    enabled: bool = True
    top_k_archive: int = 3
    pool_carryover_k: int = 3
    cluster_pareto: bool = False
    gate_delta: float = 0.01
    every_n_cycles: int = 1
    scope: str = "base_guidelines"  # base_guidelines | full_prompt


@dataclass
class ActiveLearningCfg:
    n_cycles: int = 6
    seed: int = 42
    selection_mode: str = "lexicographic"  # lexicographic | weighted
    selection_w_cvar: float = 0.7
    selection_w_global: float = 0.3
    proxy_validation_min_cycles: int = 2


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


@dataclass
class PrimeConfig:
    dataset: DatasetCfg = field(default_factory=DatasetCfg)
    clusters: ClusterCfg = field(default_factory=ClusterCfg)
    ensemble: EnsembleCfg = field(default_factory=EnsembleCfg)
    fitness: FitnessCfg = field(default_factory=FitnessCfg)
    acquisition: AcquisitionCfg = field(default_factory=AcquisitionCfg)
    evolution: EvolutionCfg = field(default_factory=EvolutionCfg)
    consolidation: ConsolidationCfg = field(default_factory=ConsolidationCfg)
    active_learning: ActiveLearningCfg = field(default_factory=ActiveLearningCfg)
    experiment: ExperimentCfg = field(default_factory=ExperimentCfg)
    prompt_path: str = "prompts/initial_prompt.txt"
    openevolve_config_path: Optional[str] = None

    def validate(self) -> None:
        if self.acquisition.hard_ratio < 0 or self.acquisition.hard_ratio > 1:
            raise ValueError("acquisition.hard_ratio must be in [0, 1]")
        if self.fitness.mode not in ("cvar", "global", "v1_weighted"):
            raise ValueError(f"Unknown fitness.mode: {self.fitness.mode}")
        policies = ("random", "qbc_d", "hardest", "lexicographic", "group_aware")
        if self.acquisition.policy not in policies:
            raise ValueError(f"acquisition.policy must be one of {policies}")
        if self.acquisition.expansion_policy not in ("disagreement", "farthest"):
            raise ValueError("acquisition.expansion_policy must be disagreement or farthest")
        if self.active_learning.selection_mode not in ("lexicographic", "weighted"):
            raise ValueError("active_learning.selection_mode must be lexicographic or weighted")
        if self.clusters.pca_components < 1:
            raise ValueError("clusters.pca_components must be >= 1")
        if self.consolidation.every_n_cycles < 1:
            raise ValueError("consolidation.every_n_cycles must be >= 1")
        if self.consolidation.scope not in ("base_guidelines", "full_prompt"):
            raise ValueError("consolidation.scope must be base_guidelines or full_prompt")


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
    return WorkerSpec(
        name=str(d["name"]),
        temperature=float(d.get("temperature", 0.0)),
        max_tokens=int(d.get("max_tokens", 64)),
    )


def load_config(path: Path, overrides: Optional[Dict[str, Any]] = None) -> PrimeConfig:
    """Load YAML config; optional dict overrides merged recursively."""
    raw = _load_yaml_tree(path)
    if overrides:
        raw = _merge_dict(raw, overrides)

    ds = raw.get("dataset", {})
    cl = raw.get("clusters", {})
    ens = raw.get("ensemble", {})
    fit = raw.get("fitness", {})
    acq = raw.get("acquisition", {})
    evo = raw.get("evolution", {})
    cons = raw.get("consolidation", {})
    al = raw.get("active_learning", {})
    exp = raw.get("experiment", {})

    workers_raw = ens.get("workers", [])
    workers = [_dict_to_worker_spec(w) for w in workers_raw]
    if not workers:
        workers = [
            WorkerSpec("openai/gpt-4o-mini"),
            WorkerSpec("google/gemini-2.5-flash"),
            WorkerSpec("anthropic/claude-3.5-haiku"),
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
            tie_break=str(ens.get("tie_break", "lowest_rating")),
        ),
        fitness=FitnessCfg(**{k: fit[k] for k in FitnessCfg.__dataclass_fields__ if k in fit}),
        acquisition=AcquisitionCfg(
            **{k: acq[k] for k in AcquisitionCfg.__dataclass_fields__ if k in acq}
        ),
        evolution=EvolutionCfg(**{k: evo[k] for k in EvolutionCfg.__dataclass_fields__ if k in evo}),
        consolidation=ConsolidationCfg(
            **{k: cons[k] for k in ConsolidationCfg.__dataclass_fields__ if k in cons}
        ),
        active_learning=ActiveLearningCfg(
            **{k: al[k] for k in ActiveLearningCfg.__dataclass_fields__ if k in al}
        ),
        experiment=ExperimentCfg(**{k: exp[k] for k in ExperimentCfg.__dataclass_fields__ if k in exp}),
        prompt_path=str(raw.get("prompt_path", "prompts/initial_prompt.txt")),
        openevolve_config_path=raw.get("openevolve_config_path"),
    )
    cfg.validate()
    return cfg

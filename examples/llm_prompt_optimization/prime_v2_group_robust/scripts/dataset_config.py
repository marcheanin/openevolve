"""Per-dataset settings for the S11 protocol analysis (`analyze_s11.py`, `statistic_reproducibility.py`,
`ci_calibration_check.py`, `run_s11_final.py`).

The dataset is chosen by the environment variable `S11_DATASET`:

  civil  (default)  CivilComments, results/S11_protocol_matrix. Every default below reproduces what the
                    scripts hard-coded before this module existed; the regression check is that
                    `run_s11_final.py` gives the same files as the pre-refactor code.
  mnli              MultiNLI, results/S13_mnli_matrix: 10 genres as groups (ids 1..10), cells of
                    genre x label with 250 rows each, binary label (1 = contradiction).
  toxlang           Multilingual toxicity, results/S15_toxlang_matrix: 6 languages as groups
                    (ids 1..6: en, de, ru, ar, hi, am), cells of language x label with 250 rows
                    each. Same task and same prompts as `civil` (binary toxicity, seed prompt
                    `prompts/initial_prompt_civilcomments.txt`); only the text source and the
                    group axis change.

Four further variables override paths, so that a test on synthetic data cannot overwrite real files:

  S11_SETS_DIR      directory holding <set>.json
  S11_PREDS_DIR     directory holding <set>/<name with ':' replaced by '__'>.npy
  S11_OUTPUTS_DIR   where power_summary.json, lottery_summary_*.json and final/ are written
  S11_PROMPTS_JSON  {name: text} file the length penalty reads word counts from

A relative override is taken relative to the project root (not the current directory), because
`run_s11_final.py` starts its children with cwd = project root and they inherit the environment.

`python scripts/dataset_config.py` prints the resolved configuration.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]

CIVIL_GROUP_NAMES = ("none", "male", "female", "LGBTQ", "christian", "muslim", "other_religions",
                     "black", "white")

# Jobs of run_s11_final.py that read experiments/E5_civilcomments or results/E5_*: there is no
# MultiNLI counterpart of that data, so they only run for `civil`.
CIVIL_ONLY_JOBS = ("gate_a_recheck", "draw_lottery", "repeats_vs_rows", "pool_spread")


def _env_path(var: str, default: Optional[Path]) -> Optional[Path]:
    v = os.environ.get(var, "").strip()
    if not v:
        return default
    p = Path(v)
    return p if p.is_absolute() else ROOT / p


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    sets_dir: Path            # <set>.json files (labels, cluster_ids, fingerprint, ...)
    preds_dir: Path           # <set>/<name>.npy files, one int16 vector per prompt
    outputs: Path             # power_summary.json, lottery_summary_*.json; final/ goes under it
    test_set: str             # the large set that plays the role of the truth
    dev_set: str              # the universe validation draws are taken from
    ids: tuple                # groups the worst-group metrics are computed over
    sample_groups: tuple      # groups the validation draws are sampled from (may include a group
                              # that the metrics ignore: CivilComments group 0 "none")
    uniform_per_cell: int     # rows per group x label cell in the uniform allocation
    pilot_per_cell: int       # rows per cell in the pilot that finds the worst groups
    dev_budget: int           # total validation rows an allocation may use
    final_prefix: str         # optimizer finals are named `<prefix>:<seed>_<method>...`
    method_sep: Optional[str]  # cut the method off the tail at this separator (None = keep the tail)
    prompts_json: Optional[Path]  # {name: text}; None = collect_prompts() of score_s11_matrix.py
    expected_prompts: Optional[int]  # prompts a complete matrix holds; None = read it from the data
    names: Optional[tuple]    # fixed group names indexed by id; None = read them from the set JSON

    @property
    def final_dir(self) -> Path:
        return self.outputs / "final"

    @property
    def pools(self) -> list:
        """Values of --pool: the whole pool, the optimizer finals (+ seed), the one-line edits (+ seed)."""
        return ["all", self.final_prefix, "r15"]

    def set_path(self, name: str) -> Path:
        return self.sets_dir / f"{name}.json"

    def group_names(self, rec: Optional[dict] = None) -> dict:
        """{group id: name} for printing; from the set JSON for datasets without a fixed list."""
        if self.names is not None:
            return {i: n for i, n in enumerate(self.names)}
        rec = rec or {}
        if rec.get("group_names"):
            return {int(k): v for k, v in rec["group_names"].items()}
        if rec.get("genre") and rec.get("cluster_ids"):
            return {int(g): str(n) for g, n in zip(rec["cluster_ids"], rec["genre"])}
        return {int(g): str(g) for g in self.ids}

    def method_of(self, name: str) -> Optional[str]:
        """Optimizer of a final: `s9:42_evoprompt_ga` -> `evoprompt_ga`; with a protocol suffix,
        `s13:42_ape__soft_min` -> `ape`. Anything else (seed, one-line edits) -> None."""
        if not (name.startswith(self.final_prefix + ":") and "_" in name):
            return None
        tail = name.split(":", 1)[1].split("_", 1)[1]
        return tail if self.method_sep is None else tail.split(self.method_sep, 1)[0]

    def prompt_words(self) -> dict:
        """{prompt name: word count}, the input of the length penalty."""
        if self.prompts_json is None:
            sys.path.insert(0, str(ROOT / "scripts"))
            from score_s11_matrix import collect_prompts
            return {n: len(t.split()) for n, t in collect_prompts()}
        return {n: len(t.split()) for n, t in self._read_prompts().items()}

    def _read_prompts(self) -> dict:
        if not self.prompts_json.is_file():
            raise SystemExit(f"prompts file not found: {self.prompts_json} "
                             "(a {name: text} JSON, see scripts/collect_prompts_json.py)")
        return json.loads(self.prompts_json.read_text(encoding="utf-8"))

    def expected_count(self, have: dict) -> int:
        """How many prompts a finished matrix holds. `have` = {set name: prompts scored so far}."""
        if self.expected_prompts is not None:
            return self.expected_prompts
        if self.prompts_json is not None and self.prompts_json.is_file():
            return len(self._read_prompts())
        return max(have.values(), default=0)


def _civil() -> DatasetConfig:
    base = ROOT / "results/S11_protocol_matrix"
    return DatasetConfig(
        key="civil",
        sets_dir=_env_path("S11_SETS_DIR", base / "fixed_sets"),
        preds_dir=_env_path("S11_PREDS_DIR", base / "preds"),
        outputs=_env_path("S11_OUTPUTS_DIR", base),
        test_set="truth_large", dev_set="dev_universe",
        ids=tuple(range(1, 9)), sample_groups=tuple(range(9)),
        uniform_per_cell=50, pilot_per_cell=20, dev_budget=900,
        final_prefix="s9", method_sep=None,
        prompts_json=_env_path("S11_PROMPTS_JSON", None),
        expected_prompts=37, names=CIVIL_GROUP_NAMES)


def _mnli() -> DatasetConfig:
    base = ROOT / "results/S13_mnli_matrix"
    return DatasetConfig(
        key="mnli",
        sets_dir=_env_path("S11_SETS_DIR", base / "fixed_sets"),
        preds_dir=_env_path("S11_PREDS_DIR", base / "scorer_gemma/preds"),
        outputs=_env_path("S11_OUTPUTS_DIR", base),
        test_set="truth_mnli", dev_set="dev_universe_mnli",
        ids=tuple(range(1, 11)), sample_groups=tuple(range(1, 11)),
        uniform_per_cell=45, pilot_per_cell=20, dev_budget=900,
        final_prefix="s13", method_sep="__",
        prompts_json=_env_path("S11_PROMPTS_JSON", base / "prompts.json"),
        expected_prompts=None, names=None)


def _toxlang() -> DatasetConfig:
    base = ROOT / "results/S15_toxlang_matrix"
    return DatasetConfig(
        key="toxlang",
        sets_dir=_env_path("S11_SETS_DIR", base / "fixed_sets"),
        preds_dir=_env_path("S11_PREDS_DIR", base / "scorer_gemma/preds"),
        outputs=_env_path("S11_OUTPUTS_DIR", base),
        test_set="truth_tox", dev_set="dev_universe_tox",
        ids=tuple(range(1, 7)), sample_groups=tuple(range(1, 7)),
        uniform_per_cell=45, pilot_per_cell=20, dev_budget=900,
        final_prefix="s15", method_sep="__",
        prompts_json=_env_path("S11_PROMPTS_JSON", base / "prompts.json"),
        expected_prompts=None, names=None)


_BUILDERS = {"civil": _civil, "mnli": _mnli, "toxlang": _toxlang}
KEY = (os.environ.get("S11_DATASET") or "civil").strip().lower()
if KEY not in _BUILDERS:
    raise SystemExit(f"S11_DATASET={KEY!r} is not one of {sorted(_BUILDERS)}")
cfg: DatasetConfig = _BUILDERS[KEY]()


def require_civil(script: str) -> None:
    """Stop with a readable message when a CivilComments-only script is started for another dataset."""
    if cfg.key != "civil":
        sys.exit(f"{script}: not applicable with S11_DATASET={cfg.key!r}. It reads the old CivilComments "
                 "E5 data (experiments/E5_civilcomments, results/E5_*), which has no MultiNLI counterpart. "
                 "Run it with S11_DATASET unset or set to 'civil'.")


if __name__ == "__main__":
    for k, v in vars(cfg).items():
        print(f"{k:18s} {v}")
    print(f"{'final_dir':18s} {cfg.final_dir}")
    print(f"{'pools':18s} {cfg.pools}")

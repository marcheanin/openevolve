"""Experiment run context: results dir, config snapshot, git hash, seed."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from prime.config import PrimeConfig


def git_hash(cwd: Optional[Path] = None) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd or Path.cwd()),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


class RunContext:
    """Manages reproducible experiment output directory."""

    def __init__(self, cfg: PrimeConfig, config_path: Path, project_root: Path) -> None:
        self.cfg = cfg
        self.config_path = config_path.resolve()
        self.project_root = project_root.resolve()
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        base = cfg.experiment.results_dir or str(project_root / "results")
        self.run_dir = (Path(base) / cfg.experiment.name / f"seed{cfg.active_learning.seed}_{ts}").resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def snapshot(self) -> Path:
        """Copy config, write metadata (seed, git hash), attach protocol notes."""
        snap = self.run_dir / "config_used.yaml"
        shutil.copy2(self.config_path, snap)
        meta = {
            "seed": self.cfg.active_learning.seed,
            "git_hash": git_hash(self.project_root),
            "experiment": self.cfg.experiment.name,
            "config_source": str(self.config_path),
            "smoke": self.cfg.experiment.smoke,
            "cluster_geometry": getattr(self.cfg.clusters, "geometry", "style"),
            "fitness_mode": self.cfg.fitness.mode,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        (self.run_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        resolved = asdict(self.cfg)
        (self.run_dir / "config_snapshot.json").write_text(
            json.dumps(resolved, indent=2, default=str),
            encoding="utf-8",
        )
        # Fully-resolved YAML for OE evaluator subprocesses. The raw copy above
        # keeps `includes:` with paths relative to experiments/, which break when
        # reloaded from run_dir → empty workers → expensive base_v3 defaults
        # (OBSERVATIONS O28). OpenEvolve reads PRIME_CONFIG_PATH from this file.
        resolved_path = self.run_dir / "config_resolved.yaml"
        with open(resolved_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(resolved, f, sort_keys=False, allow_unicode=True)
        # Keep lessons next to the run so post-mortems don't depend on repo edits.
        obs = self.project_root / "experiments" / "OBSERVATIONS.md"
        if obs.is_file():
            shutil.copy2(obs, self.run_dir / "OBSERVATIONS.md")
        return snap

    def cycle_dir(self, cycle: int) -> Path:
        d = self.run_dir / f"al_iter_{cycle}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_json(self, name: str, payload: Dict[str, Any]) -> Path:
        path = self.run_dir / name
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

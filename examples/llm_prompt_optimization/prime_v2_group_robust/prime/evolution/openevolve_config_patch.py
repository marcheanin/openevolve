"""Write OpenEvolve config snapshot with QD feature_dimensions matching actual K."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from prime.evolution.qd_features import qd_feature_dimension_names


def patch_openevolve_feature_dimensions(
    source_path: Optional[Path],
    dest_path: Path,
    n_clusters: int,
) -> Path:
    """
    Copy OpenEvolve YAML (if present) and set database.feature_dimensions to
    cluster_acc_0..K-1 + prompt_length for the fitted K.
    """
    raw: Dict[str, Any] = {}
    if source_path is not None and Path(source_path).is_file():
        with open(source_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    db = dict(raw.get("database") or {})
    db["feature_dimensions"] = qd_feature_dimension_names(n_clusters)
    raw["database"] = db

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, sort_keys=False, allow_unicode=True)
    return dest_path

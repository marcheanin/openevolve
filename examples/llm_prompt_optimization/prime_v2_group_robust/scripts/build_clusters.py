#!/usr/bin/env python3
"""Offline: fit style clusters on WILDS Amazon train and save artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))

from prime.config import load_config
from prime.data.clustering import fit_style_clusters
from prime.data.wilds_loader import load_amazon_splits


def main() -> int:
    parser = argparse.ArgumentParser(description="Build style cluster artifacts")
    parser.add_argument("--config", type=Path, default=PKG_ROOT / "configs" / "base.yaml")
    parser.add_argument("--output", type=Path, default=PKG_ROOT / "data" / "clusters.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    splits = load_amazon_splits(cfg.dataset, seed=cfg.active_learning.seed)
    art = fit_style_clusters(splits["train"], cfg.clusters, cfg.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    art.save(args.output)
    print(f"Saved {art.n_clusters} clusters for {len(art.user_to_cluster)} users -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Evaluate a prompt on the capped official test split and persist reusable artifacts.

Writes under --out-dir (default: <run-dir>/evals/<tag>/):
  metrics.json                  ensemble + per-worker metrics
  ensemble_predictions.npy      shape (N,)
  worker_predictions.npy        shape (W, N)  — one row per worker
  labels.npy, user_ids.npy, cluster_ids.npy
  meta.json                     prompt hash, worker names, n, tag
  prompt.txt                    the evaluated prompt

Reuse: if metrics.json + worker_predictions.npy already exist for the same
prompt hash, the script skips the API call and just reprints the metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))


def _prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _load_test_slice(run_dir: Path, cfg):
    """Load the same capped test examples the run used (by user_id / assign)."""
    # Prefer exact ordering from a prior eval_test dump when present.
    et = run_dir / "evals" / "eval_test"
    if (et / "user_ids.npy").is_file() and (et / "labels.npy").is_file():
        want_uids = [str(u) for u in np.load(et / "user_ids.npy", allow_pickle=True)]
        want_labels = np.load(et / "labels.npy")
        want_cids = (
            np.load(et / "cluster_ids.npy")
            if (et / "cluster_ids.npy").is_file()
            else None
        )
    else:
        want_uids = want_labels = want_cids = None

    if cfg.dataset.name == "civilcomments":
        from prime.data.civilcomments_loader import load_civilcomments_splits

        test = load_civilcomments_splits(cfg.dataset, seed=cfg.active_learning.seed)["test"]
    else:
        from prime.data.wilds_loader import load_amazon_splits

        test = load_amazon_splits(cfg.dataset, seed=cfg.active_learning.seed)["test"]

    assign_path = run_dir / "cluster_assign_test.json"
    user_to_cluster = {}
    if assign_path.is_file():
        assign = json.loads(assign_path.read_text(encoding="utf-8"))
        user_to_cluster = {
            str(k): int(v) for k, v in assign.items() if not str(k).startswith("_")
        }

    uid_to_idx = {str(u): i for i, u in enumerate(test.user_ids)}
    if want_uids is not None:
        idxs = []
        for uid in want_uids:
            if uid not in uid_to_idx:
                raise RuntimeError(f"eval_test user_id {uid} missing from reloaded test split")
            idxs.append(uid_to_idx[uid])
        labels = np.asarray([test.labels[i] for i in idxs], dtype=np.int16)
        if not np.array_equal(labels, want_labels.astype(np.int16)):
            raise RuntimeError("reloaded test labels disagree with evals/eval_test/labels.npy")
        if want_cids is not None:
            cluster_ids = np.asarray(want_cids, dtype=np.int16)
        elif test.example_cluster_ids:
            cluster_ids = np.asarray(
                [test.example_cluster_ids[i] for i in idxs], dtype=np.int16
            )
        else:
            cluster_ids = np.asarray(
                [user_to_cluster[str(test.user_ids[i])] for i in idxs], dtype=np.int16
            )
        return {
            "texts": [test.texts[i] for i in idxs],
            "labels": labels,
            "user_ids": np.asarray([test.user_ids[i] for i in idxs]),
            "cluster_ids": cluster_ids,
        }

    idxs = [i for i, u in enumerate(test.user_ids) if str(u) in user_to_cluster]
    if not idxs and test.example_cluster_ids:
        # Oracle CC: assign file may be optional; use full capped split.
        idxs = list(range(len(test.texts)))
        cluster_ids = np.asarray(test.example_cluster_ids, dtype=np.int16)
    elif not idxs:
        raise RuntimeError("cluster_assign_test.json matched no test users")
    else:
        cluster_ids = np.asarray(
            [user_to_cluster[str(test.user_ids[i])] for i in idxs], dtype=np.int16
        )
    return {
        "texts": [test.texts[i] for i in idxs],
        "labels": np.asarray([test.labels[i] for i in idxs], dtype=np.int16),
        "user_ids": np.asarray([test.user_ids[i] for i in idxs]),
        "cluster_ids": cluster_ids,
    }


def _per_worker_metrics(
    wp: np.ndarray,
    labels: np.ndarray,
    user_ids: np.ndarray,
    cluster_ids: np.ndarray,
    worker_names: List[str],
    cfg,
) -> Dict[str, Any]:
    from prime.fitness.metrics import compute_metrics

    out: Dict[str, Any] = {}
    for w, name in enumerate(worker_names):
        m = compute_metrics(
            wp[w],
            labels,
            user_ids,
            cluster_ids=cluster_ids,
            cvar_quantile=cfg.fitness.cvar_quantile,
            beta_a=cfg.fitness.beta_a,
            beta_b=cfg.fitness.beta_b,
            tail_quantile=cfg.active_learning.tail_quantile,
        )
        out[name] = {
            k: m.get(k)
            for k in [
                "R_global",
                "R_worst",
                "R_tail",
                "CVaR_cluster",
                "mae",
                "cluster_accuracies",
            ]
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--prompt",
        type=Path,
        default=None,
        help="Prompt file (default: <run-dir>/initial_prompt.txt)",
    )
    parser.add_argument(
        "--tag",
        default="initial_prompt",
        help="Subdir name under evals/ (e.g. initial_prompt, final_selected)",
    )
    parser.add_argument("--force", action="store_true", help="Ignore cached artifacts")
    args = parser.parse_args()

    from prime.config import load_config
    from prime.fitness.metrics import compute_metrics
    from prime.workers.ensemble import (
        build_workers,
        load_dotenv_if_present,
        parallel_predict,
    )

    load_dotenv_if_present()
    run_dir = args.run_dir.resolve()
    prompt_path = args.prompt or (run_dir / "initial_prompt.txt")
    prompt = prompt_path.read_text(encoding="utf-8")
    ph = _prompt_hash(prompt)
    out_dir = run_dir / "evals" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / "meta.json"
    metrics_path = out_dir / "metrics.json"
    wp_path = out_dir / "worker_predictions.npy"
    if (
        not args.force
        and metrics_path.is_file()
        and wp_path.is_file()
        and meta_path.is_file()
    ):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("prompt_hash") == ph:
            print(f"[eval] cache hit {out_dir} (hash={ph})", flush=True)
            print(metrics_path.read_text(encoding="utf-8"), flush=True)
            return 0

    cfg_path = run_dir / "config_resolved.yaml"
    if not cfg_path.is_file():
        cfg_path = run_dir / "config_used.yaml"
    cfg = load_config(cfg_path)
    slice_ = _load_test_slice(run_dir, cfg)
    texts, labels, user_ids, cluster_ids = (
        slice_["texts"],
        slice_["labels"],
        slice_["user_ids"],
        slice_["cluster_ids"],
    )
    workers = build_workers(cfg.ensemble)
    worker_names = [w.model_name for w in workers]
    print(
        f"[eval] tag={args.tag} n={len(texts)} users={len(set(map(str, user_ids.tolist())))} "
        f"workers={worker_names} hash={ph}",
        flush=True,
    )
    ensemble, wp = parallel_predict(
        workers,
        texts,
        prompt,
        max_parallel=cfg.ensemble.max_parallel,
        tie_break=cfg.ensemble.tie_break,
        aggregation=cfg.ensemble.aggregation,
        label_space=cfg.dataset.label_space,
    )
    ens = np.asarray(ensemble, dtype=np.int16)
    wp_arr = np.asarray(wp, dtype=np.int16)  # (W, N)

    ens_metrics = compute_metrics(
        ens,
        labels,
        user_ids,
        worker_predictions=[wp_arr[w] for w in range(wp_arr.shape[0])],
        cluster_ids=cluster_ids,
        cvar_quantile=cfg.fitness.cvar_quantile,
        beta_a=cfg.fitness.beta_a,
        beta_b=cfg.fitness.beta_b,
        tail_quantile=cfg.active_learning.tail_quantile,
        shrink_prior_weight=cfg.fitness.shrink_prior_weight,
        class_balanced=cfg.fitness.class_balanced,
    )
    ens_keys = [
        "R_global",
        "R_macro",
        "R_worst",
        "R_worst_group",
        "R_tail",
        "CVaR_cluster",
        "CVaR_cluster_shrunk",
        "CVaR_cluster_balanced_shrunk",
        "mae",
        "num_users",
        "cluster_accuracies",
        "accuracy_per_class",
        "mean_kappa",
        "disagreement_rate",
    ]
    payload = {
        "tag": args.tag,
        "prompt_hash": ph,
        "prompt_path": str(prompt_path),
        "n_examples": int(len(texts)),
        "n_users": int(len(set(map(str, user_ids.tolist())))),
        "worker_names": worker_names,
        "ensemble": {k: ens_metrics.get(k) for k in ens_keys},
        "per_worker": _per_worker_metrics(
            wp_arr, labels, user_ids, cluster_ids, worker_names, cfg
        ),
    }

    np.save(out_dir / "ensemble_predictions.npy", ens)
    np.save(wp_path, wp_arr)
    np.save(out_dir / "labels.npy", labels)
    np.save(out_dir / "user_ids.npy", user_ids)
    np.save(out_dir / "cluster_ids.npy", cluster_ids)
    (out_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    metrics_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    # Stable alias next to run root for the initial-prompt case.
    if args.tag == "initial_prompt":
        (run_dir / "baseline_initial_prompt_test.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        np.save(run_dir / "baseline_initial_worker_predictions.npy", wp_arr)
        np.save(run_dir / "baseline_initial_ensemble_predictions.npy", ens)
        # Canonical shared baseline for E4 arms (seed prompt is fixed).
        shared = PKG_ROOT / "experiments" / "E4_civilcomments" / "baseline_initial_prompt_test"
        if cfg.dataset.name == "civilcomments":
            shared.mkdir(parents=True, exist_ok=True)
            (shared / "metrics.json").write_text(
                json.dumps(payload, indent=2, default=str), encoding="utf-8"
            )
            np.save(shared / "ensemble_predictions.npy", ens)
            np.save(shared / "worker_predictions.npy", wp_arr)
            np.save(shared / "labels.npy", labels)
            np.save(shared / "user_ids.npy", user_ids)
            np.save(shared / "cluster_ids.npy", cluster_ids)
            (shared / "prompt.txt").write_text(prompt, encoding="utf-8")
            (shared / "source_run.txt").write_text(str(run_dir) + "\n", encoding="utf-8")
            print(f"[eval] also wrote shared baseline {shared}", flush=True)
    meta_path.write_text(
        json.dumps(
            {
                "tag": args.tag,
                "prompt_hash": ph,
                "worker_names": worker_names,
                "n_examples": len(texts),
                "shapes": {
                    "ensemble": list(ens.shape),
                    "worker_predictions": list(wp_arr.shape),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(payload["ensemble"], indent=2, default=str), flush=True)
    print(f"[eval] wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

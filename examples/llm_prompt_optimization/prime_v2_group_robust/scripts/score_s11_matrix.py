#!/usr/bin/env python
"""Score the prompt pool on the S11 fixed sets, resumably.

One `.npy` of predictions per (prompt, set). Re-running skips whatever is already
on disk, so the job can be interrupted and continued. The seed prompt is scored
first and again at the end of each set: the two passes bound the scorer drift that
OBSERVATIONS M29 flagged, instead of assuming it away.

Spend is read from the OpenRouter key before and after every prompt and written to
`spend_log.jsonl`; the run stops if `--max-spend` would be exceeded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "results/S11_protocol_matrix"
MATRIX = ROOT / "results/E5_s9_matrix"
POOLS = ROOT / "experiments/E5_civilcomments/pools/strictness_sweep"
METHODS = ["ape", "ape_k48", "ape_ut", "apo", "gpo", "evoprompt_ga", "evoprompt_de",
           "gepa", "prime", "random_al", "oracle"]
# Scored first, so a partial run already answers the main questions.
PRIORITY = ["seed", "s9:42_gpo", "s9:42_prime", "s9:44_apo", "s9:44_evoprompt_de",
            "r15:nl_strict", "r15:nl_lenient_max", "s9:42_ape", "s9:44_ape"]


def key_usage(retries: int = 3) -> float | None:
    """Spend so far on this key, or None if the endpoint is unreachable.

    Never raises: a flaky balance check must not kill a multi-hour scoring job.
    The key's own hard limit remains the real backstop.
    """
    req = urllib.request.Request("https://openrouter.ai/api/v1/auth/key",
                                 headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return float(json.load(r)["data"]["usage"])
        except Exception as exc:
            if attempt == retries - 1:
                print(f"[budget] usage check unavailable ({type(exc).__name__}); continuing", flush=True)
            else:
                time.sleep(3 * (attempt + 1))
    return None


def collect_prompts() -> list[tuple[str, str]]:
    """Unique prompt texts across the optimizer finals and the one-line edit pool."""
    entries = [("seed", (ROOT / "prompts/initial_prompt_civilcomments.txt").read_text(encoding="utf-8"))]
    for s in ("42", "43", "44"):
        for m in METHODS:
            p = MATRIX / f"seed{s}" / m / "best_prompt.txt"
            if p.is_file():
                entries.append((f"s9:{s}_{m}", p.read_text(encoding="utf-8")))
    man = json.loads((POOLS / "manifest.json").read_text(encoding="utf-8"))
    for c in sorted(man["candidates"], key=lambda c: c["rank"]):
        entries.append((f"r15:{c['name']}", (POOLS / c["file"]).read_text(encoding="utf-8")))

    seen, unique = {}, []
    for name, txt in entries:
        h = hashlib.sha256(txt.encode("utf-8")).hexdigest()[:12]
        if h in seen:
            continue
        seen[h] = name
        unique.append((name, txt))
    order = {n: i for i, n in enumerate(PRIORITY)}
    unique.sort(key=lambda kv: (order.get(kv[0], 999), kv[0]))
    return unique


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "experiments/E5_civilcomments/config_prime_main.yaml")
    ap.add_argument("--sets", default="truth_large,dev_universe")
    ap.add_argument("--max-parallel", type=int, default=4,
                    help="4 is the measured ceiling before the provider returns 429 storms")
    ap.add_argument("--chunk", type=int, default=400, help="rows per progress/checkpoint block")
    ap.add_argument("--max-spend", type=float, default=25.0, help="USD ceiling for this job")
    ap.add_argument("--limit-prompts", type=int, default=0)
    args = ap.parse_args()

    from e5_stable_eval import _score_once
    from prime.config import load_config
    from prime.data.civilcomments_loader import load_civilcomments_splits
    from prime.workers.ensemble import load_dotenv_if_present

    load_dotenv_if_present()
    cfg = load_config(args.config)
    cfg.ensemble.max_parallel = args.max_parallel
    cfg.dataset.max_val_users = max(int(cfg.dataset.max_val_users or 0), 60_000)
    cfg.dataset.max_test_users = max(int(cfg.dataset.max_test_users or 0), 200_000)
    splits = load_civilcomments_splits(cfg.dataset, seed=42)

    prompts = collect_prompts()
    if args.limit_prompts:
        prompts = prompts[: args.limit_prompts]
    print(f"[pool] {len(prompts)} unique prompts", flush=True)

    start_spend = key_usage()
    log = (OUT / "spend_log.jsonl").open("a", encoding="utf-8")
    print(f"[budget] start usage {'unknown' if start_spend is None else f'${start_spend:.4f}'}, "
          f"ceiling +${args.max_spend:.2f}", flush=True)

    for set_name in args.sets.split(","):
        rec = json.loads((OUT / "fixed_sets" / f"{set_name}.json").read_text(encoding="utf-8"))
        split = splits[rec["source_split"]]
        idx = rec["indices"]
        texts = [split.texts[i] for i in idx]
        labels, clusters = rec["labels"], rec["cluster_ids"]
        users = [str(split.user_ids[i]) for i in idx] if getattr(split, "user_ids", None) is not None \
            else [str(i) for i in idx]
        preds_dir = OUT / "preds" / set_name
        preds_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[set] {set_name}: n={len(texts)} fp={rec['fingerprint']}", flush=True)

        todo = [(n, t) for n, t in prompts
                if not (preds_dir / f"{n.replace(':', '__')}.npy").is_file()]
        print(f"[set] {len(prompts) - len(todo)} cached, {len(todo)} to score "
              f"(~{len(todo) * len(texts)} calls)", flush=True)

        for i, (name, txt) in enumerate(todo, 1):
            now = key_usage()
            spent = None if (now is None or start_spend is None) else now - start_spend
            if spent is not None and spent > args.max_spend:
                print(f"[stop] spend ${spent:.2f} exceeds ceiling; stopping cleanly", flush=True)
                return 2
            # Scored in chunks: progress is visible, a killed run resumes mid-prompt,
            # and because the row order is shuffled a partial file is still a
            # stratified random subsample rather than a few whole cells.
            stem = name.replace(":", "__")
            part = preds_dir / f"{stem}.partial.npy"
            done = np.load(part) if part.is_file() else np.zeros(0, dtype=np.int16)
            t0, scored_now = time.time(), 0
            while len(done) < len(texts):
                a, b = len(done), min(len(done) + args.chunk, len(texts))
                chunk, _ = _score_once(texts[a:b], labels[a:b], clusters[a:b], users[a:b], txt, cfg)
                done = np.concatenate([done, np.asarray(chunk, dtype=np.int16)])
                scored_now += b - a
                np.save(part, done)
                el = max(time.time() - t0, 1e-6)
                print(f"      {name:26s} {len(done):5d}/{len(texts)} rows  "
                      f"{scored_now/el:4.1f}/s  eta {(len(texts)-len(done))/(scored_now/el)/60:5.1f} min",
                      flush=True)
            np.save(preds_dir / f"{stem}.npy", done)
            part.unlink(missing_ok=True)
            dt = time.time() - t0
            after = key_usage()
            total = None if (after is None or start_spend is None) else after - start_spend
            log.write(json.dumps({"set": set_name, "prompt": name, "n": len(texts),
                                  "seconds": round(dt, 1), "usage_usd": after,
                                  "spent_total": total}) + "\n")
            log.flush()
            print(f"[{set_name}] {i:3d}/{len(todo)} {name:26s} {dt:6.0f}s "
                  f"({len(texts)/dt:4.1f}/s)  total "
                  f"{'n/a' if total is None else f'${total:.3f}'}", flush=True)

        # drift guard: re-score the seed at the end of the set (M29)
        drift_path = preds_dir / "_seed_rescore.npy"
        if not drift_path.is_file() and (preds_dir / "seed.npy").is_file():
            preds, _ = _score_once(texts, labels, clusters, users, dict(prompts)["seed"], cfg)
            np.save(drift_path, np.asarray(preds, dtype=np.int16))
            first = np.load(preds_dir / "seed.npy")
            print(f"[drift] {set_name}: seed first vs last pass agreement "
                  f"{float((first == np.asarray(preds)).mean()):.4f}", flush=True)

    end = key_usage()
    total = None if (end is None or start_spend is None) else end - start_spend
    print(f"\n[done] total spend {'n/a' if total is None else f'${total:.3f}'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Собирает JSON {имя: текст промпта} для скорера с остановкой (`--prompts-json`).

Порядок в файле — порядок скоринга при прерывании: сначала seed, затем однострочные правки, затем
финалы прогонов. Повторы по тексту отбрасываются (первое имя остаётся), как в S11: два прогона,
пришедшие к одному и тому же промпту, не должны считаться дважды.

Примеры:
  MultiNLI, пул и финалы:
    python scripts/collect_prompts_json.py --seed-prompt experiments/S13_mnli/prompts/seed.txt \
        --pool-dir experiments/S13_mnli/pools/strictness_sweep \
        --finals-root results/S13_mnli_loop --prefix s13 --out results/S13_mnli_matrix/prompts.json
  Финалы S12 (CivilComments) вместе с seed для сверки:
    python scripts/collect_prompts_json.py --seed-prompt prompts/initial_prompt_civilcomments.txt \
        --finals-root results/S12_live_loop --prefix s12 --out results/S12_live_loop/finals_prompts.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed-prompt", type=Path, required=True)
    ap.add_argument("--pool-dir", type=Path, default=None, help="каталог с manifest.json пула однострочных правок")
    ap.add_argument("--finals-root", type=Path, default=None, help="каталог прогонов: seed*/<метод>__<протокол>/")
    ap.add_argument("--prefix", default="run", help="префикс имён финалов, например s12 или s13")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    entries: list[tuple[str, str]] = [("seed", args.seed_prompt.read_text(encoding="utf-8"))]
    if args.pool_dir:
        man = json.loads((args.pool_dir / "manifest.json").read_text(encoding="utf-8"))
        for c in sorted(man["candidates"], key=lambda c: c["rank"]):
            entries.append((f"r15:{c['name']}", (args.pool_dir / c["file"]).read_text(encoding="utf-8")))
    n_runs = 0
    if args.finals_root:
        for d in sorted(args.finals_root.glob("seed*/*")):
            if not (d / "done.json").is_file():
                continue  # недописанный прогон
            seed = d.parent.name.replace("seed", "")
            entries.append((f"{args.prefix}:{seed}_{d.name}", (d / "best_prompt.txt").read_text(encoding="utf-8")))
            n_runs += 1

    seen: dict[str, str] = {}
    unique: dict[str, str] = {}
    dup: list[tuple[str, str]] = []
    for name, txt in entries:
        h = hashlib.sha256(txt.encode("utf-8")).hexdigest()[:12]
        if h in seen:
            dup.append((name, seen[h]))
            continue
        seen[h] = name
        unique[name] = txt

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(unique, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(unique)} уникальных промптов записано в {args.out} "
          f"(seed 1, правок {sum(k.startswith('r15:') for k in unique)}, финалов {sum(k.startswith(args.prefix + ':') for k in unique)} из {n_runs} готовых прогонов)")
    for name, first in dup:
        print(f"  повтор по тексту: {name} совпал с {first}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

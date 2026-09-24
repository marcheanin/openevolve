#!/usr/bin/env python
"""Запуск PRIME на стендах S16 (mnli / toxlang). См. scripts/run_e5_prime_main.py (CivilComments).

Использование:
  python scripts/run_s16_prime.py --stand toxlang --seed 42
  python scripts/run_s16_prime.py --stand mnli --seed 43
  python scripts/run_s16_prime.py --stand toxlang --smoke-mock     # без реальных вызовов API
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# openevolve.api._prepare_program writes the evolving prompt with a bare open(path, "w")
# (no encoding=). On Windows the default text encoding is the system codepage (cp1252 here),
# which cannot encode toxlang's non-Latin few-shot text (Cyrillic/Arabic/Devanagari/Ethiopic
# injected verbatim, O22) and crashes with UnicodeEncodeError as soon as evolution starts.
# Python's UTF-8 mode (PEP 540) fixes this, but it can only be set at interpreter start (an
# in-process os.environ write after that point is too late, and self-re-exec via os.execv was
# tried and silently lost stdout on this Windows Python) — so this refuses to run rather than
# fail 15 stages into a paid live run. Invoke as: PYTHONUTF8=1 python scripts/run_s16_prime.py ...
if sys.flags.utf8_mode == 0:
    raise SystemExit(
        "PYTHONUTF8 не включён (sys.flags.utf8_mode == 0). Без него запись эволюционирующего "
        "промпта в openevolve.api._prepare_program падает на не-ASCII символах (toxlang: "
        "многоязычные few-shot). Запускайте как: PYTHONUTF8=1 python scripts/run_s16_prime.py ..."
    )

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CFG_DIR = ROOT / "experiments" / "S16_prime_stands"
SEED_SUFFIX = {42: "", 43: "_seed43", 44: "_seed44"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stand", required=True, choices=("mnli", "toxlang"))
    ap.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    ap.add_argument("--smoke-mock", action="store_true",
                    help="мок-дым (force_mock: true в конфиге, ни одного реального вызова API)")
    args, rest = ap.parse_known_args()

    if args.smoke_mock:
        cfg = CFG_DIR / f"config_prime_{args.stand}_smoke_mock.yaml"
    else:
        cfg = CFG_DIR / f"config_prime_{args.stand}{SEED_SUFFIX[args.seed]}.yaml"
    if not cfg.is_file():
        raise SystemExit(f"конфиг не найден: {cfg}")

    from prime.cli import main as cli_main

    sys.argv = ["prime", "--config", str(cfg)] + rest
    return int(cli_main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())

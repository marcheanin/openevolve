#!/usr/bin/env python3
"""Диагностика строк INVALID в основных матрицах: сбой транспорта или неразбираемый ответ?

Скорер пишет -1 и когда исчерпаны повторы (транспорт), и когда ответ модели не разобран
(содержательный сбой). Различить можно только повторным запросом: при температуре 0 содержательный
сбой воспроизводится, транспортный — исправляется. Скрипт переспрашивает ровно пары (промпт, строка)
с -1, сохраняет сырой ответ модели и НЕ трогает основные предсказания — результат пишется отдельно
в results/requery_invalid/<стенд>.json.

  python scripts/requery_invalid.py --stand toxlang [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stand", required=True, choices=("civil", "mnli", "toxlang"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    os.environ["S11_DATASET"] = args.stand
    os.environ.pop("S11_PREDS_DIR", None)

    from analyze_s11 import load_preds, load_set
    from dataset_config import cfg

    y, c, rec = load_set(cfg.test_set)
    P = load_preds(cfg.test_set, len(y), controls=False)
    if "texts" not in rec:
        # CivilComments: набор хранит индексы в сплите, тексты материализуются так же, как в
        # scripts/score_s11_matrix.py (тот же конфиг, те же расширенные капы, seed=42).
        from prime.config import load_config
        from prime.data.civilcomments_loader import load_civilcomments_splits

        ccfg = load_config(ROOT / "experiments/E5_civilcomments/config_prime_main.yaml")
        ccfg.dataset.max_val_users = max(int(ccfg.dataset.max_val_users or 0), 60_000)
        ccfg.dataset.max_test_users = max(int(ccfg.dataset.max_test_users or 0), 200_000)
        split = load_civilcomments_splits(ccfg.dataset, seed=42)[rec["source_split"]]
        rec = dict(rec, texts=[split.texts[i] for i in rec["indices"]])
        assert [int(split.labels[i]) for i in rec["indices"]] == [int(v) for v in y], \
            "материализованные метки не совпали с набором — тексты не те"
    if cfg.prompts_json is None:
        from score_s11_matrix import collect_prompts
        texts_of = dict(collect_prompts())
    else:
        texts_of = json.loads(cfg.prompts_json.read_text(encoding="utf-8"))
    pairs = [(n, int(i)) for n in sorted(P) for i in np.flatnonzero(P[n] < 0)]
    print(f"{args.stand}: пар (промпт, строка) с INVALID: {len(pairs)}")
    missing = sorted({n for n, _ in pairs if n not in texts_of})
    if missing:
        print(f"[!] нет текста промпта для {missing}")
        return 1
    if args.dry_run:
        return 0

    from prime.config import load_config
    from prime.workers.ensemble import INVALID, load_dotenv_if_present, parse_label
    from score_s11_scorer2 import build_worker

    load_dotenv_if_present()
    wcfg = load_config(ROOT / "experiments/S15_toxlang/config_toxlang.yaml")
    worker = build_worker("google/gemma-3-12b-it", wcfg, provider=None, use_logprobs=False)

    out = []
    for n, i in pairs:
        prompt = texts_of[n].format(review=rec["texts"][i])
        try:
            text, _ = worker.call_lp(prompt)
            lab = parse_label(text, "binary", fail_closed=True)
            kind = "recovered" if lab != INVALID else "content"
            out.append({"prompt": n, "row": i, "group": int(c[i]), "gold": int(y[i]),
                        "raw": text, "label": int(lab), "kind": kind})
        except Exception as exc:  # noqa: BLE001
            out.append({"prompt": n, "row": i, "group": int(c[i]), "gold": int(y[i]),
                        "raw": None, "label": -1, "kind": "transport", "error": str(exc)[:200]})
    kinds = {k: sum(1 for o in out if o["kind"] == k) for k in ("recovered", "content", "transport")}
    dst = ROOT / "results" / "requery_invalid" / f"{args.stand}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps({"stand": args.stand, "set": cfg.test_set, "n_pairs": len(pairs),
                               "kinds": kinds, "pairs": out}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"  итог: {kinds}  → {dst.relative_to(ROOT)}")
    for o in out[:5]:
        print(f"    {o['prompt']:<34} строка {o['row']:>4}  {o['kind']:<9} {repr(o['raw'])[:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

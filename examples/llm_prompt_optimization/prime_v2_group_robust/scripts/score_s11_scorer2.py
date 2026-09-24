#!/usr/bin/env python
"""Второй скорер на матрице S11: можно остановить в любой момент и продолжить с того же места.

Зачем второй скорер. Все прежние числа получены на gemma-3-12b-it. Если однострочные правки,
отбор на формат вывода или лотерея протокола — свойство именно этой модели, то на другой
модели они пропадут. Скорер отличается от основного только моделью: тот же шаблон, тот же
разбор ответа (`parse_label`), тот же fail-closed. Дополнительно запрашиваются логпробы,
из них берётся логарифм отношения шансов «токсично / нет» — он нужен для анализа рабочей
точки при равном пороге.

Как устроена остановка.
  * Порядок «по префиксу строк», а не «по промптам». План — это список шагов вида
    `множество:длина`; шаг доводит ВСЕ промпты до первых `длина` строк множества. Строки в
    множестве перемешаны, поэтому любой префикс — стратифицированная подвыборка. После
    прерывания матрица прямоугольна до длины самого отстающего промпта, и анализ можно
    запускать сразу; в прежнем порядке при остановке одни промпты были бы готовы целиком, а
    другие не начаты.
  * Состояние — только файлы: длина `<промпт>.npy` и есть «сколько строк сделано». Ни
    отдельного счётчика, ни базы: нечему рассинхронизироваться.
  * Запись атомарна (временный файл + `os.replace`): убийство процесса в любой момент
    оставляет прежний целый файл. Логарифмы шансов пишутся раньше меток, поэтому при обрыве
    между двумя записями лишнее просто отбрасывается.
  * Мягкая пауза: файл `STOP` в каталоге результатов — процесс закончит текущий блок и выйдет.
  * Блок по умолчанию 120 строк (около 15 секунд работы): столько теряется при жёсткой остановке.

Запуск и продолжение — одна и та же команда.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from score_s11_matrix import collect_prompts, key_usage  # noqa: E402

OUT = ROOT / "results/S11_protocol_matrix"
# Порядок шагов: сначала грубая проверка на обоих множествах (900 строк = 50 на ячейку),
# затем тест до конца, затем валидация. Тест раньше валидации, потому что для лотереи нужна
# валидация не меньше ~2700 строк, а для мощности и рабочей точки достаточно теста.
DEFAULT_PLAN = ("truth_large:900,dev_universe:900,truth_large:1800,truth_large:3600,"
                "dev_universe:2700,dev_universe:4325")
# Логарифм вероятности токена, которого нет среди top_logprobs. Модель почти всегда уверена
# (альтернативная цифра лежит на -18..-27), так что прежний пол -25 обрезал значения; -60
# срабатывает только когда альтернативы действительно нет в ответе.
FLOOR = -60.0

_stop = False


def _on_signal(signum, frame):  # noqa: ARG001
    global _stop
    _stop = True
    print(f"[pause] получен сигнал {signum}: закончу текущий блок и выйду", flush=True)


def atomic_save(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, arr)
    os.replace(tmp, path)


def load_state(pdir: Path, stem: str) -> tuple[np.ndarray, np.ndarray]:
    f, g = pdir / f"{stem}.npy", pdir / f"{stem}.lo.npy"
    pred = np.load(f) if f.is_file() else np.zeros(0, dtype=np.int16)
    lo = np.load(g) if g.is_file() else np.zeros(0, dtype=np.float32)
    if len(lo) >= len(pred):
        lo = lo[: len(pred)]  # хвост мог остаться после обрыва между двумя записями
    else:
        lo = np.concatenate([lo, np.full(len(pred) - len(lo), np.nan, dtype=np.float32)])
    return pred, lo


def label_logodds(choice) -> float:
    """log P('1') - log P('0') в позиции цифры метки; NaN, если логпробов нет."""
    lp = getattr(choice, "logprobs", None)
    content = getattr(lp, "content", None) if lp is not None else None
    if not content:
        return float("nan")
    for tok in content:
        if (getattr(tok, "token", "") or "").strip() in ("0", "1"):
            best = {"0": FLOOR, "1": FLOOR}
            for alt in [tok, *(getattr(tok, "top_logprobs", None) or [])]:
                s = (getattr(alt, "token", "") or "").strip()
                if s in best:
                    best[s] = max(best[s], float(alt.logprob))
            return best["1"] - best["0"]
    return float("nan")


def build_worker(model: str, cfg, provider: str | None = None, use_logprobs: bool = True):
    from prime.workers.ensemble import LLMWorker, _message_text

    class LogprobWorker(LLMWorker):
        """Тот же воркер, но ответ возвращается вместе с логарифмом шансов метки."""

        retries = 0
        providers: Counter = Counter()
        pin_provider: str | None = None  # OpenRouter иначе чередует OpenAI и Azure внутри одного прогона
        use_logprobs = True  # gemma логпробов не отдаёт: для неё --no-logprobs

        def call_lp(self, prompt: str):
            last = None
            for attempt in range(1, self.max_retries + 1):
                try:
                    extra: dict = {}
                    if self.use_logprobs:
                        extra.update(logprobs=True, top_logprobs=20)
                    if self.pin_provider:
                        extra["extra_body"] = {"provider": {"order": [self.pin_provider],
                                                            "allow_fallbacks": False}}
                    resp = self.client.chat.completions.create(
                        model=self.model_uri, messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature, max_tokens=self.max_tokens,
                        timeout=self.timeout, **extra)
                    if not resp.choices:
                        raise RuntimeError("ответ без choices")
                    ch = resp.choices[0]
                    text = _message_text(ch.message)
                    if not text:
                        raise RuntimeError(f"пустой ответ (finish_reason={getattr(ch, 'finish_reason', None)})")
                    prov = getattr(resp, "provider", None)
                    if prov:
                        LogprobWorker.providers[str(prov)] += 1
                    return text, label_logodds(ch)
                except Exception as exc:  # noqa: BLE001
                    last = exc
                    LogprobWorker.retries += 1
                    slow = "429" in str(exc) or "rate-limited" in str(exc).lower()
                    time.sleep(min(30.0, 4.0 * attempt) if slow else 1.5 * attempt)
            raise RuntimeError(f"вызов не удался после {self.max_retries} попыток: {last}")

    LogprobWorker.pin_provider = provider
    LogprobWorker.use_logprobs = use_logprobs
    return LogprobWorker(
        model_name=model, api_base=cfg.ensemble.api_base, temperature=0.0, max_tokens=16,
        timeout=cfg.ensemble.timeout, max_retries=cfg.ensemble.max_retries,
        reasoning_effort="none", label_space="binary", fail_closed=True)


def score_rows(worker, texts: list[str], template: str, max_parallel: int):
    from prime.workers.ensemble import INVALID, parse_label

    def one(i: int):
        try:
            prompt = template.format(review=texts[i])
        except (KeyError, ValueError, IndexError):
            return INVALID, float("nan")  # шаблон не форматируется: fail-closed, как в основном пути
        try:
            text, lo = worker.call_lp(prompt)
            return parse_label(text, "binary", fail_closed=True), lo
        except Exception:  # noqa: BLE001
            return INVALID, float("nan")

    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        res = list(pool.map(one, range(len(texts))))
    return (np.asarray([r[0] for r in res], dtype=np.int16),
            np.asarray([r[1] for r in res], dtype=np.float32))


def parse_plan(text: str) -> list[tuple[str, int]]:
    steps = []
    for item in text.split(","):
        name, _, length = item.strip().partition(":")
        steps.append((name, int(length)))
    return steps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="короткое имя скорера, например gpt4omini")
    ap.add_argument("--model", required=True, help="идентификатор модели в OpenRouter")
    ap.add_argument("--plan", default=DEFAULT_PLAN)
    ap.add_argument("--provider", default=None,
                    help="закрепить провайдера в OpenRouter (для gpt-4o-mini: OpenAI); без этого "
                         "запросы чередуются между OpenAI и Azure")
    ap.add_argument("--config", type=Path, default=ROOT / "experiments/E5_civilcomments/config_prime_main.yaml")
    ap.add_argument("--max-parallel", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=120)
    ap.add_argument("--max-spend", type=float, default=30.0, help="потолок расходов этого запуска, USD")
    ap.add_argument("--limit-prompts", type=int, default=0)
    ap.add_argument("--sets-dir", type=Path, default=None,
                    help="каталог фиксированных множеств; по умолчанию S11 (CivilComments)")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="каталог результатов; по умолчанию results/S11_protocol_matrix/scorer2_<tag>")
    ap.add_argument("--prompts-json", type=Path, default=None,
                    help="JSON {имя: текст промпта}; по умолчанию пул S11 из collect_prompts()")
    ap.add_argument("--no-logprobs", action="store_true", help="не запрашивать логпробы (нужно для gemma)")
    ap.add_argument("--dry-run", action="store_true", help="посчитать объём работы и выйти, не обращаясь к API")
    args = ap.parse_args()

    from prime.config import load_config
    from prime.data.civilcomments_loader import load_civilcomments_splits
    from prime.workers.ensemble import load_dotenv_if_present

    signal.signal(signal.SIGINT, _on_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _on_signal)

    load_dotenv_if_present()
    cfg = load_config(args.config)
    cfg.dataset.max_val_users = max(int(cfg.dataset.max_val_users or 0), 60_000)
    cfg.dataset.max_test_users = max(int(cfg.dataset.max_test_users or 0), 200_000)
    splits_holder: dict = {}

    def get_splits():
        if "s" not in splits_holder:  # CivilComments грузится только если есть множество без inline-текстов
            splits_holder["s"] = load_civilcomments_splits(cfg.dataset, seed=42)
        return splits_holder["s"]

    worker = build_worker(args.model, cfg, args.provider, use_logprobs=not args.no_logprobs)

    root = args.out_root or OUT / f"scorer2_{args.tag}"
    sets_dir = args.sets_dir or OUT / "fixed_sets"
    stop_file = root / "STOP"
    (root / "preds").mkdir(parents=True, exist_ok=True)
    (root / "meta.json").write_text(json.dumps(
        {"tag": args.tag, "model": args.model, "provider": args.provider, "plan": args.plan,
         "max_tokens": 16, "temperature": 0.0, "logprobs": not args.no_logprobs, "top_logprobs": 20,
         "chunk": args.chunk, "sets_dir": str(sets_dir),
         "prompts_json": str(args.prompts_json) if args.prompts_json else None}, indent=2), encoding="utf-8")

    prompts = (list(json.loads(args.prompts_json.read_text(encoding="utf-8")).items())
               if args.prompts_json else collect_prompts())
    if args.limit_prompts:
        prompts = prompts[: args.limit_prompts]
    plan = parse_plan(args.plan)

    sets: dict[str, dict] = {}
    for set_name, _ in plan:
        if set_name in sets:
            continue
        rec = json.loads((sets_dir / f"{set_name}.json").read_text(encoding="utf-8"))
        if "texts" in rec:  # MultiNLI: тексты лежат прямо в JSON
            texts = rec["texts"]
        else:
            split = get_splits()[rec["source_split"]]
            texts = [split.texts[i] for i in rec["indices"]]
        sets[set_name] = {"texts": texts, "fp": rec["fingerprint"]}
        (root / "preds" / set_name).mkdir(parents=True, exist_ok=True)

    # сколько вызовов осталось — до первого обращения к API
    remaining = 0
    final_target: dict[str, int] = {}  # шаги плана перекрываются, поэтому берётся самый длинный на множество
    for set_name, length in plan:
        final_target[set_name] = max(final_target.get(set_name, 0), min(length, len(sets[set_name]["texts"])))
    for set_name, target in final_target.items():
        for name, _ in prompts:
            have = len(load_state(root / "preds" / set_name, name.replace(":", "__"))[0])
            remaining += max(0, target - have)
    print(f"[plan] {args.plan}\n[plan] {len(prompts)} промптов, осталось ~{remaining} вызовов", flush=True)
    if args.dry_run:
        return 0

    start_spend = key_usage()
    last_check, spent = time.time(), 0.0
    print(f"[budget] потолок +${args.max_spend:.2f}", flush=True)
    log = (root / "spend_log.jsonl").open("a", encoding="utf-8")
    t_start, scored_total = time.time(), 0

    def want_stop() -> bool:
        return _stop or stop_file.exists()

    for step, (set_name, length) in enumerate(plan, 1):
        texts = sets[set_name]["texts"]
        target = min(length, len(texts))
        pdir = root / "preds" / set_name
        print(f"\n[шаг {step}/{len(plan)}] {set_name} до {target} строк (fp {sets[set_name]['fp']})", flush=True)
        for pi, (name, tmpl) in enumerate(prompts, 1):
            stem = name.replace(":", "__")
            pred, lo = load_state(pdir, stem)
            while len(pred) < target:
                if want_stop():
                    print("[pause] остановка по запросу; состояние сохранено, продолжение — той же командой",
                          flush=True)
                    return 0
                if time.time() - last_check > 60:
                    now = key_usage()
                    if now is not None and start_spend is not None:
                        spent = now - start_spend
                    last_check = time.time()
                    if spent > args.max_spend:
                        print(f"[stop] расход ${spent:.2f} превысил потолок; выхожу чисто", flush=True)
                        return 2
                a, b = len(pred), min(len(pred) + args.chunk, target)
                t0 = time.time()
                p, l = score_rows(worker, texts[a:b], tmpl, args.max_parallel)
                pred, lo = np.concatenate([pred, p]), np.concatenate([lo, l])
                atomic_save(pdir / f"{stem}.lo.npy", lo)   # логарифмы раньше меток (см. описание)
                atomic_save(pdir / f"{stem}.npy", pred)
                scored_total += b - a
                el = max(time.time() - t_start, 1e-6)
                bad = int((p < 0).sum())
                print(f"  {name:26s} {len(pred):5d}/{target}  {(b - a) / max(time.time() - t0, 1e-6):4.1f}/с  "
                      f"общий {scored_total / el:4.1f}/с  invalid {bad}  повторов {worker.retries}  "
                      f"расход ${spent:.2f}", flush=True)
            if pi % 5 == 0 or pi == len(prompts):
                (root / "status.json").write_text(json.dumps(
                    {"updated": time.strftime("%Y-%m-%d %H:%M:%S"), "step": step, "set": set_name,
                     "target": target, "prompts_done_in_step": pi, "of": len(prompts),
                     "rows_scored_this_run": scored_total, "spent_usd": round(spent, 3),
                     "providers": dict(worker.providers)}, indent=2), encoding="utf-8")
        log.write(json.dumps({"step": step, "set": set_name, "length": target,
                              "spent": round(spent, 3), "time": time.strftime("%H:%M:%S")}) + "\n")
        log.flush()
        print(f"[шаг {step}] готов; расход ${spent:.2f}; провайдеры {dict(worker.providers)}", flush=True)

    print(f"\n[готово] план выполнен, расход ${spent:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

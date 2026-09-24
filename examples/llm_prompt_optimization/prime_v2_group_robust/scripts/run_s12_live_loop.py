#!/usr/bin/env python
"""S12: живая петля в миниатюре — протокол отбора внутри поиска, а не после него.

Зачем. Вся лотерея S11 — симуляция: готовый пул промптов, отбор по нему разными протоколами.
Она не отвечает на вопрос, что происходит, когда протокол встроен в поиск и сам определяет,
какие промпты вообще будут порождены. Здесь три базовых метода запускаются по-настоящему, в
полном факторном плане «метод × протокол × сид».

Что такое протокол здесь. Все базовые методы (APE, APO, EvoPrompt, GEPA) отбирают кандидатов
по одному числу — `R_soft_min_gba`, которое читают из общей функции `compute_fitness`. Раннер
подменяет именно это число на статистику выбранного протокола, ничего не меняя в коде методов:
  soft_min  — как в S9 (без подмены);
  hard_min  — худшая группа, `R_worst_gba`;
  mean_gba  — среднее по группам, `R_gba_mean`;
  global    — общая сбалансированная точность, `R_global`.
Отбраковка по доле неразобранных ответов и по вырожденной доле «токсично» остаётся прежней
для всех протоколов: меняется только то, по чему ранжируются допущенные кандидаты.

Что сохраняется. Помимо финального промпта — ПОЛНЫЙ архив: каждый промпт, который петля
оценивала, со всеми оценками. В S9 промежуточные кандидаты не сохранялись, поэтому отбор внутри
настоящего прогона нельзя было воспроизвести; теперь можно, и любой другой протокол можно
проиграть на архиве того же прогона без нового поиска.

Второй датасет. `--dataset mnli` запускает те же методы на MultiNLI (S13): профиль задачи
переключается на MNLI, обучающие примеры берутся из train, отбор идёт по префиксу
`dev_universe_mnli`; результаты в results/S13_mnli_loop.

Третий датасет. `--dataset toxlang` запускает те же методы на многоязычной токсичности (S15):
задача и seed-промпт — те же, что у `civil` (профиль задачи НЕ переключается, остаётся CIVIL),
меняется только источник текста и ось групп (язык вместо демографической категории); отбор идёт
по префиксу `dev_universe_tox`; результаты в results/S15_toxlang_loop.

Остановка. Единица работы — один прогон (метод, протокол, сид), 20–40 минут. Признак готовности —
файл `done.json`, который пишется последним и атомарно. Недописанный прогон при продолжении
удаляется и начинается заново: у методов нет промежуточных контрольных точек, а из-за
стохастичности мутатора «продолженный» прогон всё равно не был бы тем же прогоном.
  * файл STOP в каталоге результатов — закончить текущий прогон и выйти;
  * жёсткое убийство теряет только текущий прогон.
Порядок «сид снаружи, метод и протокол внутри»: после каждого сида остаётся полная решётка.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "results/S12_live_loop"

# протокол -> ключ метрики, по которому петля выбирает лучшего
PROTOCOLS = {
    "soft_min": None,  # без подмены: ровно то, что делал S9
    "hard_min": "R_worst_gba",
    "mean_gba": "R_gba_mean",
    "global": "R_global",
}
# параметры — как в S9 для сидов 43 и 44 (run_e5_s9_overnight.py)
EVO_POP, EVO_GEN, GEPA_REFLECTIONS, APO_ROUNDS = 4, 3, 8, 2
LOGGED_KEYS = ("R_soft_min_gba", "R_worst_gba", "R_gba_mean", "R_global", "R_worst_group",
               "toxic_recall", "invalid_rate", "pred_pos_rate")


class ArchiveTap:
    """Перехватывает compute_fitness: пишет каждую оценку и подменяет число для отбора."""

    def __init__(self, protocol: str):
        self.key = PROTOCOLS[protocol]
        self.records: dict[str, dict] = {}
        self._orig = None

    def install(self) -> None:
        import prime.fitness.objective as obj

        self._orig = obj.compute_fitness
        tap = self

        def wrapped(predictions, gold, user_ids, prompt, cfg, *args, **kwargs):
            fit = tap._orig(predictions, gold, user_ids, prompt, cfg, *args, **kwargs)
            rejected = fit.get("reject_reason")
            used = fit.get("R_soft_min_gba", fit.get("fitness"))
            if not rejected and tap.key and tap.key in fit:
                used = float(fit[tap.key])
                # baselines читают либо R_soft_min_gba, либо fitness — подменяем оба
                fit["R_soft_min_gba"] = used
                fit["fitness"] = fit["combined_score"] = used
            h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
            rec = tap.records.setdefault(h, {"hash": h, "prompt": prompt, "evals": []})
            ev = {"rows": int(len(predictions)), "selected_on": None if used is None else round(float(used), 5),
                  "rejected": rejected}
            for k in LOGGED_KEYS:
                if k in fit and fit[k] is not None:
                    try:
                        ev[k] = round(float(fit[k]), 5)
                    except (TypeError, ValueError):
                        pass
            rec["evals"].append(ev)
            return fit

        obj.compute_fitness = wrapped

    def uninstall(self) -> None:
        import prime.fitness.objective as obj

        if self._orig is not None:
            obj.compute_fitness = self._orig


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def mnli_train_set(seed: int, group_names: dict, per_cell: int = 36):
    """Обучающие примеры для методов: по `per_cell` строк на ячейку (жанр x метка) из train MultiNLI.

    В train всего пять жанров (fiction, government, slate, telephone, travel), поэтому выходит
    360 строк, как и S_source в CivilComments. Остальные пять жанров методы не видят вовсе, и на
    них проверочные множества дают естественный сдвиг.
    """
    import numpy as np
    from baselines.api import LabeledSet
    from build_s13_mnli import render
    from datasets import load_dataset

    ds = load_dataset("nyu-mll/multi_nli", split="train")
    gid = {v: int(k) for k, v in group_names.items()}
    cells: dict = {}
    for i, (g, lab) in enumerate(zip(ds["genre"], ds["label"])):
        if int(lab) in (0, 1, 2):
            cells.setdefault((g, int(int(lab) == 2)), []).append(i)
    rng = np.random.RandomState(seed)
    rows = [(int(i), key) for key in sorted(cells) for i in rng.choice(cells[key], per_cell, replace=False)]
    texts, labels, groups = [], [], []
    for j in rng.permutation(len(rows)):
        i, (g, y) = rows[j]
        r = ds[i]
        texts.append(render({"premise": r["premise"], "hypothesis": r["hypothesis"]}))
        labels.append(y)
        groups.append(gid[g])
    return LabeledSet(texts=texts, labels=labels, group_ids=groups)


def toxlang_train_set(seed: int, group_names: dict, per_cell: int = 36):
    """Обучающие примеры для методов: по `per_cell` строк на ячейку (язык x метка) из того же
    источника, что и S15 (multilingual_toxicity_dataset), за вычетом строк, уже занятых под
    truth_tox/dev_universe_tox — иначе методы подглядывали бы в проверочные множества.

    per_cell=36 — то же значение, что у S13 (mnli_train_set) и что использовалось для
    CivilComments; при 6 языках x 2 метках получается 432 строки, того же порядка, что и там.
    """
    import numpy as np
    from baselines.api import LabeledSet
    from build_s15_toxlang import DATASET, LANGS
    from datasets import load_dataset

    used = set()
    for name in ("truth_tox", "dev_universe_tox"):
        rec = json.loads((ROOT / "results/S15_toxlang_matrix/fixed_sets" / f"{name}.json")
                         .read_text(encoding="utf-8"))
        used.update(rec["texts"])

    gid = {v: int(k) for k, v in group_names.items()}
    rng = np.random.RandomState(seed)
    rows = []  # (text, label, group_id)
    for lang in LANGS.values():
        ds = load_dataset(DATASET, split=lang)
        cells: dict = {}
        for i, (t, y) in enumerate(zip(ds["text"], ds["toxic"])):
            if t in used:
                continue
            cells.setdefault(int(y), []).append(i)
        for y in sorted(cells):
            for i in rng.choice(cells[y], per_cell, replace=False):
                rows.append((ds[int(i)]["text"], y, gid[lang]))
    texts, labels, groups = [], [], []
    for j in rng.permutation(len(rows)):
        t, y, g = rows[j]
        texts.append(t)
        labels.append(y)
        groups.append(g)
    return LabeledSet(texts=texts, labels=labels, group_ids=groups)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", default="ape,evoprompt_de,gepa")
    ap.add_argument("--protocols", default="soft_min,hard_min,global")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--config", type=Path, default=ROOT / "experiments/E5_civilcomments/config_prime_main.yaml")
    ap.add_argument("--fixed-dir", type=Path, default=ROOT / "experiments/E5_civilcomments/fixed_sets")
    ap.add_argument("--max-parallel", type=int, default=4,
                    help="измеренный потолок для gemma через OpenRouter; при 16 — 78%% ошибок 429")
    ap.add_argument("--max-spend", type=float, default=25.0, help="потолок расходов запуска, USD")
    ap.add_argument("--u-target-n", type=int, default=4000)
    ap.add_argument("--dataset", choices=["civil", "mnli", "toxlang"], default="civil",
                    help="civil = CivilComments (S12); mnli = MultiNLI (S13), результаты в results/S13_mnli_loop; "
                         "toxlang = многоязычная токсичность (S15), результаты в results/S15_toxlang_loop")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    protocols = [p.strip() for p in args.protocols.split(",") if p.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    for p in protocols:
        if p not in PROTOCOLS:
            raise SystemExit(f"неизвестный протокол {p!r}; есть {sorted(PROTOCOLS)}")

    from baselines.ape import APEOptimizer
    from baselines.apo import APOOptimizer
    from baselines.api import LabeledSet, Task, UnlabeledSet
    from baselines.evoprompt import EvoPromptOptimizer
    from baselines.gepa_baseline import GEPAOptimizer
    from baselines.optimizer_llm import build_optimizer_llm
    from baselines.regime_sets import load_or_build_regime_sets
    from prime.config import load_config
    from prime.data.civilcomments_loader import load_civilcomments_splits
    from prime.data.fixed_sets import FixedSet, materialize_split
    from prime.experiment.budget import TokenTracker
    from prime.workers.ensemble import load_dotenv_if_present
    from prime.workers.scorer import Scorer
    from score_s11_matrix import key_usage

    load_dotenv_if_present()
    cfg = load_config(args.config)
    cfg.ensemble.max_parallel = args.max_parallel
    cfg.dataset.max_val_users = max(int(cfg.dataset.max_val_users or 0), 50_000)
    cfg.dataset.max_test_users = max(int(cfg.dataset.max_test_users or 0), 80_000)
    group_names = None
    if args.dataset == "civil":
        # индексы d_dev зафиксированы под seed=42 загрузчика; менять нельзя (см. run_e5_s9_matrix.py)
        splits = load_civilcomments_splits(cfg.dataset, seed=42)
        d_dev = FixedSet.load(args.fixed_dir / "d_dev.json")
        mat_dev = materialize_split(splits["validation"], d_dev.indices)
        dev_set = LabeledSet(texts=mat_dev["texts"], labels=mat_dev["labels"], group_ids=mat_dev["cluster_ids"])
        seed_prompt = (ROOT / "prompts/initial_prompt_civilcomments.txt").read_text(encoding="utf-8")
    elif args.dataset == "mnli":
        from baselines import task_profile

        task_profile.set_active(task_profile.MNLI)
        splits = None
        rec = json.loads((ROOT / "results/S13_mnli_matrix/fixed_sets/dev_universe_mnli.json")
                         .read_text(encoding="utf-8"))
        n_dev = 900  # префикс перемешанного множества = стратифицированная подвыборка, ~45 на ячейку
        dev_set = LabeledSet(texts=rec["texts"][:n_dev], labels=rec["labels"][:n_dev],
                             group_ids=rec["cluster_ids"][:n_dev])
        group_names = rec["group_names"]
        seed_prompt = (ROOT / "experiments/S13_mnli/prompts/seed.txt").read_text(encoding="utf-8")
    else:  # toxlang
        # профиль задачи НЕ меняется: task_profile.set_active(...) сюда не добавляем. По умолчанию
        # активен профиль CIVIL, и это ровно то, что нужно — задача здесь та же самая бинарная
        # токсичность, что у CivilComments, меняется только язык текста и ось групп (язык вместо
        # демографической категории). Если кто-то «поправит» это на MNLI/toxlang-профиль — это
        # будет ошибкой: подмена профиля задачи здесь не нужна и не предусмотрена.
        splits = None
        rec = json.loads((ROOT / "results/S15_toxlang_matrix/fixed_sets/dev_universe_tox.json")
                         .read_text(encoding="utf-8"))
        n_dev = 900  # префикс перемешанного множества = стратифицированная подвыборка, ~75 на ячейку (12 ячеек)
        dev_set = LabeledSet(texts=rec["texts"][:n_dev], labels=rec["labels"][:n_dev],
                             group_ids=rec["cluster_ids"][:n_dev])
        group_names = rec["group_names"]
        seed_prompt = (ROOT / "prompts/initial_prompt_civilcomments.txt").read_text(encoding="utf-8")
    scorer = Scorer(cfg.ensemble, label_space="binary", use_mock=False)
    cfg.budget.on_exhausted = "warn"

    if args.dataset == "civil":
        out_root = OUT
    elif args.dataset == "mnli":
        out_root = ROOT / "results/S13_mnli_loop"
    else:
        out_root = ROOT / "results/S15_toxlang_loop"
    out_root.mkdir(parents=True, exist_ok=True)
    stop_file = out_root / "STOP"
    (out_root / "meta.json").write_text(json.dumps(
        {"dataset": args.dataset, "methods": methods, "protocols": protocols, "seeds": seeds, "evo_pop": EVO_POP,
         "evo_generations": EVO_GEN, "gepa_reflections": GEPA_REFLECTIONS, "apo_rounds": APO_ROUNDS,
         "max_parallel": args.max_parallel,
         "dev_fixed_set": {"civil": "d_dev", "mnli": "dev_universe_mnli",
                           "toxlang": "dev_universe_tox"}[args.dataset],
         "dev_n": len(dev_set.texts)},
        indent=2), encoding="utf-8")

    plan = [(s, m, p) for s in seeds for m in methods for p in protocols]
    todo = [t for t in plan if not (out_root / f"seed{t[0]}" / f"{t[1]}__{t[2]}" / "done.json").is_file()]
    print(f"[plan] {len(plan)} прогонов, готово {len(plan) - len(todo)}, осталось {len(todo)}", flush=True)

    start_spend = key_usage()
    spent = 0.0
    regime_cache: dict[int, dict] = {}

    for idx, (seed, method, protocol) in enumerate(todo, 1):
        if stop_file.exists():
            print("[pause] найден STOP; выхожу между прогонами, всё готовое сохранено", flush=True)
            return 0
        now = key_usage()
        if now is not None and start_spend is not None:
            spent = now - start_spend
        if spent > args.max_spend:
            print(f"[stop] расход ${spent:.2f} превысил потолок ${args.max_spend:.2f}", flush=True)
            return 2

        rundir = out_root / f"seed{seed}" / f"{method}__{protocol}"
        if rundir.exists():
            shutil.rmtree(rundir)  # недописанный прогон: продолжать бессмысленно (см. описание)
        rundir.mkdir(parents=True)
        print(f"\n=== [{idx}/{len(todo)}] seed={seed} method={method} protocol={protocol} "
              f"(расход ${spent:.2f}) ===", flush=True)

        cfg.active_learning.seed = int(seed)
        if seed not in regime_cache:
            if args.dataset == "civil":
                regime_cache[seed] = load_or_build_regime_sets(
                    splits["train"], args.fixed_dir, seed=int(seed), s_per_cell=45, u_n=int(args.u_target_n))
            elif args.dataset == "mnli":
                regime_cache[seed] = {"s_source": mnli_train_set(int(seed), group_names)}
            else:
                regime_cache[seed] = {"s_source": toxlang_train_set(int(seed), group_names)}
        train_set = regime_cache[seed]["s_source"]
        budget = TokenTracker.from_cfg(cfg.budget)

        def task(unlabeled, temp: float) -> Task:
            return Task(train=train_set, dev=dev_set, unlabeled=unlabeled, label_budget=240, scorer=scorer,
                        optimizer_llm=build_optimizer_llm(cfg, temperature=temp), budget=budget,
                        seed_prompt=seed_prompt, meta={"seed": int(seed)})

        tap = ArchiveTap(protocol)
        tap.install()
        t0 = time.time()
        try:
            if method == "ape":
                result = APEOptimizer(k_prompts=6, n_shots=36, temperature=0.0).run(
                    task(UnlabeledSet(texts=[]), 0.0))
            elif method == "evoprompt_de":
                result = EvoPromptOptimizer(mode="de", population_size=EVO_POP, generations=EVO_GEN).run(
                    task(UnlabeledSet(texts=[]), 0.8))
            elif method == "evoprompt_ga":
                result = EvoPromptOptimizer(mode="ga", population_size=EVO_POP, generations=EVO_GEN).run(
                    task(UnlabeledSet(texts=[]), 0.8))
            elif method == "gepa":
                result = GEPAOptimizer(max_metric_calls=20_000, reflections=GEPA_REFLECTIONS).run(
                    task(UnlabeledSet(texts=[]), 0.7))
            elif method == "apo":
                result = APOOptimizer(rounds=APO_ROUNDS, beam_size=4).run(task(UnlabeledSet(texts=[]), 0.7))
            else:
                raise SystemExit(f"метод {method!r} в S12 не предусмотрен")
        finally:
            tap.uninstall()

        atomic_write_text(rundir / "best_prompt.txt", result.best_prompt)
        atomic_write_text(rundir / "archive.json", json.dumps(
            {"protocol": protocol, "method": method, "seed": seed,
             "all_candidates": list(result.all_candidates or []),
             "evaluated": list(tap.records.values())}, ensure_ascii=False))
        atomic_write_text(rundir / "optimizer_result.json", json.dumps(
            {"n_candidates": len(result.all_candidates or []), "trace": result.trace,
             "scorer_calls": result.scorer_calls, "optimizer_calls": result.optimizer_calls},
            ensure_ascii=False, indent=2, default=str))
        dt = time.time() - t0
        # признак готовности — последним
        atomic_write_text(rundir / "done.json", json.dumps(
            {"seconds": round(dt, 1), "unique_prompts_evaluated": len(tap.records),
             "scorer_calls": result.scorer_calls, "optimizer_calls": result.optimizer_calls,
             "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
        print(f"    готово за {dt / 60:.1f} мин: оценено {len(tap.records)} уникальных промптов, "
              f"scorer_calls {result.scorer_calls}, optimizer_calls {result.optimizer_calls}", flush=True)

    print("\n[готово] весь план выполнен", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

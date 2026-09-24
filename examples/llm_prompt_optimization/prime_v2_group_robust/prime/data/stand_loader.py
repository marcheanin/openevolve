"""Загрузчик MultiNLI и multilingual-toxicity ("toxlang") в контракт ReviewSplit.

CivilComments и Amazon грузят свои собственные официальные сплиты (см.
``civilcomments_loader.py`` / ``wilds_loader.py``). Для этих двух стендов validation/test
уже существуют как готовые, зафиксированные наборы — ``dev_universe_*.json`` и
``truth_*.json`` под ``results/S13_mnli_matrix`` и ``results/S15_toxlang_matrix`` — те же
самые файлы, на которых считаются базовые методы (APE/EvoPrompt/GEPA/APO) в
``scripts/run_s12_live_loop.py``. Единственное, чего у PRIME нет — обучающего пула, из
которого он берёт D_select/D_anchor: он строится здесь из исходного источника HF, по тем же
принципам, что ``mnli_train_set`` / ``toxlang_train_set`` в живом цикле (сбалансированные
ячейки группа x метка, тексты проверочных множеств исключены).

Инвариант выравнивания: validation/test читаются из JSON КАК ЕСТЬ, без перестановки строк —
``d_dev.json`` / ``test_fixed.json`` (см. ``scripts/build_s16_prime_fixed_sets.py``) хранят
индексы 0..899 / 0..999 в порядке этих файлов, и это должно совпадать с тем, что видят базовые
методы (первые 900/1000 строк).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from prime.config import DatasetCfg
from prime.data.cache import resolve_cache_dir
from prime.data.wilds_loader import ReviewSplit

ROOT = Path(__file__).resolve().parents[2]

# id группы 0 = "none"-заглушка (ни у одной строки такого id нет; нужна только чтобы
# ORACLE_GROUP_NAMES-подобная схема id-в-имя была единообразна с CivilComments, где
# 0 = "none" — реальная группа). Держим её ради общего кода контроллера (balanced_cells,
# GBA-дашборд), не ради семантики.
STAND_GROUP_NAMES: Dict[str, Tuple[str, ...]] = {
    "toxlang": ("none", "en", "de", "ru", "ar", "hi", "am"),
    "mnli": ("none", "facetoface", "fiction", "government", "letters", "nineeleven",
            "oup", "slate", "telephone", "travel", "verbatim"),
}

# toxlang's tightest cell (German, toxic=1) has 1890 rows left after excluding the fixed
# val/test texts (see scripts/build_s16_prime_fixed_sets.py's exclusion set); 1800 leaves
# margin. MNLI's train genres have tens of thousands of rows per cell, so this default is
# binding only for toxlang.
TRAIN_PER_CELL_DEFAULT = 1800
POOL_SEED = 0  # обучающий пул фиксирован по сиду и не зависит от active_learning.seed прогона
CACHE_VERSION = 1


def stand_group_names(name: str) -> Tuple[str, ...]:
    return STAND_GROUP_NAMES[name]


def _fixed_json(stand: str, kind: str) -> Path:
    d = {"toxlang": ROOT / "results/S15_toxlang_matrix/fixed_sets",
         "mnli": ROOT / "results/S13_mnli_matrix/fixed_sets"}[stand]
    f = {"validation": "dev_universe_tox.json" if stand == "toxlang" else "dev_universe_mnli.json",
         "test": "truth_tox.json" if stand == "toxlang" else "truth_mnli.json"}[kind]
    return d / f


def _load_fixed_split(stand: str, kind: str, offset: int) -> ReviewSplit:
    import json

    rec = json.loads(_fixed_json(stand, kind).read_text(encoding="utf-8"))
    n = len(rec["texts"])
    return ReviewSplit(
        name=kind,
        texts=list(rec["texts"]),
        labels=[int(x) for x in rec["labels"]],
        user_ids=[offset + i for i in range(n)],
        example_cluster_ids=[int(x) for x in rec["cluster_ids"]],
    )


def _toxlang_pool(per_cell: int, exclude: set) -> List[Tuple[str, int, int]]:
    from build_s15_toxlang import DATASET, LANGS  # scripts/ on sys.path via cli entrypoint
    from datasets import load_dataset

    rng = np.random.RandomState(POOL_SEED)
    rows: List[Tuple[str, int, int]] = []
    for gid, lang in LANGS.items():
        ds = load_dataset(DATASET, split=lang)
        cells: Dict[int, List[int]] = {}
        for i, (t, y) in enumerate(zip(ds["text"], ds["toxic"])):
            if t.strip() in exclude:
                continue
            cells.setdefault(int(y), []).append(i)
        for y in sorted(cells):
            pool = cells[y]
            if len(pool) < per_cell:
                raise ValueError(f"toxlang pool cell (lang={lang}, y={y}) has only "
                                 f"{len(pool)} rows after exclusion, need {per_cell}")
            for i in rng.choice(pool, per_cell, replace=False):
                rows.append((ds[int(i)]["text"].strip(), y, gid))
    return rows


def _mnli_pool(per_cell: int, exclude: set) -> List[Tuple[str, int, int]]:
    from build_s13_mnli import render  # scripts/
    from datasets import load_dataset

    rng = np.random.RandomState(POOL_SEED)
    ds = load_dataset("nyu-mll/multi_nli", split="train")
    name_to_id = {n: i for i, n in enumerate(STAND_GROUP_NAMES["mnli"])}
    cells: Dict[Tuple[str, int], List[int]] = {}
    for i, (g, lab) in enumerate(zip(ds["genre"], ds["label"])):
        if int(lab) not in (0, 1, 2):
            continue
        text = render({"premise": ds[i]["premise"], "hypothesis": ds[i]["hypothesis"]})
        if text in exclude:
            continue
        cells.setdefault((g, int(int(lab) == 2)), []).append(i)
    rows: List[Tuple[str, int, int]] = []
    for (g, y), pool in sorted(cells.items()):
        if len(pool) < per_cell:
            raise ValueError(f"mnli pool cell (genre={g}, y={y}) has only {len(pool)} rows "
                             f"after exclusion, need {per_cell}")
        gid = name_to_id[g]
        for i in rng.choice(pool, per_cell, replace=False):
            r = ds[int(i)]
            rows.append((render({"premise": r["premise"], "hypothesis": r["hypothesis"]}), y, gid))
    return rows


def _build_pool(cfg: DatasetCfg, name: str, per_cell: int) -> List[Tuple[str, int, int]]:
    exclude: set = set()
    for kind in ("validation", "test"):
        rec_path = _fixed_json(name, kind)
        import json

        exclude.update(json.loads(rec_path.read_text(encoding="utf-8"))["texts"])
    if name == "toxlang":
        rows = _toxlang_pool(per_cell, exclude)
    elif name == "mnli":
        rows = _mnli_pool(per_cell, exclude)
    else:
        raise ValueError(f"unsupported stand: {name}")
    overlap = exclude & {t for t, _, _ in rows}
    if overlap:
        raise RuntimeError(f"{name}: train pool overlaps validation/test on "
                           f"{len(overlap)} texts — disjointness violated")
    return rows


def load_stand_splits(cfg: DatasetCfg, seed: int = 42) -> Dict[str, ReviewSplit]:
    """dataset.name in ('mnli', 'toxlang') -> {'train','validation','test'} как ReviewSplit."""
    name = cfg.name
    if name not in STAND_GROUP_NAMES:
        raise ValueError(f"load_stand_splits: unsupported dataset {name!r}")

    validation = _load_fixed_split(name, "validation", offset=10_000_000)
    test = _load_fixed_split(name, "test", offset=20_000_000)

    per_cell = int(getattr(cfg, "stand_train_per_cell", 0) or TRAIN_PER_CELL_DEFAULT)
    cache_root = resolve_cache_dir(cfg.data_root, cfg.cache_dir)
    cache_path = cache_root / f"stand_{name}_train_pc{per_cell}_v{CACHE_VERSION}.pkl"
    rows = None
    if cfg.use_cache and cache_path.is_file():
        with open(cache_path, "rb") as fh:
            payload = pickle.load(fh)
        if payload.get("version") == CACHE_VERSION and payload.get("per_cell") == per_cell:
            rows = payload["rows"]
    if rows is None:
        rows = _build_pool(cfg, name, per_cell)
        if cfg.use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as fh:
                pickle.dump({"version": CACHE_VERSION, "per_cell": per_cell, "rows": rows}, fh,
                           protocol=pickle.HIGHEST_PROTOCOL)

    order = np.random.RandomState(POOL_SEED).permutation(len(rows))
    texts = [rows[i][0] for i in order]
    labels = [rows[i][1] for i in order]
    group_ids = [rows[i][2] for i in order]
    max_n = cfg.max_train_users
    if max_n and max_n < len(texts):
        texts, labels, group_ids = texts[:max_n], labels[:max_n], group_ids[:max_n]
    train = ReviewSplit(name="train", texts=texts, labels=labels,
                        user_ids=list(range(len(texts))), example_cluster_ids=group_ids)

    train_texts = set(train.texts)
    overlap = train_texts & (set(validation.texts) | set(test.texts))
    if overlap:
        raise RuntimeError(f"{name}: train/validation/test overlap on {len(overlap)} texts")

    return {"train": train, "validation": validation, "test": test}

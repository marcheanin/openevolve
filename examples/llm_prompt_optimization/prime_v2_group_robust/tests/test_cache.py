import numpy as np

from prime.data.cache import (
    embeddings_cache_path,
    load_raw_split_pickle,
    raw_split_cache_path,
    resolve_cache_dir,
    save_embeddings_pickle,
    save_raw_split_pickle,
    splits_memory_key,
    load_embeddings_pickle,
)


def test_raw_split_pickle_roundtrip(tmp_path):
    path = raw_split_cache_path(tmp_path, "train", None, 1)
    save_raw_split_pickle(path, ["a", "b"], [1, 2], [10, 11])
    data = load_raw_split_pickle(path)
    assert data is not None
    assert data["texts"] == ["a", "b"]
    assert data["labels"] == [1, 2]
    assert data["user_ids"] == [10, 11]


def test_embeddings_pickle_roundtrip(tmp_path):
    path = embeddings_cache_path(tmp_path, "all-MiniLM-L6-v2", "train")
    emb = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    save_embeddings_pickle(path, emb, 2)
    loaded = load_embeddings_pickle(path)
    assert loaded is not None
    assert loaded.shape == (2, 2)
    assert float(loaded[0, 0]) == 1.0


def test_splits_memory_key_stable():
    k1 = splits_memory_key("./data", None, 1, 100, 50, 50, 10, 42)
    k2 = splits_memory_key("./data", None, 1, 100, 50, 50, 10, 42)
    k3 = splits_memory_key("./data", None, 1, 100, 50, 50, 10, 43)
    assert k1 == k2
    assert k1 != k3


def test_resolve_cache_dir_default(tmp_path):
    p = resolve_cache_dir(str(tmp_path / "wilds_data"), None)
    assert p.name == ".prime_cache"
    assert p.is_dir()

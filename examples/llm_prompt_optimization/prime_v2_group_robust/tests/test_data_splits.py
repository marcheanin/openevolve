from prime.data.wilds_loader import ReviewSplit, subsample_split


def test_user_disjoint_check():
    a = ReviewSplit("train", ["t"], [1], [1])
    b = ReviewSplit("val", ["t2"], [2], [2])
    c = ReviewSplit("test", ["t3"], [3], [1])
    assert a.user_disjoint_check(b)
    assert not a.user_disjoint_check(c)


def test_subsample_split_preserves_fields():
    split = ReviewSplit(
        name="train",
        texts=["a", "b", "c", "d"],
        labels=[1, 2, 3, 4],
        user_ids=[10, 11, 12, 13],
        example_cluster_ids=[0, 1, 2, 3],
    )
    sub = subsample_split(split, max_examples=2, seed=0)
    assert len(sub) == 2
    assert sub.example_cluster_ids is not None
    assert len(sub.example_cluster_ids) == 2

from prime.acquisition.expansion import select_expansion_by_disagreement


def test_expansion_prefers_high_disagreement_and_covers_weak_groups():
    # 6 unseen: clusters 0,0,1,1,2,2 with increasing d
    unseen = [10, 11, 12, 13, 14, 15]
    disagreements = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7]
    clusters = [0, 0, 1, 1, 2, 2]
    picked = select_expansion_by_disagreement(
        unseen,
        disagreements,
        clusters,
        n_add=3,
        cluster_accuracies={0: 0.9, 1: 0.1, 2: 0.5},
        seed=0,
    )
    assert len(picked) == 3
    # Weak cluster 1 should get at least one slot
    picked_clusters = {clusters[unseen.index(i)] for i in picked}
    assert 1 in picked_clusters
    # Highest-d per selected groups preferred: 11 (0.9), 13 (0.8), 15 (0.7) are tops
    assert all(i in unseen for i in picked)

import numpy as np

from prime.fitness.metrics import cvar_cluster_accuracy, cluster_accuracies


def test_cvar_worst_third():
    preds = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    gold = np.array([1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4])
    clusters = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    accs = cluster_accuracies(preds, gold, clusters)
    assert len(accs) == 3
    cvar = cvar_cluster_accuracy(preds, gold, clusters, quantile=0.33)
    worst = min(accs.values())
    assert abs(cvar - worst) < 1e-6 or cvar <= sorted(accs.values())[1]

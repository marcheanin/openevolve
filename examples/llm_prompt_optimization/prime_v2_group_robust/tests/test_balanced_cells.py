"""F9 balanced cell sampler + fingerprints."""

from prime.data.balanced_cells import (
    cell_counts,
    fingerprint_indices,
    sample_balanced_cells,
)
from prime.data.fixed_sets import assert_disjoint


def test_exact_cell_counts():
    labels = []
    groups = []
    for g in (1, 2):
        for lab in (0, 1):
            labels.extend([lab] * 30)
            groups.extend([g] * 30)
    idx = sample_balanced_cells(labels, groups, groups=(1, 2), per_cell=10, seed=0)
    assert len(idx) == 40
    counts = cell_counts(labels, groups, idx)
    assert all(v == 10 for v in counts.values())


def test_fingerprint_stable():
    a = fingerprint_indices([3, 1, 2])
    b = fingerprint_indices([3, 1, 2])
    c = fingerprint_indices([1, 2, 3])
    assert a == b
    assert a != c


def test_disjoint_with_exclude():
    labels = []
    groups = []
    for g in (0, 1):
        for lab in (0, 1):
            labels.extend([lab] * 40)
            groups.extend([g] * 40)
    idx1 = sample_balanced_cells(labels, groups, groups=(0, 1), per_cell=5, seed=1)
    idx2 = sample_balanced_cells(
        labels, groups, groups=(0, 1), per_cell=5, seed=2, exclude=set(idx1)
    )
    assert_disjoint(idx1, idx2)
    assert len(idx1) == 20
    assert len(idx2) == 20

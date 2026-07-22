from prime.experiment.proxy_validation import pearson_correlation, proxy_validation_report
import pytest


def test_pearson_perfect_correlation():
    assert pearson_correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == 1.0


def test_proxy_report_not_ready_with_one_cycle():
    report = proxy_validation_report([(0.5, 0.4)], min_cycles=2)
    assert report["ready"] is False
    assert report["correlation"] is None


def test_proxy_report_ready_with_two_cycles():
    report = proxy_validation_report([(0.4, 0.3), (0.6, 0.5)], min_cycles=2)
    assert report["ready"] is True
    assert report["correlation"] == pytest.approx(1.0)

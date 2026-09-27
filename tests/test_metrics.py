"""Regression tests for threshold coverage and calibration boundaries."""
import pytest

from benchmark import coverage_at_precision, ece


def test_coverage_cannot_split_equal_confidences():
    assert coverage_at_precision([0.9, 0.9], [1, 0]) == 0.0
    assert coverage_at_precision([0.9, 0.9], [0, 1]) == 0.0


def test_coverage_accepts_whole_tie_groups():
    assert coverage_at_precision([0.9, 0.8, 0.8, 0.7], [1, 1, 1, 0]) == 0.75
    assert coverage_at_precision([0.9, 0.8, 0.8], [1, 1, 0]) == pytest.approx(1 / 3)
    assert coverage_at_precision([0.9, 0.9], [1, 1]) == 1.0


def test_coverage_empty_input():
    assert coverage_at_precision([], []) == 0.0


@pytest.mark.parametrize("conf,correct,target", [
    ([0.9], [], 0.95), ([float("nan")], [1], 0.95),
    ([1.1], [1], 0.95), ([0.9], [2], 0.95),
    ([0.9], [1], 0), ([0.9], [1], 1.1),
])
def test_coverage_rejects_invalid_inputs(conf, correct, target):
    with pytest.raises(ValueError):
        coverage_at_precision(conf, correct, target)


def test_ece_includes_zero_confidence():
    assert ece([0.0, 1.0], [1, 1]) == pytest.approx(0.5)

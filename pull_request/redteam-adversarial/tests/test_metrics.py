import pytest

from redteam_adversarial.metrics import expected_calibration_error


def test_ece_is_zero_when_calibrated():
    probs = [0.25, 0.25, 0.25, 0.25]
    labels = [1, 0, 0, 0]
    assert expected_calibration_error(probs, labels) == pytest.approx(0.0)


def test_ece_penalizes_overconfidence():
    probs = [0.95, 0.95, 0.95, 0.95]
    labels = [1, 0, 1, 0]
    assert expected_calibration_error(probs, labels) == pytest.approx(0.45)


def test_ece_rejects_length_mismatch():
    with pytest.raises(ValueError):
        expected_calibration_error([0.5], [1, 0])

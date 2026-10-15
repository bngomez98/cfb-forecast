"""The calibration check that weekly validation reports."""
import numpy as np
import pytest

from cfb_forecast.evaluation.evaluate import calibration, calibration_test


def test_honest_probabilities_pass_and_biased_ones_fail():
    rng = np.random.default_rng(7)
    p = rng.uniform(0.05, 0.95, 4000)
    honest = rng.random(4000) < p
    assert calibration_test(honest, p)["calibrated_at_5_percent"]
    # Outcomes that come true less often than stated for favorites and more for
    # underdogs: the classic overconfident forecast.
    shrunk = 0.5 + 0.6 * (p - 0.5)
    overconfident = rng.random(4000) < shrunk
    report = calibration_test(overconfident, p)
    assert not report["calibrated_at_5_percent"]
    assert report["p_value"] < 0.001
    assert report["bins_outside_95_range"] >= 2


def test_bin_range_matches_hand_calculation():
    p = np.array([0.62, 0.64, 0.66, 0.68])
    y = np.array([1, 1, 0, 0])
    row = calibration(y, p).iloc[0]
    assert row.games == 4
    assert row.mean_probability == pytest.approx(0.65)
    assert row.gap == pytest.approx(0.5 - 0.65)
    assert row.gap_standard_error == pytest.approx(np.sqrt(np.sum(p * (1 - p))) / 4)
    assert not row.outside_95_range


def test_spiegelhalter_z_matches_formula():
    p = np.array([0.2, 0.4, 0.7, 0.9, 0.55])
    y = np.array([0, 1, 1, 1, 0])
    z = np.sum((y - p) * (1 - 2 * p)) / np.sqrt(np.sum((1 - 2 * p) ** 2 * p * (1 - p)))
    assert calibration_test(y, p)["spiegelhalter_z"] == pytest.approx(z)

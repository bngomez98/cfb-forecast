"""Behavioral invariants for the existing score model and causal adjustments."""

import numpy as np
import pandas as pd
import pytest

from cfb_forecast.modeling.context_model import apply_score_context, select_context_model
from cfb_forecast.modeling.forecast import _out_of_fold_location
from cfb_forecast.modeling.forecast_model import ForecastModel, KNOWN_COLUMNS


def test_legacy_winner_curve_and_slate_shrinkage_cannot_override_scores():
    model = ForecastModel.identity()
    # Old artifacts still load, but their independent winner curve and
    # request-dependent shrinkage cannot alter the common score distribution.
    model.logistic_ready = True
    model.logistic_intercept = -10
    model.logistic_coef = -1
    model.reconcile = 0.8
    model = ForecastModel.from_dict(model.to_dict())
    means = np.array([[31., 21.], [7., 63.]])
    known = np.full((2, len(KNOWN_COLUMNS)), np.nan)
    cov = [[100., 15.], [15., 100.]]
    markets = [{"home_spread": 0., "total_line": 52.5}, {}]
    together = model.apply(means, known, cov, markets, [1, 2])
    alone = model.apply(means[:1], known[:1], cov, markets[:1], [1])[0]
    assert together[0] == alone
    assert alone["home_win_probability"] > .5
    assert alone["home_win_probability"] == pytest.approx(alone["home_cover_probability"])
    assert alone["home_win_probability"] == alone["gaussian_home_win_probability"]
    assert alone["home_poisson_rate"] == alone["expected_home_score"]


def test_missing_pregame_information_is_not_treated_as_observed_average():
    rng = np.random.default_rng(17)
    n = 300
    location = np.full((n, 2), 24.)
    known = np.full((n, len(KNOWN_COLUMNS)), np.nan)
    values = rng.normal(size=n)
    missing = np.arange(n) % 3 == 0
    known[:, 3] = np.where(missing, np.nan, values)
    actual = location.copy()
    actual[:, 0] += np.where(missing, 6., 2. * values)
    wins = (actual[:, 0] > actual[:, 1]).astype(int)
    model = ForecastModel.fit(location, actual, known, wins)
    query = np.full((2, len(KNOWN_COLUMNS)), np.nan)
    query[0, 3] = np.nanmean(known[:, 3])
    rows = model.apply(location[:2], query, np.diag([64., 64.]), [{}, {}], [1, 2])
    assert rows[1]["expected_home_score"] - rows[0]["expected_home_score"] > 5
    restored = ForecastModel.from_dict(model.to_dict())
    assert restored.apply(location[:2], query, np.diag([64., 64.]), [{}, {}], [1, 2]) == rows


def test_walk_forward_locations_do_not_learn_from_later_seasons():
    rng = np.random.default_rng(3)
    features = rng.normal(size=(180, 2, 26))
    observed = rng.normal(25., 5., size=(180, 2, 5))
    seasons = np.repeat([2023, 2024, 2025], 60)
    mask = np.ones(180, dtype=bool)
    weights = np.ones(180)
    model = {"variant": "score", "head": {"alpha": 10.}}
    before = _out_of_fold_location(model, features, observed, mask, weights, seasons)
    altered = observed.copy()
    altered[seasons >= 2024] *= 10
    after = _out_of_fold_location(model, features, altered, mask, weights, seasons)
    assert np.isnan(before[:60]).all()
    assert np.isfinite(before[60:]).all()
    np.testing.assert_allclose(before[60:120], after[60:120])
    assert not np.allclose(before[120:], after[120:])


def test_context_moves_scores_and_recalculates_every_probability():
    model = ForecastModel.identity()
    cov = [[100., 15.], [15., 100.]]
    original = model.apply(np.array([[28., 24.]]), np.zeros((1, 9)), cov,
                           [{"home_spread": 0., "total_line": 52.}], [1])
    spec = {"schema_version": 1, "feature_columns": ["weather_wind"],
            "center": [0.], "scale": [1.], "coefficients": [1., 0.],
            "max_logit_change": .2}
    frame = pd.DataFrame([{**original[0], "home_spread": 0., "total_line": 52.}])
    result = apply_score_context(frame, pd.DataFrame({"weather_wind": [1.]}), spec, cov).iloc[0]
    assert result.home_win_probability > original[0]["home_win_probability"]
    assert result.expected_home_score > original[0]["expected_home_score"]
    assert result.projected_total == pytest.approx(52.)
    assert result.home_win_probability == pytest.approx(result.home_cover_probability)
    assert result.home_win_probability == result.gaussian_home_win_probability
    assert result.over_probability + result.under_probability + result.total_push_probability == pytest.approx(1.)
    assert result.home_poisson_rate == result.expected_home_score


@pytest.mark.parametrize("schema", [1, 2])
def test_existing_adjustment_artifacts_remain_readable(schema):
    saved = ForecastModel.identity().to_dict()
    saved["schema_version"] = schema
    saved.pop("missing_koopman")
    saved.pop("input_missing_koopman")
    restored = ForecastModel.from_dict(saved)
    row = restored.apply(np.array([[28., 24.]]), np.zeros((1, 9)),
                         np.diag([100., 100.]), [{}], [1])[0]
    assert .5 < row["home_win_probability"] < 1.


def test_context_validation_uses_latest_time_not_alphabetical_run_id():
    history = pd.DataFrame({
        "run_id": np.repeat(["week-8", "week-9", "week-10"], 50),
        "as_of": np.repeat(["2025-10-19", "2025-10-26", "2025-11-02"], 50),
        "home_win": np.tile([0, 1], 75),
        "base_probability": .5,
        "weather_wind": np.tile(np.arange(50), 3),
    })
    _, report = select_context_model(history)
    assert report["validation_run"] == "week-10"
    assert report["training_games"] == 100
    assert report["validation_games"] == 50

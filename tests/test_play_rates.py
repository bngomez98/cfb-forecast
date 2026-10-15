import numpy as np
import pandas as pd
import pytest

from cfb_forecast.config import METRICS, StateConfig
from cfb_forecast.features import PLAY_RATE_NAMES, feature_columns, observations, play_rate_observations
from cfb_forecast.ratings import DynamicRatings


def test_recorded_scores_stay_on_the_original_metrics_without_plays():
    games = pd.DataFrame({"game_id": [1], "home_id": [10], "away_id": [20], "home_points": [28.0], "away_points": [14.0]})
    observed = observations(games, pd.DataFrame())
    assert observed.shape == (1, 2, len(METRICS))
    assert feature_columns("score") == [0, 5, 10, 20, 22, 23]
    assert feature_columns("pbp") == list(range(26))


def test_kneel_down_enters_the_rusher_rate():
    plays = pd.DataFrame([
        {"game_id": 1, "pos_team_id": 10, "def_pos_team_id": 20, "rusher_player_name": "QB One", "rush": True, "kneel_down": False, "statYardage": 5, "EPA": 0.2},
        {"game_id": 1, "pos_team_id": 10, "def_pos_team_id": 20, "rusher_player_name": "QB One", "rush": False, "kneel_down": True, "statYardage": -1, "EPA": -0.4},
    ])
    games = pd.DataFrame({"game_id": [1], "home_id": [10], "away_id": [20]})
    rates = play_rate_observations(games, plays)
    column = PLAY_RATE_NAMES.index("rb_yards_per_play")
    epa = PLAY_RATE_NAMES.index("rb_epa")
    assert rates[0, 0, column] == pytest.approx(2.0)
    assert rates[0, 0, epa] == pytest.approx(-0.1)
    assert np.isnan(rates[0, 1, column])


def test_play_rates_enter_the_hierarchical_state():
    width = len(METRICS) + len(PLAY_RATE_NAMES)
    state = DynamicRatings([10, 20], np.zeros(width), np.ones(width), StateConfig())
    values = np.full((2, width), np.nan)
    values[:, 0] = [2.0, -2.0]
    epa = len(METRICS) + PLAY_RATE_NAMES.index("epa")
    values[:, epa] = [1.5, -1.5]
    state.update(10, 20, True, values, "2024-09-01T00:00Z")
    assert state.mean.shape[0] == width
    assert state.mean[epa, 0] > 0
    assert state.mean[epa, state.n + 1] < 0
    features = state.features(10, 20, False)
    assert features.shape == (2, 4 * width + 6)
    columns = feature_columns("score", width)
    assert epa in columns
    assert features.shape[1] > max(columns)
    np.testing.assert_array_equal(state.features(10, 20, True)[:, 4 * width + 2], [0, 0])

import pandas as pd
import pytest

from cfb_forecast.data import normalize_schedule
from cfb_forecast.layered_model import LayeredBayesModel


def layered_games():
    return normalize_schedule(pd.DataFrame([
        {"game_id": 1, "season": 2021, "week": 1, "start_date": "2021-09-04T16:00Z", "completed": True, "neutral_site": False, "home_id": 1, "away_id": 2, "home_team": "A", "away_team": "B", "home_points": 35, "away_points": 14, "fbs_game": True},
        {"game_id": 2, "season": 2022, "week": 1, "start_date": "2022-09-04T16:00Z", "completed": True, "neutral_site": False, "home_id": 1, "away_id": 2, "home_team": "A", "away_team": "B", "home_points": 31, "away_points": 17, "fbs_game": True},
        {"game_id": 3, "season": 2023, "week": 1, "start_date": "2023-09-04T16:00Z", "completed": True, "neutral_site": False, "home_id": 2, "away_id": 1, "home_team": "B", "away_team": "A", "home_points": 20, "away_points": 27, "fbs_game": True},
    ]))


def test_layered_model_trains_and_returns_all_layers():
    model = LayeredBayesModel().fit(layered_games())
    result = model.predict(1, 2)
    assert result["expected_home_score"] > result["expected_away_score"]
    assert 0 < result["home_win_probability"] < 1
    assert result["home_score_poisson_pmf"] > 0
    assert "gaussian_margin" in result["model_layers"]


def test_layered_model_roundtrip():
    model = LayeredBayesModel().fit(layered_games())
    restored = LayeredBayesModel.from_dict(model.to_dict())
    assert restored.predict(1, 2)["projected_home_margin"] == pytest.approx(model.predict(1, 2)["projected_home_margin"])


def test_layered_model_requires_prior_outcomes():
    games = layered_games()
    games.loc[:, "result_usable"] = False
    with pytest.raises(ValueError):
        LayeredBayesModel().fit(games)

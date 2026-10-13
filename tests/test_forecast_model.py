import numpy as np
import pandas as pd
import pytest

from cfb_forecast.database import connect, record_weekly_validation
from cfb_forecast.evaluate import spread_regression
from cfb_forecast.forecast_model import METHOD, ForecastModel, known_future_matrix, rest_days
from cfb_forecast.publisher import board_row


def test_koopman_shift_follows_known_future_and_logistic_picks_the_higher_score():
    rng = np.random.default_rng(7)
    known = rng.normal(size=(80, 9))
    location = np.full((80, 2), 24.0)
    true = np.zeros((9, 2))
    true[0, 0] = 2.0
    true[0, 1] = -1.0
    actual = location + (known - known.mean(0)) / known.std(0) @ true
    actual = actual - actual.mean(0) + 24
    wins = (actual[:, 0] > actual[:, 1]).astype(int)
    # Keep both outcomes represented.
    wins[0] = 0
    wins[1] = 1
    model = ForecastModel.fit(location, actual, known, wins)
    assert model.ready
    assert model.logistic_ready
    favored = location[:1] + 0
    known_row = np.zeros((1, 9))
    known_row[0, 0] = known[:, 0].mean() + 2 * known[:, 0].std()
    out = model.apply(favored, known_row, [[36, 4], [4, 36]], [{}], [11], draws=500)
    assert out[0]["distribution_method"] == "forecast_model"
    assert out[0]["win_link"] == "logistic"
    assert 0 < out[0]["home_win_probability"] < 1
    assert out[0]["expected_home_score"] >= 0
    assert "posterior" not in METHOD


def test_partial_known_future_is_filled_and_scores_stay_inside_observed_bounds():
    rng = np.random.default_rng(3)
    known = rng.normal(size=(60, 9))
    known[50:, 1] = np.nan
    location = rng.normal(21, 2, size=(60, 2))
    actual = np.clip(location + rng.normal(0, 1, size=(60, 2)), 3, 45)
    wins = (actual[:, 0] > actual[:, 1]).astype(int)
    wins[:2] = [0, 1]
    model = ForecastModel.fit(location, actual, known, wins)
    extreme = np.array([[200.0, -20.0]])
    blank = np.full((1, 9), np.nan)
    out = model.apply(extreme, blank, [[25, 2], [2, 25]], [{}], [4], draws=200)
    assert model.score_low[0] <= out[0]["expected_home_score"] <= model.score_high[0]
    assert model.score_low[1] <= out[0]["expected_away_score"] <= model.score_high[1]


def test_monte_carlo_spread_prices_sum_and_market_sign_matches_the_margin():
    model = ForecastModel.identity()
    means = np.array([[31.0, 21.0]])
    out = model.apply(means, np.zeros((1, 9)), [[49, 6], [6, 49]], [{"home_spread": -7.0, "total_line": 52.0}], [9], draws=2000)
    row = out[0]
    assert row["model_spread"] == pytest.approx(-10.0)
    assert row["home_cover_probability"] + row["away_cover_probability"] + row["spread_push_probability"] == pytest.approx(1)
    assert row["over_probability"] + row["under_probability"] + row["total_push_probability"] == pytest.approx(1)
    assert row["monte_carlo_draws"] == 2000


def test_rest_uses_only_the_previous_kickoff():
    games = pd.DataFrame({
        "game_id": [1, 2],
        "home_id": [10, 10],
        "away_id": [20, 30],
        "start_date": pd.to_datetime(["2026-09-05T16:00Z", "2026-09-12T16:00Z"], utc=True),
    })
    rested = rest_days(games, games)
    second = rested.loc[rested.game_id == 2].iloc[0]
    assert pd.isna(rested.loc[rested.game_id == 1, "home_rest_days"]).all()
    assert second.home_rest_days == pytest.approx(7)


def test_known_future_matrix_removes_vig():
    frame = pd.DataFrame([{"home_ml_decimal": 1.5, "away_ml_decimal": 3.0, "home_spread": -3}])
    values = known_future_matrix(frame)
    assert values[0, 2] == pytest.approx((1 / 1.5) / (1 / 1.5 + 1 / 3))


def test_board_and_database_keep_the_spread():
    row = board_row({
        "game_id": 1, "season": 2026, "home_team": "Home", "away_team": "Away",
        "kickoff": "2026-10-17T19:00:00Z", "expected_home_score": 30,
        "expected_away_score": 20, "home_win_probability": 0.7,
        "projected_home_margin": 10, "model_spread": -10, "home_spread": -7,
        "home_cover_probability": 0.62,
    })
    assert row["winner"] == "Home"
    assert row["model_spread"] == -10
    assert row["market_spread"] == -7
    fit = spread_regression([3, 7, 10, 14, -3, 6, 9, 1], [-3, -7, -10, -14, 3, -6, -9, -1])
    assert fit["games"] == 8


def test_existing_forecast_log_gains_spread_columns(tmp_path):
    path = tmp_path / "cfb.sqlite"
    conn = connect(path)
    conn.execute("DROP TABLE forecast_log")
    conn.execute(
        """CREATE TABLE forecast_log (
            id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, season INTEGER NOT NULL,
            game_id INTEGER NOT NULL, kickoff TEXT NOT NULL, home_team TEXT NOT NULL,
            away_team TEXT NOT NULL, expected_home_score REAL NOT NULL,
            expected_away_score REAL NOT NULL, winner TEXT NOT NULL, confidence REAL NOT NULL,
            home_win_probability REAL NOT NULL, logged_at TEXT NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{}', UNIQUE(run_id, game_id))"""
    )
    conn.execute(
        """INSERT INTO forecast_log(run_id, season, game_id, kickoff, home_team, away_team,
            expected_home_score, expected_away_score, winner, confidence, home_win_probability, logged_at)
            VALUES('2026-week-7', 2026, 1, '2026-10-17T19:00:00Z', 'Home', 'Away', 28, 21, 'Home', 0.6, 0.6, '2026-10-13T00:00:00Z')"""
    )
    conn.commit()
    conn.close()
    conn = connect(path)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(forecast_log)")}
    assert {"model_spread", "market_spread", "home_cover_probability"} <= columns
    assert conn.execute("SELECT COUNT(*) FROM forecast_log").fetchone()[0] == 1
    record_weekly_validation(conn, {
        "season": 2026, "week": 6, "run_id": "2026-week-6", "validated_at": "2026-10-13T00:00:00Z",
        "games": 40, "status": "settled", "log_loss": 0.6, "brier": 0.2, "margin_mae": 10, "spread_mae": 10,
    })
    assert conn.execute("SELECT games FROM weekly_validation").fetchone()[0] == 40
    conn.close()

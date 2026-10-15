from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cfb_forecast.database import connect, record_weekly_validation
from cfb_forecast.decision import probability_to_american
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
    out = model.apply(favored, known_row, [[36, 4], [4, 36]], [{}], [11])
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
    out = model.apply(extreme, blank, [[25, 2], [2, 25]], [{}], [4])
    assert model.score_low[0] <= out[0]["expected_home_score"] <= model.score_high[0]
    assert model.score_low[1] <= out[0]["expected_away_score"] <= model.score_high[1]


def test_spread_and_total_prices_are_calculated_directly_without_simulation():
    from scipy.stats import norm

    model = ForecastModel.identity()
    means = np.array([[31.0, 21.0]])
    out = model.apply(means, np.zeros((1, 9)), [[49, 6], [6, 49]], [{"home_spread": -7.0, "total_line": 52.0}], [9])
    row = out[0]
    assert row["model_spread"] == pytest.approx(-10.0)
    margin_sd = np.sqrt(49 + 49 - 2 * 6)
    total_sd = np.sqrt(49 + 49 + 2 * 6)
    # Whole-number lines: a 7-point win and a 52-point total are pushes.
    assert row["home_cover_probability"] == pytest.approx(norm.sf((7.5 - 10.0) / margin_sd))
    assert row["spread_push_probability"] == pytest.approx(norm.cdf((7.5 - 10.0) / margin_sd) - norm.cdf((6.5 - 10.0) / margin_sd))
    assert row["over_probability"] == pytest.approx(norm.sf((52.5 - 52.0) / total_sd))
    assert row["under_probability"] == pytest.approx(norm.cdf((51.5 - 52.0) / total_sd))
    assert "monte_carlo_draws" not in row


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


def test_fair_american_moneyline_matches_sportsbook_sign():
    assert probability_to_american(0.5) == 100
    assert probability_to_american(0.7) == -233
    assert probability_to_american(0.3) == 233


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
    assert row["home_moneyline"] == -233
    assert row["away_moneyline"] == 233
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
    assert {"model_spread", "market_spread", "home_cover_probability", "home_moneyline", "away_moneyline"} <= columns
    assert conn.execute("SELECT COUNT(*) FROM forecast_log").fetchone()[0] == 1
    record_weekly_validation(conn, {
        "season": 2026, "week": 6, "run_id": "2026-week-6", "validated_at": "2026-10-13T00:00:00Z",
        "games": 40, "status": "settled", "log_loss": 0.6, "brier": 0.2, "margin_mae": 10, "spread_mae": 10,
    })
    assert conn.execute("SELECT games FROM weekly_validation").fetchone()[0] == 40
    conn.close()


def test_recorded_scoring_history_moves_the_home_score():
    rng = np.random.default_rng(4)
    n = 80
    known = np.full((n, 9), np.nan)
    location = np.full((n, 2), 24.0)
    history = np.full((n, 20), np.nan)
    history[:, 0] = rng.normal(28, 6, n)
    history[:, 1] = rng.normal(22, 6, n)
    actual = np.column_stack(
        [
            0.45 * history[:, 0] + 0.55 * 24 + rng.normal(0, 1, n),
            0.45 * history[:, 1] + 0.55 * 24 + rng.normal(0, 1, n),
        ]
    )
    wins = (actual[:, 0] > actual[:, 1]).astype(int)
    wins[0] = 0
    wins[1] = 1
    model = ForecastModel.fit(location, actual, known, wins, history=history)
    assert model.input_columns
    assert "posterior" not in METHOD
    high = history[:1].copy()
    high[0, 0] = 42
    high[0, 1] = 14
    low = history[:1].copy()
    low[0, 0] = 14
    low[0, 1] = 42
    blank = np.full((1, 9), np.nan)
    cov = [[36.0, 4.0], [4.0, 36.0]]
    up = model.apply(location[:1], blank, cov, [{}], [1], history=high)
    down = model.apply(location[:1], blank, cov, [{}], [2], history=low)
    assert up[0]["expected_home_score"] > down[0]["expected_home_score"]


def test_pregame_history_uses_prior_games_in_the_same_season(tmp_path):
    from cfb_forecast.features import pregame_history

    completed = pd.DataFrame(
        {
            "game_id": [1, 2],
            "season": [2025, 2025],
            "week": [1, 2],
            "home_id": [10, 10],
            "away_id": [20, 30],
            "home_points": [35.0, 21.0],
            "away_points": [10.0, 28.0],
            "start_date": pd.to_datetime(["2025-09-01T18:00:00Z", "2025-09-08T18:00:00Z"]),
        }
    )
    matrix = pregame_history(completed, tmp_path, completed)
    assert matrix is not None
    assert np.isnan(matrix[0, 0])
    assert matrix[1, 0] == pytest.approx(35.0)
    assert matrix[1, 2] == pytest.approx(10.0)


def test_database_team_facts_enter_the_pregame_rates(tmp_path):
    from cfb_forecast.database import connect, record_history_facts
    from cfb_forecast.features import pregame_history

    db = tmp_path / "facts.sqlite"
    conn = connect(db)
    record_history_facts(conn, pd.DataFrame([{
        "game_id": 1, "team_id": 10, "season": 2025, "week": 1,
        "points_for": 41.0, "points_against": 7.0, "yards_per_play": 7.2,
    }]))
    conn.close()
    completed = pd.DataFrame({
        "game_id": [1, 2],
        "season": [2025, 2025],
        "week": [1, 2],
        "home_id": [10, 10],
        "away_id": [20, 30],
        "home_points": [41.0, 24.0],
        "away_points": [7.0, 21.0],
        "start_date": pd.to_datetime(["2025-09-01T18:00:00Z", "2025-09-08T18:00:00Z"]),
    })
    matrix = pregame_history(completed, tmp_path, completed, database=db)
    assert matrix is not None
    assert matrix[1, 0] == pytest.approx(41.0)
    assert matrix[1, 4] == pytest.approx(7.2)


def test_pregame_history_replays_bowl_games_after_the_regular_season(tmp_path):
    from cfb_forecast.features import pregame_history

    # Postseason games restart week numbering at 1; the December bowl must not
    # feed the same team's September week-3 rates.
    completed = pd.DataFrame(
        {
            "game_id": [1, 2, 3],
            "season": [2025, 2025, 2025],
            "week": [1, 1, 3],
            "season_type": ["regular", "postseason", "regular"],
            "home_id": [10, 10, 10],
            "away_id": [20, 40, 30],
            "home_points": [35.0, 70.0, 21.0],
            "away_points": [10.0, 0.0, 28.0],
            "start_date": pd.to_datetime(
                ["2025-09-01T18:00:00Z", "2025-12-20T18:00:00Z", "2025-09-15T18:00:00Z"]
            ),
        }
    )
    matrix = pregame_history(completed, tmp_path, completed)
    assert matrix[2, 0] == pytest.approx(35.0)
    # Every same-season game counts equally: the bowl sees both regular-season games.
    assert matrix[1, 0] == pytest.approx((35.0 + 21.0) / 2)


def test_pregame_history_starts_from_the_previous_season_only(tmp_path):
    from cfb_forecast.features import pregame_history

    # 2024 is two seasons back, so only 2025 sets the starting mean for 2026.
    completed = pd.DataFrame(
        {
            "game_id": [1, 2, 3, 4],
            "season": [2024, 2025, 2025, 2026],
            "week": [1, 1, 2, 1],
            "home_id": [10, 10, 10, 10],
            "away_id": [20, 30, 40, 50],
            "home_points": [10.0, 30.0, 40.0, 0.0],
            "away_points": [0.0, 0.0, 0.0, 0.0],
            "start_date": pd.to_datetime(
                ["2024-09-01T18:00:00Z", "2025-09-01T18:00:00Z", "2025-09-08T18:00:00Z", "2026-09-01T18:00:00Z"]
            ),
        }
    )
    matrix = pregame_history(completed.tail(1), tmp_path, completed.head(3))
    assert matrix[0, 0] == pytest.approx(35.0)
    # The bowl sees both regular-season games, averaged (35 and 21 points).
    assert matrix[1, 0] == pytest.approx((35.0 + 21.0) / 2)


def test_postseason_week_one_is_not_early_season():
    from types import SimpleNamespace
    from cfb_forecast.ratings import early_season

    assert early_season(SimpleNamespace(week=2, season_type="regular"))
    assert not early_season(SimpleNamespace(week=1, season_type="postseason"))
    assert not early_season(SimpleNamespace(week=7, season_type="regular"))
    assert early_season(SimpleNamespace(week=3))


def test_rest_days_count_games_against_non_fbs_opponents():
    schedule = pd.DataFrame(
        {
            "game_id": [1, 2, 3],
            "home_id": [1, 1, 1],
            "away_id": [2, 99, 3],
            "start_date": pd.to_datetime(
                ["2025-09-01T18:00:00Z", "2025-09-08T18:00:00Z", "2025-09-15T18:00:00Z"]
            ),
            "fbs_game": [True, False, True],
        }
    )
    rested = rest_days(schedule[schedule.game_id == 3], schedule)
    assert rested.home_rest_days.iloc[0] == pytest.approx(7)


def test_season_backtest_forecasts_bowls_separately_from_week_one(monkeypatch, tmp_path):
    import cfb_forecast.season_backtest as season_backtest

    schedule = pd.DataFrame(
        {
            "game_id": [1, 2],
            "season": [2025, 2025],
            "week": [1, 1],
            "season_type": ["regular", "postseason"],
            "start_date": pd.to_datetime(["2025-08-30T18:00:00Z", "2025-12-20T18:00:00Z"]),
            "fbs_game": [True, True],
            "completed": [True, True],
            "home_points": [30.0, 24.0],
            "away_points": [20.0, 27.0],
        }
    )
    calls = []

    def fake_forecast(**kwargs):
        calls.append((kwargs["as_of"], kwargs["target_game_ids"]))
        Path(kwargs["out_root"]).mkdir(parents=True, exist_ok=True)
        return pd.DataFrame(
            {
                "game_id": kwargs["target_game_ids"],
                "home_win_probability": 0.6,
                "expected_home_score": 28.0,
                "expected_away_score": 21.0,
                "projected_home_margin": 7.0,
                "projected_total": 49.0,
            }
        )

    monkeypatch.setattr(season_backtest, "load_data", lambda root: (schedule, None, None))
    monkeypatch.setattr(season_backtest, "forecast", fake_forecast)
    reports, summary = season_backtest.run_season_backtest(tmp_path, tmp_path, tmp_path, 2025, lead_hours=48)
    assert [ids for _, ids in calls] == [[1], [2]]
    assert pd.Timestamp(calls[1][0]) == pd.Timestamp("2025-12-18T18:00:00Z")
    assert summary["games"] == 2


def test_pickem_line_cannot_push_because_ties_go_to_overtime():
    model = ForecastModel.identity()
    row = model.apply(np.array([[27.0, 24.0]]), np.zeros((1, 9)), [[49, 6], [6, 49]], [{"home_spread": 0.0}], [5])[0]
    assert row["spread_push_probability"] == pytest.approx(0.0)
    assert row["home_cover_probability"] == pytest.approx(row["gaussian_home_win_probability"])


def test_closing_lines_skip_unavailable_spreads(tmp_path):
    from cfb_forecast.market import closing_lines

    path = tmp_path / "closing.csv"
    pd.DataFrame(
        {
            "game_id": [1, 2],
            "home_team_spread": [-7.5, -2.5],
            "over_under": [55.5, 48.0],
            "game_spread_available": [True, False],
        }
    ).to_csv(path, index=False)
    assert closing_lines(path) == {1: {"home_spread": -7.5, "total_line": 55.5}}
    assert closing_lines(tmp_path / "missing.csv") == {}


def test_games_without_a_line_use_the_fit_that_never_saw_lines():
    from cfb_forecast.forecast_model import KNOWN_COLUMNS, ForecastModel

    rng = np.random.default_rng(7)
    n = 400
    location = np.column_stack([rng.normal(30, 3, n), rng.normal(25, 3, n)])
    spread = rng.normal(-5, 8, n)
    actual = location + np.column_stack([-spread / 2, spread / 2]) + rng.normal(0, 7, (n, 2))
    known = np.full((n, len(KNOWN_COLUMNS)), np.nan)
    known[:, KNOWN_COLUMNS.index("home_spread")] = spread
    known[:, KNOWN_COLUMNS.index("home_rest_days")] = rng.integers(6, 9, n)
    wins = (actual[:, 0] > actual[:, 1]).astype(int)
    fitted = ForecastModel.fit(location, actual, known, wins)
    assert "home_spread" in fitted.columns
    assert fitted.no_line is not None and "home_spread" not in fitted.no_line.columns

    target = known[:2].copy()
    target[1, KNOWN_COLUMNS.index("home_spread")] = np.nan
    rows = fitted.apply(location[:2], target, np.diag([100.0, 100.0]), [{}, {}], [1, 2])
    assert rows[0]["uses_market_line"] and not rows[1]["uses_market_line"]
    alone = fitted.no_line.apply(location[1:2], target[1:2], np.diag([100.0, 100.0]), [{}], [2])
    assert rows[1]["projected_home_margin"] == pytest.approx(alone[0]["projected_home_margin"])

    restored = ForecastModel.from_dict(fitted.to_dict())
    assert restored.no_line is not None and restored.no_line.columns == fitted.no_line.columns

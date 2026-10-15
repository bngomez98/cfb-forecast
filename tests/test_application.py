"""Daily-run outputs, named player tables, ESPN pregame odds, the app launcher and the dashboard."""
import numpy as np
import pandas as pd
import pytest

from cfb_forecast import app as app_module
from cfb_forecast.market import read_market_history
from cfb_forecast.players import player_table, fit_player_ratings, team_table, recent_usage
from cfb_forecast.results import capture_espn_odds, event_odds
from cfb_forecast.weekly import forecast_changes, previous_snapshot, prune_snapshots


# ------------------------------------------------------------------ daily changes + retention
def _forecast_frame(prob_a, prob_b, as_of):
    return pd.DataFrame({
        "game_id": [1, 2], "away_team": ["A", "C"], "home_team": ["B", "D"], "kickoff": ["2026-10-17T19:00:00Z"] * 2,
        "as_of": [as_of] * 2, "home_win_probability": [prob_a, prob_b],
        "projected_home_margin": [3.0, -2.0], "projected_total": [50.0, 47.0],
    })


def test_forecast_changes_orders_biggest_movers_first_and_marks_new_games(tmp_path):
    prev = tmp_path / "forecasts" / "20261008T090000Z"
    prev.mkdir(parents=True)
    _forecast_frame(0.60, 0.50, "2026-10-08T09:00:00Z").drop(index=[1]).to_csv(prev / "forecasts.csv", index=False)
    current = _forecast_frame(0.55, 0.52, "2026-10-09T09:00:00Z")
    out = forecast_changes(current, prev).set_index("game_id")
    assert out.loc[1, "win_probability_change"] == pytest.approx(-0.05)
    assert pd.isna(out.loc[2, "prior_home_win_probability"])  # game 2 is new in this run
    assert out.index[0] == 1                                    # moved games sort before new/unchanged ones
    first = forecast_changes(current, None)
    assert first.win_probability_change.isna().all() and len(first) == 2


def test_snapshot_lookup_and_retention(tmp_path):
    for stamp in ("20261006T090000Z", "20261007T090000Z", "20261008T090000Z"):
        d = tmp_path / "forecasts" / stamp
        d.mkdir(parents=True)
        (d / "forecasts.csv").write_text("game_id\n1\n")
    assert previous_snapshot(tmp_path, "20261008T090000Z").name == "20261007T090000Z"
    assert previous_snapshot(tmp_path, "20261006T090000Z") is None
    assert prune_snapshots(tmp_path, keep=2) == 1
    assert sorted(p.name for p in (tmp_path / "forecasts").iterdir()) == ["20261007T090000Z", "20261008T090000Z"]


# ------------------------------------------------------------------ named player tables
def _players_agg():
    rng = np.random.default_rng(1)
    rows, start = [], pd.Timestamp("2026-08-30T17:00:00Z")
    for week in range(1, 7):
        for team, opp, pid, name, mean in ((10, 11, 101, "Ace QB", 8.0), (11, 10, 202, "Backup QB", 4.0)):
            yards = rng.normal(mean, 3, 30)
            rows.append({"game_id": week * 10 + (team - 10), "role": "pass", "player_id": pid, "player_name": name,
                         "team_id": team, "def_id": opp, "plays": 30, "yards": float(yards.sum()),
                         "yards_per_play": float(yards.mean()), "season": 2026, "week": week,
                         "start_date": start + pd.Timedelta(weeks=week)})
    return pd.DataFrame(rows)


def test_player_and_team_tables_are_named_and_show_usage():
    agg, cutoff = _players_agg(), "2026-12-01T00:00:00Z"
    ratings = fit_player_ratings(agg, cutoff)
    names = {10: "Alpha", 11: "Beta"}
    players = player_table(agg, ratings, cutoff, names, min_plays=10)
    best = players.iloc[0]
    assert (best.player_name, best.team, best.role) == ("Ace QB", "Alpha", "pass")
    assert best.rating > players.iloc[1].rating and best.usage_share == pytest.approx(1.0)
    teams = team_table(ratings, recent_usage(agg, cutoff), names)
    assert teams.iloc[0].team == "Alpha" and set(["pass_offense", "player_defense", "net_rating"]) <= set(teams.columns)
    # a game still inside the result lag (kickoff 17:00 + 8h) is invisible at 20:00 ...
    inside = player_table(agg, fit_player_ratings(agg, "2026-09-06T20:00:00Z"), "2026-09-06T20:00:00Z", names, min_plays=1)
    assert inside.empty
    # ... and counts once the lag has passed, while the following week's game still does not
    after = player_table(agg, fit_player_ratings(agg, "2026-09-08T00:00:00Z"), "2026-09-08T00:00:00Z", names, min_plays=1)
    assert after.games.max() == 1


# ------------------------------------------------------------------ ESPN pregame odds
def _event(event_id, details, home="UGA", away="AUB", state="pre", spread=None, flags=(True, False), over_under=52.5):
    return {
        "id": str(event_id), "date": "2026-10-17T19:00:00Z", "season": {"year": 2026}, "week": {"number": 7},
        "competitions": [{
            "status": {"type": {"state": state, "completed": state == "post", "name": "STATUS_SCHEDULED"}},
            "competitors": [
                {"homeAway": "home", "team": {"displayName": "Georgia Bulldogs", "abbreviation": home, "id": "61"}},
                {"homeAway": "away", "team": {"displayName": "Auburn Tigers", "abbreviation": away, "id": "2"}}],
            "odds": [{"provider": {"name": "ESPN BET"}, "details": details, "overUnder": over_under, "spread": spread,
                      "homeTeamOdds": {"favorite": flags[0], "moneyLine": -250}, "awayTeamOdds": {"favorite": flags[1], "moneyLine": 200}}],
        }]}


def test_event_odds_signs_and_prices():
    home_fav = event_odds(_event(1, "UGA -6.5"))
    assert home_fav["home_spread"] == -6.5 and home_fav["total_line"] == 52.5
    assert home_fav["home_ml_decimal"] == pytest.approx(1.4) and home_fav["away_ml_decimal"] == pytest.approx(3.0)
    away_fav = event_odds(_event(2, "AUB -3.0", flags=(False, True)))
    assert away_fav["home_spread"] == 3.0                                  # away favourite: home gets the points
    assert event_odds(_event(3, "EVEN", flags=(False, False)))["home_spread"] == 0.0
    contradictory = event_odds(_event(4, "UGA -6.5", flags=(False, True)))
    assert contradictory["home_spread"] is None and contradictory["total_line"] == 52.5   # sign not guessed; total kept
    assert event_odds(_event(5, "UGA -6.5", state="in")) is None            # started games are not pregame
    assert event_odds(_event(6, "UGA -6.5", state="post")) is None


def test_capture_espn_odds_appends_only_pregame_lines_to_market_history(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    pd.DataFrame({"game_id": [11, 12], "start_date": ["2026-10-17T19:00:00Z", "2026-10-10T19:00:00Z"],
                  "home_team": ["Georgia", "Ohio"], "away_team": ["Auburn", "Kent"]}).to_csv(data / "schedule_2026.csv", index=False)
    payload = {"events": [_event(11, "UGA -6.5"), _event(99, "UGA -1.0")]}   # 99 is not a scheduled game
    monkeypatch.setattr("cfb_forecast.results._request_json", lambda url: payload)
    store = tmp_path / "market_history.csv"
    rows = capture_espn_odds(data, 2026, store, days_ahead=8, now="2026-10-12T00:00:00Z")
    assert rows.game_id.tolist() == [11]          # game 12 already started; game 99 is unscheduled
    history = read_market_history(store)
    assert history.provider.iloc[0] == "espn:espn_bet" and history.home_spread.iloc[0] == -6.5
    assert history.observed_at.iloc[0] < pd.Timestamp("2026-10-17T19:00:00Z")


# ------------------------------------------------------------------ the application launcher
def test_app_runs_scheduler_and_stops_dashboard(tmp_path, monkeypatch):
    events = []

    class FakeProcess:
        def poll(self):
            return None

        def terminate(self):
            events.append("dashboard-terminated")

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(app_module, "start_dashboard", lambda data, artifacts, port: events.append(f"dashboard-port-{port}") or FakeProcess())
    monkeypatch.setattr(app_module.Scheduler, "forever", lambda self: events.append("scheduler-ran"))
    args = app_module.add_arguments(__import__("argparse").ArgumentParser()).parse_args(
        ["--data", str(tmp_path / "d"), "--artifacts", str(tmp_path / "a"), "--season", "2026", "--port", "8123"])
    assert app_module.run_from_args(args) == 0
    assert events == ["dashboard-port-8123", "scheduler-ran", "dashboard-terminated"]


def test_app_refuses_to_start_with_nothing_enabled(tmp_path):
    args = app_module.add_arguments(__import__("argparse").ArgumentParser()).parse_args(
        ["--artifacts", str(tmp_path), "--no-dashboard", "--no-scheduler"])
    with pytest.raises(SystemExit):
        app_module.run_from_args(args)


# ------------------------------------------------------------------ dashboard, run for real
def test_dashboard_renders_every_view_from_a_daily_run(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    artifacts, data = tmp_path / "artifacts", tmp_path / "data"
    latest = artifacts / "latest"
    latest.mkdir(parents=True)
    data.mkdir()
    games = _forecast_frame(0.6, 0.45, "2026-10-09T09:00:00Z").assign(
        kickoff=["2026-10-17T19:00:00Z", "2026-10-17T23:00:00Z"], expected_home_score=[27.0, 21.0], expected_away_score=[24.0, 23.0],
        margin_interval_80=["[-10.0, 16.0]", "[-14.0, 10.0]"], total_interval_80=["[35.0, 62.0]", "[33.0, 60.0]"])
    games.to_csv(latest / "forecasts.csv", index=False)
    forecast_changes(games, None).to_csv(latest / "changes.csv", index=False)
    pd.DataFrame({"team_id": [10], "team": ["Alpha"], "pass_offense": [1.0], "rush_offense": [0.2], "player_offense": [0.7],
                  "player_defense": [0.3], "net_rating": [1.0]}).to_csv(latest / "team_ratings.csv", index=False)
    pd.DataFrame({"role": ["pass"], "player_id": [101], "player_name": ["Ace QB"], "team_id": [10], "team": ["Alpha"], "rating": [1.2],
                  "usage_share": [0.9], "plays": [300], "recent_plays": [120], "games": [10]}).to_csv(latest / "player_ratings.csv", index=False)
    (latest / "forecast_run_manifest.json").write_text('[{"season": 2026, "next_week": 7, "games": 2}]')
    monkeypatch.setenv("CFB_ARTIFACTS_DIR", str(artifacts))
    monkeypatch.setenv("CFB_DATA_DIR", str(data))
    script = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")
    at = AppTest.from_file(script, default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    at.sidebar.radio[0].set_value("Forecast desk").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("What changed" in s.value for s in at.subheader)
    for view in ("Overview", "Players & teams", "Web sources", "Sources", "Calculation pipeline", "Historical training"):
        at.sidebar.radio[0].set_value(view).run()
        assert not at.exception, (view, [e.value for e in at.exception])
    at.sidebar.radio[0].set_value("Players & teams").run()
    assert any("Player ratings" in s.value for s in at.subheader)

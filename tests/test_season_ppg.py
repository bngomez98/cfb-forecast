"""Tests for the season-to-date PPG module."""

from __future__ import annotations

import pandas as pd

from cfb_forecast.data import normalize_schedule
from cfb_forecast.season_ppg import (
    PPG_COLUMNS,
    attach_ppg_to_forecasts,
    compute_season_ppg,
)


def _fixture_schedule() -> pd.DataFrame:
    """Three games across two weeks, two teams scoring, one FCS game excluded."""
    rows = [
        {
            "game_id": 1,
            "season": 2026,
            "week": 1,
            "start_date": "2026-08-30T18:00Z",
            "completed": True,
            "neutral_site": False,
            "home_id": 100,
            "away_id": 200,
            "home_team": "Team A",
            "away_team": "Team B",
            "home_points": 35,
            "away_points": 14,
            "fbs_game": True,
        },
        {
            "game_id": 2,
            "season": 2026,
            "week": 1,
            "start_date": "2026-08-30T20:00Z",
            "completed": True,
            "neutral_site": False,
            "home_id": 300,
            "away_id": 400,
            "home_team": "Team C",
            "away_team": "Team D",
            "home_points": 21,
            "away_points": 20,
            "fbs_game": True,
        },
        {
            "game_id": 3,
            "season": 2026,
            "week": 2,
            "start_date": "2026-09-06T19:00Z",
            "completed": True,
            "neutral_site": False,
            "home_id": 100,
            "away_id": 300,
            "home_team": "Team A",
            "away_team": "Team C",
            "home_points": 28,
            "away_points": 24,
            "fbs_game": True,
        },
        {
            "game_id": 4,
            "season": 2026,
            "week": 2,
            "start_date": "2026-09-06T22:00Z",
            "completed": True,
            "neutral_site": False,
            "home_id": 200,
            "away_id": 500,
            "home_team": "Team B",
            "away_team": "FCS Opponent",
            "home_points": 42,
            "away_points": 7,
            "fbs_game": False,
        },
    ]
    return normalize_schedule(pd.DataFrame(rows))


def test_compute_season_ppg_counts_only_games_before_cutoff():
    schedules = _fixture_schedule()
    # After week 1 only (game 1 and 2 are available ~kickoff+8h).
    ppg = compute_season_ppg(schedules, season=2026, as_of="2026-09-05T00:00:00Z")
    teams = {int(row.team_id): row for row in ppg.itertuples()}
    assert set(teams) == {100, 200, 300, 400}
    assert teams[100].games_played_to_date == 1
    assert teams[100].ppg_scored == 35.0
    assert teams[100].ppg_allowed == 14.0
    assert teams[100].ppg_margin == 21.0
    assert teams[200].ppg_scored == 14.0


def test_compute_season_ppg_includes_week_two_once_available():
    schedules = _fixture_schedule()
    ppg = compute_season_ppg(schedules, season=2026, as_of="2026-09-08T12:00:00Z")
    teams = {int(row.team_id): row for row in ppg.itertuples()}
    # Team 100 played both games (35 scored, 14 allowed) and (28 scored, 24 allowed).
    row_100 = teams[100]
    assert row_100.games_played_to_date == 2
    assert row_100.points_scored_total == 63
    assert row_100.points_allowed_total == 38
    assert row_100.ppg_scored == 31.5
    assert row_100.ppg_allowed == 19.0


def test_compute_season_ppg_excludes_fbs_vs_fcs_by_default():
    schedules = _fixture_schedule()
    ppg = compute_season_ppg(schedules, season=2026, as_of="2026-09-10T00:00:00Z")
    teams = {int(row.team_id): row for row in ppg.itertuples()}
    # Team 200's FCS game should not count: only their week 1 game contributes.
    assert teams[200].games_played_to_date == 1
    assert teams[200].ppg_scored == 14.0


def test_compute_season_ppg_includes_fcs_when_flag_disabled():
    schedules = _fixture_schedule()
    ppg = compute_season_ppg(
        schedules,
        season=2026,
        as_of="2026-09-10T00:00:00Z",
        fbs_only=False,
    )
    teams = {int(row.team_id): row for row in ppg.itertuples()}
    assert teams[200].games_played_to_date == 2
    assert teams[200].points_scored_total == 14 + 42
    assert 500 in teams


def test_compute_season_ppg_empty_before_any_games():
    schedules = _fixture_schedule()
    ppg = compute_season_ppg(schedules, season=2026, as_of="2026-08-25T00:00:00Z")
    assert list(ppg.columns) == PPG_COLUMNS
    assert len(ppg) == 0


def test_attach_ppg_to_forecasts_adds_both_sides_columns():
    schedules = _fixture_schedule()
    ppg = compute_season_ppg(schedules, season=2026, as_of="2026-09-10T00:00:00Z")
    forecasts = pd.DataFrame(
        [{"game_id": 99, "home_id": 100, "away_id": 300, "kickoff": "2026-09-13T19:00Z"}]
    )
    enriched = attach_ppg_to_forecasts(forecasts, ppg)
    assert enriched.loc[0, "home_ppg_scored"] == 31.5
    assert enriched.loc[0, "home_ppg_allowed"] == 19.0
    assert enriched.loc[0, "home_games_played_to_date"] == 2
    assert enriched.loc[0, "away_ppg_scored"] == 22.5
    assert enriched.loc[0, "away_ppg_allowed"] == 24.0
    assert enriched.loc[0, "away_ppg_margin"] == -1.5


def test_attach_ppg_to_forecasts_handles_empty_ppg_table():
    forecasts = pd.DataFrame([{"game_id": 1, "home_id": 100, "away_id": 300, "kickoff": "2026-08-20T19:00Z"}])
    empty_ppg = pd.DataFrame(columns=PPG_COLUMNS)
    enriched = attach_ppg_to_forecasts(forecasts, empty_ppg)
    assert pd.isna(enriched.loc[0, "home_ppg_scored"])
    assert pd.isna(enriched.loc[0, "away_ppg_scored"])


def test_attach_ppg_to_forecasts_leaves_unknown_teams_as_nan():
    schedules = _fixture_schedule()
    ppg = compute_season_ppg(schedules, season=2026, as_of="2026-09-05T00:00:00Z")
    forecasts = pd.DataFrame([{"game_id": 1, "home_id": 100, "away_id": 999, "kickoff": "2026-09-13T19:00Z"}])
    enriched = attach_ppg_to_forecasts(forecasts, ppg)
    assert enriched.loc[0, "home_ppg_scored"] == 35.0
    assert pd.isna(enriched.loc[0, "away_ppg_scored"])

"""Tests for the player-PPG module."""

from __future__ import annotations

import pandas as pd

from cfb_forecast.player_ppg import (
    PLAYER_PPG_COLUMNS,
    compute_player_points_column,
    compute_player_ppg,
    summarize_player_ppg,
)


def _fixture_player_games() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # QB plays two games, throws 3 TDs across them.
            {
                "player_id": "P1",
                "player_name": "QB One",
                "team_id": 100,
                "position": "QB",
                "game_id": 1,
                "game_date": "2026-08-30T22:00Z",
                "season": 2026,
                "passing_td": 2,
                "rushing_td": 0,
                "receiving_td": 0,
                "field_goals_made": 0,
                "extra_points_made": 0,
                "two_point_conversions": 0,
            },
            {
                "player_id": "P1",
                "player_name": "QB One",
                "team_id": 100,
                "position": "QB",
                "game_id": 2,
                "game_date": "2026-09-06T22:00Z",
                "season": 2026,
                "passing_td": 1,
                "rushing_td": 1,
                "receiving_td": 0,
                "field_goals_made": 0,
                "extra_points_made": 0,
                "two_point_conversions": 0,
            },
            # Kicker: 4 FG, 3 XP in one game.
            {
                "player_id": "P2",
                "player_name": "Kicker Two",
                "team_id": 100,
                "position": "PK",
                "game_id": 1,
                "game_date": "2026-08-30T22:00Z",
                "season": 2026,
                "passing_td": 0,
                "rushing_td": 0,
                "receiving_td": 0,
                "field_goals_made": 4,
                "extra_points_made": 3,
                "two_point_conversions": 0,
            },
            # WR: a receiving TD plus a two-point conversion.
            {
                "player_id": "P3",
                "player_name": "WR Three",
                "team_id": 200,
                "position": "WR",
                "game_id": 3,
                "game_date": "2026-09-06T19:00Z",
                "season": 2026,
                "passing_td": 0,
                "rushing_td": 0,
                "receiving_td": 1,
                "field_goals_made": 0,
                "extra_points_made": 0,
                "two_point_conversions": 1,
            },
            # Future game that must not leak before its date.
            {
                "player_id": "P1",
                "player_name": "QB One",
                "team_id": 100,
                "position": "QB",
                "game_id": 4,
                "game_date": "2026-09-13T22:00Z",
                "season": 2026,
                "passing_td": 5,
                "rushing_td": 0,
                "receiving_td": 0,
                "field_goals_made": 0,
                "extra_points_made": 0,
                "two_point_conversions": 0,
            },
        ]
    )


def test_compute_player_points_column_sums_scoring_weights():
    frame = _fixture_player_games()
    points = compute_player_points_column(frame)
    # Row 0: 2 passing TDs * 6 = 12
    assert points.iloc[0] == 12
    # Row 2: 4 FG * 3 + 3 XP = 15
    assert points.iloc[2] == 15
    # Row 3: 1 rec TD * 6 + 1 2PT * 2 = 8
    assert points.iloc[3] == 8


def test_compute_player_ppg_respects_cutoff():
    frame = _fixture_player_games()
    result = compute_player_ppg(frame, as_of="2026-09-10T00:00:00Z")
    players = {row.player_id: row for row in result.itertuples()}
    # P1 has two games (12 + 12 points), the future row at 2026-09-13 is excluded.
    assert players["P1"].games_played_to_date == 2
    assert players["P1"].points_scored_total == 24
    assert players["P1"].ppg_scored == 12.0
    assert "P4" not in players
    # P2 kicker: 1 game, 15 points.
    assert players["P2"].ppg_scored == 15.0


def test_compute_player_ppg_empty_before_games():
    frame = _fixture_player_games()
    result = compute_player_ppg(frame, as_of="2026-08-01T00:00:00Z")
    assert list(result.columns) == PLAYER_PPG_COLUMNS
    assert len(result) == 0


def test_compute_player_ppg_min_games_filter():
    frame = _fixture_player_games()
    result = compute_player_ppg(frame, as_of="2026-09-10T00:00:00Z", min_games=2)
    ids = set(result.player_id)
    # P1 has 2 games before cutoff, P2 and P3 only one. Only P1 survives.
    assert ids == {"P1"}


def test_compute_player_ppg_without_date_column_raises():
    frame = _fixture_player_games().drop(columns=["game_date"])
    try:
        compute_player_ppg(frame, as_of="2026-09-10T00:00:00Z")
    except ValueError as exc:
        assert "game_date" in str(exc) or "start_date" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for missing date column")


def test_summarize_player_ppg_ranks_by_ppg():
    frame = _fixture_player_games()
    result = compute_player_ppg(frame, as_of="2026-09-10T00:00:00Z")
    summary = summarize_player_ppg(result, top_n=2)
    assert summary["players"] == 3
    assert len(summary["top_scorers"]) == 2
    # Top scorer by PPG should be P2 (15.0) over P1 (12.0).
    assert summary["top_scorers"][0]["player_id"] == "P2"
    assert summary["top_scorers"][1]["player_id"] == "P1"


def test_season_filter_narrows_to_requested_year():
    frame = _fixture_player_games().copy()
    frame.loc[frame.player_id.eq("P3"), "season"] = 2025
    result = compute_player_ppg(frame, as_of="2026-09-10T00:00:00Z", season=2026)
    assert set(result.player_id) == {"P1", "P2"}

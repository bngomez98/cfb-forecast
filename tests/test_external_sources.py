import pandas as pd

from cfb_forecast.external_cli import select_next_week
from cfb_forecast.ingest.external_sources import validate_schedule_rows
from cfb_forecast.ingest.data import normalize_schedule


def _games():
    return normalize_schedule(
        pd.DataFrame(
            [
                dict(game_id=1, season=2026, week=7, start_date="2026-10-17T16:00Z", completed=False, neutral_site=False, home_id=1, away_id=2, home_team="Ohio State", away_team="Michigan", home_points=None, away_points=None, fbs_game=True),
                dict(game_id=2, season=2026, week=8, start_date="2026-10-24T16:00Z", completed=False, neutral_site=False, home_id=3, away_id=4, home_team="Texas", away_team="Oklahoma", home_points=None, away_points=None, fbs_game=True),
            ]
        )
    )


def test_schedule_validation_matches_both_team_names():
    rows = [
        ["Week 7", "Michigan at Ohio State", "12:00 PM"],
        ["Week 7", "Other at Team", "3:30 PM"],
    ]
    result = validate_schedule_rows(_games().iloc[:1], rows, "cfbtxt")
    assert result[0]["status"] == "matched"
    assert result[0]["provider"] == "cfbtxt"


def test_schedule_validation_does_not_match_one_team_only():
    rows = [["Ohio State", "12:00 PM"]]
    result = validate_schedule_rows(_games().iloc[:1], rows, "cfbtxt")
    assert result[0]["status"] == "not_found"


def test_next_week_selection_is_future_fbs_week_only():
    games = _games()
    selected = select_next_week(games, 2026, "2026-10-12T12:00:00Z")
    assert selected.game_id.tolist() == [1]

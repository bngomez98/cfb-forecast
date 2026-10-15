"""Tests for the data cleaning and organization layer (cfb_forecast.clean)."""

import json

import pandas as pd
import pytest

from cfb_forecast.clean import (
    REASON_DUPLICATE_KEY,
    REASON_MISSING_SCORE,
    REASON_TIE,
    REASON_ZERO_ZERO,
    build_division_lookup,
    clean_all,
    clean_market_history,
)


def raw_schedule():
    rows = [
        # clean FBS game, home win
        dict(
            game_id=1,
            season=2025,
            week=1,
            season_type="regular",
            start_date="2025-08-30T00:00Z",
            completed="true",
            neutral_site="false",
            home_id=10,
            home_team="Alpha State",
            home_division="fbs",
            away_id=11,
            away_team="Beta Tech",
            away_division="fbs",
            home_points=31,
            away_points=14,
        ),
        # away win
        dict(
            game_id=2,
            season=2025,
            week=2,
            season_type="regular",
            start_date="2025-09-06T00:00Z",
            completed="true",
            neutral_site="false",
            home_id=10,
            home_team="Alpha State",
            home_division="fbs",
            away_id=12,
            away_team="Gamma College",
            away_division="fcs",
            home_points=17,
            away_points=24,
        ),
        # missing division on away side; Gamma College is labelled in game 2 -> inferable
        dict(
            game_id=3,
            season=2025,
            week=3,
            season_type="regular",
            start_date="2025-09-13T00:00Z",
            completed="true",
            neutral_site="false",
            home_id=12,
            home_team="Gamma College",
            home_division=None,
            away_id=10,
            away_team="Alpha State",
            away_division="fbs",
            home_points=40,
            away_points=10,
        ),
        # completed with null scores -> quarantine
        dict(
            game_id=4,
            season=2025,
            week=3,
            season_type="regular",
            start_date="2025-09-13T12:00Z",
            completed="true",
            neutral_site="false",
            home_id=13,
            home_team="Delta University",
            home_division="iii",
            away_id=14,
            away_team="Epsilon College",
            away_division="iii",
            home_points=None,
            away_points=None,
        ),
        # completed 0-0 -> cancelled artifact, quarantine
        dict(
            game_id=5,
            season=2025,
            week=4,
            season_type="regular",
            start_date="2025-09-20T00:00Z",
            completed="true",
            neutral_site="false",
            home_id=13,
            home_team="Delta University",
            home_division="iii",
            away_id=14,
            away_team="Epsilon College",
            away_division="iii",
            home_points=0,
            away_points=0,
        ),
        # genuine D3 tie, nonzero -> excluded from orientation but auditable
        dict(
            game_id=6,
            season=2025,
            week=4,
            season_type="regular",
            start_date="2025-09-20T12:00Z",
            completed="true",
            neutral_site="false",
            home_id=13,
            home_team="Delta University",
            home_division="iii",
            away_id=14,
            away_team="Epsilon College",
            away_division="iii",
            home_points=28,
            away_points=28,
        ),
        # duplicate game id -> second copy quarantined
        dict(
            game_id=1,
            season=2025,
            week=1,
            season_type="regular",
            start_date="2025-08-30T00:00Z",
            completed="true",
            neutral_site="false",
            home_id=10,
            home_team="Alpha State",
            home_division="fbs",
            away_id=11,
            away_team="Beta Tech",
            away_division="fbs",
            home_points=31,
            away_points=14,
        ),
        # unplayed future game kept with result_usable False
        dict(
            game_id=7,
            season=2025,
            week=9,
            season_type="regular",
            start_date="2025-10-25T00:00Z",
            completed="false",
            neutral_site="false",
            home_id=11,
            home_team="Beta Tech",
            home_division="fbs",
            away_id=12,
            away_team="Gamma College",
            away_division="fcs",
            home_points=None,
            away_points=None,
        ),
    ]
    cols = [
        "game_id",
        "season",
        "week",
        "season_type",
        "start_date",
        "completed",
        "neutral_site",
        "home_id",
        "home_team",
        "home_division",
        "away_id",
        "away_team",
        "away_division",
        "home_points",
        "away_points",
    ]
    return pd.DataFrame(rows)[cols]


@pytest.fixture()
def cleaned(tmp_path):
    raw_schedule().to_csv(tmp_path / "schedule_2025.csv", index=False)
    market = pd.DataFrame(
        [
            dict(game_id=1, provider="p", observed_at="2025-08-29T00:00Z", home_spread=-13.5, source="s"),
            dict(game_id=1, provider="p", observed_at="2025-08-29T12:00Z", home_spread=-14.0, source="s"),
            dict(
                game_id=2, provider="p", observed_at="2025-09-01T00:00Z", home_spread=float("nan"), source="s"
            ),
        ]
    )
    market.to_csv(tmp_path / "market_history.csv", index=False)
    return clean_all(tmp_path)


def test_quarantine_rules(cleaned):
    q = cleaned["quarantine"]
    reasons = set(q.reason)
    assert {REASON_MISSING_SCORE, REASON_ZERO_ZERO, REASON_TIE, REASON_DUPLICATE_KEY} <= reasons
    assert not (q.game_id == 1).any() or q[q.game_id == 1].reason.eq(REASON_DUPLICATE_KEY).all()


def test_games_orientation_exact(cleaned):
    g = cleaned["games"].set_index("game_id")
    assert g.loc[1, "winner_id"] == 10 and g.loc[1, "margin"] == 17 and g.loc[1, "total_points"] == 45
    assert g.loc[2, "winner_id"] == 12 and g.loc[2, "home_win"] == False  # noqa: E712
    # ties never appear in oriented games
    assert not g.is_tie.any()
    # future game kept, no winner
    assert g.loc[7, "result_usable"] == False  # noqa: E712
    assert pd.isna(g.loc[7, "winner_id"])


def test_division_inference_from_other_rows(cleaned):
    g = cleaned["games"].set_index("game_id")
    assert g.loc[3, "home_division"] == "fcs"  # inferred via team-id lookup
    assert bool(g.loc[3, "home_division_inferred"])
    assert g.loc[3, "fbs_game"] == False  # noqa: E712


def test_team_games_long_format(cleaned):
    t = cleaned["team_games"]
    assert len(t) == 2 * len(cleaned["games"])
    one = t[t.game_id == 1].set_index("team_id")
    assert one.loc[10, "points_for"] == 31 and one.loc[10, "won"] == True  # noqa: E712
    assert one.loc[11, "margin_for"] == -17


def test_market_latest_and_bad_rows():
    keep, quar = clean_market_history(raw_market())
    assert len(keep) == 1 and keep.iloc[0].home_spread == -14.0  # latest observation wins
    assert len(quar) == 1 and quar.iloc[0].reason == "nonfinite_spread"


def raw_market():
    return pd.DataFrame(
        [
            dict(game_id=1, provider="p", observed_at="2025-08-29T00:00Z", home_spread=-13.5, source="s"),
            dict(game_id=1, provider="p", observed_at="2025-08-29T12:00Z", home_spread=-14.0, source="s"),
            dict(
                game_id=2, provider="p", observed_at="2025-09-01T00:00Z", home_spread=float("nan"), source="s"
            ),
        ]
    )


def test_report_written(tmp_path):
    raw_schedule().to_csv(tmp_path / "schedule_2025.csv", index=False)
    clean_all(tmp_path)
    report = json.loads((tmp_path / "clean" / "cleaning_report.json").read_text())
    assert report["files"]["schedule_2025.csv"]["raw_rows"] == 8
    assert report["quarantine_reasons"][REASON_ZERO_ZERO] == 1
    assert (tmp_path / "clean" / "games.parquet").exists()
    assert (tmp_path / "clean" / "team_games.parquet").exists()
    assert (tmp_path / "clean" / "quarantine.csv").exists()


def test_raw_files_never_modified(tmp_path):
    src = tmp_path / "schedule_2025.csv"
    raw_schedule().to_csv(src, index=False)
    before = src.read_bytes()
    clean_all(tmp_path)
    assert src.read_bytes() == before


def test_division_lookup_conflicts_dropped():
    df = raw_schedule()
    conflict = df.iloc[[0]].copy()
    conflict["away_division"] = "iii"
    frames = [df, conflict]
    lookup = build_division_lookup(frames)
    # Beta Tech maps to both fbs and iii across conflicting frames -> dropped
    assert lookup.get(11) is None
    assert lookup[10] == "fbs"

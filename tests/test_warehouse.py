import sqlite3

import pandas as pd

from cfb_forecast.data import load_data
from cfb_forecast.warehouse import load_data_from_database, update_database


def files(tmp_path):
    rows = [dict(game_id=i, season=2026, week=i, start_date=f"2026-09-{i + 5:02d}T16:00Z", completed=i < 3,
                 neutral_site=False, home_id=1, away_id=2, home_team="A", away_team="B",
                 home_points=30 if i < 3 else None, away_points=20 if i < 3 else None, fbs_game=True) for i in range(1, 5)]
    pd.DataFrame(rows).to_csv(tmp_path / "schedule_2026.csv", index=False)
    adv = [dict(game_id=i, pos_team_id=t, yards_per_play=5.0 + t, yards_per_pass=7.0, yards_per_rush=None if i == 1 else 4.0,
                first_downs_created_rate=0.3) for i in range(1, 3) for t in (1, 2)]
    # the real SportsDataverse file carries its own season column
    pd.DataFrame(adv).assign(season=2026).to_parquet(tmp_path / "adv_team_2026.parquet", index=False)


def test_database_round_trips_forecast_inputs_and_keeps_missing_rates_missing(tmp_path):
    files(tmp_path)
    db = tmp_path / "f.sqlite"
    summary = update_database(tmp_path, tmp_path / "art", 2026, db, collect=False)
    assert summary["games"] == 4 and summary["team_game_facts"] == 4
    schedules, adv, _ = load_data(tmp_path)
    s2, a2, paths = load_data(db)
    assert paths == [db]
    cols = ["game_id", "season", "week", "start_date", "completed", "home_id", "away_id", "home_points", "away_points", "result_usable", "result_available_at"]
    pd.testing.assert_frame_equal(schedules[cols].reset_index(drop=True), s2[cols].reset_index(drop=True), check_dtype=False)
    _, a2 = load_data_from_database(db)
    pd.testing.assert_frame_equal(adv.drop(columns=["season"]).sort_values(["game_id", "pos_team_id"]).reset_index(drop=True), a2, check_dtype=False)
    assert a2.loc[a2.game_id.eq(1), "yards_per_rush"].isna().all()


def test_update_is_idempotent_and_writes_both_team_rows(tmp_path):
    files(tmp_path)
    db = tmp_path / "f.sqlite"
    for _ in range(2):
        update_database(tmp_path, tmp_path / "art", 2026, db, collect=False)
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM games").fetchone()[0] == 4
        assert conn.execute("SELECT COUNT(*) FROM game_teams").fetchone()[0] == 8
        assert conn.execute("SELECT COUNT(*) FROM team_game_facts").fetchone()[0] == 4


def test_sources_and_validation_rows_are_stored(tmp_path, monkeypatch):
    from cfb_forecast import warehouse
    files(tmp_path)
    report = {"collected_at": "2026-10-15T00:00:00+00:00",
              "sources": [{"name": "cfbtrack", "url": "https://cfbtrack.com/", "status": "ok", "retrieved_at": "2026-10-15T00:00:00+00:00", "sha256": "ab", "bytes": 5},
                          {"name": "collegefootballdata", "url": "https://api.collegefootballdata.com", "status": "not_configured"}],
              "schedule_validation": [{"game_id": 3, "provider": "cfbtxt", "status": "match"}]}
    monkeypatch.setattr(warehouse, "collect_extended_sources", lambda games, season, root: report)
    db = tmp_path / "f.sqlite"
    summary = update_database(tmp_path, tmp_path / "art", 2026, db)
    assert (summary["sources"], summary["source_rows"]) == (2, 1)
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT status FROM source_retrievals WHERE name='collegefootballdata'").fetchone()[0] == "not_configured"
        assert conn.execute("SELECT source,game_id FROM source_rows").fetchone() == ("cfbtxt", 3)
        assert conn.execute("SELECT COUNT(*) FROM source_registry WHERE name='cfbtrack'").fetchone()[0] == 1
